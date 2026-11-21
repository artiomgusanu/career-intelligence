"""Reusable JSON-over-HTTP client for source adapters.

Guarantees:

- Explicit timeouts. Settings default to ``load_http_settings()``, so the
  ``HTTP_TIMEOUT_SECONDS``, ``HTTP_MAX_RETRIES`` and ``HTTP_BACKOFF_SECONDS``
  environment variables control every client not given explicit settings.
- Limited retries with exponential backoff, only for transient failures:
  timeouts, transport errors and HTTP 429/500/502/503/504.
- Redirects are not followed (a redirect could downgrade to HTTP and leak
  query-string credentials); a 3xx response is an ``HttpStatusError``.
- HTTPS is required unless ``allow_insecure_http=True``, and even then a
  request carrying a credential (sensitive query parameter or auth header)
  is refused before anything is sent.
- Errors are ``HttpClientError`` subclasses whose messages contain only the
  redacted URL. They are raised outside ``except`` blocks, so no underlying
  httpx exception (which holds the full URL) is chained to them.
- Only redacted URLs are logged; headers are never logged.

Tests inject an ``httpx.MockTransport`` through ``transport``.
"""

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Self

import httpx

from career_intelligence.config import HttpSettings, load_http_settings
from career_intelligence.ingestion.contracts import IngestionError
from career_intelligence.logging_config import (
    SENSITIVE_HEADERS,
    SENSITIVE_PARAMS,
    protect_third_party_loggers,
    redact_url,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

type QueryParams = Mapping[str, str | int | float | None]


class HttpClientError(IngestionError):
    """Base class for client failures. Messages never contain credentials."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class HttpStatusError(HttpClientError):
    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(
            f"GET {url} returned HTTP {status_code}",
            retryable=status_code in RETRYABLE_STATUS_CODES,
        )
        self.status_code = status_code


class HttpTimeoutError(HttpClientError):
    def __init__(self, url: str) -> None:
        super().__init__(f"GET {url} timed out", retryable=True)


class HttpTransportError(HttpClientError):
    def __init__(self, url: str, error_type: str) -> None:
        super().__init__(f"GET {url} failed: {error_type}", retryable=True)


class InvalidJsonError(HttpClientError):
    def __init__(self, url: str) -> None:
        super().__init__(f"GET {url} returned a body that is not valid JSON")


class InsecureTransportError(HttpClientError):
    """Raised instead of sending a credential over plain HTTP."""


class HttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        settings: HttpSettings | None = None,
        headers: Mapping[str, str] | None = None,
        sensitive_params: frozenset[str] = frozenset(),
        transport: httpx.BaseTransport | None = None,
        allow_insecure_http: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        scheme = httpx.URL(base_url).scheme
        if scheme not in {"http", "https"}:
            raise ValueError("base_url must be an http(s) URL.")
        if scheme == "http" and not allow_insecure_http:
            raise ValueError("base_url must use HTTPS (pass allow_insecure_http=True to override).")

        headers = dict(headers or {})
        self._insecure = scheme == "http"
        if self._insecure and any(name.lower() in SENSITIVE_HEADERS for name in headers):
            raise InsecureTransportError("Refusing to send credential headers over plain HTTP.")

        # Without explicit settings, HTTP_* environment variables apply.
        self._settings = settings if settings is not None else load_http_settings()
        self._sensitive_params = SENSITIVE_PARAMS | {p.lower() for p in sensitive_params}
        self._sleep = sleep

        protect_third_party_loggers()
        self._client = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": self._settings.user_agent, **headers},
            timeout=httpx.Timeout(self._settings.timeout_seconds),
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_json(self, path: str, params: QueryParams | None = None) -> Any:
        """GET ``path`` and return the decoded JSON body. ``None`` params are omitted."""
        query = {name: value for name, value in (params or {}).items() if value is not None}

        if self._insecure and any(name.lower() in self._sensitive_params for name in query):
            raise InsecureTransportError("Refusing to send a credential over plain HTTP.")

        request = self._client.build_request("GET", path, params=query)
        safe_url = redact_url(request.url, self._sensitive_params)
        attempts = self._settings.max_retries + 1

        for attempt in range(1, attempts + 1):
            response, failure = self._send(request, safe_url, attempt)
            if response is not None:
                return self._decode_json(response, safe_url)

            assert failure is not None
            if not failure.retryable or attempt == attempts:
                raise failure

            delay = self._settings.backoff_seconds * 2 ** (attempt - 1)
            logger.warning(
                "http_retry method=GET url=%s attempt=%d/%d delay_s=%.2f reason=%r",
                safe_url,
                attempt,
                attempts,
                delay,
                str(failure),
            )
            self._sleep(delay)

        raise AssertionError("unreachable")  # pragma: no cover

    def _send(
        self, request: httpx.Request, safe_url: str, attempt: int
    ) -> tuple[httpx.Response | None, HttpClientError | None]:
        logger.debug("http_request method=GET url=%s attempt=%d", safe_url, attempt)
        started = time.monotonic()

        try:
            response = self._client.send(request)
        except httpx.TimeoutException:
            return None, HttpTimeoutError(safe_url)
        except httpx.TransportError as exc:
            return None, HttpTransportError(safe_url, type(exc).__name__)

        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info(
            "http_response method=GET url=%s status=%d attempt=%d elapsed_ms=%.0f",
            safe_url,
            response.status_code,
            attempt,
            elapsed_ms,
        )

        if response.is_success:
            return response, None
        return None, HttpStatusError(response.status_code, safe_url)

    @staticmethod
    def _decode_json(response: httpx.Response, safe_url: str) -> Any:
        try:
            return response.json()
        except ValueError:
            pass
        raise InvalidJsonError(safe_url)
