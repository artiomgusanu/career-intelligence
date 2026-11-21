"""Reusable JSON-over-HTTP client for source adapters.

Guarantees:

- Explicit timeouts. Settings default to ``load_http_settings()``, so the
  ``HTTP_*`` environment variables control every client not given explicit
  settings.
- Limited retries, only for transient failures: timeouts, transport errors
  and HTTP 429/500/502/503/504. The wait before a retry is exponential
  backoff, or the server's ``Retry-After`` for 429/503 (seconds or HTTP-date,
  compared in UTC). A ``Retry-After`` above ``max_retry_after_seconds`` fails
  the request at once instead of waiting.
- Optional minimum interval between requests (``min_interval_seconds``),
  measured from the end of the previous request. Before each attempt the
  client waits once, for ``max(retry delay, remaining interval)``: waits are
  never added together.
- Redirects are not followed (a redirect could downgrade to HTTP and leak
  query-string credentials); a 3xx response is an ``HttpStatusError``.
- HTTPS is required unless ``allow_insecure_http=True``. A plain-HTTP client
  never carries credentials: a base URL with userinfo or sensitive query
  parameters is always refused, and so is a request with a sensitive
  parameter or a credential header.
- Errors are ``HttpClientError`` subclasses whose messages contain only the
  redacted URL. They are raised outside ``except`` blocks, so no underlying
  httpx exception (which holds the full URL) is chained to them.
- Only redacted URLs are logged; headers are never logged.

Concurrency: a client may be shared by several threads. Each attempt (the
wait before it, the request and recording its end) runs under an internal
lock, so requests through one client are serialized and the minimum interval
holds between any two of them. Separate clients, and separate processes, do
not coordinate: each enforces its own interval. Not designed for asyncio.

Lifecycle:

- ``close()`` takes the same lock, so it waits for the HTTP attempt in
  progress on another thread to finish, including that attempt's wait
  (interval, backoff or ``Retry-After``, the latter bounded by
  ``max_retry_after_seconds``). It does not wait for the whole ``get_json()``
  call: if ``close()`` runs between two attempts, the next attempt fails with
  ``HttpClientError("HTTP client is closed")``.
- Never call ``close()`` from inside a request on the same thread (for
  example from a transport, an httpx event hook or a logging handler invoked
  while a request runs): the lock is not reentrant and that call would
  deadlock.

Tests inject ``transport`` (an ``httpx.MockTransport``), ``sleep``, ``clock``
(monotonic seconds) and ``now`` (aware UTC datetime).
"""

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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
RETRY_AFTER_STATUS_CODES = frozenset({429, 503})

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


class HttpRetryAfterExceededError(HttpStatusError):
    """The server asked to wait longer than ``max_retry_after_seconds``."""

    def __init__(self, status_code: int, url: str, retry_after: float, limit: float) -> None:
        HttpClientError.__init__(
            self,
            f"GET {url} returned HTTP {status_code} with Retry-After {retry_after:.0f}s, "
            f"above the {limit:.0f}s limit",
            retryable=False,
        )
        self.status_code = status_code
        self.retry_after = retry_after
        self.limit = limit


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


@dataclass(frozen=True)
class _PendingRetry:
    delay: float  # requested by backoff or Retry-After, before the interval is applied
    source: str  # "backoff" or "retry-after"
    reason: str  # message of the failure being retried (redacted URL only)


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
        min_interval_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"}:
            raise ValueError("base_url must be an http(s) URL.")
        if not math.isfinite(min_interval_seconds) or min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be a finite number >= 0.")

        self._sensitive_params = SENSITIVE_PARAMS | {p.lower() for p in sensitive_params}
        headers = dict(headers or {})
        self._insecure = url.scheme == "http"

        if self._insecure:
            # Checked before allow_insecure_http: these are refused in every case.
            if url.username or url.password:
                raise InsecureTransportError(
                    "Refusing a plain-HTTP base_url that contains credentials."
                )
            if any(name.lower() in self._sensitive_params for name in url.params):
                raise InsecureTransportError(
                    "Refusing a plain-HTTP base_url with a credential query parameter."
                )
            if not allow_insecure_http:
                raise ValueError(
                    "base_url must use HTTPS (pass allow_insecure_http=True to override)."
                )
            if any(name.lower() in SENSITIVE_HEADERS for name in headers):
                raise InsecureTransportError(
                    "Refusing to send credential headers over plain HTTP."
                )

        # Without explicit settings, HTTP_* environment variables apply.
        self._settings = settings if settings is not None else load_http_settings()
        self._min_interval = min_interval_seconds
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._lock = threading.Lock()
        self._last_finished: float | None = None
        self._closed = False

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
        """Close the underlying connection pool. Safe to call more than once.

        Waits for the HTTP attempt in progress on another thread to finish, not
        for its whole ``get_json()`` call. Must not be called from inside a
        request on the same thread (see module docs).
        """
        with self._lock:
            if not self._closed:
                self._client.close()
                self._closed = True

    def get_json(self, path: str, params: QueryParams | None = None) -> Any:
        """GET ``path`` and return the decoded JSON body. ``None`` params are omitted."""
        query = {name: value for name, value in (params or {}).items() if value is not None}

        if self._insecure and any(name.lower() in self._sensitive_params for name in query):
            raise InsecureTransportError("Refusing to send a credential over plain HTTP.")

        request = self._client.build_request("GET", path, params=query)
        safe_url = redact_url(request.url, self._sensitive_params)
        attempts = self._settings.max_retries + 1
        retry: _PendingRetry | None = None

        for attempt in range(1, attempts + 1):
            response, failure, retry_after = self._attempt(
                request, safe_url, attempt, attempts, retry
            )
            if response is not None:
                return self._decode_json(response, safe_url)

            assert failure is not None
            if not failure.retryable or attempt == attempts:
                raise failure

            if retry_after is not None:
                limit = self._settings.max_retry_after_seconds
                if retry_after > limit:
                    assert isinstance(failure, HttpStatusError)
                    raise HttpRetryAfterExceededError(
                        failure.status_code, safe_url, retry_after, limit
                    )
                retry = _PendingRetry(retry_after, "retry-after", str(failure))
            else:
                backoff = self._settings.backoff_seconds * 2 ** (attempt - 1)
                retry = _PendingRetry(backoff, "backoff", str(failure))

        raise AssertionError("unreachable")  # pragma: no cover

    def _attempt(
        self,
        request: httpx.Request,
        safe_url: str,
        attempt: int,
        attempts: int,
        retry: _PendingRetry | None,
    ) -> tuple[httpx.Response | None, HttpClientError | None, float | None]:
        """Wait once for max(retry delay, remaining interval), then send. Serialized."""
        with self._lock:
            if self._closed:
                raise HttpClientError("HTTP client is closed")

            requested = retry.delay if retry is not None else 0.0
            wait = max(requested, self._remaining_interval())
            if retry is not None:
                # Logged here, where the effective wait is known.
                logger.warning(
                    "http_retry method=GET url=%s attempt=%d/%d delay_s=%.2f delay_source=%s "
                    "requested_delay_s=%.2f reason=%r",
                    safe_url,
                    attempt - 1,
                    attempts,
                    wait,
                    retry.source,
                    requested,
                    retry.reason,
                )
            elif wait > 0:
                logger.debug("http_wait url=%s wait_s=%.2f", safe_url, wait)
            if wait > 0:
                self._sleep(wait)

            try:
                return self._send(request, safe_url, attempt)
            finally:
                self._last_finished = self._clock()

    def _remaining_interval(self) -> float:
        if self._min_interval == 0 or self._last_finished is None:
            return 0.0
        return max(0.0, self._min_interval - (self._clock() - self._last_finished))

    def _send(
        self, request: httpx.Request, safe_url: str, attempt: int
    ) -> tuple[httpx.Response | None, HttpClientError | None, float | None]:
        logger.debug("http_request method=GET url=%s attempt=%d", safe_url, attempt)
        started = self._clock()

        try:
            response = self._client.send(request)
        except httpx.TimeoutException:
            return None, HttpTimeoutError(safe_url), None
        except httpx.TransportError as exc:
            return None, HttpTransportError(safe_url, type(exc).__name__), None

        elapsed_ms = (self._clock() - started) * 1000
        logger.info(
            "http_response method=GET url=%s status=%d attempt=%d elapsed_ms=%.0f",
            safe_url,
            response.status_code,
            attempt,
            elapsed_ms,
        )

        if response.is_success:
            return response, None, None

        retry_after = None
        if response.status_code in RETRY_AFTER_STATUS_CODES:
            retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
        return None, HttpStatusError(response.status_code, safe_url), retry_after

    def _parse_retry_after(self, value: str | None) -> float | None:
        """Seconds to wait per ``Retry-After``, or None if absent or invalid.

        Accepts delay-seconds (digits only) or an HTTP-date. Dates are compared
        in UTC; a date without zone information (``-0000``) is taken as UTC and
        a date in the past means no wait.
        """
        if value is None:
            return None
        value = value.strip()
        if value.isascii() and value.isdigit():
            return float(value)

        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)

        current = self._now()
        if current.tzinfo is None:
            raise ValueError("now() must return a timezone-aware datetime.")
        return max(0.0, (when - current.astimezone(UTC)).total_seconds())

    @staticmethod
    def _decode_json(response: httpx.Response, safe_url: str) -> Any:
        try:
            return response.json()
        except ValueError:
            pass
        raise InvalidJsonError(safe_url)
