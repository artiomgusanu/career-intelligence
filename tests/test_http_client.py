import json
import logging
import traceback
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from career_intelligence.config import ConfigurationError, HttpSettings
from career_intelligence.ingestion.http import (
    HttpClient,
    HttpClientError,
    HttpStatusError,
    HttpTimeoutError,
    HttpTransportError,
    InsecureTransportError,
    InvalidJsonError,
)

BASE_URL = "https://api.example.com"
SECRET = "sk_live_DO_NOT_LEAK_123"
FAST = HttpSettings(timeout_seconds=1.0, max_retries=2, backoff_seconds=0.1)

type Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """MockTransport handler that replays scripted outcomes and records requests."""

    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(
    handler: Handler,
    *,
    settings: HttpSettings = FAST,
    sleeps: list[float] | None = None,
    base_url: str = BASE_URL,
    **kwargs: Any,
) -> HttpClient:
    recorded = sleeps if sleeps is not None else []
    return HttpClient(
        base_url,
        settings=settings,
        transport=httpx.MockTransport(handler),
        sleep=recorded.append,
        **kwargs,
    )


def ok(payload: object) -> httpx.Response:
    return httpx.Response(200, json=payload)


# --- Success path --------------------------------------------------------------


def test_get_json_returns_payload_and_sends_params() -> None:
    recorder = Recorder(ok({"results": [1, 2]}))
    with make_client(recorder) as client:
        payload = client.get_json("/job/list.json", {"page": 2, "limit": 50, "company": None})

    assert payload == {"results": [1, 2]}
    request = recorder.requests[0]
    assert request.method == "GET"
    assert request.url.path == "/job/list.json"
    assert dict(request.url.params) == {"page": "2", "limit": "50"}  # None omitted
    assert request.headers["User-Agent"] == FAST.user_agent


def test_environment_settings_control_a_client_without_explicit_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("HTTP_MAX_RETRIES", "1")
    monkeypatch.setenv("HTTP_BACKOFF_SECONDS", "0.25")
    recorder = Recorder(httpx.Response(503))
    sleeps: list[float] = []
    client = HttpClient(BASE_URL, transport=httpx.MockTransport(recorder), sleep=sleeps.append)

    with client, pytest.raises(HttpStatusError):
        client.get_json("/x")

    assert len(recorder.requests) == 2  # HTTP_MAX_RETRIES=1
    assert sleeps == [0.25]  # HTTP_BACKOFF_SECONDS
    assert recorder.requests[0].extensions["timeout"]["read"] == 4  # HTTP_TIMEOUT_SECONDS


def test_invalid_environment_settings_fail_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_MAX_RETRIES", "99")

    with pytest.raises(ConfigurationError):
        HttpClient(BASE_URL)


def test_explicit_settings_take_precedence_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_MAX_RETRIES", "5")
    recorder = Recorder(httpx.Response(503))

    with (
        make_client(recorder, settings=HttpSettings(max_retries=0)) as client,
        pytest.raises(HttpStatusError),
    ):
        client.get_json("/x")

    assert len(recorder.requests) == 1


def test_timeout_is_explicit_on_requests() -> None:
    recorder = Recorder(ok({}))
    with make_client(recorder, settings=HttpSettings(timeout_seconds=2.5)) as client:
        client.get_json("/x")

    assert recorder.requests[0].extensions["timeout"] == {
        "connect": 2.5,
        "read": 2.5,
        "write": 2.5,
        "pool": 2.5,
    }


# --- HTTP errors and retries ---------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    recorder = Recorder(httpx.Response(status))
    sleeps: list[float] = []

    with make_client(recorder, sleeps=sleeps) as client, pytest.raises(HttpStatusError) as exc:
        client.get_json("/x")

    assert exc.value.status_code == status
    assert exc.value.retryable is False
    assert len(recorder.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_is_retried_then_succeeds(status: int) -> None:
    recorder = Recorder(httpx.Response(status), httpx.Response(status), ok({"ok": True}))
    sleeps: list[float] = []

    with make_client(recorder, sleeps=sleeps) as client:
        assert client.get_json("/x") == {"ok": True}

    assert len(recorder.requests) == 3
    assert sleeps == [0.1, 0.2]  # exponential backoff


def test_retries_are_limited_then_last_error_is_raised() -> None:
    recorder = Recorder(httpx.Response(503))
    sleeps: list[float] = []

    with make_client(recorder, sleeps=sleeps) as client, pytest.raises(HttpStatusError) as exc:
        client.get_json("/x")

    assert exc.value.status_code == 503
    assert len(recorder.requests) == FAST.max_retries + 1
    assert len(sleeps) == FAST.max_retries


def test_zero_retries_means_single_attempt() -> None:
    recorder = Recorder(httpx.Response(503))
    settings = HttpSettings(max_retries=0)

    with make_client(recorder, settings=settings) as client, pytest.raises(HttpStatusError):
        client.get_json("/x")

    assert len(recorder.requests) == 1


def test_redirects_are_not_followed() -> None:
    recorder = Recorder(httpx.Response(302, headers={"Location": "http://elsewhere.example/"}))

    with make_client(recorder) as client, pytest.raises(HttpStatusError) as exc:
        client.get_json("/x")

    assert exc.value.status_code == 302
    assert len(recorder.requests) == 1


# --- Timeouts, transport errors, invalid JSON ------------------------------------


def test_timeout_is_retried_then_raised() -> None:
    recorder = Recorder(httpx.ReadTimeout("read timed out"))

    with make_client(recorder) as client, pytest.raises(HttpTimeoutError):
        client.get_json("/x")

    assert len(recorder.requests) == FAST.max_retries + 1


def test_timeout_then_success() -> None:
    recorder = Recorder(httpx.ConnectTimeout("connect timed out"), ok([1]))

    with make_client(recorder) as client:
        assert client.get_json("/x") == [1]


def test_transport_error_is_retried_then_raised() -> None:
    recorder = Recorder(httpx.ConnectError("connection refused"))

    with make_client(recorder) as client, pytest.raises(HttpTransportError) as exc:
        client.get_json("/x")

    assert "ConnectError" in str(exc.value)
    assert len(recorder.requests) == FAST.max_retries + 1


@pytest.mark.parametrize("body", [b"<html>maintenance</html>", b"", b"{'single': 'quotes'}"])
def test_invalid_json_raises_without_retry(body: bytes) -> None:
    recorder = Recorder(httpx.Response(200, content=body))

    with make_client(recorder) as client, pytest.raises(InvalidJsonError):
        client.get_json("/x")

    assert len(recorder.requests) == 1


# --- Transport security ------------------------------------------------------------


def test_plain_http_base_url_is_rejected_by_default() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        HttpClient("http://api.example.com")


def test_non_http_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        HttpClient("ftp://api.example.com")


def test_insecure_http_allowed_only_without_credentials() -> None:
    recorder = Recorder(ok({"ok": True}))
    with make_client(
        recorder, base_url="http://sandbox.example.com", allow_insecure_http=True
    ) as client:
        assert client.get_json("/x", {"page": 1}) == {"ok": True}

        with pytest.raises(InsecureTransportError):
            client.get_json("/x", {"api_key": SECRET})

    assert len(recorder.requests) == 1  # the credentialed request was never sent


def test_insecure_http_refuses_credential_headers() -> None:
    with pytest.raises(InsecureTransportError):
        make_client(
            Recorder(ok({})),
            base_url="http://sandbox.example.com",
            allow_insecure_http=True,
            headers={"Authorization": f"Bearer {SECRET}"},
        )


# --- Credentials never leak --------------------------------------------------------

LEAK_SCENARIOS = {
    "success": Recorder(ok({"ok": True})),
    "http_404": Recorder(httpx.Response(404, text=f"unknown key {SECRET}")),
    "http_503_retried": Recorder(httpx.Response(503)),
    "timeout": Recorder(httpx.ReadTimeout(f"timed out for ?api_key={SECRET}")),
    "transport": Recorder(httpx.ConnectError(f"failed ?api_key={SECRET}")),
    "invalid_json": Recorder(httpx.Response(200, content=b"not json")),
}


@pytest.mark.parametrize("scenario", sorted(LEAK_SCENARIOS))
def test_credentials_never_appear_in_logs_or_exceptions(
    scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = LEAK_SCENARIOS[scenario]
    recorder.requests.clear()
    caught: HttpClientError | None = None

    with caplog.at_level(logging.DEBUG):
        client = make_client(
            recorder,
            headers={"Authorization": f"Bearer {SECRET}", "X-Api-Key": SECRET},
            sensitive_params=frozenset({"api_key"}),
        )
        try:
            client.get_json("/job/list.json", {"api_key": SECRET, "page": 1})
        except HttpClientError as exc:
            caught = exc
        finally:
            client.close()

    # The credentials were really sent...
    sent = recorder.requests[0]
    assert sent.url.params["api_key"] == SECRET
    assert sent.headers["Authorization"] == f"Bearer {SECRET}"
    # ...but appear nowhere in logs (including httpx's own request log line).
    assert SECRET not in caplog.text
    assert "api_key=REDACTED" in caplog.text
    if caught is not None:
        assert SECRET not in str(caught)
        assert caught.__cause__ is None and caught.__context__ is None
        assert SECRET not in "".join(traceback.format_exception(caught))


def test_logs_are_key_value_and_include_status() -> None:
    recorder = Recorder(ok({}))
    handler_records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            handler_records.append(record)

    logger = logging.getLogger("career_intelligence.ingestion.http")
    collect = Collect(level=logging.DEBUG)
    logger.addHandler(collect)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with make_client(recorder) as client:
            client.get_json("/x", {"page": 3})
    finally:
        logger.removeHandler(collect)
        logger.setLevel(previous)

    messages = [record.getMessage() for record in handler_records]
    assert any(m.startswith("http_request method=GET") for m in messages)
    response_line = next(m for m in messages if m.startswith("http_response"))
    assert "status=200" in response_line and "page=3" in response_line


def test_json_payload_types_are_preserved() -> None:
    payload = {"total": 2, "results": [{"id": 1}, {"id": 2}], "query": None}
    recorder = Recorder(httpx.Response(200, content=json.dumps(payload).encode()))

    with make_client(recorder) as client:
        assert client.get_json("/x") == payload
