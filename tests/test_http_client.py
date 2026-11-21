import faulthandler
import json
import logging
import os
import re
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from email.utils import format_datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest

from career_intelligence.config import ConfigurationError, HttpSettings
from career_intelligence.ingestion.contracts import IngestionError
from career_intelligence.ingestion.http import (
    HttpClient,
    HttpClientError,
    HttpRetryAfterExceededError,
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


# =====================================================================================
# 2.0B-1: lifecycle, minimum interval, Retry-After, base URL security
# =====================================================================================

WALL_START = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)

# Upper bound for any operation that could block on the client lock.
DEADLINE_SECONDS = 5.0
# Hard stop for a whole test in this module: dumps all thread stacks to the real
# stderr and ends the process, so a deadlock anywhere fails the run (non-zero exit,
# visible traceback) instead of hanging it. Overridable only to test the watchdog.
WATCHDOG_SECONDS = float(os.environ.get("CAREER_INTELLIGENCE_TEST_WATCHDOG_SECONDS", "60"))


@pytest.fixture(autouse=True)
def deadlock_watchdog(real_stderr_fd: int) -> Iterator[None]:
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True, file=real_stderr_fd)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


INNER_DEADLOCK_TEST = """\
import threading

from test_http_client import deadlock_watchdog  # noqa: F401 - the fixture under test


def test_simulated_deadlock():
    lock = threading.Lock()
    lock.acquire()
    lock.acquire()  # blocks forever
"""


def test_watchdog_turns_a_deadlock_into_a_visible_failure(tmp_path: Path) -> None:
    """A deadlocked test under normal output capture must exit non-zero and show
    the traceback with the test name and the blocking line."""
    inner = tmp_path / "test_inner_deadlock.py"
    inner.write_text(INNER_DEADLOCK_TEST, encoding="utf-8")
    blocking_line = INNER_DEADLOCK_TEST.splitlines().index(
        "    lock.acquire()  # blocks forever"
    ) + 1
    tests_dir = Path(__file__).resolve().parent
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(tests_dir), os.environ.get("PYTHONPATH")])),
        "CAREER_INTELLIGENCE_TEST_WATCHDOG_SECONDS": "2",
    }

    completed = subprocess.run(
        # Default capture (no -s). The real tests/conftest.py is loaded as a plugin so
        # the inner run uses the same real_stderr_fd fixture as this suite.
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "conftest", "-q",
         inner.name],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "Timeout (0:00:02)!" in output
    assert re.search(
        rf'File ".*{inner.name}", line {blocking_line} in test_simulated_deadlock', output
    ), output


class FakeTime:
    """Deterministic time: the monotonic clock only moves through sleep() or advance().

    The wall clock (for HTTP-date comparisons) follows the monotonic clock, so a
    simulated sleep also moves "now".
    """

    def __init__(self) -> None:
        self.monotonic = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.monotonic

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.monotonic += seconds

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds

    def now(self) -> datetime:
        return WALL_START + timedelta(seconds=self.monotonic - 1000.0)


class TimedRecorder(Recorder):
    """Recorder that notes the fake clock when each request starts and simulates latency."""

    def __init__(
        self, fake: FakeTime, *outcomes: httpx.Response | Exception, latency: float = 0.0
    ) -> None:
        super().__init__(*outcomes)
        self.fake = fake
        self.latency = latency
        self.sent_at: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent_at.append(self.fake.clock())
        self.fake.advance(self.latency)
        return super().__call__(request)


def timed_client(
    recorder: TimedRecorder,
    *,
    min_interval: float = 0.0,
    settings: HttpSettings = FAST,
    **kwargs: Any,
) -> HttpClient:
    fake = recorder.fake
    return HttpClient(
        BASE_URL,
        settings=settings,
        transport=httpx.MockTransport(recorder),
        min_interval_seconds=min_interval,
        sleep=fake.sleep,
        clock=fake.clock,
        now=fake.now,
        **kwargs,
    )


def gaps(times: list[float]) -> list[float]:
    return [round(b - a, 6) for a, b in pairwise(times)]


def retry_after(status: int, value: str) -> httpx.Response:
    return httpx.Response(status, headers={"Retry-After": value})


# --- Lifecycle -------------------------------------------------------------------------


def test_context_manager_closes_the_client_and_blocks_further_requests() -> None:
    recorder = Recorder(ok({}))
    with make_client(recorder) as client:
        client.get_json("/x")

    with pytest.raises(HttpClientError, match="closed") as exc_info:
        client.get_json("/x")

    assert isinstance(exc_info.value, IngestionError)
    assert len(recorder.requests) == 1  # nothing sent after close


def test_context_manager_closes_even_when_the_block_raises() -> None:
    recorder = Recorder(ok({}))

    with pytest.raises(RuntimeError, match="boom"), make_client(recorder) as client:
        raise RuntimeError("boom")

    with pytest.raises(HttpClientError, match="closed"):
        client.get_json("/x")
    assert recorder.requests == []


def test_close_is_idempotent() -> None:
    client = make_client(Recorder(ok({})))

    client.close()
    client.close()

    with pytest.raises(HttpClientError, match="closed"):
        client.get_json("/x")


# --- Minimum interval --------------------------------------------------------------------


def test_first_request_never_waits() -> None:
    fake = FakeTime()
    with timed_client(TimedRecorder(fake, ok({})), min_interval=1.0) as client:
        client.get_json("/x")

    assert fake.sleeps == []


def test_consecutive_requests_are_spaced_by_the_minimum_interval() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, ok({}))

    with timed_client(recorder, min_interval=1.0) as client:
        for _ in range(3):
            client.get_json("/x")

    assert fake.sleeps == [1.0, 1.0]
    assert gaps(recorder.sent_at) == [1.0, 1.0]


def test_interval_is_measured_from_the_end_of_the_previous_request() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, ok({}), latency=0.4)

    with timed_client(recorder, min_interval=1.0) as client:
        for _ in range(3):
            client.get_json("/x")

    # Latency does not count towards the interval: a full 1.0s wait after each end.
    assert fake.sleeps == [1.0, 1.0]
    assert gaps(recorder.sent_at) == [pytest.approx(1.4), pytest.approx(1.4)]


def test_only_the_remaining_interval_is_waited() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, ok({}))

    with timed_client(recorder, min_interval=1.0) as client:
        client.get_json("/x")
        fake.advance(0.4)  # caller work between requests
        client.get_json("/x")

    assert fake.sleeps == [pytest.approx(0.6)]
    assert gaps(recorder.sent_at) == [pytest.approx(1.0)]


def test_no_wait_when_enough_time_passed_since_the_previous_request() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, ok({}))

    with timed_client(recorder, min_interval=1.0) as client:
        client.get_json("/x")
        fake.advance(1.5)
        client.get_json("/x")

    assert fake.sleeps == []


def test_zero_interval_never_sleeps() -> None:
    fake = FakeTime()

    with timed_client(TimedRecorder(fake, ok({})), min_interval=0.0) as client:
        for _ in range(3):
            client.get_json("/x")

    assert fake.sleeps == []


def test_interval_also_follows_a_failed_request() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, httpx.Response(404), ok({}))

    with timed_client(recorder, min_interval=1.0) as client:
        with pytest.raises(HttpStatusError):
            client.get_json("/missing")
        client.get_json("/x")

    assert fake.sleeps == [1.0]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_min_interval_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="min_interval_seconds"):
        HttpClient(BASE_URL, settings=FAST, min_interval_seconds=value)


def test_interval_holds_across_threads_sharing_one_client() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, ok({}), latency=0.25)
    client = timed_client(recorder, min_interval=1.0)
    start = threading.Barrier(2)

    def worker() -> None:
        start.wait(timeout=5)
        for _ in range(3):
            client.get_json("/x")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=DEADLINE_SECONDS)
    assert not any(thread.is_alive() for thread in threads), "workers stuck: possible deadlock"
    client.close()

    assert len(recorder.sent_at) == 6
    # Serialized by the client: each request starts a full interval after the previous
    # one ended (0.25s latency + 1.0s interval), whichever thread sent it.
    assert gaps(sorted(recorder.sent_at)) == [pytest.approx(1.25)] * 5
    assert fake.sleeps == [1.0] * 5


# --- Lock release on exception paths (M1) ---------------------------------------------------
#
# Every potentially blocking call runs in a daemon thread with a deadline, so a lock
# that is never released fails the test instead of hanging the suite.


class SleepInterrupted(Exception):
    pass


def run_with_deadline(operation: Callable[[], Any]) -> dict[str, Any]:
    """Run ``operation`` in a daemon thread; fail if it does not finish in time."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = operation()
        except BaseException as exc:  # noqa: BLE001 - captured and asserted by the caller
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=DEADLINE_SECONDS)
    assert not thread.is_alive(), "operation did not finish: lock never released (deadlock)"
    return outcome


def assert_lock_released(client: HttpClient) -> None:
    acquired = client._lock.acquire(timeout=DEADLINE_SECONDS)
    assert acquired, "client lock is still held after the exception"
    client._lock.release()


def test_lock_is_released_when_the_transport_raises() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, RuntimeError("transport bug"), ok({"ok": True}))
    client = timed_client(recorder, min_interval=1.0)

    first = run_with_deadline(lambda: client.get_json("/x"))
    assert isinstance(first.get("error"), RuntimeError)
    assert_lock_released(client)

    # A later request (from another thread) proceeds normally, and the failed
    # attempt still counted for the interval (its end was recorded).
    second = run_with_deadline(lambda: client.get_json("/x"))
    assert second == {"value": {"ok": True}}
    assert fake.sleeps == [1.0]
    run_with_deadline(client.close)


def test_lock_is_released_when_sleep_raises() -> None:
    fake = FakeTime()
    calls = {"count": 0}

    def flaky_sleep(seconds: float) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise SleepInterrupted("interrupted while waiting")
        fake.sleep(seconds)

    recorder = TimedRecorder(fake, ok({"ok": True}))
    client = HttpClient(
        BASE_URL,
        settings=FAST,
        transport=httpx.MockTransport(recorder),
        min_interval_seconds=1.0,
        sleep=flaky_sleep,
        clock=fake.clock,
        now=fake.now,
    )

    assert run_with_deadline(lambda: client.get_json("/x")) == {"value": {"ok": True}}
    interrupted = run_with_deadline(lambda: client.get_json("/x"))
    assert isinstance(interrupted.get("error"), SleepInterrupted)
    assert len(recorder.requests) == 1  # nothing sent by the interrupted attempt
    assert_lock_released(client)

    assert run_with_deadline(lambda: client.get_json("/x")) == {"value": {"ok": True}}
    assert len(recorder.requests) == 2
    run_with_deadline(client.close)


def test_lock_is_released_when_now_lacks_a_timezone() -> None:
    header = format_datetime(WALL_START + timedelta(seconds=30), usegmt=True)
    recorder = Recorder(retry_after(429, header), ok({"ok": True}))
    client = HttpClient(
        BASE_URL,
        settings=FAST,
        transport=httpx.MockTransport(recorder),
        sleep=lambda _: None,
        now=lambda: datetime(2026, 11, 22, 12, 0),  # noqa: DTZ001 - deliberately naive
    )

    failed = run_with_deadline(lambda: client.get_json("/x"))
    assert isinstance(failed.get("error"), ValueError)
    assert "timezone-aware" in str(failed["error"])
    assert_lock_released(client)

    assert run_with_deadline(lambda: client.get_json("/x")) == {"value": {"ok": True}}
    run_with_deadline(client.close)


# --- Single wait: max(retry delay, remaining interval) ------------------------------------


@pytest.mark.parametrize(
    ("backoff", "interval", "expected_wait"),
    [(0.1, 1.0, 1.0), (2.0, 1.0, 2.0), (1.0, 1.0, 1.0)],
)
def test_retry_waits_once_for_the_larger_of_backoff_and_interval(
    backoff: float, interval: float, expected_wait: float
) -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, httpx.Response(503), ok({}))
    settings = HttpSettings(max_retries=1, backoff_seconds=backoff)

    with timed_client(recorder, min_interval=interval, settings=settings) as client:
        assert client.get_json("/x") == {}

    assert fake.sleeps == [expected_wait]  # one wait, never backoff + interval
    assert gaps(recorder.sent_at) == [expected_wait]


RETRY_LOG = re.compile(
    r"^http_retry .* delay_s=(?P<delay>[\d.]+) delay_source=(?P<source>\S+) "
    r"requested_delay_s=(?P<requested>[\d.]+) "
)


def retry_log_entries(caplog: pytest.LogCaptureFixture) -> list[re.Match[str]]:
    matches = [RETRY_LOG.match(record.getMessage()) for record in caplog.records]
    return [match for match in matches if match]


@pytest.mark.parametrize(
    ("first_response", "backoff", "interval", "expected"),
    [
        # interval dominates the backoff: logged delay is the interval, source preserved
        (httpx.Response(503), 0.1, 1.0, ("1.00", "backoff", "0.10")),
        # backoff dominates
        (httpx.Response(503), 2.0, 1.0, ("2.00", "backoff", "2.00")),
        # interval dominates a short Retry-After
        (retry_after(429, "0"), 0.1, 1.0, ("1.00", "retry-after", "0.00")),
        # Retry-After dominates
        (retry_after(429, "3"), 0.1, 1.0, ("3.00", "retry-after", "3.00")),
    ],
    ids=["interval>backoff", "backoff>interval", "interval>retry-after", "retry-after>interval"],
)
def test_retry_log_reports_the_effective_wait(
    first_response: httpx.Response,
    backoff: float,
    interval: float,
    expected: tuple[str, str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, first_response, ok({}))
    settings = HttpSettings(max_retries=1, backoff_seconds=backoff)

    with (
        caplog.at_level(logging.WARNING, logger="career_intelligence.ingestion.http"),
        timed_client(recorder, min_interval=interval, settings=settings) as client,
    ):
        client.get_json("/x")

    entries = retry_log_entries(caplog)
    assert len(entries) == 1
    logged = entries[0]
    assert (logged["delay"], logged["source"], logged["requested"]) == expected
    # The logged delay is exactly the wait that was applied.
    assert fake.sleeps == [float(logged["delay"])]


@pytest.mark.parametrize(("header", "expected_wait"), [("3", 3.0), ("0", 1.0)])
def test_retry_after_and_interval_are_not_added(header: str, expected_wait: float) -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(429, header), ok({}))

    with timed_client(recorder, min_interval=1.0) as client:
        client.get_json("/x")

    assert fake.sleeps == [expected_wait]


# --- Retry-After ------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_seconds_replaces_backoff(status: int) -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(status, "7"), ok({"ok": True}))

    with timed_client(recorder) as client:
        assert client.get_json("/x") == {"ok": True}

    assert fake.sleeps == [7.0]
    assert len(recorder.requests) == 2


def test_retry_after_http_date_is_compared_in_utc() -> None:
    fake = FakeTime()
    header = format_datetime(fake.now() + timedelta(seconds=30), usegmt=True)
    recorder = TimedRecorder(fake, retry_after(503, header), ok({}))

    with timed_client(recorder) as client:
        client.get_json("/x")

    assert fake.sleeps == [30.0]


def test_retry_after_http_date_with_another_offset_is_normalized_to_utc() -> None:
    fake = FakeTime()
    plus_one = timezone(timedelta(hours=1))
    header = format_datetime((fake.now() + timedelta(seconds=45)).astimezone(plus_one))
    assert "+0100" in header
    recorder = TimedRecorder(fake, retry_after(429, header), ok({}))

    with timed_client(recorder) as client:
        client.get_json("/x")

    assert fake.sleeps == [45.0]


def test_retry_after_http_date_without_zone_is_taken_as_utc() -> None:
    fake = FakeTime()
    header = (fake.now() + timedelta(seconds=20)).strftime("%a, %d %b %Y %H:%M:%S -0000")
    recorder = TimedRecorder(fake, retry_after(429, header), ok({}))

    with timed_client(recorder) as client:
        client.get_json("/x")

    assert fake.sleeps == [20.0]


def test_retry_after_date_in_the_past_retries_without_waiting() -> None:
    fake = FakeTime()
    header = format_datetime(fake.now() - timedelta(minutes=5), usegmt=True)
    recorder = TimedRecorder(fake, retry_after(503, header), ok({}))

    with timed_client(recorder) as client:
        client.get_json("/x")

    assert fake.sleeps == []
    assert len(recorder.requests) == 2


def test_retry_after_above_the_limit_fails_immediately_without_waiting() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(429, "120"), ok({}))
    settings = HttpSettings(max_retries=3, max_retry_after_seconds=60)

    with (
        timed_client(recorder, settings=settings) as client,
        pytest.raises(HttpRetryAfterExceededError) as exc_info,
    ):
        client.get_json("/x")

    error = exc_info.value
    assert isinstance(error, HttpStatusError) and isinstance(error, IngestionError)
    assert (error.status_code, error.retry_after, error.limit) == (429, 120.0, 60.0)
    assert error.retryable is False
    assert "Retry-After 120s" in str(error) and "60s limit" in str(error)
    assert fake.sleeps == []
    assert len(recorder.requests) == 1


def test_retry_after_equal_to_the_limit_is_honoured() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(429, "60"), ok({}))
    settings = HttpSettings(max_retries=1, max_retry_after_seconds=60)

    with timed_client(recorder, settings=settings) as client:
        client.get_json("/x")

    assert fake.sleeps == [60.0]


def test_non_ascii_digits_are_not_accepted_as_retry_after_seconds() -> None:
    # Cannot be sent through httpx as a header value; checked at the parser directly.
    client = HttpClient(BASE_URL, settings=FAST)
    try:
        assert client._parse_retry_after("²") is None
        assert client._parse_retry_after(" 12 ") == 12.0
    finally:
        client.close()


@pytest.mark.parametrize("header", ["soon", "-5", "1.5", "", "Mon, 99 Foo 2026"])
def test_invalid_retry_after_falls_back_to_backoff(header: str) -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(429, header), ok({}))

    with timed_client(recorder) as client:
        client.get_json("/x")

    assert fake.sleeps == [FAST.backoff_seconds]


@pytest.mark.parametrize("status", [500, 502, 504])
def test_retry_after_is_ignored_for_other_statuses(status: int) -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(status, "30"), ok({}))

    with timed_client(recorder) as client:
        client.get_json("/x")

    assert fake.sleeps == [FAST.backoff_seconds]


def test_retry_after_on_the_last_attempt_raises_without_waiting() -> None:
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(429, "5"))
    settings = HttpSettings(max_retries=0)

    with timed_client(recorder, settings=settings) as client, pytest.raises(HttpStatusError) as exc:
        client.get_json("/x")

    assert type(exc.value) is HttpStatusError
    assert fake.sleeps == []


def test_retry_after_limit_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_MAX_RETRY_AFTER_SECONDS", "10")
    fake = FakeTime()
    recorder = TimedRecorder(fake, retry_after(503, "11"), ok({}))
    client = HttpClient(
        BASE_URL,
        transport=httpx.MockTransport(recorder),
        sleep=fake.sleep,
        clock=fake.clock,
        now=fake.now,
    )

    with client, pytest.raises(HttpRetryAfterExceededError):
        client.get_json("/x")


# --- Base URL security -------------------------------------------------------------------


@pytest.mark.parametrize("allow_insecure_http", [False, True])
@pytest.mark.parametrize(
    "base_url",
    [
        f"http://user:{SECRET}@sandbox.example.com",
        "http://user@sandbox.example.com",
        f"http://sandbox.example.com/?api_key={SECRET}",
        f"http://sandbox.example.com/?TOKEN={SECRET}&page=1",
    ],
)
def test_plain_http_base_url_with_credentials_is_always_refused(
    base_url: str, allow_insecure_http: bool
) -> None:
    recorder = Recorder(ok({}))

    with pytest.raises(InsecureTransportError) as exc_info:
        HttpClient(
            base_url,
            settings=FAST,
            transport=httpx.MockTransport(recorder),
            allow_insecure_http=allow_insecure_http,
        )

    assert SECRET not in str(exc_info.value)
    assert recorder.requests == []


def test_plain_http_without_credentials_still_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        HttpClient("http://sandbox.example.com/?page=1", settings=FAST)


def test_https_base_url_with_userinfo_is_allowed_and_redacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = Recorder(ok({}))

    with (
        caplog.at_level(logging.DEBUG),
        make_client(recorder, base_url=f"https://user:{SECRET}@api.example.com") as client,
    ):
        client.get_json("/x")

    assert len(recorder.requests) == 1
    assert SECRET not in caplog.text


# --- Credentials never leak on the new paths -----------------------------------------------


def _scenario_retry_after_exceeded(fake: FakeTime) -> HttpClient:
    return timed_client(
        TimedRecorder(fake, retry_after(429, "999")),
        sensitive_params=frozenset({"api_key"}),
    )


def _scenario_closed_client(fake: FakeTime) -> HttpClient:
    client = timed_client(TimedRecorder(fake, ok({})), sensitive_params=frozenset({"api_key"}))
    client.close()
    return client


def _scenario_interval_and_retry(fake: FakeTime) -> HttpClient:
    return timed_client(
        TimedRecorder(fake, retry_after(503, "1"), ok({})),
        min_interval=2.0,
        sensitive_params=frozenset({"api_key"}),
    )


@pytest.mark.parametrize(
    "build",
    [_scenario_retry_after_exceeded, _scenario_closed_client, _scenario_interval_and_retry],
    ids=["retry_after_exceeded", "closed_client", "interval_and_retry"],
)
def test_credentials_never_leak_on_new_paths(
    build: Callable[[FakeTime], HttpClient], caplog: pytest.LogCaptureFixture
) -> None:
    caught: HttpClientError | None = None

    with caplog.at_level(logging.DEBUG):
        client = build(FakeTime())
        try:
            client.get_json("/job/list.json", {"api_key": SECRET, "page": 1})
            client.get_json("/job/list.json", {"api_key": SECRET, "page": 2})
        except HttpClientError as exc:
            caught = exc
        finally:
            client.close()

    assert SECRET not in caplog.text
    if caught is not None:
        assert SECRET not in str(caught)
        assert caught.__cause__ is None and caught.__context__ is None
        assert SECRET not in "".join(traceback.format_exception(caught))
