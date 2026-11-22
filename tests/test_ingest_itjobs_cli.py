"""CLI tests for the ``itjobs`` source. No network, no database, no real credentials.

The API is simulated with ``httpx.MockTransport`` through ``itjobs_client_factory``
(or by replacing ``HttpClient`` in the default factory). ``FAKE_API_KEY`` is a
placeholder, never a credential; an autouse fixture sets it so a key from a
developer's ``.env`` can never be used here.
"""

import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from career_intelligence import ingest
from career_intelligence.config import HttpSettings
from career_intelligence.ingest import ITJOBS_API_KEY_ENV, create_itjobs_client, main
from career_intelligence.ingestion.http import HttpClient
from career_intelligence.sources.itjobs import (
    DEFAULT_PAGE_SIZE,
    ITJOBS_BASE_URL,
    ITJOBS_LIST_PATH,
    ITJOBS_MIN_INTERVAL_SECONDS,
    ITJobsAdapter,
)

FAKE_API_KEY = "test-key-not-real"

type Body = dict[str, Any]
type Tamper = Callable[[int, Body], Body | httpx.Response]


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ITJOBS_API_KEY_ENV, FAKE_API_KEY)


def forbidden_session_factory():
    raise AssertionError("the database must not be contacted")


def record(posting_id: int) -> dict[str, Any]:
    return {
        "id": posting_id,
        "title": f"Engineer {posting_id}",
        "company": {"name": "Example Tech"},
    }


class FakeTime:
    """Monotonic clock that only moves when the client sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeListApi:
    """Simulated ``job/list.json`` with page-number pagination."""

    def __init__(
        self, postings: list[Any], *, total: int | None = None, tamper: Tamper | None = None
    ) -> None:
        self.postings = postings
        self.total = len(postings) if total is None else total
        self.tamper = tamper
        self.requests: list[httpx.Request] = []

    @property
    def pages(self) -> list[int]:
        return [int(request.url.params["page"]) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        page = int(request.url.params["page"])
        limit = int(request.url.params["limit"])
        start = (page - 1) * limit
        body: Body = {
            "total": self.total,
            "page": page,
            "limit": limit,
            "results": self.postings[start : start + limit],
        }
        if self.tamper is not None:
            tampered = self.tamper(page, body)
            if isinstance(tampered, httpx.Response):
                return tampered
            body = tampered
        return httpx.Response(200, json=body)


class TrackingClient(HttpClient):
    """Real ``HttpClient`` that records ``close()`` calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class ClientFactory:
    """``itjobs_client_factory`` over a mock transport and a fake clock."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response], **settings: Any) -> None:
        self.handler = handler
        self.settings = HttpSettings(**{"max_retries": 0, "backoff_seconds": 0, **settings})
        self.time = FakeTime()
        self.clients: list[TrackingClient] = []

    def __call__(self) -> TrackingClient:
        client = TrackingClient(
            ITJOBS_BASE_URL,
            settings=self.settings,
            transport=httpx.MockTransport(self.handler),
            min_interval_seconds=ITJOBS_MIN_INTERVAL_SECONDS,
            sleep=self.time.sleep,
            clock=self.time.clock,
        )
        self.clients.append(client)
        return client


def run(factory: ClientFactory, *flags: str) -> int:
    return main(
        [*flags, "itjobs"],
        session_factory=forbidden_session_factory,
        itjobs_client_factory=factory,
    )


def assert_key_not_exposed(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> tuple[str, str]:
    captured = capsys.readouterr()
    assert FAKE_API_KEY not in captured.out
    assert FAKE_API_KEY not in captured.err
    assert FAKE_API_KEY not in caplog.text
    return captured.out, captured.err


# --- Dry run --------------------------------------------------------------------------------


def test_dry_run_collects_two_pages_and_never_touches_the_database(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    size = DEFAULT_PAGE_SIZE
    api = FakeListApi([record(n) for n in range(1, size + 3)])  # one full page + 2
    factory = ClientFactory(api)

    exit_code = run(factory)

    out, _ = assert_key_not_exposed(capsys, caplog)
    assert exit_code == 0
    assert api.pages == [1, 2]
    assert f"source='ITJobs.pt' mode=dry-run fetched={size + 2} valid=0 rejected={size + 2}" in out
    assert "validation: job_url: Field required" in out
    assert "dry run: nothing was written" in out
    assert "persisting to" not in out
    assert "persisted" not in out
    # The key was actually sent, only ever over the documented request.
    for request in api.requests:
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert request.url.path == ITJOBS_LIST_PATH
        assert request.url.params["api_key"] == FAKE_API_KEY


def test_requests_respect_the_minimum_interval() -> None:
    api = FakeListApi([record(n) for n in range(1, DEFAULT_PAGE_SIZE * 2 + 2)])  # 3 pages
    factory = ClientFactory(api)

    assert run(factory) == 0

    assert api.pages == [1, 2, 3]
    assert factory.time.sleeps == [ITJOBS_MIN_INTERVAL_SECONDS, ITJOBS_MIN_INTERVAL_SECONDS]


# Runs in a fresh interpreter, so modules loaded by other tests cannot hide an
# import. Sockets are disabled there too: the mock transport is the only "server".
_DRY_RUN_PROCESS = textwrap.dedent(
    """
    import socket
    import sys

    def _no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    socket.socket.connect = _no_network
    socket.getaddrinfo = _no_network

    import httpx

    from career_intelligence.ingest import main
    from career_intelligence.ingestion.http import HttpClient
    from career_intelligence.sources.itjobs import (
        DEFAULT_PAGE_SIZE,
        ITJOBS_BASE_URL,
        ITJOBS_MIN_INTERVAL_SECONDS,
    )

    POSTINGS = [
        {"id": n, "title": f"Engineer {n}", "company": {"name": "Example Tech"}}
        for n in range(1, DEFAULT_PAGE_SIZE + 3)
    ]
    pages = []

    def handler(request):
        page = int(request.url.params["page"])
        limit = int(request.url.params["limit"])
        pages.append(page)
        body = {
            "total": len(POSTINGS),
            "page": page,
            "limit": limit,
            "results": POSTINGS[(page - 1) * limit : page * limit],
        }
        return httpx.Response(200, json=body)

    def factory():
        return HttpClient(
            ITJOBS_BASE_URL,
            transport=httpx.MockTransport(handler),
            min_interval_seconds=ITJOBS_MIN_INTERVAL_SECONDS,
            sleep=lambda seconds: None,
        )

    code = main(["itjobs"], itjobs_client_factory=factory)
    loaded = [
        m
        for m in ("sqlalchemy", "psycopg", "career_intelligence.db.session")
        if m in sys.modules
    ]
    assert not loaded, loaded
    assert pages == [1, 2], pages
    sys.exit(code)
    """
)


def test_dry_run_process_never_imports_the_database_layer() -> None:
    env = {
        **os.environ,
        ITJOBS_API_KEY_ENV: FAKE_API_KEY,
        # Unroutable (RFC 5737): any attempt to use it would fail, never reach a server.
        "DATABASE_URL": "postgresql+psycopg://nobody@192.0.2.1:1/never",
    }

    completed = subprocess.run(
        [sys.executable, "-c", _DRY_RUN_PROCESS],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    total = DEFAULT_PAGE_SIZE + 2
    assert f"source='ITJobs.pt' mode=dry-run fetched={total} valid=0 rejected={total}" in (
        completed.stdout
    )
    assert "dry run: nothing was written" in completed.stdout
    assert FAKE_API_KEY not in completed.stdout
    assert FAKE_API_KEY not in completed.stderr


# --- Missing configuration: zero requests ---------------------------------------------------------


def never_called_factory() -> HttpClient:
    raise AssertionError("no HTTP client may be created")


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n"], ids=["unset", "empty", "spaces", "ws"])
def test_missing_or_blank_api_key_fails_before_any_request(
    value: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if value is None:
        monkeypatch.delenv(ITJOBS_API_KEY_ENV)
    else:
        monkeypatch.setenv(ITJOBS_API_KEY_ENV, value)

    exit_code = main(
        ["itjobs"],
        session_factory=forbidden_session_factory,
        itjobs_client_factory=never_called_factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "error: invalid configuration: ITJOBS_API_KEY is not set.\n"
    assert "fetched=" not in captured.out


def test_missing_api_key_with_persist_also_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ITJOBS_API_KEY_ENV)

    exit_code = main(
        ["--persist", "itjobs"],
        session_factory=forbidden_session_factory,
        itjobs_client_factory=never_called_factory,
    )

    assert exit_code == 2
    assert "ITJOBS_API_KEY is not set" in capsys.readouterr().err


def test_invalid_http_configuration_fails_before_reading_the_key_or_any_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HTTP_MAX_RETRIES", "99")

    exit_code = main(
        ["itjobs"],
        session_factory=forbidden_session_factory,
        itjobs_client_factory=never_called_factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "HTTP_MAX_RETRIES" in captured.err
    assert FAKE_API_KEY not in captured.err


def test_the_key_is_not_a_command_line_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["itjobs", "--api-key", FAKE_API_KEY], session_factory=forbidden_session_factory)

    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


# --- HTTP and pagination errors ----------------------------------------------------------------------


def server_error(page: int, body: Body) -> httpx.Response:
    return httpx.Response(500)


def invalid_json(page: int, body: Body) -> httpx.Response:
    return httpx.Response(200, text="<html>")


def api_error_envelope(page: int, body: Body) -> Body:
    return {"error": {"message": f"Invalid api_key={FAKE_API_KEY}"}}


def total_changes_on_page_2(page: int, body: Body) -> Body:
    if page == 2:
        body["total"] += 1
    return body


def repeats_page_1(page: int, body: Body) -> Body:
    return {**body, "page": 1, "results": [record(n) for n in range(1, DEFAULT_PAGE_SIZE + 1)]}


def unauthorized(page: int, body: Body) -> httpx.Response:
    return httpx.Response(401)


@pytest.mark.parametrize(
    ("tamper", "postings", "expected_pages", "message"),
    [
        (server_error, 3, [1, 1, 1], "HTTP 500"),
        (unauthorized, 3, [1], "HTTP 401"),
        (invalid_json, 3, [1], "not valid JSON"),
        (api_error_envelope, 3, [1], "ITJobs API returned an error response"),
        (total_changes_on_page_2, 2 * DEFAULT_PAGE_SIZE + 1, [1, 2], "total changed"),
        (repeats_page_1, 2 * DEFAULT_PAGE_SIZE + 1, [1, 2], "page 2 was requested"),
    ],
    ids=["http_500", "http_401", "invalid_json", "api_error", "total_changed", "repeated_page"],
)
def test_ingestion_errors_exit_1_close_the_client_and_never_expose_the_key(
    tamper: Tamper,
    postings: int,
    expected_pages: list[int],
    message: str,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    api = FakeListApi([record(n) for n in range(1, postings + 1)], tamper=tamper)
    factory = ClientFactory(api, max_retries=2)

    exit_code = run(factory)

    _, err = assert_key_not_exposed(capsys, caplog)
    assert exit_code == 1
    assert api.pages == expected_pages
    assert message in err
    assert [client.close_calls for client in factory.clients] == [1]


def test_too_many_pages_exits_1_after_one_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ingest, "ITJobsAdapter", _adapter_with(max_pages=2))
    api = FakeListApi([record(n) for n in range(1, DEFAULT_PAGE_SIZE * 2 + 2)])  # needs 3 pages

    exit_code = run(ClientFactory(api))

    assert exit_code == 1
    assert api.pages == [1]
    assert "more than max_pages=2" in capsys.readouterr().err


def _adapter_with(**overrides: Any) -> Callable[..., Any]:
    def build(client: Any, **kwargs: Any) -> ITJobsAdapter:
        return ITJobsAdapter(client, **{**kwargs, **overrides})

    return build


# --- Client lifecycle -------------------------------------------------------------------------------


def test_client_is_closed_after_success() -> None:
    factory = ClientFactory(FakeListApi([record(1)]))

    assert run(factory) == 0

    assert len(factory.clients) == 1
    assert factory.clients[0].close_calls == 1


def test_client_is_closed_after_an_unexpected_error() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("unexpected failure in transport")

    factory = ClientFactory(broken)

    with pytest.raises(RuntimeError, match="unexpected failure") as excinfo:
        run(factory)

    assert FAKE_API_KEY not in str(excinfo.value)
    assert factory.clients[0].close_calls == 1


# --- Default client factory ---------------------------------------------------------------------------


def test_default_factory_uses_the_itjobs_base_url_and_minimum_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_http_client(*args: Any, **kwargs: Any) -> str:
        captured.append((args, kwargs))
        return "client"

    monkeypatch.setattr(ingest, "HttpClient", fake_http_client)

    assert create_itjobs_client() == "client"
    assert captured == [((ITJOBS_BASE_URL,), {"min_interval_seconds": ITJOBS_MIN_INTERVAL_SECONDS})]


def test_main_uses_the_default_factory_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeListApi([record(n) for n in range(1, DEFAULT_PAGE_SIZE + 2)])  # 2 pages
    time = FakeTime()
    created: list[TrackingClient] = []

    def http_client(base_url: str, **kwargs: Any) -> TrackingClient:
        # Keeps the default factory's arguments; only the transport and clock are fake.
        client = TrackingClient(
            base_url,
            transport=httpx.MockTransport(api),
            sleep=time.sleep,
            clock=time.clock,
            **kwargs,
        )
        created.append(client)
        return client

    monkeypatch.setattr(ingest, "HttpClient", http_client)

    exit_code = main(["itjobs"], session_factory=forbidden_session_factory)

    assert exit_code == 0
    assert [str(request.url.copy_with(query=None)) for request in api.requests] == [
        f"{ITJOBS_BASE_URL}{ITJOBS_LIST_PATH}"
    ] * 2
    assert time.sleeps == [ITJOBS_MIN_INTERVAL_SECONDS]
    assert [client.close_calls for client in created] == [1]
    assert "source='ITJobs.pt' mode=dry-run" in capsys.readouterr().out


# --- Persistence rule -------------------------------------------------------------------------------


def test_persist_with_every_posting_rejected_exits_1_without_touching_the_database(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    api = FakeListApi([record(n) for n in range(1, DEFAULT_PAGE_SIZE + 3)])  # 2 pages
    factory = ClientFactory(api)

    exit_code = run(factory, "--persist")

    out, err = assert_key_not_exposed(capsys, caplog)
    total = DEFAULT_PAGE_SIZE + 2
    assert exit_code == 1
    assert api.pages == [1, 2]
    assert f"mode=persist fetched={total} valid=0 rejected={total}" in out
    assert "validation: job_url: Field required" in out
    assert err == f"error: refusing to persist: all {total} fetched postings were rejected\n"
    assert "persisting to" not in out
    assert [client.close_calls for client in factory.clients] == [1]


def test_persist_with_an_empty_listing_exits_0_without_touching_the_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = FakeListApi([])
    factory = ClientFactory(api)

    exit_code = run(factory, "--persist")

    out = capsys.readouterr().out
    assert exit_code == 0
    assert api.pages == [1]
    assert "mode=persist fetched=0 valid=0 rejected=0" in out
    assert "nothing to persist: the source returned no postings" in out
    assert [client.close_calls for client in factory.clients] == [1]
