"""ITJobs CLI with ``--persist`` against the PostgreSQL test database.

Every path reachable today must leave the database untouched: ITJobs records
carry no ``job_url``, so they are all rejected and nothing may be written,
not even the source row. Persisting valid ITJobs postings is not tested until
a confirmed source for ``job_url`` exists; no URL is invented here.

The API is simulated with ``httpx.MockTransport``; no request leaves the
process. ``FAKE_API_KEY`` is a placeholder, never a credential, and an autouse
fixture sets it so a key from a developer's ``.env`` can never be used.
"""

import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from career_intelligence.config import HttpSettings
from career_intelligence.ingest import ITJOBS_API_KEY_ENV, main
from career_intelligence.ingestion.http import HttpClient
from career_intelligence.models import JobPosting, Source
from career_intelligence.sources.itjobs import (
    DEFAULT_PAGE_SIZE,
    ITJOBS_BASE_URL,
    ITJOBS_MIN_INTERVAL_SECONDS,
)

pytestmark = pytest.mark.postgres

FAKE_API_KEY = "test-key-not-real"
TWO_PAGES = DEFAULT_PAGE_SIZE + 2


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ITJOBS_API_KEY_ENV, FAKE_API_KEY)


@pytest.fixture
def test_sessions(committed_sessions: sessionmaker[Session]) -> sessionmaker[Session]:
    # committed_sessions is bound to the validated TEST_DATABASE_URL; re-checked here
    # because these tests pass it to the CLI explicitly, never the default SessionLocal.
    database = committed_sessions.kw["bind"].url.database
    assert database is not None and database.endswith("_test")
    return committed_sessions


def record(posting_id: int) -> dict[str, Any]:
    """A synthetic ITJobs list record. Deliberately no URL: the API provides none."""
    return {"id": posting_id, "title": f"Engineer {posting_id}", "company": {"name": "Example"}}


class ListingApi:
    """Minimal ``job/list.json``: page-number pagination, optional failure on one page."""

    def __init__(self, postings: list[dict[str, Any]], *, fail_on_page: int | None = None) -> None:
        self.postings = postings
        self.fail_on_page = fail_on_page
        self.pages: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        limit = int(request.url.params["limit"])
        self.pages.append(page)
        if page == self.fail_on_page:
            return httpx.Response(500)
        start = (page - 1) * limit
        return httpx.Response(
            200,
            json={
                "total": len(self.postings),
                "page": page,
                "limit": limit,
                "results": self.postings[start : start + limit],
            },
        )


class TrackingClient(HttpClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class ClientFactory:
    """HTTP clients over the mock API, without retries and with a fake clock."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.clients: list[TrackingClient] = []
        self.now = 0.0

    def _sleep(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> TrackingClient:
        client = TrackingClient(
            ITJOBS_BASE_URL,
            settings=HttpSettings(max_retries=0, backoff_seconds=0),
            transport=httpx.MockTransport(self.handler),
            min_interval_seconds=ITJOBS_MIN_INTERVAL_SECONDS,
            sleep=self._sleep,
            clock=lambda: self.now,
        )
        self.clients.append(client)
        return client

    @property
    def close_calls(self) -> list[int]:
        return [client.close_calls for client in self.clients]


class RecordingSessions:
    """Session factory over the test database that records whether the CLI opened one."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions
        self.opened = 0

    def __call__(self) -> Session:
        self.opened += 1
        return self.sessions()


def persist_itjobs(sessions: RecordingSessions, factory: Callable[[], HttpClient]) -> int:
    return main(["--persist", "itjobs"], session_factory=sessions, itjobs_client_factory=factory)


def assert_database_empty(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        for model in (Source, JobPosting):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def assert_key_not_exposed(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> tuple[str, str]:
    captured = capsys.readouterr()
    for text in (captured.out, captured.err, caplog.text):
        assert FAKE_API_KEY not in text
    return captured.out, captured.err


def test_every_posting_rejected_exits_1_and_writes_nothing(
    test_sessions: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    api = ListingApi([record(n) for n in range(1, TWO_PAGES + 1)])
    factory = ClientFactory(api)
    sessions = RecordingSessions(test_sessions)

    exit_code = persist_itjobs(sessions, factory)

    out, err = assert_key_not_exposed(capsys, caplog)
    assert exit_code == 1
    assert api.pages == [1, 2]
    assert f"mode=persist fetched={TWO_PAGES} valid=0 rejected={TWO_PAGES}" in out
    assert f"{TWO_PAGES}  validation: job_url: Field required" in out
    assert err == f"error: refusing to persist: all {TWO_PAGES} fetched postings were rejected\n"
    assert factory.close_calls == [1]
    assert sessions.opened == 0
    assert_database_empty(test_sessions)


def test_empty_listing_exits_0_and_writes_nothing(
    test_sessions: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    api = ListingApi([])
    factory = ClientFactory(api)
    sessions = RecordingSessions(test_sessions)

    exit_code = persist_itjobs(sessions, factory)

    out, err = assert_key_not_exposed(capsys, caplog)
    assert exit_code == 0
    assert api.pages == [1]
    assert "mode=persist fetched=0 valid=0 rejected=0" in out
    assert "nothing to persist: the source returned no postings" in out
    assert err == ""
    assert factory.close_calls == [1]
    assert sessions.opened == 0
    assert_database_empty(test_sessions)


def test_failure_on_page_2_exits_1_and_writes_nothing(
    test_sessions: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    api = ListingApi([record(n) for n in range(1, TWO_PAGES + 1)], fail_on_page=2)
    factory = ClientFactory(api)
    sessions = RecordingSessions(test_sessions)

    exit_code = persist_itjobs(sessions, factory)

    out, err = assert_key_not_exposed(capsys, caplog)
    assert exit_code == 1
    assert api.pages == [1, 2]
    assert err.startswith("error: ")
    assert "HTTP 500" in err
    assert "fetched=" not in out  # aborted: no partial collection is reported
    assert factory.close_calls == [1]
    assert sessions.opened == 0
    assert_database_empty(test_sessions)


def test_missing_api_key_exits_2_without_requests_and_writes_nothing(
    test_sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(ITJOBS_API_KEY_ENV)
    api = ListingApi([record(1)])
    factory = ClientFactory(api)
    sessions = RecordingSessions(test_sessions)

    exit_code = persist_itjobs(sessions, factory)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "error: invalid configuration: ITJOBS_API_KEY is not set.\n"
    assert api.pages == []
    assert factory.clients == []  # no client is even created
    assert sessions.opened == 0
    assert_database_empty(test_sessions)
