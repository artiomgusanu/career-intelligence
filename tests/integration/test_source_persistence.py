import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from career_intelligence.db.repositories import (
    SourceInUseError,
    delete_source,
    get_or_create_source_by_name,
    upsert_job_postings,
)
from career_intelligence.db.session import get_db
from career_intelligence.main import app
from career_intelligence.models import Source
from career_intelligence.schemas import JobPostingCreate

pytestmark = pytest.mark.postgres


def register(session: Session, name: str = "Example Board", **overrides: str):
    return get_or_create_source_by_name(
        session,
        name=name,
        url=overrides.get("url", "https://board.example.com/"),
        source_type=overrides.get("source_type", "job_board"),
    )


def count_sources(session: Session) -> int:
    return session.execute(select(func.count()).select_from(Source)).scalar_one()


def add_job(session: Session, source_id: int) -> None:
    upsert_job_postings(
        session,
        source_id=source_id,
        postings=[
            JobPostingCreate(
                external_id="1",
                title="Engineer",
                company_name="Acme",
                job_url="https://jobs.example.com/1",
            )
        ],
    )


# --- get_or_create_source_by_name -------------------------------------------


def test_get_or_create_creates_new_source(db_session: Session) -> None:
    registration = register(db_session)

    assert registration.created is True
    assert registration.source.id is not None
    assert registration.source.name == "Example Board"
    assert count_sources(db_session) == 1


def test_get_or_create_reuses_existing_source_unchanged(db_session: Session) -> None:
    original = register(db_session).source

    again = register(
        db_session, url="https://different.example.com/", source_type="other"
    )

    assert again.created is False
    assert again.source.id == original.id
    assert again.source.url == "https://board.example.com/"
    assert again.source.source_type == "job_board"
    assert count_sources(db_session) == 1


def test_get_or_create_is_safe_under_concurrent_registration(
    committed_sessions: sessionmaker[Session],
) -> None:
    """B's insert must block on A's uncommitted row, then reuse it after A commits."""
    a_inserted = threading.Event()
    timeline: dict[str, float] = {}

    def worker_a() -> tuple[int, bool]:
        try:
            with committed_sessions() as session:
                registration = register(session)
                a_inserted.set()
                time.sleep(0.5)  # hold the uncommitted row while B tries to insert
                timeline["a_committed"] = time.monotonic()
                session.commit()
                return registration.source.id, registration.created
        finally:
            a_inserted.set()  # never leave B waiting if A fails

    def worker_b() -> tuple[int, bool]:
        assert a_inserted.wait(timeout=5)
        with committed_sessions() as session:
            registration = register(session)
            timeline["b_returned"] = time.monotonic()
            session.commit()
            return registration.source.id, registration.created

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a, future_b = pool.submit(worker_a), pool.submit(worker_b)
        # .result() re-raises any exception from the worker thread.
        a_id, a_created = future_a.result(timeout=15)
        b_id, b_created = future_b.result(timeout=15)

    assert a_created is True
    assert (b_id, b_created) == (a_id, False)
    assert timeline["b_returned"] >= timeline["a_committed"]  # B really waited for A
    with committed_sessions() as session:
        assert count_sources(session) == 1


def test_get_or_create_creates_when_concurrent_insert_rolls_back(
    committed_sessions: sessionmaker[Session],
) -> None:
    a_inserted = threading.Event()

    def worker_a() -> None:
        try:
            with committed_sessions() as session:
                register(session)
                a_inserted.set()
                time.sleep(0.5)
                session.rollback()
        finally:
            a_inserted.set()

    def worker_b() -> bool:
        assert a_inserted.wait(timeout=5)
        with committed_sessions() as session:
            created = register(session).created
            session.commit()
            return created

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a, future_b = pool.submit(worker_a), pool.submit(worker_b)
        future_a.result(timeout=15)
        b_created = future_b.result(timeout=15)

    assert b_created is True
    with committed_sessions() as session:
        assert count_sources(session) == 1


def test_get_or_create_does_not_commit(committed_sessions: sessionmaker[Session]) -> None:
    with committed_sessions() as session:
        register(session)
        session.rollback()

    with committed_sessions() as session:
        assert count_sources(session) == 0


# --- Foreign key RESTRICT ---------------------------------------------------


def test_database_restricts_deleting_source_with_job_postings(db_session: Session) -> None:
    source_id = register(db_session).source.id
    add_job(db_session, source_id)

    with pytest.raises(IntegrityError) as exc_info, db_session.begin_nested():
        db_session.execute(delete(Source).where(Source.id == source_id))

    assert exc_info.value.orig.sqlstate == "23503"


def test_delete_source_raises_source_in_use_and_keeps_data(db_session: Session) -> None:
    source_id = register(db_session).source.id
    add_job(db_session, source_id)
    db_session.commit()  # release the setup savepoint; the outer transaction still isolates

    with pytest.raises(SourceInUseError):
        delete_source(db_session, source_id)

    assert db_session.get(Source, source_id) is not None


def test_delete_source_without_job_postings_succeeds(db_session: Session) -> None:
    source_id = register(db_session).source.id

    assert delete_source(db_session, source_id) is True
    assert count_sources(db_session) == 0


@pytest.fixture
def pg_client(db_session: Session) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_delete_endpoint_returns_409_when_source_has_job_postings(
    db_session: Session, pg_client: TestClient
) -> None:
    source_id = register(db_session).source.id
    add_job(db_session, source_id)
    db_session.commit()  # release the setup savepoint; the outer transaction still isolates

    response = pg_client.delete(f"/sources/{source_id}")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Source has associated job postings and cannot be deleted."
    }
    assert pg_client.get(f"/sources/{source_id}").status_code == 200
