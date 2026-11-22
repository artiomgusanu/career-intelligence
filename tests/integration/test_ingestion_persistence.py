import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from career_intelligence.db.repositories import UpsertResult
from career_intelligence.ingest import main
from career_intelligence.ingestion.contracts import DEMO_SOURCE_TYPE
from career_intelligence.ingestion.persistence import persist_collection
from career_intelligence.ingestion.service import collect_postings
from career_intelligence.models import JobPosting, Source
from career_intelligence.sources.local_file import DEFAULT_METADATA, LocalFileAdapter

pytestmark = pytest.mark.postgres

FIXTURE = Path(__file__).parents[1] / "fixtures" / "local_postings.json"


def count(session: Session, model: type) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def assert_database_empty(committed_sessions: sessionmaker[Session]) -> None:
    with committed_sessions() as session:
        assert count(session, Source) == 0
        assert count(session, JobPosting) == 0


def test_persist_collection_registers_source_and_upserts(db_session: Session) -> None:
    collection = collect_postings(LocalFileAdapter(FIXTURE))

    result = persist_collection(db_session, collection)

    assert result.source_created is True
    assert (result.fetched, result.valid, result.rejected) == (8, 4, 4)
    assert result.upsert == UpsertResult(inserted=3, updated=0, duplicates_in_batch=1)
    stored = db_session.scalars(select(JobPosting).where(JobPosting.external_id == "be-001")).one()
    # In-batch duplicates collapse to the last occurrence as a whole record, before SQL:
    # fields only present in the earlier occurrence are not merged (COALESCE applies only
    # against rows already stored).
    assert stored.title == "Backend Engineer (Python) - updated"
    assert stored.salary_currency is None


def test_persist_collection_does_not_commit(committed_sessions: sessionmaker[Session]) -> None:
    collection = collect_postings(LocalFileAdapter(FIXTURE))

    with committed_sessions() as session:
        persist_collection(session, collection)
        session.rollback()

    assert_database_empty(committed_sessions)


# --- CLI scenarios against the test database ------------------------------------------


def test_cli_without_flags_writes_nothing(
    committed_sessions: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["local-file", str(FIXTURE)], session_factory=committed_sessions) == 0

    assert "dry run: nothing was written" in capsys.readouterr().out
    assert_database_empty(committed_sessions)


def test_cli_persist_alone_refuses_demo_data(
    committed_sessions: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--persist", "local-file", str(FIXTURE)], session_factory=committed_sessions)

    assert exit_code == 2
    assert "--allow-demo-data" in capsys.readouterr().err
    assert_database_empty(committed_sessions)


def test_cli_persist_with_allow_demo_data_commits_and_is_idempotent(
    committed_sessions: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--persist", "--allow-demo-data", "local-file", str(FIXTURE)]

    assert main(argv, session_factory=committed_sessions) == 0
    first = capsys.readouterr().out
    assert main(argv, session_factory=committed_sessions) == 0
    second = capsys.readouterr().out

    assert "source_created=True inserted=3 updated=0 duplicates_in_batch=1" in first
    assert "source_created=False inserted=0 updated=3 duplicates_in_batch=1" in second
    with committed_sessions() as session:
        assert count(session, JobPosting) == 3
        source = session.scalars(select(Source)).one()
        assert (source.name, source.url, source.source_type) == (
            DEFAULT_METADATA.name,
            DEFAULT_METADATA.url,
            DEMO_SOURCE_TYPE,
        )


def test_cli_shows_database_target_without_credentials(
    committed_sessions: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    engine_url = committed_sessions.kw["bind"].url

    main(
        ["--persist", "--allow-demo-data", "local-file", str(FIXTURE)],
        session_factory=committed_sessions,
    )

    out = capsys.readouterr().out
    target_line = next(line for line in out.splitlines() if line.startswith("persisting to"))
    assert target_line == (
        f"persisting to database {engine_url.host}:{engine_url.port}/{engine_url.database}"
    )
    assert engine_url.database.endswith("_test")
    assert engine_url.password and engine_url.password not in out
    assert engine_url.username not in target_line


# --- Persistence rule against the test database ---------------------------------------------
# Built from records of the existing fixture: the first three are valid and
# distinct, the last four are rejected.

PERSIST = ["--persist", "--allow-demo-data", "local-file"]


def write_postings(tmp_path: Path, records: list[object]) -> str:
    path = tmp_path / "postings.json"
    path.write_text(json.dumps({"postings": records}), encoding="utf-8")
    return str(path)


def fixture_records() -> list[object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["postings"]


def test_cli_persist_of_an_empty_collection_registers_no_source(
    committed_sessions: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, [])

    assert main([*PERSIST, path], session_factory=committed_sessions) == 0

    assert "nothing to persist" in capsys.readouterr().out
    assert_database_empty(committed_sessions)


def test_cli_persist_with_every_posting_rejected_registers_no_source(
    committed_sessions: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, fixture_records()[4:])

    assert main([*PERSIST, path], session_factory=committed_sessions) == 1

    assert "refusing to persist: all 4 fetched postings were rejected" in capsys.readouterr().err
    assert_database_empty(committed_sessions)


def test_cli_persist_of_only_valid_postings_commits_and_is_idempotent(
    committed_sessions: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [*PERSIST, write_postings(tmp_path, fixture_records()[:3])]

    assert main(argv, session_factory=committed_sessions) == 0
    first = capsys.readouterr().out
    assert main(argv, session_factory=committed_sessions) == 0
    second = capsys.readouterr().out

    assert "fetched=3 valid=3 rejected=0" in first
    assert "source_created=True inserted=3 updated=0 duplicates_in_batch=0" in first
    assert "source_created=False inserted=0 updated=3 duplicates_in_batch=0" in second
    with committed_sessions() as session:
        assert count(session, Source) == 1
        assert count(session, JobPosting) == 3


def test_cli_persist_failure_rolls_back_the_source_and_postings(
    committed_sessions: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from career_intelligence.ingestion import persistence

    sources_in_transaction: list[int] = []

    def failing_upsert(session: Session, **kwargs: object) -> None:
        # The source row exists inside the open transaction when the failure happens.
        sources_in_transaction.append(count(session, Source))
        raise RuntimeError("upsert failed after the source was registered")

    monkeypatch.setattr(persistence, "upsert_job_postings", failing_upsert)
    path = write_postings(tmp_path, fixture_records())

    with pytest.raises(RuntimeError, match="upsert failed"):
        main([*PERSIST, path], session_factory=committed_sessions)

    assert sources_in_transaction == [1]
    assert_database_empty(committed_sessions)
