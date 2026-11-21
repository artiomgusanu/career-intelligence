import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.orm import Session, selectinload, sessionmaker

from career_intelligence.db.repositories import (
    UpsertResult,
    get_job_posting_by_id,
    get_or_create_source_by_name,
    list_job_postings_by_source,
    upsert_job_postings,
)
from career_intelligence.models import JobPosting, Source
from career_intelligence.schemas import JobPostingCreate, JobPostingResponse

pytestmark = pytest.mark.postgres


def make_posting(external_id: str, **overrides: object) -> JobPostingCreate:
    fields: dict[str, object] = {
        "external_id": external_id,
        "title": f"Engineer {external_id}",
        "company_name": "Acme",
        "job_url": f"https://jobs.example.com/{external_id}",
    }
    fields.update(overrides)
    return JobPostingCreate(**fields)


def make_source(session: Session, name: str = "Example Board") -> Source:
    return get_or_create_source_by_name(
        session, name=name, url="https://board.example.com/", source_type="job_board"
    ).source


def fetch_posting(session: Session, source_id: int, external_id: str) -> JobPosting:
    statement = (
        select(JobPosting)
        .where(JobPosting.source_id == source_id, JobPosting.external_id == external_id)
        .execution_options(populate_existing=True)
    )
    return session.scalars(statement).one()


def count_postings(session: Session, source_id: int | None = None) -> int:
    statement = select(func.count()).select_from(JobPosting)
    if source_id is not None:
        statement = statement.where(JobPosting.source_id == source_id)
    return session.execute(statement).scalar_one()


# --- Bulk insert and statistics --------------------------------------------


def test_bulk_insert_across_multiple_chunks(db_session: Session) -> None:
    source = make_source(db_session)
    postings = [make_posting(str(i)) for i in range(2500)]

    result = upsert_job_postings(
        db_session, source_id=source.id, postings=postings, chunk_size=1000
    )

    assert result == UpsertResult(inserted=2500, updated=0, duplicates_in_batch=0)
    assert count_postings(db_session, source.id) == 2500


def test_mixed_batch_statistics_match_database_state(db_session: Session) -> None:
    source = make_source(db_session)
    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[make_posting(str(i)) for i in range(2500)],
        chunk_size=1000,
    )

    # 2000 already exist (500..2499), 500 are new (2500..2999); chunks straddle both.
    result = upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[make_posting(str(i)) for i in range(500, 3000)],
        chunk_size=700,
    )

    assert result == UpsertResult(inserted=500, updated=2000, duplicates_in_batch=0)
    assert count_postings(db_session, source.id) == 3000


def test_reupserting_identical_postings_counts_as_updated(db_session: Session) -> None:
    source = make_source(db_session)
    postings = [make_posting("a"), make_posting("b")]
    upsert_job_postings(db_session, source_id=source.id, postings=postings)

    result = upsert_job_postings(db_session, source_id=source.id, postings=postings)

    assert result == UpsertResult(inserted=0, updated=2, duplicates_in_batch=0)


def test_empty_batch_is_a_no_op(db_session: Session) -> None:
    source = make_source(db_session)

    result = upsert_job_postings(db_session, source_id=source.id, postings=[])

    assert result == UpsertResult(inserted=0, updated=0, duplicates_in_batch=0)
    assert count_postings(db_session) == 0


def test_chunk_size_must_be_positive(db_session: Session) -> None:
    with pytest.raises(ValueError):
        upsert_job_postings(db_session, source_id=1, postings=[], chunk_size=0)


# --- Duplicate handling -----------------------------------------------------


def test_duplicates_within_batch_are_collapsed_last_wins(db_session: Session) -> None:
    source = make_source(db_session)
    postings = [
        make_posting("a", title="First version"),
        make_posting("b"),
        make_posting("a", title="Second version"),
    ]

    result = upsert_job_postings(db_session, source_id=source.id, postings=postings)

    assert result == UpsertResult(inserted=2, updated=0, duplicates_in_batch=1)
    assert fetch_posting(db_session, source.id, "a").title == "Second version"


def test_same_external_id_in_different_sources_is_not_a_duplicate(
    db_session: Session,
) -> None:
    board_a = make_source(db_session, "Board A")
    board_b = make_source(db_session, "Board B")

    upsert_job_postings(db_session, source_id=board_a.id, postings=[make_posting("42")])
    result = upsert_job_postings(
        db_session, source_id=board_b.id, postings=[make_posting("42")]
    )

    assert result.inserted == 1
    assert count_postings(db_session) == 2


# --- Update semantics -------------------------------------------------------


def test_upsert_updates_required_fields(db_session: Session) -> None:
    source = make_source(db_session)
    upsert_job_postings(db_session, source_id=source.id, postings=[make_posting("a")])

    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[
            make_posting(
                "a",
                title="Senior Engineer",
                company_name="Acme Corp",
                job_url="https://jobs.example.com/a-v2",
            )
        ],
    )

    stored = fetch_posting(db_session, source.id, "a")
    assert stored.title == "Senior Engineer"
    assert stored.company_name == "Acme Corp"
    assert stored.job_url == "https://jobs.example.com/a-v2"


def test_null_values_preserve_existing_data(db_session: Session) -> None:
    source = make_source(db_session)
    published = datetime(2026, 11, 1, 9, 30, tzinfo=UTC)
    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[
            make_posting(
                "a",
                description="Original description",
                salary_min=50000,
                salary_max=70000,
                salary_currency="eur",
                published_at=published,
            )
        ],
    )

    # Second sighting omits the optional fields but changes one of them.
    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[make_posting("a", salary_max=75000)],
    )

    stored = fetch_posting(db_session, source.id, "a")
    assert stored.description == "Original description"
    assert stored.salary_min == 50000
    assert stored.salary_max == 75000
    assert stored.salary_currency == "EUR"
    assert stored.published_at == published


def test_last_seen_refreshed_and_first_seen_preserved(
    committed_sessions: sessionmaker[Session],
) -> None:
    # now() is fixed per transaction, so each sighting must be its own committed
    # transaction for the timestamps to differ.
    with committed_sessions() as session:
        source_id = make_source(session).id
        upsert_job_postings(session, source_id=source_id, postings=[make_posting("a")])
        session.commit()

    with committed_sessions() as session:
        first = fetch_posting(session, source_id, "a")
        first_seen, last_seen = first.first_seen_at, first.last_seen_at

    time.sleep(0.05)

    with committed_sessions() as session:
        upsert_job_postings(
            session, source_id=source_id, postings=[make_posting("a", title="Updated")]
        )
        session.commit()

    with committed_sessions() as session:
        second = fetch_posting(session, source_id, "a")
        assert second.title == "Updated"
        assert second.first_seen_at == first_seen
        assert second.last_seen_at > last_seen
        assert second.last_seen_at > second.first_seen_at


# --- Transactions ------------------------------------------------------------


def test_rollback_discards_upsert(committed_sessions: sessionmaker[Session]) -> None:
    with committed_sessions() as session:
        source_id = make_source(session).id
        session.commit()

    with committed_sessions() as session:
        result = upsert_job_postings(
            session,
            source_id=source_id,
            postings=[make_posting(str(i)) for i in range(10)],
        )
        assert result.inserted == 10
        session.rollback()

    with committed_sessions() as session:
        assert count_postings(session) == 0


def test_upsert_does_not_commit(committed_sessions: sessionmaker[Session]) -> None:
    with committed_sessions() as session:
        source_id = make_source(session).id
        session.commit()

    with committed_sessions() as writer, committed_sessions() as reader:
        upsert_job_postings(writer, source_id=source_id, postings=[make_posting("a")])
        assert count_postings(reader) == 0  # uncommitted work is invisible
        writer.commit()
        assert count_postings(reader) == 1


# --- Retrieval and relationships -------------------------------------------


def test_get_job_posting_by_id(db_session: Session) -> None:
    source = make_source(db_session)
    upsert_job_postings(db_session, source_id=source.id, postings=[make_posting("a")])
    stored_id = fetch_posting(db_session, source.id, "a").id

    found = get_job_posting_by_id(db_session, stored_id)

    assert found is not None
    assert JobPostingResponse.model_validate(found).external_id == "a"
    assert get_job_posting_by_id(db_session, 999_999) is None


def test_list_job_postings_by_source_filters_and_paginates(db_session: Session) -> None:
    board_a = make_source(db_session, "Board A")
    board_b = make_source(db_session, "Board B")
    upsert_job_postings(
        db_session,
        source_id=board_a.id,
        postings=[make_posting(str(i)) for i in range(5)],
    )
    upsert_job_postings(db_session, source_id=board_b.id, postings=[make_posting("x")])

    first_page = list_job_postings_by_source(db_session, board_a.id, limit=3)
    second_page = list_job_postings_by_source(db_session, board_a.id, limit=3, offset=3)

    assert [p.external_id for p in first_page] == ["0", "1", "2"]
    assert [p.external_id for p in second_page] == ["3", "4"]
    assert all(p.source_id == board_a.id for p in first_page + second_page)


def test_relationships_load_in_both_directions(db_session: Session) -> None:
    source = make_source(db_session)
    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[make_posting("a"), make_posting("b")],
    )
    db_session.expire_all()

    loaded = db_session.scalars(
        select(Source)
        .where(Source.id == source.id)
        .options(selectinload(Source.job_postings))
    ).one()

    assert sorted(p.external_id for p in loaded.job_postings) == ["a", "b"]
    assert all(p.source is loaded for p in loaded.job_postings)


# --- Hardening: M1 monotonic last_seen_at ------------------------------------


def test_last_seen_never_moves_backwards_with_concurrent_transactions(
    committed_sessions: sessionmaker[Session],
) -> None:
    """An older transaction finishing after a newer one must not regress last_seen_at.

    now() is the transaction start time. The older transaction starts first,
    the newer one inserts and commits, then the older one updates the row.
    """
    with committed_sessions() as session:
        source_id = make_source(session).id
        session.commit()

    with committed_sessions() as older:
        older_started = older.execute(text("SELECT now()")).scalar_one()  # pins now()
        time.sleep(0.05)

        with committed_sessions() as newer:
            upsert_job_postings(newer, source_id=source_id, postings=[make_posting("a")])
            newer.commit()

        with committed_sessions() as reader:
            after_newer = fetch_posting(reader, source_id, "a")
            newer_first_seen = after_newer.first_seen_at
            newer_last_seen = after_newer.last_seen_at

        upsert_job_postings(
            older,
            source_id=source_id,
            postings=[make_posting("a", title="Written by older transaction")],
        )
        older.commit()

    with committed_sessions() as reader:
        final = fetch_posting(reader, source_id, "a")

    assert older_started < newer_last_seen
    assert final.title == "Written by older transaction"  # the update itself applied
    assert final.last_seen_at == newer_last_seen  # not regressed to older_started
    assert final.first_seen_at == newer_first_seen
    assert final.last_seen_at >= final.first_seen_at


# --- Hardening: M2 blank optional strings -------------------------------------


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_optional_strings_preserve_existing_data(
    db_session: Session, blank: str
) -> None:
    source = make_source(db_session)
    original = {
        "description": "Original description",
        "employment_type": "full_time",
        "application_url": "https://apply.example.com/a",
        "salary_currency": "EUR",
        "location_restrictions": "Europe",
    }
    upsert_job_postings(
        db_session, source_id=source.id, postings=[make_posting("a", **original)]
    )

    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[make_posting("a", **dict.fromkeys(original, blank))],
    )

    stored = fetch_posting(db_session, source.id, "a")
    for field, value in original.items():
        assert getattr(stored, field) == value, field


# --- Hardening: M3 deterministic write order ----------------------------------


def test_rows_are_written_in_external_id_order_regardless_of_input_order(
    db_session: Session,
) -> None:
    source = make_source(db_session)

    upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[make_posting(key) for key in ("d", "b", "a", "c")],
        chunk_size=1,  # one statement per row: id order == execution order
    )

    by_id = db_session.scalars(
        select(JobPosting.external_id)
        .where(JobPosting.source_id == source.id)
        .order_by(JobPosting.id)
    ).all()
    assert by_id == ["a", "b", "c", "d"]


def test_sorting_keeps_last_occurrence_wins_and_statistics(db_session: Session) -> None:
    source = make_source(db_session)
    upsert_job_postings(db_session, source_id=source.id, postings=[make_posting("c")])

    result = upsert_job_postings(
        db_session,
        source_id=source.id,
        postings=[
            make_posting("b"),
            make_posting("a", title="First"),
            make_posting("c", title="Updated c"),
            make_posting("a", title="Last"),
        ],
        chunk_size=2,
    )

    assert result == UpsertResult(inserted=2, updated=1, duplicates_in_batch=1)
    assert fetch_posting(db_session, source.id, "a").title == "Last"
    assert fetch_posting(db_session, source.id, "c").title == "Updated c"


def test_concurrent_upserts_in_opposite_input_order_do_not_deadlock(
    committed_sessions: sessionmaker[Session],
) -> None:
    """Two writers touching the same rows must lock them in the same order.

    Writer 1 pauses after its first chunk while holding that row's lock.
    Writer 2 then upserts the same keys in reverse input order. With a
    consistent (sorted) order, writer 2 waits on the first row; without it,
    writer 2 would lock the second row first and PostgreSQL would abort one
    writer with a deadlock.
    """
    with committed_sessions() as session:
        source_id = make_source(session).id
        upsert_job_postings(
            session, source_id=source_id, postings=[make_posting("a"), make_posting("b")]
        )
        session.commit()

    first_chunk_done = threading.Event()

    def writer_1() -> UpsertResult:
        try:
            with committed_sessions() as session:
                connection = session.connection()
                statements = 0

                def pause_after_first_chunk(*args: object) -> None:
                    nonlocal statements
                    statement = str(args[2])
                    if statement.lstrip().upper().startswith("INSERT INTO JOB_POSTINGS"):
                        statements += 1
                        if statements == 1:
                            first_chunk_done.set()
                            time.sleep(0.5)

                event.listen(connection, "after_cursor_execute", pause_after_first_chunk)
                result = upsert_job_postings(
                    session,
                    source_id=source_id,
                    postings=[make_posting("a"), make_posting("b")],
                    chunk_size=1,
                )
                session.commit()
                return result
        finally:
            first_chunk_done.set()

    def writer_2() -> UpsertResult:
        assert first_chunk_done.wait(timeout=5)
        with committed_sessions() as session:
            result = upsert_job_postings(
                session,
                source_id=source_id,
                postings=[make_posting("b"), make_posting("a")],
                chunk_size=1,
            )
            session.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_1, future_2 = pool.submit(writer_1), pool.submit(writer_2)
        result_1 = future_1.result(timeout=20)
        result_2 = future_2.result(timeout=20)

    expected = UpsertResult(inserted=0, updated=2, duplicates_in_batch=0)
    assert result_1 == expected
    assert result_2 == expected
