from collections.abc import Sequence
from dataclasses import dataclass
from itertools import batched
from typing import NamedTuple

from sqlalchemy import func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from career_intelligence.models import JobPosting, Source
from career_intelligence.schemas import JobPostingCreate, SourceUpdate

# PostgreSQL SQLSTATE for foreign_key_violation.
_FOREIGN_KEY_VIOLATION = "23503"

_JOB_POSTINGS = JobPosting.__table__
_SOURCES = Source.__table__


class SourceInUseError(Exception):
    """Raised when a source cannot be deleted because job postings reference it."""

    def __init__(self, source_id: int) -> None:
        super().__init__(f"Source {source_id} has associated job postings.")
        self.source_id = source_id


class SourceRegistration(NamedTuple):
    source: Source
    created: bool  # True: newly inserted. False: an existing source was reused.


@dataclass(frozen=True)
class UpsertResult:
    inserted: int
    updated: int
    duplicates_in_batch: int


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def create_source(
    session: Session,
    *,
    name: str,
    url: str,
    source_type: str,
) -> Source:
    source = Source(
        name=name,
        url=url,
        source_type=source_type,
    )

    session.add(source)
    session.commit()
    session.refresh(source)

    return source


def get_source_by_id(
    session: Session,
    source_id: int,
) -> Source | None:
    return session.get(Source, source_id)


def list_sources(session: Session) -> list[Source]:
    statement = select(Source).order_by(Source.id)
    return list(session.scalars(statement).all())


def update_source(
    session: Session,
    source_id: int,
    updates: SourceUpdate,
) -> Source | None:
    source = session.get(Source, source_id)

    if source is None:
        return None

    update_data = updates.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if field == "url":
            value = str(value)

        setattr(source, field, value)

    session.commit()
    session.refresh(source)

    return source


def delete_source(session: Session, source_id: int) -> bool:
    """Delete a source. Raises SourceInUseError if job postings reference it."""
    source = session.get(Source, source_id)

    if source is None:
        return False

    session.delete(source)

    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        if getattr(exc.orig, "sqlstate", None) == _FOREIGN_KEY_VIOLATION:
            raise SourceInUseError(source_id) from exc
        raise

    return True


def get_or_create_source_by_name(
    session: Session,
    *,
    name: str,
    url: str,
    source_type: str,
) -> SourceRegistration:
    """Return the source called ``name``, creating it if it does not exist.

    An existing source is returned unchanged: ``url`` and ``source_type`` are
    only used when a new source is created.

    Concurrency-safe without catching IntegrityError: ``ON CONFLICT DO NOTHING``
    waits for any concurrent transaction inserting the same name. If that
    transaction commits, nothing is inserted here and the follow-up SELECT
    (a new snapshot under READ COMMITTED) sees the committed row; if it rolls
    back, this insert proceeds.

    Does not commit: the caller owns the transaction. PostgreSQL only.
    """
    statement = (
        pg_insert(_SOURCES)
        .values(name=name, url=url, source_type=source_type)
        .on_conflict_do_nothing(index_elements=[_SOURCES.c.name])
        .returning(_SOURCES.c.id)
    )
    new_id = session.execute(statement).scalar_one_or_none()

    if new_id is not None:
        source = session.get(Source, new_id)
        assert source is not None
        return SourceRegistration(source=source, created=True)

    existing = session.scalars(select(Source).where(Source.name == name)).one()
    return SourceRegistration(source=existing, created=False)


# ---------------------------------------------------------------------------
# Job postings
# ---------------------------------------------------------------------------

# Always overwritten on conflict: NOT NULL columns, so a value is always provided.
_ALWAYS_UPDATED_COLUMNS = ("title", "company_name", "job_url")

# Overwritten only when the incoming value is not NULL (COALESCE). A missing
# value never erases stored data. Explicitly clearing a field is not supported
# yet and would need an opt-in mechanism.
_NULL_PRESERVING_COLUMNS = tuple(
    name
    for name in JobPostingCreate.model_fields
    if name != "external_id" and name not in _ALWAYS_UPDATED_COLUMNS
)


def upsert_job_postings(
    session: Session,
    *,
    source_id: int,
    postings: Sequence[JobPostingCreate],
    chunk_size: int = 1000,
) -> UpsertResult:
    """Insert or update a batch of job postings for one source.

    Matching is on ``(source_id, external_id)``. For existing postings:
    required fields are overwritten, optional fields are overwritten only when
    the incoming value is not None (None preserves stored data),
    ``last_seen_at`` is refreshed, and ``first_seen_at`` is never changed.

    Postings repeating an ``external_id`` within the batch are collapsed (the
    last occurrence wins) and counted in ``duplicates_in_batch``.

    Rows are written in ``external_id`` order (plain string ordering), whatever
    the input order. Concurrent upserts over overlapping postings therefore
    lock rows in the same order and cannot deadlock each other.

    ``last_seen_at`` only ever moves forward: an older transaction that commits
    after a newer one keeps the newer timestamp.

    Executes one multi-row statement per chunk. Does not commit: the caller
    owns the transaction. Rows are written with Core statements, so ORM
    objects already loaded in ``session`` are not refreshed. PostgreSQL only.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    unique_rows: dict[str, dict[str, object]] = {}
    for posting in postings:
        unique_rows[posting.external_id] = {"source_id": source_id, **posting.model_dump()}

    duplicates_in_batch = len(postings) - len(unique_rows)
    ordered_rows = [unique_rows[external_id] for external_id in sorted(unique_rows)]
    inserted = 0
    updated = 0

    for chunk in batched(ordered_rows, chunk_size):
        chunk_inserted = _upsert_chunk(session, list(chunk))
        inserted += chunk_inserted
        updated += len(chunk) - chunk_inserted

    return UpsertResult(
        inserted=inserted,
        updated=updated,
        duplicates_in_batch=duplicates_in_batch,
    )


def _upsert_chunk(session: Session, rows: list[dict[str, object]]) -> int:
    """Upsert rows and return how many were newly inserted.

    Implementation detail: a row created by this statement has system column
    ``xmax = 0``, while a row updated through ON CONFLICT DO UPDATE carries
    the updating transaction's id. This is the standard PostgreSQL technique
    for telling inserts from updates in a single round trip.
    """
    statement = pg_insert(_JOB_POSTINGS).values(rows)
    excluded = statement.excluded

    set_: dict[str, object] = {name: excluded[name] for name in _ALWAYS_UPDATED_COLUMNS}
    set_.update(
        {
            name: func.coalesce(excluded[name], _JOB_POSTINGS.c[name])
            for name in _NULL_PRESERVING_COLUMNS
        }
    )
    # now() is the transaction start time; GREATEST keeps last_seen_at monotonic
    # when an older transaction updates the row after a newer one committed.
    set_["last_seen_at"] = func.greatest(_JOB_POSTINGS.c.last_seen_at, func.now())

    statement = statement.on_conflict_do_update(
        constraint="uq_job_postings_source_external_id",
        set_=set_,
    ).returning(literal_column("(xmax = 0)").label("inserted"))

    flags = session.execute(statement).scalars().all()
    return sum(1 for flag in flags if flag)


def get_job_posting_by_id(
    session: Session,
    job_posting_id: int,
) -> JobPosting | None:
    return session.get(JobPosting, job_posting_id)


def list_job_postings_by_source(
    session: Session,
    source_id: int,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[JobPosting]:
    statement = (
        select(JobPosting)
        .where(JobPosting.source_id == source_id)
        .order_by(JobPosting.id)
        .limit(limit)
        .offset(offset)
    )
    return list(session.scalars(statement).all())
