"""Persistence stage of the ingestion pipeline.

Delegates to the repository layer. Does not commit: the caller owns the
transaction, so a failed run leaves no partial writes.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from career_intelligence.db.repositories import (
    UpsertResult,
    get_or_create_source_by_name,
    upsert_job_postings,
)
from career_intelligence.ingestion.service import CollectionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    source_id: int
    source_created: bool
    fetched: int
    valid: int
    rejected: int
    upsert: UpsertResult


def persist_collection(session: Session, collection: CollectionResult) -> IngestionResult:
    metadata = collection.source
    registration = get_or_create_source_by_name(
        session,
        name=metadata.name,
        url=metadata.url,
        source_type=metadata.source_type,
    )
    upsert = upsert_job_postings(
        session,
        source_id=registration.source.id,
        postings=collection.postings,
    )

    result = IngestionResult(
        source_id=registration.source.id,
        source_created=registration.created,
        fetched=collection.fetched,
        valid=len(collection.postings),
        rejected=len(collection.rejected),
        upsert=upsert,
    )
    logger.info(
        "ingestion_persisted source=%r source_id=%d source_created=%s inserted=%d updated=%d "
        "duplicates_in_batch=%d",
        metadata.name,
        result.source_id,
        result.source_created,
        upsert.inserted,
        upsert.updated,
        upsert.duplicates_in_batch,
    )
    return result
