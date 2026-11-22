"""Collection stage of the ingestion pipeline: fetch, normalize and validate.

Database-free by design (no SQLAlchemy import), so dry runs cannot write.
Persistence lives in ``career_intelligence.ingestion.persistence``.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from career_intelligence.ingestion.contracts import (
    IngestionError,
    MappingError,
    SourceAdapter,
    SourceMetadata,
)
from career_intelligence.schemas import JobPostingCreate, SourceCreate

logger = logging.getLogger(__name__)

# Most frequent rejection reasons shown in the aggregated warning and by the CLI.
MAX_REASONS_SHOWN = 10


class AdapterContractError(IngestionError):
    """Raised when an adapter's metadata cannot be registered as a source."""


@dataclass(frozen=True)
class RejectedPosting:
    position: int  # 0-based position in the fetched stream
    external_id: str | None
    reason: str  # field names and messages only, never raw values


@dataclass(frozen=True)
class CollectionResult:
    source: SourceMetadata
    postings: tuple[JobPostingCreate, ...]
    rejected: tuple[RejectedPosting, ...]

    @property
    def fetched(self) -> int:
        return len(self.postings) + len(self.rejected)


def count_rejection_reasons(rejected: Iterable[RejectedPosting]) -> list[tuple[str, int]]:
    """(reason, count) pairs, most frequent first, ties broken by reason. Deterministic."""
    counts = Counter(item.reason for item in rejected)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def validate_metadata(metadata: SourceMetadata) -> None:
    try:
        SourceCreate(name=metadata.name, url=metadata.url, source_type=metadata.source_type)
    except ValidationError as exc:
        raise AdapterContractError(f"Invalid source metadata: {_summarize(exc)}") from None


def collect_postings(adapter: SourceAdapter) -> CollectionResult:
    """Fetch every raw record, map and validate it.

    Invalid records are rejected individually and reported. Fetch failures
    propagate: a partial collection is never returned as if it were complete.
    """
    metadata = adapter.metadata
    validate_metadata(metadata)
    logger.info("collection_started source=%r", metadata.name)

    postings: list[JobPostingCreate] = []
    rejected: list[RejectedPosting] = []

    for position, raw in enumerate(adapter.fetch_raw_postings()):
        if not isinstance(raw, Mapping):
            rejected.append(RejectedPosting(position, None, "raw posting is not an object"))
            continue

        try:
            fields = adapter.map_posting(raw)
        except MappingError as exc:
            rejected.append(RejectedPosting(position, None, f"mapping: {exc}"))
            continue

        external_id = fields.get("external_id")
        try:
            postings.append(JobPostingCreate.model_validate(fields))
        except ValidationError as exc:
            rejected.append(
                RejectedPosting(
                    position,
                    str(external_id) if external_id is not None else None,
                    f"validation: {_summarize(exc)}",
                )
            )

    # One line per rejection at DEBUG; a single aggregated WARNING per collection.
    for item in rejected:
        logger.debug(
            "posting_rejected source=%r position=%d external_id=%r reason=%r",
            metadata.name,
            item.position,
            item.external_id,
            item.reason,
        )
    if rejected:
        reasons = count_rejection_reasons(rejected)
        logger.warning(
            "rejections_summary source=%r rejected=%d distinct_reasons=%d omitted_reasons=%d "
            "top_reasons=%r",
            metadata.name,
            len(rejected),
            len(reasons),
            max(len(reasons) - MAX_REASONS_SHOWN, 0),
            dict(reasons[:MAX_REASONS_SHOWN]),
        )
    logger.info(
        "collection_finished source=%r fetched=%d valid=%d rejected=%d",
        metadata.name,
        len(postings) + len(rejected),
        len(postings),
        len(rejected),
    )
    return CollectionResult(source=metadata, postings=tuple(postings), rejected=tuple(rejected))


def _summarize(exc: ValidationError) -> str:
    """Field locations and messages only; input values are deliberately omitted."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'record'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )
