"""Contract between job sources and the ingestion pipeline.

Pipeline: ``fetch_raw_postings`` (collection) -> ``map_posting`` (normalization)
-> ``JobPostingCreate`` (validation, done by the service) -> repository
(persistence, done outside the adapter).

Adapters must not import SQLAlchemy or FastAPI, nor touch the database.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

type RawPosting = Mapping[str, Any]

# Sources with this type hold demonstration data, which must never be persisted
# without an explicit opt-in (see the ``--allow-demo-data`` CLI flag).
DEMO_SOURCE_TYPE = "demo"


@dataclass(frozen=True)
class SourceMetadata:
    """Identity under which a source is registered (see ``sources`` table)."""

    name: str
    url: str
    source_type: str


class IngestionError(Exception):
    """Base class for failures that abort an ingestion run.

    Messages must never contain credentials. A run that raises one of these
    has not produced a complete collection and must not be persisted.
    """


class SourceFetchError(IngestionError):
    """Raised by an adapter when its raw data cannot be obtained or read."""


class MappingError(ValueError):
    """Raised by ``map_posting`` when a raw record cannot be normalized."""


class SourceAdapter(Protocol):
    @property
    def metadata(self) -> SourceMetadata: ...

    def fetch_raw_postings(self) -> Iterable[RawPosting]:
        """Yield raw records as delivered by the source.

        Failures must be raised as ``IngestionError`` subclasses (e.g.
        ``SourceFetchError``, ``HttpClientError``, ``IncompletePaginationError``);
        any of them aborts the run.
        """
        ...

    def map_posting(self, raw: RawPosting) -> dict[str, Any]:
        """Normalize one raw record into ``JobPostingCreate`` field names.

        Must be free of I/O. Raise ``MappingError`` for records that cannot be
        mapped; omit fields that have no safe correspondence.
        """
        ...
