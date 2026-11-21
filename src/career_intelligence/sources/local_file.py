"""Local JSON file adapter: a DEMONSTRATION source.

Not a production job source: it runs the ingestion pipeline from a local file
for dry runs, demos and tests, with no network access. Its ``source_type`` is
``DEMO_SOURCE_TYPE``, so the CLI refuses to persist it unless both
``--persist`` and ``--allow-demo-data`` are given. Its URL uses the reserved
``.invalid`` top-level domain (RFC 2606), which can never resolve.

Expected file format::

    {
      "postings": [
        {
          "id": "abc-1",                       # string or integer (required)
          "title": "Backend Engineer",         # required
          "company": {"name": "Acme", "slug": "acme"},   # name required
          "url": "https://example.com/jobs/abc-1",       # required
          "employment_type": "Full-time",
          "locations": ["Lisboa", "Porto"],
          "description": "...",
          "salary": {"min": 30000, "max": 45000, "currency": "EUR", "period": "year"},
          "published_at": "2026-11-20T09:00:00+00:00"    # must include a UTC offset
        }
      ]
    }

No timezone, currency or salary period is assumed: values are passed through
as given and validated by ``JobPostingCreate``.
"""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from career_intelligence.ingestion.contracts import (
    DEMO_SOURCE_TYPE,
    MappingError,
    RawPosting,
    SourceFetchError,
    SourceMetadata,
)

DEFAULT_METADATA = SourceMetadata(
    name="Demo: local fixture file",
    url="https://demo.invalid/local-fixture",
    source_type=DEMO_SOURCE_TYPE,
)


class LocalFileAdapter:
    def __init__(self, path: Path, *, metadata: SourceMetadata = DEFAULT_METADATA) -> None:
        self._path = path
        self._metadata = metadata

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SourceFetchError(f"Cannot read {self._path}: {exc.strerror}") from None

        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceFetchError(
                f"{self._path} is not valid JSON (line {exc.lineno}, column {exc.colno})"
            ) from None

        postings = document.get("postings") if isinstance(document, dict) else None
        if not isinstance(postings, list):
            raise SourceFetchError(f'{self._path} must contain an object with a "postings" list')

        yield from postings

    def map_posting(self, raw: RawPosting) -> dict[str, Any]:
        return map_local_posting(raw)


def map_local_posting(raw: RawPosting) -> dict[str, Any]:
    """Normalize one local record into ``JobPostingCreate`` fields. Pure function."""
    external_id = raw.get("id")
    if isinstance(external_id, bool) or not isinstance(external_id, str | int):
        raise MappingError("id must be a string or an integer")

    company = raw.get("company")
    if not isinstance(company, Mapping):
        raise MappingError("company must be an object")

    fields: dict[str, Any] = {
        "external_id": str(external_id),
        "title": raw.get("title"),
        "company_name": company.get("name"),
        "company_slug": company.get("slug"),
        "job_url": raw.get("url"),
        "employment_type": raw.get("employment_type"),
        "location_restrictions": _join_locations(raw.get("locations")),
        "description": raw.get("description"),
        "published_at": raw.get("published_at"),
    }

    salary = raw.get("salary")
    if salary is not None:
        if not isinstance(salary, Mapping):
            raise MappingError("salary must be an object")
        fields.update(
            salary_min=salary.get("min"),
            salary_max=salary.get("max"),
            salary_currency=salary.get("currency"),
            salary_period=salary.get("period"),
        )

    return fields


def _join_locations(locations: object) -> str | None:
    if locations is None:
        return None
    if not isinstance(locations, list) or not all(isinstance(item, str) for item in locations):
        raise MappingError("locations must be a list of strings")
    names = [item.strip() for item in locations if item.strip()]
    return ", ".join(names) or None
