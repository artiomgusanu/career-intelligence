"""ITJobs.pt source: list-endpoint adapter and record mapper.

The record and envelope shapes are taken from the public API documentation
(``/api/docs/job``). They have not been confirmed against real responses.

Adapter
-------

``ITJobsAdapter`` pages through ``GET /job/list.json`` (``api_key``, ``page``,
``limit``) with an injected client that only needs ``get_json`` (the
``JsonClient`` protocol). Callers pass an ``HttpClient`` built for
``ITJOBS_BASE_URL`` with ``min_interval_seconds=ITJOBS_MIN_INTERVAL_SECONDS``,
and own its lifecycle (``with``); the adapter never closes it. ``httpx`` is
deliberately not imported here, so importing this module (for the mapper)
does not load the HTTP stack.

The API key is injected by the caller and only ever sent as the ``api_key``
query parameter, which ``HttpClient`` redacts in logs and errors. The adapter
never puts it, or any text returned by the API, into its own messages.

A collection is all or nothing. Any inconsistency raises an
``IngestionError`` and aborts the run, so a partial collection can never be
persisted:

- envelope: not an object, an ``error`` member, ``total``/``page``/``limit``
  missing or not integers, ``page`` or ``limit`` differing from the request,
  ``results`` not a list or longer than ``limit``;
- across pages: ``total`` changing, a page other than the last not full, the
  last page not holding exactly the remaining postings, a posting ``id``
  seen twice, or the collected count differing from ``total``;
- size: more pages than ``max_pages`` (checked after the first page, before
  any further request).

Exactly ``max(1, ceil(total / page_size))`` pages are requested.

Mapper
------

Mapped fields:

- ``id`` (positive integer)  -> ``external_id``
- ``title``                  -> ``title``
- ``company.name``           -> ``company_name`` (a ``company`` object is required)
- ``company.slug``           -> ``company_slug``
- ``types[].name``           -> ``employment_type`` (names as given, joined)
- ``body`` (HTML, untrusted) -> ``description`` (stored as given; sanitize on display)

``job_url`` is never produced. The API has no field with the posting's public
URL and the site's URL pattern is not documented, so none is derived from
``id``, ``slug`` or any other field. ``job_url`` is required by
``JobPostingCreate``, so every ITJobs record is rejected at validation
(``job_url: Field required``) until a confirmed source for it exists.

Deliberately not read:

- ``publishedAt`` / ``updatedAt``: no timezone, none is assumed.
- ``salaryMin`` / ``salaryMax`` / ``wage``: currency and period undocumented.
- ``locations`` / ``country``: semantics not confirmed.
- ``contracts`` / ``workModel``: separate dimensions, never mixed into ``types``.
- ``company`` contact data (``email``, ``phone``, ``fax``, ``address``) and
  ``company.url`` / ``logo`` / ``description``, ``companyId``, ``ref``, ``slug``.

Nothing is inferred about a posting's status: no record is marked as expired
or closed. Scalar types (``title``, ``company.name``, ``body``) are left to
``JobPostingCreate``; this module validates structure only.
"""

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from career_intelligence.ingestion.contracts import (
    MappingError,
    RawPosting,
    SourceFetchError,
    SourceMetadata,
)
from career_intelligence.ingestion.pagination import (
    IncompletePaginationError,
    Page,
    PaginationLimitError,
    paginate,
)

logger = logging.getLogger(__name__)

ITJOBS_BASE_URL = "https://api.itjobs.pt"
ITJOBS_LIST_PATH = "/job/list.json"
ITJOBS_MIN_INTERVAL_SECONDS = 1.0  # robots.txt Crawl-delay; no API rate limit is documented
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_PAGES = 100

ITJOBS_METADATA = SourceMetadata(
    name="ITJobs.pt",
    url="https://www.itjobs.pt/",
    source_type="job_board",
)


class JsonClient(Protocol):
    """The part of ``HttpClient`` the adapter uses."""

    def get_json(
        self, path: str, params: Mapping[str, str | int | float | None] | None = None
    ) -> Any: ...


@dataclass(frozen=True)
class ListPage:
    total: int
    page: int
    limit: int
    results: Sequence[Any]


def parse_list_envelope(payload: object, *, requested_page: int, requested_limit: int) -> ListPage:
    """Validate one ``job/list`` response body. Pure function.

    Raises ``SourceFetchError`` with a generic message: neither the payload
    nor any API-provided text is included.
    """
    if not isinstance(payload, Mapping):
        raise SourceFetchError("ITJobs list response is not a JSON object")
    if "error" in payload:
        raise SourceFetchError("ITJobs API returned an error response")

    total = _envelope_int(payload, "total", minimum=0)
    page = _envelope_int(payload, "page", minimum=1)
    limit = _envelope_int(payload, "limit", minimum=1)
    results = payload.get("results")

    if page != requested_page:
        raise SourceFetchError(
            f"ITJobs list response is for page {page}, but page {requested_page} was requested"
        )
    if limit != requested_limit:
        raise SourceFetchError(
            f"ITJobs list response has limit {limit}, but limit {requested_limit} was requested"
        )
    if not isinstance(results, list):
        raise SourceFetchError('ITJobs list response field "results" must be a list')
    if len(results) > limit:
        raise SourceFetchError(
            f"ITJobs list response has {len(results)} results, more than its limit {limit}"
        )
    return ListPage(total=total, page=page, limit=limit, results=results)


def _envelope_int(payload: Mapping[str, Any], name: str, *, minimum: int) -> int:
    value = payload.get(name)
    # bool is a subclass of int and must not pass as a count.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SourceFetchError(
            f'ITJobs list response field "{name}" must be an integer >= {minimum}'
        )
    return value


class ITJobsAdapter:
    """Collects every posting from ``job/list``. See the module docs for guarantees."""

    def __init__(
        self,
        client: JsonClient,
        *,
        api_key: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = DEFAULT_MAX_PAGES,
    ) -> None:
        # Messages never echo the rejected value: it may be a credential.
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")
        for name, value in (("page_size", page_size), ("max_pages", max_pages)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1.")

        self._client = client
        self._api_key = api_key
        self._page_size = page_size
        self._max_pages = max_pages

    @property
    def metadata(self) -> SourceMetadata:
        return ITJOBS_METADATA

    def map_posting(self, raw: RawPosting) -> dict[str, Any]:
        return map_itjobs_posting(raw)

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        """Yield every posting, then verify the collection was complete.

        State is local to each call, so the adapter can be run more than once.
        """
        expected_total: int | None = None
        seen_ids: set[int] = set()
        collected = 0

        def fetch_page(page_number: int) -> Page[RawPosting]:
            nonlocal expected_total, collected
            payload = self._client.get_json(
                ITJOBS_LIST_PATH,
                {"api_key": self._api_key, "page": page_number, "limit": self._page_size},
            )
            listing = parse_list_envelope(
                payload, requested_page=page_number, requested_limit=self._page_size
            )

            if expected_total is None:
                expected_total = listing.total
                pages_needed = max(1, math.ceil(expected_total / self._page_size))
                if pages_needed > self._max_pages:
                    raise PaginationLimitError(
                        f"ITJobs reports {expected_total} postings ({pages_needed} pages of "
                        f"{self._page_size}), more than max_pages={self._max_pages}"
                    )
            elif listing.total != expected_total:
                raise IncompletePaginationError(
                    f"ITJobs total changed from {expected_total} to {listing.total} "
                    f"during the collection (page {page_number})"
                )

            remaining = expected_total - (page_number - 1) * self._page_size
            has_next = remaining > self._page_size
            expected_count = self._page_size if has_next else max(remaining, 0)
            if len(listing.results) != expected_count:
                raise IncompletePaginationError(
                    f"ITJobs page {page_number} has {len(listing.results)} postings, "
                    f"expected {expected_count}"
                )

            for item in listing.results:
                posting_id = item.get("id") if isinstance(item, Mapping) else None
                if isinstance(posting_id, int) and not isinstance(posting_id, bool):
                    if posting_id in seen_ids:
                        raise IncompletePaginationError(
                            f"ITJobs posting id {posting_id} was returned twice "
                            f"(page {page_number})"
                        )
                    seen_ids.add(posting_id)

            collected += len(listing.results)
            logger.info(
                "itjobs_page page=%d results=%d total=%d",
                page_number,
                len(listing.results),
                expected_total,
            )
            return Page(items=listing.results, has_next=has_next)

        yield from paginate(fetch_page, max_pages=self._max_pages)

        if collected != expected_total:  # pragma: no cover - guarded per page above
            raise IncompletePaginationError(
                f"ITJobs collection has {collected} postings, expected {expected_total}"
            )


def map_itjobs_posting(raw: RawPosting) -> dict[str, Any]:
    """Normalize one ITJobs record into ``JobPostingCreate`` fields. Pure function.

    Reads a fixed set of keys and never iterates the record, so unknown or
    excluded fields cannot leak into the result. ``raw`` is not modified.
    """
    company = raw.get("company")
    if not isinstance(company, Mapping):
        raise MappingError("company must be an object")

    return {
        "external_id": _external_id(raw.get("id")),
        "title": raw.get("title"),
        "company_name": company.get("name"),
        "company_slug": company.get("slug"),
        "employment_type": _employment_type(raw.get("types")),
        "description": raw.get("body"),
    }


def _external_id(value: object) -> str:
    # bool is a subclass of int and must not pass as an id.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MappingError("id must be a positive integer")
    return str(value)


def _employment_type(types: object) -> str | None:
    if types is None:
        return None
    if not isinstance(types, list):
        raise MappingError("types must be a list")

    names: list[str] = []
    for item in types:
        if not isinstance(item, Mapping):
            raise MappingError("types items must be objects")
        name = item.get("name")
        if not isinstance(name, str):
            raise MappingError("types[].name must be a string")
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names) or None
