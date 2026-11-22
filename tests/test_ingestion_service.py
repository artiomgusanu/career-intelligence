"""Collection pipeline tests: contract, normalization, validation, HTTP end to end."""

import itertools
import logging
import subprocess
import sys
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
import pytest

from career_intelligence.config import HttpSettings
from career_intelligence.ingestion.contracts import (
    MappingError,
    RawPosting,
    SourceFetchError,
    SourceMetadata,
)
from career_intelligence.ingestion.http import HttpClient, HttpStatusError
from career_intelligence.ingestion.pagination import Page, paginate
from career_intelligence.ingestion.service import (
    MAX_REASONS_SHOWN,
    AdapterContractError,
    RejectedPosting,
    collect_postings,
    count_rejection_reasons,
)

METADATA = SourceMetadata(name="Fake Board", url="https://fake.example.com/", source_type="test")


def raw(external_id: str, **overrides: Any) -> dict[str, Any]:
    record = {
        "ref": external_id,
        "headline": f"Engineer {external_id}",
        "employer": "Acme",
        "link": f"https://fake.example.com/jobs/{external_id}",
    }
    record.update(overrides)
    return record


def map_fake(record: RawPosting) -> dict[str, Any]:
    if "ref" not in record:
        raise MappingError("missing ref")
    return {
        "external_id": record["ref"],
        "title": record.get("headline"),
        "company_name": record.get("employer"),
        "job_url": record.get("link"),
    }


class ListAdapter:
    """In-memory adapter: exercises the contract without I/O."""

    def __init__(self, records: Iterable[object], metadata: SourceMetadata = METADATA) -> None:
        self._records = records
        self._metadata = metadata
        self.fetch_calls = 0

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        self.fetch_calls += 1
        yield from self._records  # type: ignore[misc]

    def map_posting(self, record: RawPosting) -> dict[str, Any]:
        return map_fake(record)


# --- Collection, normalization, validation -------------------------------------


def test_valid_records_become_validated_postings() -> None:
    result = collect_postings(ListAdapter([raw("1"), raw("2")]))

    assert [p.external_id for p in result.postings] == ["1", "2"]
    assert result.postings[0].title == "Engineer 1"
    assert result.rejected == ()
    assert result.fetched == 2
    assert result.source == METADATA


def test_invalid_records_are_rejected_individually_with_reasons() -> None:
    records = [
        raw("ok"),
        "not-a-dict",
        {"headline": "no ref"},
        raw("blank-title", headline="   "),
        raw("long-title", headline="x" * 301),
    ]

    result = collect_postings(ListAdapter(records))

    assert [p.external_id for p in result.postings] == ["ok"]
    reasons = {item.position: item for item in result.rejected}
    assert reasons[1].reason == "raw posting is not an object"
    assert reasons[2].reason == "mapping: missing ref"
    assert reasons[3].external_id == "blank-title"
    assert reasons[3].reason.startswith("validation: title")
    assert reasons[4].reason.startswith("validation: title")
    assert result.fetched == 5


def test_rejection_reasons_do_not_include_raw_values() -> None:
    sensitive_value = "x" * 301 + "PRIVATE"

    result = collect_postings(ListAdapter([raw("1", headline=sensitive_value)]))

    assert "PRIVATE" not in result.rejected[0].reason


def test_mapper_cannot_introduce_fields_without_a_model_column() -> None:
    class ExtraFieldAdapter(ListAdapter):
        def map_posting(self, record: RawPosting) -> dict[str, Any]:
            return {**map_fake(record), "work_model": "remote"}

    result = collect_postings(ExtraFieldAdapter([raw("1")]))

    assert result.postings == ()
    assert "work_model" in result.rejected[0].reason


# --- Rejection logging and grouping -----------------------------------------------

SERVICE_LOGGER = "career_intelligence.ingestion.service"


def rejections(*reasons: str) -> list[RejectedPosting]:
    return [RejectedPosting(i, None, reason) for i, reason in enumerate(reasons)]


class ReasonAdapter(ListAdapter):
    """Rejects every record at mapping, with the reason the record names."""

    def map_posting(self, record: RawPosting) -> dict[str, Any]:
        raise MappingError(record["why"])


def records_with_reasons(counts: dict[str, int]) -> list[dict[str, str]]:
    return [{"why": reason} for reason, count in counts.items() for _ in range(count)]


def log_records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(message)]


def test_count_rejection_reasons_counts_each_reason() -> None:
    reasons = rejections("a", "c", "a", "b", "c", "a")

    assert count_rejection_reasons(reasons) == [("a", 3), ("c", 2), ("b", 1)]


def test_count_rejection_reasons_breaks_ties_alphabetically() -> None:
    assert count_rejection_reasons(rejections("z", "m", "a", "z", "a", "m")) == [
        ("a", 2),
        ("m", 2),
        ("z", 2),
    ]


def test_count_rejection_reasons_is_independent_of_input_order() -> None:
    reasons = rejections("b", "a", "c", "a", "d", "c")
    expected = [("a", 2), ("c", 2), ("b", 1), ("d", 1)]

    for permutation in itertools.permutations(reasons):
        assert count_rejection_reasons(permutation) == expected


def test_count_rejection_reasons_of_nothing_is_empty() -> None:
    assert count_rejection_reasons([]) == []


def test_each_rejection_is_logged_at_debug_not_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=SERVICE_LOGGER)

    collect_postings(ListAdapter([raw("ok"), "not-a-dict", {"headline": "no ref"}]))

    individual = log_records(caplog, "posting_rejected")
    assert len(individual) == 2
    assert {r.levelno for r in individual} == {logging.DEBUG}


def test_a_collection_with_rejections_logs_exactly_one_aggregated_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=SERVICE_LOGGER)
    records = [raw("ok"), "not-a-dict", {"headline": "no ref"}, "also-not-a-dict"]

    collect_postings(ListAdapter(records))

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "rejections_summary source='Fake Board' rejected=3 distinct_reasons=2 "
        "omitted_reasons=0 top_reasons={'raw posting is not an object': 2, "
        "'mapping: missing ref': 1}"
    )


def test_aggregated_warning_shows_at_most_the_most_frequent_reasons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=SERVICE_LOGGER)
    counts = {f"reason-{k:02d}": k % 3 + 1 for k in range(MAX_REASONS_SHOWN + 2)}

    result = collect_postings(ReasonAdapter(records_with_reasons(counts)))

    (warning,) = log_records(caplog, "rejections_summary")
    expected = count_rejection_reasons(result.rejected)
    message = warning.getMessage()
    assert f"rejected={sum(counts.values())} distinct_reasons={len(counts)} " in message
    assert "omitted_reasons=2 " in message
    assert message.endswith(f"top_reasons={dict(expected[:MAX_REASONS_SHOWN])!r}")
    assert all(reason not in message for reason, _ in expected[MAX_REASONS_SHOWN:])


@pytest.mark.parametrize("records", [[], [raw("1"), raw("2")]], ids=["empty", "all_valid"])
def test_collections_without_rejections_log_no_rejection_warning(
    records: list[Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=SERVICE_LOGGER)

    collect_postings(ListAdapter(records))

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert log_records(caplog, "posting_rejected") == []
    assert log_records(caplog, "rejections_summary") == []


def test_rejection_logs_never_contain_record_values(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sentinel = "PRIVATE-SENTINEL"
    records = [
        raw("long-title", headline="x" * 301 + sentinel, employer=sentinel),
        raw("blank-title", headline="   ", employer=sentinel),
        {"headline": sentinel, "employer": sentinel},  # mapping failure
        [sentinel],  # not an object
    ]

    result = collect_postings(ListAdapter(records))

    assert len(result.rejected) == 4
    assert log_records(caplog, "posting_rejected")
    assert log_records(caplog, "rejections_summary")
    assert sentinel not in caplog.text


# --- Contract -------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata",
    [
        SourceMetadata(name="", url="https://fake.example.com/", source_type="test"),
        SourceMetadata(name="x" * 151, url="https://fake.example.com/", source_type="test"),
        SourceMetadata(name="Fake", url="not a url", source_type="test"),
        SourceMetadata(name="Fake", url="https://fake.example.com/", source_type=""),
    ],
)
def test_invalid_metadata_is_rejected_before_fetching(metadata: SourceMetadata) -> None:
    adapter = ListAdapter([raw("1")], metadata=metadata)

    with pytest.raises(AdapterContractError):
        collect_postings(adapter)

    assert adapter.fetch_calls == 0


def test_fetch_failures_propagate_instead_of_returning_partial_results() -> None:
    class FailingAdapter(ListAdapter):
        def fetch_raw_postings(self) -> Iterator[RawPosting]:
            yield raw("1")
            raise SourceFetchError("page 2 unavailable")

    with pytest.raises(SourceFetchError):
        collect_postings(FailingAdapter([]))


def test_contract_and_adapters_do_not_load_database_or_web_frameworks() -> None:
    code = (
        "import sys\n"
        "import career_intelligence.ingestion.contracts\n"
        "import career_intelligence.ingestion.service\n"
        "import career_intelligence.ingestion.http\n"
        "import career_intelligence.sources.local_file\n"
        "loaded = [m for m in ('sqlalchemy', 'fastapi', 'psycopg') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )

    assert completed.returncode == 0, completed.stderr


# --- HTTP adapter end to end (MockTransport, no network) ------------------------


class PagedHttpAdapter:
    """Test-only adapter for a fictional paged JSON API."""

    def __init__(self, client: HttpClient, page_size: int = 2) -> None:
        self._client = client
        self._page_size = page_size

    @property
    def metadata(self) -> SourceMetadata:
        return METADATA

    def _fetch_page(self, page_number: int) -> Page[RawPosting]:
        payload = self._client.get_json("/jobs", {"page": page_number, "limit": self._page_size})
        return Page(
            items=payload["results"],
            has_next=page_number * self._page_size < payload["total"],
        )

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        return paginate(self._fetch_page)

    def map_posting(self, record: RawPosting) -> dict[str, Any]:
        return map_fake(record)


def paged_api(records: list[dict[str, Any]], fail_on_page: int | None = None):
    requested_pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        limit = int(request.url.params["limit"])
        requested_pages.append(page)
        if page == fail_on_page:
            return httpx.Response(500)
        start = (page - 1) * limit
        return httpx.Response(
            200, json={"total": len(records), "results": records[start : start + limit]}
        )

    return handler, requested_pages


def test_http_adapter_collects_all_pages_through_the_pipeline() -> None:
    records = [raw(str(n)) for n in range(5)] + [{"headline": "unmappable"}]
    handler, requested_pages = paged_api(records)
    client = HttpClient(
        "https://api.fake.example.com",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )

    with client:
        result = collect_postings(PagedHttpAdapter(client))

    assert requested_pages == [1, 2, 3]
    assert [p.external_id for p in result.postings] == ["0", "1", "2", "3", "4"]
    assert len(result.rejected) == 1
    assert result.fetched == 6


def test_http_failure_mid_pagination_aborts_the_collection() -> None:
    handler, requested_pages = paged_api([raw(str(n)) for n in range(6)], fail_on_page=2)
    client = HttpClient(
        "https://api.fake.example.com",
        settings=HttpSettings(max_retries=1, backoff_seconds=0),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )

    with client, pytest.raises(HttpStatusError):
        collect_postings(PagedHttpAdapter(client))

    assert requested_pages == [1, 2, 2]  # page 2 retried once, then aborted
