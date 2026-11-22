"""ITJobs adapter tests. No network, no database, no real credentials.

The API is simulated with ``httpx.MockTransport``; envelopes are built here
from synthetic records. ``FAKE_API_KEY`` is a placeholder, not a credential.
"""

import json
import logging
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest

from career_intelligence.config import HttpSettings
from career_intelligence.ingestion.contracts import IngestionError, SourceFetchError
from career_intelligence.ingestion.http import HttpClient, HttpStatusError, InvalidJsonError
from career_intelligence.ingestion.pagination import (
    IncompletePaginationError,
    PaginationLimitError,
)
from career_intelligence.ingestion.service import collect_postings, validate_metadata
from career_intelligence.sources.itjobs import (
    DEFAULT_MAX_PAGES,
    DEFAULT_PAGE_SIZE,
    ITJOBS_BASE_URL,
    ITJOBS_LIST_PATH,
    ITJOBS_METADATA,
    ITJOBS_MIN_INTERVAL_SECONDS,
    ITJobsAdapter,
    ListPage,
    parse_list_envelope,
)

FAKE_API_KEY = "test-key-not-real"
FIXTURE = Path(__file__).parent / "fixtures" / "itjobs_documented_records.json"

type Body = dict[str, Any]
type Tamper = Callable[[int, Body], Body | httpx.Response]


def record(posting_id: int) -> dict[str, Any]:
    return {
        "id": posting_id,
        "title": f"Engineer {posting_id}",
        "company": {"name": "Example Tech"},
    }


def records(count: int, start: int = 1) -> list[dict[str, Any]]:
    return [record(n) for n in range(start, start + count)]


class FakeTime:
    """Monotonic clock that only moves when the client sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeListApi:
    """Simulated ``job/list.json``: page-number pagination over ``postings``."""

    def __init__(
        self,
        postings: list[Any],
        *,
        total: int | None = None,
        tamper: Tamper | None = None,
        time: FakeTime | None = None,
    ) -> None:
        self.postings = postings
        self.total = len(postings) if total is None else total
        self.tamper = tamper
        self.time = time
        self.requests: list[httpx.Request] = []
        self.started_at: list[float] = []

    @property
    def pages(self) -> list[int]:
        return [int(request.url.params["page"]) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.time is not None:
            self.started_at.append(self.time.now)
        page = int(request.url.params["page"])
        limit = int(request.url.params["limit"])
        start = (page - 1) * limit
        body: Body = {
            "total": self.total,
            "page": page,
            "limit": limit,
            "results": self.postings[start : start + limit],
        }
        if self.tamper is not None:
            tampered = self.tamper(page, body)
            if isinstance(tampered, httpx.Response):
                return tampered
            body = tampered
        return httpx.Response(200, json=body)


def make_client(
    api: Callable[[httpx.Request], httpx.Response],
    *,
    time: FakeTime | None = None,
    max_retries: int = 0,
) -> HttpClient:
    time = time or FakeTime()
    return HttpClient(
        ITJOBS_BASE_URL,
        settings=HttpSettings(max_retries=max_retries, backoff_seconds=0),
        transport=httpx.MockTransport(api),
        min_interval_seconds=ITJOBS_MIN_INTERVAL_SECONDS,
        sleep=time.sleep,
        clock=time.clock,
    )


def collect(api: FakeListApi, *, page_size: int = 2, max_pages: int = 10, **kwargs: Any):
    with make_client(api, time=api.time, **kwargs) as client:
        adapter = ITJobsAdapter(
            client, api_key=FAKE_API_KEY, page_size=page_size, max_pages=max_pages
        )
        return list(adapter.fetch_raw_postings())


def envelope(**overrides: Any) -> Body:
    body: Body = {"total": 1, "page": 1, "limit": 2, "results": [record(1)]}
    body.update(overrides)
    return body


# --- Metadata and constructor ------------------------------------------------------------


def test_metadata_is_the_approved_identity_and_valid() -> None:
    assert ITJOBS_METADATA.name == "ITJobs.pt"
    assert ITJOBS_METADATA.url == "https://www.itjobs.pt/"
    assert ITJOBS_METADATA.source_type == "job_board"
    validate_metadata(ITJOBS_METADATA)


def test_defaults() -> None:
    assert DEFAULT_PAGE_SIZE == 50
    assert DEFAULT_MAX_PAGES == 100
    assert ITJOBS_MIN_INTERVAL_SECONDS == 1.0


@pytest.mark.parametrize("api_key", ["", "   ", None, 123])
def test_api_key_is_required(api_key: object) -> None:
    with pytest.raises(ValueError, match="api_key must be a non-empty string") as excinfo:
        ITJobsAdapter(object(), api_key=api_key)  # type: ignore[arg-type]

    assert str(excinfo.value) == "api_key must be a non-empty string."


@pytest.mark.parametrize("name", ["page_size", "max_pages"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10", None])
def test_page_size_and_max_pages_must_be_positive_integers(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{name} must be an integer >= 1"):
        ITJobsAdapter(object(), api_key=FAKE_API_KEY, **{name: value})  # type: ignore[arg-type]


def test_map_posting_delegates_to_the_mapper() -> None:
    adapter = ITJobsAdapter(object(), api_key=FAKE_API_KEY)  # type: ignore[arg-type]

    assert adapter.map_posting(record(7))["external_id"] == "7"


# --- Envelope (pure) -----------------------------------------------------------------------


def parse(body: object, page: int = 1, limit: int = 2) -> ListPage:
    return parse_list_envelope(body, requested_page=page, requested_limit=limit)


def test_valid_envelope() -> None:
    listing = parse(envelope(total=3, results=[record(1), record(2)]))

    assert listing == ListPage(total=3, page=1, limit=2, results=[record(1), record(2)])


@pytest.mark.parametrize("body", [[], "text", None, 1])
def test_envelope_must_be_an_object(body: object) -> None:
    with pytest.raises(SourceFetchError, match="not a JSON object"):
        parse(body)


@pytest.mark.parametrize(
    "error",
    [
        {"message": "Job not found."},
        {"message": f"invalid api_key={FAKE_API_KEY}"},
        {"message": 42},
        "plain text",
        None,
    ],
)
def test_api_error_response_gives_a_generic_message(error: object) -> None:
    with pytest.raises(SourceFetchError) as excinfo:
        parse(envelope(error=error))

    assert str(excinfo.value) == "ITJobs API returned an error response"
    assert FAKE_API_KEY not in str(excinfo.value)


@pytest.mark.parametrize("field", ["total", "page", "limit"])
@pytest.mark.parametrize("value", ["missing", None, "1", 1.0, True, -1])
def test_envelope_counts_must_be_integers(field: str, value: object) -> None:
    body = envelope()
    if value == "missing":
        del body[field]
    else:
        body[field] = value

    with pytest.raises(SourceFetchError, match=f'"{field}" must be an integer'):
        parse(body)


@pytest.mark.parametrize(("field", "value"), [("page", 0), ("limit", 0)])
def test_page_and_limit_must_be_at_least_one(field: str, value: int) -> None:
    with pytest.raises(SourceFetchError, match=f'"{field}" must be an integer >= 1'):
        parse(envelope(**{field: value}))


def test_page_must_match_the_request() -> None:
    with pytest.raises(SourceFetchError, match="for page 1, but page 2 was requested"):
        parse(envelope(), page=2)


def test_limit_must_match_the_request() -> None:
    with pytest.raises(SourceFetchError, match="limit 2, but limit 50 was requested"):
        parse(envelope(), limit=50)


@pytest.mark.parametrize("results", ["missing", None, {}, "x", 3])
def test_results_must_be_a_list(results: object) -> None:
    body = envelope()
    if results == "missing":
        del body["results"]
    else:
        body["results"] = results

    with pytest.raises(SourceFetchError, match='"results" must be a list'):
        parse(body)


def test_results_cannot_exceed_limit() -> None:
    with pytest.raises(SourceFetchError, match="3 results, more than its limit 2"):
        parse(envelope(total=3, results=records(3)))


# --- Request ---------------------------------------------------------------------------------


def test_requests_are_https_get_with_only_the_documented_parameters() -> None:
    api = FakeListApi(records(3))

    collect(api, page_size=2)

    for page, request in enumerate(api.requests, start=1):
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert request.url.host == "api.itjobs.pt"
        assert request.url.path == ITJOBS_LIST_PATH
        assert dict(request.url.params) == {
            "api_key": FAKE_API_KEY,
            "page": str(page),
            "limit": "2",
        }


# --- Pagination --------------------------------------------------------------------------------


def test_empty_listing_takes_one_request() -> None:
    api = FakeListApi([])

    assert collect(api) == []
    assert api.pages == [1]


def test_empty_total_with_results_aborts() -> None:
    api = FakeListApi(records(1), total=0)

    with pytest.raises(IncompletePaginationError, match="page 1 has 1 postings, expected 0"):
        collect(api)


def test_single_partial_page() -> None:
    api = FakeListApi(records(1))

    assert collect(api, page_size=2) == records(1)
    assert api.pages == [1]


def test_exact_multiple_needs_no_extra_request() -> None:
    api = FakeListApi(records(4))

    assert collect(api, page_size=2) == records(4)
    assert api.pages == [1, 2]


def test_collects_every_page_in_order() -> None:
    api = FakeListApi(records(5))

    assert collect(api, page_size=2) == records(5)
    assert api.pages == [1, 2, 3]


def test_short_intermediate_page_aborts() -> None:
    def short_page_2(page: int, body: Body) -> Body:
        if page == 2:
            body["results"] = body["results"][:1]
        return body

    api = FakeListApi(records(5), tamper=short_page_2)

    with pytest.raises(IncompletePaginationError, match="page 2 has 1 postings, expected 2"):
        collect(api, page_size=2)
    assert api.pages == [1, 2]


def test_empty_intermediate_page_aborts() -> None:
    def empty_page_2(page: int, body: Body) -> Body:
        if page == 2:
            body["results"] = []
        return body

    api = FakeListApi(records(5), tamper=empty_page_2)

    with pytest.raises(IncompletePaginationError, match="page 2 has 0 postings, expected 2"):
        collect(api, page_size=2)


def test_short_last_page_aborts() -> None:
    api = FakeListApi(records(4), total=5)  # claims 5, serves 4

    with pytest.raises(IncompletePaginationError, match="page 3 has 0 postings, expected 1"):
        collect(api, page_size=2)


def test_total_changing_during_collection_aborts() -> None:
    def grows(page: int, body: Body) -> Body:
        if page == 2:
            body["total"] += 1
        return body

    api = FakeListApi(records(5), tamper=grows)

    with pytest.raises(IncompletePaginationError, match="total changed from 5 to 6"):
        collect(api, page_size=2)


# --- Repeated pages and duplicate ids ---------------------------------------------------------------


def test_server_ignoring_page_parameter_aborts() -> None:
    def always_page_1(page: int, body: Body) -> Body:
        return {**body, "page": 1, "results": records(2)}

    api = FakeListApi(records(5), tamper=always_page_1)

    with pytest.raises(SourceFetchError, match="for page 1, but page 2 was requested"):
        collect(api, page_size=2)


def test_repeated_page_content_aborts_on_duplicate_ids() -> None:
    def repeat_page_1(page: int, body: Body) -> Body:
        return {**body, "results": records(2)}  # echoes page correctly, same postings

    api = FakeListApi(records(5), tamper=repeat_page_1)

    with pytest.raises(IncompletePaginationError, match="posting id 1 was returned twice"):
        collect(api, page_size=2)


def test_duplicate_id_within_a_page_aborts() -> None:
    api = FakeListApi([record(1), record(1), record(2)])

    with pytest.raises(IncompletePaginationError, match="posting id 1 was returned twice"):
        collect(api, page_size=2)


def test_records_without_integer_id_are_left_to_the_mapper() -> None:
    odd = [{"id": "1"}, {"title": "no id"}, "not an object", {"id": True}]
    api = FakeListApi([*odd, record(1)])

    assert collect(api, page_size=5) == [*odd, record(1)]


def test_state_is_reset_between_runs() -> None:
    api = FakeListApi(records(3))
    with make_client(api) as client:
        adapter = ITJobsAdapter(client, api_key=FAKE_API_KEY, page_size=2)
        first = list(adapter.fetch_raw_postings())
        second = list(adapter.fetch_raw_postings())

    assert first == second == records(3)
    assert api.pages == [1, 2, 1, 2]


# --- Page limit -----------------------------------------------------------------------------------


def test_too_many_pages_fails_after_the_first_request() -> None:
    api = FakeListApi(records(11))

    with pytest.raises(PaginationLimitError, match="11 postings \\(6 pages of 2\\)"):
        collect(api, page_size=2, max_pages=5)
    assert api.pages == [1]


def test_exactly_max_pages_is_allowed() -> None:
    api = FakeListApi(records(10))

    assert len(collect(api, page_size=2, max_pages=5)) == 10
    assert api.pages == [1, 2, 3, 4, 5]


# --- HTTP errors ----------------------------------------------------------------------------------


def test_server_error_is_retried_then_aborts() -> None:
    api = FakeListApi(records(3), tamper=lambda page, body: httpx.Response(500))

    with pytest.raises(HttpStatusError):
        collect(api, max_retries=2)
    assert api.pages == [1, 1, 1]


def test_transient_error_mid_collection_recovers() -> None:
    failures = {2: 1}

    def flaky(page: int, body: Body) -> Body | httpx.Response:
        if failures.get(page):
            failures[page] -= 1
            return httpx.Response(503)
        return body

    api = FakeListApi(records(5), tamper=flaky)

    assert collect(api, page_size=2, max_retries=1) == records(5)
    assert api.pages == [1, 2, 2, 3]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_client_errors_are_not_retried_and_do_not_leak_the_key(status: int) -> None:
    api = FakeListApi(records(3), tamper=lambda page, body: httpx.Response(status))

    with pytest.raises(HttpStatusError) as excinfo:
        collect(api, max_retries=2)

    assert api.pages == [1]
    assert FAKE_API_KEY not in str(excinfo.value)


def test_invalid_json_aborts() -> None:
    api = FakeListApi(records(3), tamper=lambda page, body: httpx.Response(200, text="<html>"))

    with pytest.raises(InvalidJsonError) as excinfo:
        collect(api)
    assert FAKE_API_KEY not in str(excinfo.value)


def test_error_envelope_with_status_200_aborts() -> None:
    api = FakeListApi(
        records(3), tamper=lambda page, body: {"error": {"message": f"api_key={FAKE_API_KEY}"}}
    )

    with pytest.raises(SourceFetchError) as excinfo:
        collect(api)
    assert str(excinfo.value) == "ITJobs API returned an error response"


# --- Minimum interval -----------------------------------------------------------------------------


def test_requests_are_at_least_one_second_apart() -> None:
    time = FakeTime()
    api = FakeListApi(records(5), time=time)

    collect(api, page_size=2)

    assert api.pages == [1, 2, 3]
    assert api.started_at[0] == 0.0  # first request does not wait
    gaps = [later - earlier for earlier, later in pairwise(api.started_at)]
    assert gaps == [ITJOBS_MIN_INTERVAL_SECONDS, ITJOBS_MIN_INTERVAL_SECONDS]


def test_retries_also_respect_the_minimum_interval() -> None:
    time = FakeTime()
    failures = {1: 1}

    def flaky(page: int, body: Body) -> Body | httpx.Response:
        if failures.get(page):
            failures[page] -= 1
            return httpx.Response(500)
        return body

    api = FakeListApi(records(1), tamper=flaky, time=time)

    collect(api, max_retries=1)

    assert api.pages == [1, 1]
    assert api.started_at[1] - api.started_at[0] >= ITJOBS_MIN_INTERVAL_SECONDS


# --- Credentials ----------------------------------------------------------------------------------


def test_api_key_never_appears_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    api = FakeListApi(
        records(3), tamper=lambda page, body: body if page == 1 else httpx.Response(500)
    )

    with pytest.raises(HttpStatusError):
        collect(api, page_size=2, max_retries=1)

    assert api.requests[0].url.params["api_key"] == FAKE_API_KEY  # it was sent
    assert caplog.records  # requests, responses and retries were logged
    assert FAKE_API_KEY not in caplog.text


def test_adapter_repr_does_not_expose_the_key() -> None:
    adapter = ITJobsAdapter(object(), api_key=FAKE_API_KEY)  # type: ignore[arg-type]

    assert FAKE_API_KEY not in repr(adapter)


# --- Pipeline integration ---------------------------------------------------------------------------


def test_pipeline_with_documented_records_rejects_all_for_missing_job_url() -> None:
    documented = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
    api = FakeListApi([*documented, record(3)])

    with make_client(api) as client:
        result = collect_postings(ITJobsAdapter(client, api_key=FAKE_API_KEY, page_size=2))

    assert api.pages == [1, 2]
    assert result.source == ITJOBS_METADATA
    assert result.fetched == 3
    assert result.postings == ()
    assert [(item.position, item.external_id, item.reason) for item in result.rejected] == [
        (0, "900001", "validation: job_url: Field required"),
        (1, "900002", "validation: job_url: Field required"),
        (2, "3", "validation: job_url: Field required"),
    ]


def test_pipeline_propagates_ingestion_errors_without_a_partial_result() -> None:
    def short_page_2(page: int, body: Body) -> Body:
        if page == 2:
            body["results"] = body["results"][:1]
        return body

    api = FakeListApi(records(5), tamper=short_page_2)

    with make_client(api) as client, pytest.raises(IngestionError):
        collect_postings(ITJobsAdapter(client, api_key=FAKE_API_KEY, page_size=2))
