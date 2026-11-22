"""ITJobs mapper tests. No network, no database.

Records come from ``fixtures/itjobs_documented_records.json``: synthetic values
in the structure of the documented API example, not real API responses.

No URL is used anywhere in this file: ``job_url`` must stay missing, and the
tests prove it is the only reason ITJobs records fail validation.
"""

import copy
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from career_intelligence.ingestion.contracts import MappingError, RawPosting, SourceMetadata
from career_intelligence.ingestion.service import collect_postings
from career_intelligence.schemas import JobPostingCreate
from career_intelligence.sources.itjobs import map_itjobs_posting
from career_intelligence.sources.local_file import DEFAULT_METADATA

FIXTURE = Path(__file__).parent / "fixtures" / "itjobs_documented_records.json"

MAPPED_KEYS = frozenset(
    {"external_id", "title", "company_name", "company_slug", "employment_type", "description"}
)


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def full_record() -> dict[str, Any]:
    return load_fixture()["records"][0]


def minimal_record() -> dict[str, Any]:
    return load_fixture()["records"][1]


def validation_errors(fields: dict[str, Any]) -> list[tuple[tuple[str | int, ...], str]]:
    with pytest.raises(ValidationError) as excinfo:
        JobPostingCreate.model_validate(fields)
    return [(error["loc"], error["type"]) for error in excinfo.value.errors()]


# --- Fixture provenance --------------------------------------------------------------


def test_fixture_is_marked_synthetic_and_contains_no_urls() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    fixture = load_fixture()

    assert fixture["_provenance"]["synthetic"] is True
    assert not re.search(r"(?i)https?:|www\.", text)
    for email in re.findall(r"[\w.+-]+@[\w.-]+", text):
        assert email.endswith(".invalid"), email


# --- Mapping -------------------------------------------------------------------------


def test_maps_full_record() -> None:
    fields = map_itjobs_posting(full_record())

    assert fields == {
        "external_id": "900001",
        "title": "Backend Engineer (Python)",
        "company_name": "Example Tech",
        "company_slug": "example-tech",
        "employment_type": "Full-time",
        "description": "<p>Build <strong>APIs</strong> with FastAPI.</p>\n<ul><li>PostgreSQL</li></ul>",
    }


def test_maps_minimal_record() -> None:
    fields = map_itjobs_posting(minimal_record())

    assert fields == {
        "external_id": "900002",
        "title": "Data Engineer",
        "company_name": "Sample Data",
        "company_slug": None,
        "employment_type": None,
        "description": None,
    }


@pytest.mark.parametrize("record", [full_record(), minimal_record()], ids=["full", "minimal"])
def test_result_keys_are_limited_to_mapped_fields(record: dict[str, Any]) -> None:
    assert set(map_itjobs_posting(record)) == MAPPED_KEYS


# --- job_url strategy ----------------------------------------------------------------


@pytest.mark.parametrize("record", [full_record(), minimal_record()], ids=["full", "minimal"])
def test_job_url_is_never_produced(record: dict[str, Any]) -> None:
    assert "job_url" not in map_itjobs_posting(record)


@pytest.mark.parametrize("record", [full_record(), minimal_record()], ids=["full", "minimal"])
def test_missing_job_url_is_the_only_validation_error(record: dict[str, Any]) -> None:
    assert validation_errors(map_itjobs_posting(record)) == [(("job_url",), "missing")]


class FixtureAdapter:
    """In-memory adapter: fixture records through the real ITJobs mapper."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    @property
    def metadata(self) -> SourceMetadata:
        # Existing demo identity: this test introduces no source URL of its own.
        return DEFAULT_METADATA

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        yield from self._records

    def map_posting(self, raw: RawPosting) -> dict[str, Any]:
        return map_itjobs_posting(raw)


def test_pipeline_rejects_every_record_for_missing_job_url() -> None:
    result = collect_postings(FixtureAdapter(load_fixture()["records"]))

    assert result.postings == ()
    assert [(item.position, item.external_id, item.reason) for item in result.rejected] == [
        (0, "900001", "validation: job_url: Field required"),
        (1, "900002", "validation: job_url: Field required"),
    ]


# --- Approved exclusions -------------------------------------------------------------


@pytest.mark.parametrize(
    "excluded",
    [
        "publishedAt",
        "updatedAt",
        "salaryMin",
        "salaryMax",
        "wage",
        "locations",
        "country",
        "contracts",
        "workModel",
        "ref",
        "slug",
        "companyId",
    ],
)
def test_excluded_fields_do_not_influence_the_result(excluded: str) -> None:
    record = full_record()
    without = {key: value for key, value in record.items() if key != excluded}

    assert excluded in record
    assert map_itjobs_posting(record) == map_itjobs_posting(without)


def test_company_contact_data_never_reaches_the_result() -> None:
    company = full_record()["company"]
    rendered = repr(map_itjobs_posting(full_record()))

    for key in ("email", "phone", "fax", "address", "description"):
        assert company[key] not in rendered, key


def test_contracts_and_work_model_are_not_mixed_into_employment_type() -> None:
    record = full_record()
    record["contracts"] = [{"id": 1, "name": "Contract X"}]
    record["workModel"] = 1

    assert map_itjobs_posting(record)["employment_type"] == "Full-time"


# --- types -> employment_type --------------------------------------------------------


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (None, None),
        ([], None),
        ([{"id": 1, "name": "  "}], None),
        ([{"id": 1, "name": " Full-time "}], "Full-time"),
        ([{"id": 1, "name": "Full-time"}, {"id": 2, "name": "Part-time"}], "Full-time, Part-time"),
        (
            [{"name": "Part-time"}, {"name": "Full-time"}, {"name": "Part-time"}],
            "Part-time, Full-time",
        ),
    ],
)
def test_types_are_joined_as_given(types: object, expected: str | None) -> None:
    record = minimal_record()
    record["types"] = types

    assert map_itjobs_posting(record)["employment_type"] == expected


def test_missing_types_key_gives_none() -> None:
    assert map_itjobs_posting(minimal_record())["employment_type"] is None


@pytest.mark.parametrize(
    ("types", "message"),
    [
        ("Full-time", "types must be a list"),
        ({"id": 1, "name": "Full-time"}, "types must be a list"),
        ([1], "types items must be objects"),
        (["Full-time"], "types items must be objects"),
        ([{"id": 1}], r"types\[\]\.name must be a string"),
        ([{"id": 1, "name": 1}], r"types\[\]\.name must be a string"),
        ([{"id": 1, "name": None}], r"types\[\]\.name must be a string"),
    ],
)
def test_invalid_types_raise_mapping_error(types: object, message: str) -> None:
    record = minimal_record()
    record["types"] = types

    with pytest.raises(MappingError, match=message):
        map_itjobs_posting(record)


# --- id -------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, 1.5, 900002.0, "900002", "", 0, -1, None])
def test_invalid_id_raises_mapping_error(value: object) -> None:
    record = minimal_record()
    record["id"] = value

    with pytest.raises(MappingError, match="id must be a positive integer"):
        map_itjobs_posting(record)


def test_missing_id_raises_mapping_error() -> None:
    record = minimal_record()
    del record["id"]

    with pytest.raises(MappingError, match="id must be a positive integer"):
        map_itjobs_posting(record)


# --- company ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["Sample Data", ["Sample Data"], None, 70001])
def test_company_must_be_an_object(value: object) -> None:
    record = minimal_record()
    record["company"] = value

    with pytest.raises(MappingError, match="company must be an object"):
        map_itjobs_posting(record)


def test_company_id_alone_is_not_enough() -> None:
    record = minimal_record()
    del record["company"]
    record["companyId"] = 70001

    with pytest.raises(MappingError, match="company must be an object"):
        map_itjobs_posting(record)


@pytest.mark.parametrize("name", [None, "", "   ", 70001])
def test_invalid_company_name_is_rejected_by_the_schema(name: object) -> None:
    record = minimal_record()
    record["company"] = {"name": name}

    errors = validation_errors(map_itjobs_posting(record))

    assert ("company_name",) in [loc for loc, _ in errors]


@pytest.mark.parametrize("title", [None, "", "   ", 123])
def test_invalid_title_is_rejected_by_the_schema(title: object) -> None:
    record = minimal_record()
    record["title"] = title

    errors = validation_errors(map_itjobs_posting(record))

    assert ("title",) in [loc for loc, _ in errors]


# --- body -> description ---------------------------------------------------------------


def test_body_html_is_preserved_verbatim() -> None:
    record = full_record()

    assert map_itjobs_posting(record)["description"] == record["body"]


@pytest.mark.parametrize("body", ["", "   ", "\n\t"])
def test_blank_body_becomes_none_in_the_schema(body: str) -> None:
    record = minimal_record()
    record["body"] = body
    fields = map_itjobs_posting(record)

    # Only job_url is missing, so the blank body did not cause a second error.
    assert validation_errors(fields) == [(("job_url",), "missing")]
    assert fields["description"] == body  # mapper passes it through unchanged


# --- Purity and isolation ----------------------------------------------------------------


def test_mapper_does_not_modify_input_and_is_deterministic() -> None:
    record = full_record()
    snapshot = copy.deepcopy(record)

    first = map_itjobs_posting(record)
    second = map_itjobs_posting(record)

    assert record == snapshot
    assert first == second


def test_module_loads_no_http_database_or_web_framework() -> None:
    code = (
        "import sys\n"
        "import career_intelligence.sources.itjobs\n"
        "loaded = [m for m in ('httpx', 'sqlalchemy', 'fastapi', 'psycopg') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )

    assert completed.returncode == 0, completed.stderr
