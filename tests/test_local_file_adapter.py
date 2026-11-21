import json
from pathlib import Path

import pytest

from career_intelligence.ingestion.contracts import MappingError, SourceFetchError
from career_intelligence.ingestion.service import collect_postings
from career_intelligence.sources.local_file import LocalFileAdapter, map_local_posting

FIXTURE = Path(__file__).parent / "fixtures" / "local_postings.json"


def load_fixture_records() -> list[object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["postings"]


# --- Mapper (pure) ----------------------------------------------------------------


def test_maps_full_record_without_assumptions() -> None:
    fields = map_local_posting(load_fixture_records()[0])  # type: ignore[arg-type]

    assert fields == {
        "external_id": "be-001",
        "title": "Backend Engineer (Python)",
        "company_name": "Example Tech",
        "company_slug": "example-tech",
        "job_url": "https://jobs.example.com/be-001",
        "employment_type": "Full-time",
        "location_restrictions": "Lisboa, Porto",
        "description": "<p>Build APIs with FastAPI.</p>",
        "published_at": "2026-11-20T09:00:00+00:00",
        "salary_min": 30000,
        "salary_max": 45000,
        "salary_currency": "eur",
        "salary_period": "year",
    }


def test_maps_minimal_record_and_integer_id() -> None:
    fields = map_local_posting(load_fixture_records()[1])  # type: ignore[arg-type]

    assert fields["external_id"] == "1002"
    assert fields["location_restrictions"] is None
    assert "salary_min" not in fields  # no salary object: nothing invented


def test_blank_locations_are_dropped() -> None:
    fields = map_local_posting(load_fixture_records()[2])  # type: ignore[arg-type]

    assert fields["location_restrictions"] == "Braga"


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"id": True, "company": {"name": "A"}}, "id"),
        ({"id": 1.5, "company": {"name": "A"}}, "id"),
        ({"id": "1"}, "company"),
        ({"id": "1", "company": "Acme"}, "company"),
        ({"id": "1", "company": {"name": "A"}, "salary": 1000}, "salary"),
        ({"id": "1", "company": {"name": "A"}, "locations": "Lisboa"}, "locations"),
        ({"id": "1", "company": {"name": "A"}, "locations": ["Lisboa", 3]}, "locations"),
    ],
)
def test_malformed_records_raise_mapping_error(record: dict[str, object], message: str) -> None:
    with pytest.raises(MappingError, match=message):
        map_local_posting(record)


# --- Adapter through the pipeline ------------------------------------------------


def test_fixture_through_pipeline_validates_and_rejects() -> None:
    result = collect_postings(LocalFileAdapter(FIXTURE))

    assert [p.external_id for p in result.postings] == ["be-001", "1002", "fe-003", "be-001"]
    first = result.postings[0]
    assert first.salary_currency == "EUR"
    assert first.published_at is not None and first.published_at.utcoffset() is not None
    assert result.postings[2].description is None  # whitespace-only became None

    rejected = {item.external_id or item.position: item.reason for item in result.rejected}
    assert set(rejected) == {4, "naive-date", "bad-salary", 7}
    assert "company" in rejected[4]
    assert "published_at" in rejected["naive-date"]
    assert "salary_min" in rejected["bad-salary"]
    assert rejected[7] == "raw posting is not an object"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ('["a list"]', '"postings" list'),
        ('{"postings": {}}', '"postings" list'),
    ],
)
def test_unreadable_files_raise_source_fetch_error(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "postings.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SourceFetchError, match=message):
        list(LocalFileAdapter(path).fetch_raw_postings())


def test_missing_file_raises_source_fetch_error(tmp_path: Path) -> None:
    with pytest.raises(SourceFetchError, match="Cannot read"):
        list(LocalFileAdapter(tmp_path / "missing.json").fetch_raw_postings())
