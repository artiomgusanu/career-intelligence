from datetime import datetime

import pytest
from pydantic import ValidationError

from career_intelligence.schemas import JobPostingCreate

REQUIRED = {
    "external_id": "abc-123",
    "title": "Backend Engineer",
    "company_name": "Acme",
    "job_url": "https://jobs.example.com/abc-123",
}


def test_minimal_posting_defaults_optional_fields_to_none() -> None:
    posting = JobPostingCreate(**REQUIRED)

    assert posting.salary_min is None
    assert posting.description is None
    assert posting.published_at is None


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_required_fields(missing: str) -> None:
    payload = {key: value for key, value in REQUIRED.items() if key != missing}

    with pytest.raises(ValidationError):
        JobPostingCreate(**payload)


def test_rejects_blank_required_string() -> None:
    with pytest.raises(ValidationError):
        JobPostingCreate(**{**REQUIRED, "title": "   "})


def test_rejects_values_longer_than_database_column() -> None:
    with pytest.raises(ValidationError):
        JobPostingCreate(**{**REQUIRED, "title": "x" * 301})


def test_rejects_naive_datetimes() -> None:
    with pytest.raises(ValidationError):
        # Deliberately naive: the schema must reject it.
        JobPostingCreate(**REQUIRED, published_at=datetime(2026, 11, 1, 9, 30))  # noqa: DTZ001


def test_rejects_inverted_salary_range() -> None:
    with pytest.raises(ValidationError, match="salary_min"):
        JobPostingCreate(**REQUIRED, salary_min=90000, salary_max=50000)


def test_rejects_negative_salary() -> None:
    with pytest.raises(ValidationError):
        JobPostingCreate(**REQUIRED, salary_min=-1)


def test_normalizes_currency_to_uppercase() -> None:
    assert JobPostingCreate(**REQUIRED, salary_currency="usd").salary_currency == "USD"


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        JobPostingCreate(**REQUIRED, source_id=1)


OPTIONAL_TEXT_FIELDS = [
    "company_slug",
    "application_url",
    "employment_type",
    "categories",
    "seniority",
    "location_restrictions",
    "timezone_restrictions",
    "salary_currency",
    "salary_period",
    "description",
]


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("field", OPTIONAL_TEXT_FIELDS)
def test_blank_optional_text_becomes_none(field: str, blank: str) -> None:
    posting = JobPostingCreate(**REQUIRED, **{field: blank})

    assert getattr(posting, field) is None


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("field", sorted(REQUIRED))
def test_blank_required_text_is_still_rejected(field: str, blank: str) -> None:
    with pytest.raises(ValidationError):
        JobPostingCreate(**{**REQUIRED, field: blank})


def test_non_blank_values_are_not_altered_beyond_trimming() -> None:
    posting = JobPostingCreate(
        **{**REQUIRED, "external_id": "  ID-42 "},
        application_url="https://apply.example.com/Path?q=A ",
        description="  Keep   inner   spacing  ",
    )

    assert posting.external_id == "ID-42"
    assert posting.application_url == "https://apply.example.com/Path?q=A"
    assert posting.description == "Keep   inner   spacing"


def test_optional_field_length_is_still_validated() -> None:
    with pytest.raises(ValidationError):
        JobPostingCreate(**REQUIRED, company_slug="x" * 201)
