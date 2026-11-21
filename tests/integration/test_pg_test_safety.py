"""Unit tests for the test-database safety guards (no database required)."""

import pytest
from pg_test_safety import UnsafeTestDatabaseError, validate_test_database_url

DEV_URL = "postgresql+psycopg://app:secret@127.0.0.1:5432/career_intelligence"


def test_accepts_dedicated_test_database() -> None:
    url = validate_test_database_url(
        "postgresql+psycopg://app:secret@127.0.0.1:5432/career_intelligence_test", DEV_URL
    )
    assert url.database == "career_intelligence_test"


@pytest.mark.parametrize(
    ("test_url", "reason"),
    [
        (None, "not set"),
        ("", "not set"),
        (DEV_URL, "must end with"),
        ("postgresql+psycopg://app:secret@127.0.0.1:5432/postgres", "must end with"),
        ("sqlite:///career_intelligence_test", "PostgreSQL"),
        ("postgresql+psycopg://app:secret@127.0.0.1:5432/Bad-Name_test", "lowercase"),
        ("postgresql+psycopg://app:secret@127.0.0.1:5432/x_test; DROP", "lowercase"),
    ],
)
def test_rejects_unsafe_test_database(test_url: str | None, reason: str) -> None:
    with pytest.raises(UnsafeTestDatabaseError, match=reason):
        validate_test_database_url(test_url, DEV_URL)


def test_rejects_test_database_with_same_name_as_development() -> None:
    dev_named_test = "postgresql+psycopg://app:secret@127.0.0.1:5432/shared_test"
    with pytest.raises(UnsafeTestDatabaseError, match="development database"):
        validate_test_database_url(dev_named_test, dev_named_test)
