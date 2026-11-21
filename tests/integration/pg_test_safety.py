"""Safety guards for the PostgreSQL integration test database.

Every destructive test operation (schema migration, TRUNCATE) must go through
this module. The guards are layered:

1. ``validate_test_database_url`` checks the configured URL before any
   connection is made: TEST_DATABASE_URL must be set explicitly, point at
   PostgreSQL, name a database ending in ``_test``, and differ from the
   development database named by DATABASE_URL.
2. ``assert_connected_to_test_database`` re-checks the *live* connection with
   ``SELECT current_database()`` immediately before destructive work.
3. ``truncate_test_tables`` is the only cleanup routine, and it calls (2) first.

Test database creation is never implicit: see ``create_test_database`` and the
``--create-test-db`` pytest option.
"""

import re

from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import URL, make_url

TEST_DATABASE_SUFFIX = "_test"
_RESERVED_DATABASES = frozenset({"postgres", "template0", "template1"})
_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9_]+$")
_TABLES_TO_TRUNCATE = ("job_postings", "sources")


class UnsafeTestDatabaseError(RuntimeError):
    """Raised when a database does not qualify as the dedicated test database."""


def validate_test_database_url(test_url: str | None, dev_url: str | None) -> URL:
    if not test_url:
        raise UnsafeTestDatabaseError(
            "TEST_DATABASE_URL is not set. PostgreSQL integration tests never fall "
            "back to DATABASE_URL."
        )

    url = make_url(test_url)
    name = url.database

    if url.get_backend_name() != "postgresql":
        raise UnsafeTestDatabaseError("TEST_DATABASE_URL must point to PostgreSQL.")

    if not name or not _SAFE_IDENTIFIER.match(name):
        raise UnsafeTestDatabaseError(
            "The test database name must contain only lowercase letters, digits "
            "and underscores."
        )

    if not name.endswith(TEST_DATABASE_SUFFIX) or name in _RESERVED_DATABASES:
        raise UnsafeTestDatabaseError(
            f"Refusing test database {name!r}: its name must end with "
            f"{TEST_DATABASE_SUFFIX!r}."
        )

    if dev_url and make_url(dev_url).database == name:
        raise UnsafeTestDatabaseError(
            f"Refusing test database {name!r}: it is the development database "
            "named by DATABASE_URL."
        )

    return url


def assert_connected_to_test_database(connection: Connection, expected_name: str) -> None:
    actual = connection.execute(text("SELECT current_database()")).scalar_one()

    if actual != expected_name or not actual.endswith(TEST_DATABASE_SUFFIX):
        raise UnsafeTestDatabaseError(
            f"Connected to database {actual!r}, expected test database "
            f"{expected_name!r}. Aborting before any destructive operation."
        )


def truncate_test_tables(connection: Connection, expected_name: str) -> None:
    """Remove all rows from application tables in the verified test database."""
    assert_connected_to_test_database(connection, expected_name)
    connection.execute(
        text(f"TRUNCATE {', '.join(_TABLES_TO_TRUNCATE)} RESTART IDENTITY CASCADE")
    )


def create_test_database(test_url: URL) -> bool:
    """Create the test database if missing. Returns True if it was created.

    Connects to the ``postgres`` maintenance database with the same
    credentials. Only ever issues CREATE DATABASE; never drops anything.
    ``test_url`` must come from ``validate_test_database_url``.
    """
    name = test_url.database
    assert name is not None and _SAFE_IDENTIFIER.match(name)
    assert name.endswith(TEST_DATABASE_SUFFIX)

    engine = create_engine(
        test_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 5},
    )
    try:
        with engine.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": name},
            ).scalar_one_or_none()
            if exists:
                return False
            connection.execute(text(f'CREATE DATABASE "{name}"'))
            return True
    finally:
        engine.dispose()
