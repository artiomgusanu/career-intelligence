"""Fixtures for PostgreSQL integration tests.

These tests run against a dedicated database named by TEST_DATABASE_URL (it
must end in ``_test``), never the development database. See README "Testing".

Isolation:
- ``db_session``: everything runs inside an outer transaction that is rolled
  back after the test (repository commits become SAVEPOINT releases).
- ``committed_sessions``: for tests that need real commits (timestamps across
  transactions, concurrency, rollback). Tables are truncated afterwards,
  through the guarded ``truncate_test_tables``.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pg_test_safety import (
    UnsafeTestDatabaseError,
    assert_connected_to_test_database,
    create_test_database,
    truncate_test_tables,
    validate_test_database_url,
)
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

# Importing the application's session module loads the project .env.
import career_intelligence.db.session  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def pg_engine(request: pytest.FixtureRequest) -> Iterator[Engine]:
    raw_test_url = os.environ.get("TEST_DATABASE_URL")
    if not raw_test_url:
        pytest.skip("TEST_DATABASE_URL is not set; PostgreSQL integration tests skipped.")

    try:
        test_url = validate_test_database_url(raw_test_url, os.environ.get("DATABASE_URL"))
    except UnsafeTestDatabaseError as exc:
        pytest.fail(str(exc), pytrace=False)

    database_name = test_url.database
    assert database_name is not None

    if request.config.getoption("--create-test-db"):
        created = create_test_database(test_url)
        print(f"\nTest database {database_name!r} {'created' if created else 'already exists'}.")

    engine = create_engine(test_url, connect_args={"connect_timeout": 5})

    try:
        with engine.begin() as connection:
            assert_connected_to_test_database(connection, database_name)

            config = Config()
            config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

            truncate_test_tables(connection, database_name)
    except OperationalError as exc:
        engine.dispose()
        pytest.fail(
            f"Cannot connect to test database {database_name!r}: {exc.orig}\n"
            "If it does not exist yet, create it explicitly with:\n"
            "    pytest --create-test-db",
            pytrace=False,
        )

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(pg_engine: Engine) -> Iterator[Session]:
    with pg_engine.connect() as connection:
        transaction = connection.begin()
        session = Session(
            bind=connection,
            join_transaction_mode="create_savepoint",
            autoflush=False,
            expire_on_commit=False,
        )
        try:
            yield session
        finally:
            session.close()
            transaction.rollback()


@pytest.fixture
def committed_sessions(pg_engine: Engine) -> Iterator[sessionmaker[Session]]:
    factory = sessionmaker(bind=pg_engine, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        with pg_engine.begin() as connection:
            # The engine URL was validated in pg_engine; the live database is re-checked.
            truncate_test_tables(connection, pg_engine.url.database)
