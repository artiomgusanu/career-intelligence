import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    # Declared here (an initial conftest) so the option is recognised for any
    # pytest invocation. Used by tests/integration/conftest.py.
    parser.addoption(
        "--create-test-db",
        action="store_true",
        default=False,
        help="Create the database named by TEST_DATABASE_URL if it does not exist "
        "(PostgreSQL integration tests). Never drops anything.",
    )
