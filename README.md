# Career Intelligence

A software engineering project focused on building an intelligence platform using Python, FastAPI, PostgreSQL, and AI.

## Goals

- Build a maintainable REST API.
- Store and manage structured data with PostgreSQL.
- Implement database migrations and automated tests.
- Use Docker for reproducible development environments.
- Automate quality checks with GitHub Actions.
- Explore AI-powered intelligence workflows as the project evolves.

## Tech Stack

- Python
- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- pytest
- Docker Compose
- GitHub Actions

## Testing

```bash
pytest                      # all tests
pytest -m "not postgres"    # only tests that need no database
pytest -m postgres          # only PostgreSQL integration tests
```

PostgreSQL integration tests (`tests/integration/`) run against a **dedicated test
database** named by `TEST_DATABASE_URL` in `.env` (see `.env.example`). They never fall
back to `DATABASE_URL`.

Safety rules, enforced in `tests/integration/pg_test_safety.py`:

- The test database name must end in `_test` and differ from the development database.
- Before migrating or truncating, the live connection is re-checked with
  `SELECT current_database()`.
- Tests apply migrations (`alembic upgrade head`) and truncate tables **only** in the
  test database. Nothing is ever dropped.

The test database is never created implicitly. Create it once, explicitly:

```bash
docker compose up -d db
pytest --create-test-db -m postgres
```

If `TEST_DATABASE_URL` is unset, integration tests are skipped with a visible reason.
If it is set but the database is unreachable, they fail.

## Development Status

Initial project setup.