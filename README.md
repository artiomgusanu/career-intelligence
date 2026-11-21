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

## Ingestion

Job sources plug into a small pipeline (`src/career_intelligence/ingestion/`):

```
SourceAdapter.fetch_raw_postings()  ->  SourceAdapter.map_posting()  ->  JobPostingCreate
   (collection, may use HttpClient)       (normalization, pure)          (validation)
                                     ->  persist_collection()  ->  repository upsert
                                         (persistence, caller-owned transaction)
```

- `ingestion/contracts.py`: `SourceAdapter` protocol. Adapters never touch the database.
- `ingestion/http.py`: `HttpClient` with explicit timeouts, limited retries for transient
  failures, no redirects, HTTPS required, and credential-free errors and logs.
- `ingestion/pagination.py`: page-number pagination that refuses to truncate silently.
- `ingestion/service.py`: collection and validation, database-free.
- `ingestion/persistence.py`: registers the source and upserts postings (no commit).

Runs are **dry runs by default**: postings are fetched, mapped and validated, and the
database layer is never imported. Writing requires `--persist`, and the target database is
shown (host, port and name only) before anything is written.

The `local-file` source is a **demonstration source** (`source_type="demo"`, see
`sources/local_file.py` for the file format). Persisting demo data additionally requires
`--allow-demo-data`:

```bash
python -m career_intelligence.ingest local-file path/to/postings.json             # dry run
python -m career_intelligence.ingest --persist --allow-demo-data local-file path/to/postings.json
```

Logging is controlled by `LOG_LEVEL`; HTTP clients by `HTTP_TIMEOUT_SECONDS`,
`HTTP_MAX_RETRIES` and `HTTP_BACKOFF_SECONDS` (validated at CLI start-up).

## Development Status

Initial project setup.