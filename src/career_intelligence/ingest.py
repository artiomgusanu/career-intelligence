"""Command-line entry point for running an ingestion.

Usage::

    python -m career_intelligence.ingest local-file PATH                    # dry run
    python -m career_intelligence.ingest itjobs                             # dry run
    python -m career_intelligence.ingest --persist SOURCE ...               # write
    python -m career_intelligence.ingest --persist --allow-demo-data local-file PATH

Runs are dry runs by default: postings are fetched, mapped and validated, and
the database layer (SQLAlchemy included) is never imported. Writing requires
``--persist``. Demonstration sources (``source_type == "demo"``) additionally
require ``--allow-demo-data``, so demo data cannot be persisted by accident.

The ``itjobs`` source reads its API key from ``ITJOBS_API_KEY`` (environment
or ``.env``), never from the command line. A missing key fails before any
request. The HTTP client is created with ``ITJOBS_BASE_URL`` and
``ITJOBS_MIN_INTERVAL_SECONDS`` and is closed once the collection ends,
whatever its outcome, before the database is touched.

Exit codes: 0 success, 1 ingestion failure, 2 invalid arguments, refused
demo persistence or invalid configuration.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from career_intelligence.config import (
    ConfigurationError,
    get_secret,
    load_environment,
    load_http_settings,
    load_log_level,
)
from career_intelligence.ingestion.contracts import DEMO_SOURCE_TYPE, IngestionError, SourceAdapter
from career_intelligence.ingestion.http import HttpClient
from career_intelligence.ingestion.service import CollectionResult, collect_postings
from career_intelligence.logging_config import configure_logging
from career_intelligence.sources.itjobs import (
    ITJOBS_BASE_URL,
    ITJOBS_MIN_INTERVAL_SECONDS,
    ITJobsAdapter,
)
from career_intelligence.sources.local_file import LocalFileAdapter

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MAX_REJECTIONS_SHOWN = 20
ITJOBS_API_KEY_ENV = "ITJOBS_API_KEY"

type HttpClientFactory = Callable[[], HttpClient]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m career_intelligence.ingest",
        description="Collect job postings from a source. Dry run unless --persist is given.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write postings to the database named by DATABASE_URL. Without this flag "
        "nothing is written and the database is never contacted.",
    )
    parser.add_argument(
        "--allow-demo-data",
        action="store_true",
        help="Required together with --persist to store data from a demonstration source.",
    )
    sources = parser.add_subparsers(dest="source", required=True, metavar="SOURCE")

    local_file = sources.add_parser(
        "local-file", help="Demonstration postings from a local JSON file."
    )
    local_file.add_argument("path", type=Path, help="Path to the JSON file.")

    sources.add_parser(
        "itjobs",
        help=f"Job postings from the ITJobs.pt API (API key read from {ITJOBS_API_KEY_ENV}).",
    )

    return parser


def create_itjobs_client() -> HttpClient:
    """HTTP client for ITJobs: fixed base URL and minimum interval, HTTP_* settings."""
    return HttpClient(ITJOBS_BASE_URL, min_interval_seconds=ITJOBS_MIN_INTERVAL_SECONDS)


def build_adapter(
    args: argparse.Namespace,
    resources: ExitStack,
    *,
    itjobs_client_factory: HttpClientFactory = create_itjobs_client,
) -> SourceAdapter:
    """Create the adapter for ``args.source``.

    Resources the adapter depends on (the ITJobs HTTP client) are registered
    on ``resources``, which the caller closes. Raises ``ConfigurationError``
    before creating any resource if required configuration is missing.
    """
    if args.source == "local-file":
        return LocalFileAdapter(args.path)
    if args.source == "itjobs":
        api_key = get_secret(ITJOBS_API_KEY_ENV)
        if api_key is None:
            raise ConfigurationError(f"{ITJOBS_API_KEY_ENV} is not set.")
        client = resources.enter_context(itjobs_client_factory())
        return ITJobsAdapter(client, api_key=api_key)
    raise AssertionError(f"unhandled source {args.source!r}")  # pragma: no cover


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: "Callable[[], Session] | None" = None,
    itjobs_client_factory: HttpClientFactory | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.allow_demo_data and not args.persist:
        parser.error("--allow-demo-data requires --persist")

    try:
        load_http_settings()  # fail fast; HTTP clients read the same settings
    except ConfigurationError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    # Resources (HTTP clients) are closed when the collection ends, whatever its
    # outcome, and before the database is touched.
    with ExitStack() as resources:
        try:
            adapter = build_adapter(
                args,
                resources,
                itjobs_client_factory=itjobs_client_factory or create_itjobs_client,
            )
        except ConfigurationError as exc:
            print(f"error: invalid configuration: {exc}", file=sys.stderr)
            return 2

        is_demo = adapter.metadata.source_type == DEMO_SOURCE_TYPE
        if args.persist and is_demo and not args.allow_demo_data:
            print(
                f"error: {adapter.metadata.name!r} is a demonstration source; "
                "refusing to persist without --allow-demo-data",
                file=sys.stderr,
            )
            return 2

        try:
            collection = collect_postings(adapter)
        except IngestionError as exc:
            logger.error("ingestion_failed source=%r error=%s", args.source, exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1

    _print_collection(collection, persist=args.persist)
    if not args.persist:
        print("dry run: nothing was written (use --persist to store postings)")
        return 0

    # Imported only when persisting, so dry runs never load the database layer.
    from career_intelligence.ingestion.persistence import persist_collection

    if session_factory is None:
        from career_intelligence.db.session import SessionLocal

        session_factory = SessionLocal

    with session_factory() as session:
        print(f"persisting to database {_describe_target(session)}")
        with session.begin():
            result = persist_collection(session, collection)

    print(
        f"persisted source_id={result.source_id} source_created={result.source_created} "
        f"inserted={result.upsert.inserted} updated={result.upsert.updated} "
        f"duplicates_in_batch={result.upsert.duplicates_in_batch}"
    )
    return 0


def _describe_target(session: "Session") -> str:
    """host:port/database only: never credentials or query parameters."""
    url = session.get_bind().engine.url
    return f"{url.host}:{url.port}/{url.database}"


def _print_collection(collection: CollectionResult, *, persist: bool) -> None:
    mode = "persist" if persist else "dry-run"
    print(
        f"source={collection.source.name!r} mode={mode} fetched={collection.fetched} "
        f"valid={len(collection.postings)} rejected={len(collection.rejected)}"
    )
    for item in collection.rejected[:MAX_REJECTIONS_SHOWN]:
        print(
            f"  rejected position={item.position} external_id={item.external_id!r}: {item.reason}"
        )
    if len(collection.rejected) > MAX_REJECTIONS_SHOWN:
        print(f"  ... {len(collection.rejected) - MAX_REJECTIONS_SHOWN} more rejected")


def run() -> None:
    load_environment()
    configure_logging(load_log_level())
    sys.exit(main())


if __name__ == "__main__":
    run()
