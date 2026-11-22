import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from career_intelligence import ingest
from career_intelligence.ingest import main
from career_intelligence.ingestion import service
from career_intelligence.ingestion.contracts import (
    MappingError,
    RawPosting,
    SourceFetchError,
    SourceMetadata,
)
from career_intelligence.ingestion.http import HttpClient
from career_intelligence.ingestion.pagination import IncompletePaginationError

FIXTURE = Path(__file__).parent / "fixtures" / "local_postings.json"


def forbidden_session_factory():
    raise AssertionError("the database must not be contacted")


# --- Default mode: dry run ----------------------------------------------------------


def test_default_run_is_a_dry_run_and_never_touches_the_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["local-file", str(FIXTURE)], session_factory=forbidden_session_factory)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "mode=dry-run fetched=8 valid=4 rejected=4" in out
    assert "external_id='naive-date'" in out
    assert "dry run: nothing was written" in out
    assert "persisting to" not in out
    assert "persisted" not in out


def test_dry_run_process_never_imports_the_database_layer() -> None:
    code = (
        "import sys\n"
        "from career_intelligence.ingest import main\n"
        f"code = main(['local-file', {str(FIXTURE)!r}])\n"
        "loaded = [m for m in ('sqlalchemy', 'psycopg', 'career_intelligence.db.session')"
        " if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "sys.exit(code)\n"
    )
    env = {**os.environ, "DATABASE_URL": "postgresql+psycopg://nobody@192.0.2.1:1/never"}

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "valid=4 rejected=4" in completed.stdout


def test_module_entry_point_runs_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "career_intelligence.ingest", "local-file", str(FIXTURE)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "collection_finished" in completed.stderr  # logging configured, key=value
    assert "fetched=8 valid=4 rejected=4" in completed.stderr


# --- Demo data protection ------------------------------------------------------------


def test_persist_without_allow_demo_data_is_refused_before_collecting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["--persist", "local-file", str(FIXTURE)], session_factory=forbidden_session_factory
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "demonstration source" in captured.err
    assert "--allow-demo-data" in captured.err
    assert "fetched=" not in captured.out  # refused before any work


def test_allow_demo_data_without_persist_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--allow-demo-data", "local-file", str(FIXTURE)])

    assert exc_info.value.code == 2
    assert "requires --persist" in capsys.readouterr().err


# --- Configuration and failures --------------------------------------------------------


def test_invalid_http_configuration_fails_at_start_up_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "not-a-number-XYZ")

    exit_code = main(["local-file", str(FIXTURE)], session_factory=forbidden_session_factory)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "HTTP_TIMEOUT_SECONDS" in captured.err
    assert "XYZ" not in captured.err
    assert "fetched=" not in captured.out


class ScriptedAdapter:
    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="Scripted", url="https://scripted.example.com/", source_type="test"
        )

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        raise self._error
        yield  # pragma: no cover - makes this a generator

    def map_posting(self, raw: RawPosting) -> dict[str, Any]:  # pragma: no cover
        return dict(raw)


def _http_404_error() -> Exception:
    client = HttpClient(
        "https://api.example.com", transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with client:
        try:
            client.get_json("/x")
        except Exception as exc:  # noqa: BLE001 - captured to replay it from an adapter
            return exc
    raise AssertionError("expected an HTTP error")


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: SourceFetchError("file vanished"),
        _http_404_error,
        lambda: IncompletePaginationError("Page 3 was empty while the source reports more"),
    ],
    ids=["source_fetch", "http_status", "incomplete_pagination"],
)
def test_every_ingestion_error_is_reported_with_exit_code_1(
    error_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = error_factory()
    monkeypatch.setattr(ingest, "build_adapter", lambda *_, **__: ScriptedAdapter(error))

    exit_code = main(["local-file", "ignored.json"], session_factory=forbidden_session_factory)

    assert exit_code == 1
    assert str(error) in capsys.readouterr().err


def test_invalid_adapter_metadata_is_reported_with_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class BadMetadataAdapter(ScriptedAdapter):
        @property
        def metadata(self) -> SourceMetadata:
            return SourceMetadata(name="", url="not a url", source_type="test")

    monkeypatch.setattr(
        ingest, "build_adapter", lambda *_, **__: BadMetadataAdapter(RuntimeError("unused"))
    )

    exit_code = main(["local-file", "ignored.json"], session_factory=forbidden_session_factory)

    assert exit_code == 1
    assert "Invalid source metadata" in capsys.readouterr().err


def test_missing_file_returns_failure_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["local-file", str(tmp_path / "missing.json")])

    assert exit_code == 1
    assert "Cannot read" in capsys.readouterr().err


def test_invalid_arguments_exit_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2


# --- Persistence rule: no valid postings, no database ---------------------------------------
# Collections are built from records of the existing fixture: its first three
# records are valid and distinct, the last four are rejected.


def fixture_records() -> list[Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["postings"]


def write_postings(tmp_path: Path, records: list[Any]) -> str:
    path = tmp_path / "postings.json"
    path.write_text(json.dumps({"postings": records}), encoding="utf-8")
    return str(path)


PERSIST = ["--persist", "--allow-demo-data"]


def test_empty_collection_dry_run_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, [])

    exit_code = main(["local-file", path], session_factory=forbidden_session_factory)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "mode=dry-run fetched=0 valid=0 rejected=0" in out
    assert "dry run: nothing was written" in out


def test_empty_collection_with_persist_exits_0_without_touching_the_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, [])

    exit_code = main([*PERSIST, "local-file", path], session_factory=forbidden_session_factory)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode=persist fetched=0 valid=0 rejected=0" in captured.out
    assert "nothing to persist: the source returned no postings" in captured.out
    assert "persisting to" not in captured.out
    assert captured.err == ""


def test_all_rejected_dry_run_exits_0_and_reports_rejections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, fixture_records()[4:])

    exit_code = main(["local-file", path], session_factory=forbidden_session_factory)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "mode=dry-run fetched=4 valid=0 rejected=4" in out
    assert "external_id='naive-date'" in out
    assert "dry run: nothing was written" in out


def test_all_rejected_with_persist_exits_1_without_touching_the_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, fixture_records()[4:])

    exit_code = main([*PERSIST, "local-file", path], session_factory=forbidden_session_factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "mode=persist fetched=4 valid=0 rejected=4" in captured.out  # results still shown
    assert "external_id='naive-date'" in captured.out
    assert captured.err == "error: refusing to persist: all 4 fetched postings were rejected\n"
    assert "persisting to" not in captured.out


@pytest.mark.parametrize(
    ("records", "valid", "rejected"),
    [(slice(0, 3), 3, 0), (slice(0, 8), 4, 4)],
    ids=["all_valid", "partial"],
)
def test_collections_with_valid_postings_open_a_session_when_persisting(
    records: slice,
    valid: int,
    rejected: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_postings(tmp_path, fixture_records()[records])
    opened: list[bool] = []

    def recording_session_factory():
        opened.append(True)
        raise RuntimeError("stop before touching any database")

    with pytest.raises(RuntimeError, match="stop before"):
        main([*PERSIST, "local-file", path], session_factory=recording_session_factory)

    assert opened == [True]
    assert f"mode=persist fetched={valid + rejected} valid={valid} rejected={rejected}" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "records", [slice(0, 0), slice(4, 8), slice(0, 3)], ids=["empty", "all_rejected", "valid"]
)
def test_dry_run_never_opens_a_session(records: slice, tmp_path: Path) -> None:
    path = write_postings(tmp_path, fixture_records()[records])

    assert main(["local-file", path], session_factory=forbidden_session_factory) == 0


# --- Rejection summary ------------------------------------------------------------------


class ReasonAdapter:
    """Rejects every record at mapping, with the reason the record names."""

    def __init__(self, records: list[dict[str, str]]) -> None:
        self._records = records

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="Reasons", url="https://reasons.example.com/", source_type="test"
        )

    def fetch_raw_postings(self) -> Iterator[RawPosting]:
        yield from self._records

    def map_posting(self, raw: RawPosting) -> dict[str, Any]:
        raise MappingError(raw["why"])


def use_reasons_adapter(monkeypatch: pytest.MonkeyPatch, counts: dict[str, int]) -> ReasonAdapter:
    records = [{"why": reason} for reason, count in counts.items() for _ in range(count)]
    adapter = ReasonAdapter(records)
    monkeypatch.setattr(ingest, "build_adapter", lambda *_, **__: adapter)
    return adapter


def reason_lines(out: str) -> list[str]:
    """Lines of the "rejection reasons" block, without the "... more reasons" line."""
    return [
        line for line in out.splitlines() if line.startswith("    ") and "more reasons" not in line
    ]


def test_cli_shows_at_most_20_individual_rejections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rejected = fixture_records()[4]  # no company object
    path = write_postings(tmp_path, [{**rejected, "id": f"r-{n}"} for n in range(25)])

    assert main(["local-file", path], session_factory=forbidden_session_factory) == 0

    out = capsys.readouterr().out
    individual = [line for line in out.splitlines() if line.startswith("  rejected position=")]
    assert len(individual) == ingest.MAX_REJECTIONS_SHOWN == 20
    assert "  ... 5 more rejected" in out
    assert "  rejection reasons:\n    25  mapping: company must be an object\n" in out


def test_cli_shows_at_most_10_reasons_and_counts_the_rest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_reasons_adapter(monkeypatch, {f"reason-{k:02d}": k % 3 + 1 for k in range(12)})

    assert main(["local-file", "ignored.json"], session_factory=forbidden_session_factory) == 0

    out = capsys.readouterr().out
    assert "  rejection reasons (top 10 of 12):" in out
    assert len(reason_lines(out)) == service.MAX_REASONS_SHOWN == 10
    assert out.endswith(
        "    ... 2 more reasons\ndry run: nothing was written (use --persist to store postings)\n"
    )


def test_cli_orders_reasons_by_frequency_then_alphabetically(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_reasons_adapter(monkeypatch, {"b-reason": 2, "a-reason": 2, "c-reason": 3, "d-reason": 1})

    assert main(["local-file", "ignored.json"], session_factory=forbidden_session_factory) == 0

    out = capsys.readouterr().out
    assert "  rejection reasons:\n" in out
    assert reason_lines(out) == [
        "    3  mapping: c-reason",
        "    2  mapping: a-reason",
        "    2  mapping: b-reason",
        "    1  mapping: d-reason",
    ]


def test_cli_and_service_group_rejections_with_the_same_helper(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert ingest.count_rejection_reasons is service.count_rejection_reasons
    assert ingest.MAX_REASONS_SHOWN is service.MAX_REASONS_SHOWN
    caplog.set_level(logging.WARNING, logger="career_intelligence.ingestion.service")
    adapter = use_reasons_adapter(monkeypatch, {f"reason-{k:02d}": k % 4 + 1 for k in range(12)})

    assert main(["local-file", "ignored.json"], session_factory=forbidden_session_factory) == 0

    expected = service.count_rejection_reasons(service.collect_postings(adapter).rejected)
    shown = expected[: service.MAX_REASONS_SHOWN]
    assert reason_lines(capsys.readouterr().out) == [f"    {n}  {r}" for r, n in shown]
    summaries = [r for r in caplog.records if r.getMessage().startswith("rejections_summary")]
    assert len(summaries) == 2  # one per collection: the CLI run, then the direct call
    assert all(r.getMessage().endswith(f"top_reasons={dict(shown)!r}") for r in summaries)


@pytest.mark.parametrize("records", [slice(0, 0), slice(0, 3)], ids=["empty", "all_valid"])
def test_cli_shows_no_reasons_block_without_rejections(
    records: slice, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_postings(tmp_path, fixture_records()[records])

    assert main(["local-file", path], session_factory=forbidden_session_factory) == 0

    assert "rejection reasons" not in capsys.readouterr().out
