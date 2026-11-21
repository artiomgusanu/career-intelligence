"""Configuration for outbound integrations, read from environment variables.

Deliberately small: frozen dataclasses and plain functions, no settings
framework. Database configuration stays in ``career_intelligence.db.session``.

Environment variables (all optional):

- ``LOG_LEVEL``: DEBUG, INFO, WARNING or ERROR (default INFO).
- ``HTTP_TIMEOUT_SECONDS``: per-request timeout (default 10).
- ``HTTP_MAX_RETRIES``: retries for transient failures, 0-5 (default 2).
- ``HTTP_BACKOFF_SECONDS``: base delay for exponential backoff (default 0.5).
- ``HTTP_MAX_RETRY_AFTER_SECONDS``: longest ``Retry-After`` honoured, 0-3600
  (default 60). A longer server-requested wait fails the request instead.

Source credentials (e.g. ``ITJOBS_API_KEY``) are read with ``get_secret``.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
MAX_HTTP_RETRIES = 5
MAX_RETRY_AFTER_LIMIT_SECONDS = 3600.0


class ConfigurationError(ValueError):
    """Raised when an environment variable holds an invalid value."""


@dataclass(frozen=True)
class HttpSettings:
    timeout_seconds: float = 10.0
    max_retries: int = 2
    backoff_seconds: float = 0.5
    max_retry_after_seconds: float = 60.0
    user_agent: str = "career-intelligence/0.1"


def load_environment() -> None:
    """Load the project ``.env`` if present. Existing variables are not overridden."""
    load_dotenv(PROJECT_ROOT / ".env")


def load_http_settings(env: Mapping[str, str] | None = None) -> HttpSettings:
    env = os.environ if env is None else env
    defaults = HttpSettings()

    timeout = _read_float(env, "HTTP_TIMEOUT_SECONDS", defaults.timeout_seconds)
    if timeout <= 0:
        raise ConfigurationError("HTTP_TIMEOUT_SECONDS must be greater than 0.")

    backoff = _read_float(env, "HTTP_BACKOFF_SECONDS", defaults.backoff_seconds)
    if backoff < 0:
        raise ConfigurationError("HTTP_BACKOFF_SECONDS cannot be negative.")

    retries = _read_int(env, "HTTP_MAX_RETRIES", defaults.max_retries)
    if not 0 <= retries <= MAX_HTTP_RETRIES:
        raise ConfigurationError(f"HTTP_MAX_RETRIES must be between 0 and {MAX_HTTP_RETRIES}.")

    retry_after_limit = _read_float(
        env, "HTTP_MAX_RETRY_AFTER_SECONDS", defaults.max_retry_after_seconds
    )
    if not 0 <= retry_after_limit <= MAX_RETRY_AFTER_LIMIT_SECONDS:
        raise ConfigurationError(
            "HTTP_MAX_RETRY_AFTER_SECONDS must be between 0 and "
            f"{MAX_RETRY_AFTER_LIMIT_SECONDS:.0f}."
        )

    return HttpSettings(
        timeout_seconds=timeout,
        max_retries=retries,
        backoff_seconds=backoff,
        max_retry_after_seconds=retry_after_limit,
    )


def load_log_level(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    level = (env.get("LOG_LEVEL") or "INFO").strip().upper()
    if level not in LOG_LEVELS:
        raise ConfigurationError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}.")
    return level


def get_secret(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """Return a credential from the environment, or None if unset or blank.

    The value is never logged or included in error messages.
    """
    env = os.environ if env is None else env
    value = (env.get(name) or "").strip()
    return value or None


def _read_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be a number.") from None


def _read_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer.") from None
