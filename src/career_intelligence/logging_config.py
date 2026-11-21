"""Logging setup and credential redaction.

Redaction is defence in depth: code in this project only logs URLs that went
through ``redact_url`` and never logs headers. ``RedactingFilter`` additionally
masks credential-looking text in any record that reaches it, including records
from third-party libraries such as httpx (which logs full request URLs at
INFO level).

Redacting logs does not make it safe to send credentials over plain HTTP;
``HttpClient`` refuses to do that separately.
"""

import logging
import re
from collections.abc import Iterable

import httpx

REDACTED = "REDACTED"

# Query parameter names treated as credentials (compared case-insensitively).
SENSITIVE_PARAMS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "client_secret",
        "key",
        "password",
        "secret",
        "signature",
        "token",
    }
)

# Header names treated as credentials (compared case-insensitively).
SENSITIVE_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization", "x-api-key"})

_PARAM_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9_])("
    + "|".join(re.escape(name) for name in sorted(SENSITIVE_PARAMS, key=len, reverse=True))
    + r")=([^&\s\"'#]+)"
)
_HEADER_PATTERN = re.compile(
    r"(?i)\b("
    + "|".join(re.escape(name) for name in sorted(SENSITIVE_HEADERS))
    + r")(['\"]?\s*[:=]\s*['\"]?)(?:(bearer|basic|token)\s+)?([^\s,;'\"]+)"
)

# Password in URL userinfo: scheme://user:PASSWORD@host
_USERINFO_PATTERN = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s/:@]*:)[^\s/@]+@")

_THIRD_PARTY_URL_LOGGERS = ("httpx",)


def redact_text(text: str) -> str:
    """Mask credential-looking query parameters, URL passwords and auth headers."""
    text = _USERINFO_PATTERN.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    text = _PARAM_PATTERN.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return _HEADER_PATTERN.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) + ' ' if m.group(3) else ''}{REDACTED}",
        text,
    )


def redact_url(url: str | httpx.URL, extra_sensitive_params: Iterable[str] = ()) -> str:
    """Return ``url`` with credential query values and any userinfo password masked."""
    parsed = httpx.URL(str(url))
    sensitive = SENSITIVE_PARAMS | {name.lower() for name in extra_sensitive_params}

    if parsed.params:
        parsed = parsed.copy_with(
            params=[
                (name, REDACTED if name.lower() in sensitive else value)
                for name, value in parsed.params.multi_items()
            ]
        )
    if parsed.password:
        parsed = parsed.copy_with(username=parsed.username, password=REDACTED)

    return str(parsed)


class RedactingFilter(logging.Filter):
    """Rewrite log records so credential-looking values never reach a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_text(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        return True


def protect_third_party_loggers() -> None:
    """Attach ``RedactingFilter`` to libraries known to log request URLs. Idempotent."""
    for name in _THIRD_PARTY_URL_LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactingFilter) for f in logger.filters):
            logger.addFilter(RedactingFilter())


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide logging for command-line entry points.

    Records are rendered as ``key=value`` pairs after a timestamp and pass
    through ``RedactingFilter``. httpx/httpcore are limited to WARNING because
    this project logs its own sanitized request lines.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
    )
    handler.addFilter(RedactingFilter())

    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    protect_third_party_loggers()
