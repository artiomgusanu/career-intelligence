import logging

import pytest

from career_intelligence.logging_config import (
    REDACTED,
    RedactingFilter,
    protect_third_party_loggers,
    redact_text,
    redact_url,
)

SECRET = "sk_live_DO_NOT_LEAK_123"


def test_redact_url_masks_sensitive_params_case_insensitively() -> None:
    url = f"https://api.example.com/job/list.json?API_KEY={SECRET}&page=2&limit=50"

    redacted = redact_url(url)

    assert SECRET not in redacted
    assert f"API_KEY={REDACTED}" in redacted
    assert "page=2" in redacted and "limit=50" in redacted


def test_redact_url_masks_extra_params_and_userinfo_password() -> None:
    url = f"https://user:{SECRET}@api.example.com/x?custom_auth={SECRET}&q=python"

    redacted = redact_url(url, extra_sensitive_params={"Custom_Auth"})

    assert SECRET not in redacted
    assert "user:" in redacted
    assert "q=python" in redacted


def test_redact_url_leaves_clean_urls_unchanged() -> None:
    url = "https://api.example.com/job/list.json?page=1&q=java"

    assert redact_url(url) == url


@pytest.mark.parametrize(
    "text",
    [
        f"GET https://api.example.com/x?api_key={SECRET}&page=1",
        f"token={SECRET}",
        f"Authorization: Bearer {SECRET}",
        f"x-api-key={SECRET}",
        f"{{'Authorization': 'Basic {SECRET}'}}",
    ],
)
def test_redact_text_masks_credentials(text: str) -> None:
    assert SECRET not in redact_text(text)


@pytest.mark.parametrize(
    "text",
    [
        f'HTTP Request: GET https://user:{SECRET}@api.example.com/x "HTTP/1.1 200 OK"',
        f"connect to postgresql+psycopg://app:{SECRET}@127.0.0.1:5432/db failed",
        f"http://:{SECRET}@host/",
    ],
)
def test_redact_text_masks_url_userinfo_passwords(text: str) -> None:
    redacted = redact_text(text)

    assert SECRET not in redacted
    assert f":{REDACTED}@" in redacted


def test_redact_text_keeps_urls_without_passwords() -> None:
    text = "GET https://user@api.example.com/x and mailto:someone@example.com"

    assert redact_text(text) == text


def test_redact_text_does_not_touch_lookalike_names() -> None:
    text = "monkey=banana keyboard=qwerty page=1"

    assert redact_text(text) == text


def test_redacting_filter_rewrites_records_with_format_args(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.redaction")
    logger.addFilter(RedactingFilter())
    try:
        with caplog.at_level(logging.INFO, logger="test.redaction"):
            logger.info("HTTP Request: GET %s", f"https://api.example.com/?api_key={SECRET}")
    finally:
        logger.filters.clear()

    assert SECRET not in caplog.text
    assert REDACTED in caplog.text


def test_httpx_logger_is_protected_and_idempotent(caplog: pytest.LogCaptureFixture) -> None:
    protect_third_party_loggers()
    protect_third_party_loggers()
    httpx_logger = logging.getLogger("httpx")

    assert sum(isinstance(f, RedactingFilter) for f in httpx_logger.filters) == 1
    with caplog.at_level(logging.INFO, logger="httpx"):
        httpx_logger.info('HTTP Request: GET %s "%s"', f"https://a.example/?key={SECRET}", "200")
    assert SECRET not in caplog.text
