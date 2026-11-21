import pytest

from career_intelligence.config import (
    ConfigurationError,
    HttpSettings,
    get_secret,
    load_http_settings,
    load_log_level,
)


def test_http_settings_defaults_when_unset() -> None:
    assert load_http_settings({}) == HttpSettings()


def test_http_settings_read_from_environment() -> None:
    settings = load_http_settings(
        {"HTTP_TIMEOUT_SECONDS": "3.5", "HTTP_MAX_RETRIES": "4", "HTTP_BACKOFF_SECONDS": "0"}
    )

    assert settings.timeout_seconds == 3.5
    assert settings.max_retries == 4
    assert settings.backoff_seconds == 0


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"HTTP_TIMEOUT_SECONDS": "abc"}, "must be a number"),
        ({"HTTP_TIMEOUT_SECONDS": "0"}, "greater than 0"),
        ({"HTTP_BACKOFF_SECONDS": "-1"}, "cannot be negative"),
        ({"HTTP_MAX_RETRIES": "1.5"}, "must be an integer"),
        ({"HTTP_MAX_RETRIES": "6"}, "between 0 and 5"),
        ({"HTTP_MAX_RETRIES": "-1"}, "between 0 and 5"),
    ],
)
def test_http_settings_reject_invalid_values(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_http_settings(env)


def test_log_level_default_and_normalization() -> None:
    assert load_log_level({}) == "INFO"
    assert load_log_level({"LOG_LEVEL": " debug "}) == "DEBUG"


def test_log_level_rejects_unknown_value() -> None:
    with pytest.raises(ConfigurationError):
        load_log_level({"LOG_LEVEL": "verbose"})


def test_get_secret_returns_value_or_none() -> None:
    assert get_secret("ITJOBS_API_KEY", {"ITJOBS_API_KEY": " abc "}) == "abc"
    assert get_secret("ITJOBS_API_KEY", {"ITJOBS_API_KEY": "   "}) is None
    assert get_secret("ITJOBS_API_KEY", {}) is None


def test_configuration_errors_never_echo_values() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        load_http_settings({"HTTP_TIMEOUT_SECONDS": "s3cr3t-looking-value"})

    assert "s3cr3t-looking-value" not in str(exc_info.value)
