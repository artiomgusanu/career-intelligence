import os
import socket
from collections.abc import Iterator

import pytest

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Duplicate of the process's real stderr, taken before pytest captures output.
# Deadlock watchdogs write their traceback dump here, so it stays visible even
# though pytest captures fd 2 while tests run.
_REAL_STDERR_FD = pytest.StashKey[int]()


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_REAL_STDERR_FD] = os.dup(2)


def pytest_unconfigure(config: pytest.Config) -> None:
    fd = config.stash.get(_REAL_STDERR_FD, None)
    if fd is not None:
        os.close(fd)
        del config.stash[_REAL_STDERR_FD]


@pytest.fixture
def real_stderr_fd(pytestconfig: pytest.Config) -> int:
    """File descriptor of the real stderr, unaffected by pytest's output capture."""
    return pytestconfig.stash[_REAL_STDERR_FD]


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


class ExternalNetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test that resolves or connects to a non-loopback host from Python.

    Loopback stays allowed for the PostgreSQL test database. psycopg connects
    through libpq (C), which does not go through Python's socket module.
    """
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def _host(address: object) -> str:
        return str(address[0]) if isinstance(address, tuple) else str(address)

    def guarded_connect(self: socket.socket, address: object) -> None:
        if _host(address) not in _LOOPBACK_HOSTS:
            raise ExternalNetworkBlockedError(f"External network blocked in tests: {_host(address)}")
        return real_connect(self, address)

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        if host is not None and str(host) not in _LOOPBACK_HOSTS:
            raise ExternalNetworkBlockedError(f"External DNS lookup blocked in tests: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield
