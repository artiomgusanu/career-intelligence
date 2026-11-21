"""The autouse guard in tests/conftest.py must really block external network access."""

import socket

import httpx
import pytest

BLOCKED = "blocked in tests"


def test_external_dns_lookup_is_blocked() -> None:
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.getaddrinfo("www.itjobs.pt", 443)


def test_external_connect_is_blocked() -> None:
    with socket.socket() as sock, pytest.raises(RuntimeError, match=BLOCKED):
        sock.connect(("192.0.2.1", 443))  # TEST-NET-1, never routable


def test_real_httpx_transport_cannot_reach_external_hosts() -> None:
    with httpx.Client(timeout=2) as client, pytest.raises(RuntimeError, match=BLOCKED):
        client.get("https://api.itjobs.pt/job/list.json")
