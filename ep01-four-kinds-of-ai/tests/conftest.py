"""
tests/conftest.py - pytest runs this file before the tests.

It makes sure tests run OFFLINE:
  1. turns off telemetry (CrewAI, Chroma, LangSmith)
  2. blocks the internet: connecting to any non-local address raises an error,
     so a test that accidentally calls a real AI model fails loudly.
"""

import os
import socket

import pytest

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Hugging Face: only use models already downloaded

LOCAL_ADDRESSES = ("127.0.0.1", "localhost", "::1")
_real_connect = socket.socket.connect


def _connect_local_only(self, address):
    """Block socket connections that are not to this computer.

    Why this exists: a test that accidentally calls Gemini should fail here
    with a clear message, instead of using the network.

    Called by: no_internet, which installs this in place of socket.socket.connect.
    """
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and host not in LOCAL_ADDRESSES and not host.startswith("/"):
        raise RuntimeError(
            f"Tests must not use the internet (tried to reach {host!r}). "
            "Pass a fake model from testing/fakes.py instead."
        )
    return _real_connect(self, address)


@pytest.fixture(autouse=True)
def no_internet(monkeypatch):
    """Turn on the local-only socket for every test. Pytest calls this itself.

    Why this exists: the suite must stay offline. autouse=True means you do
    not pass no_internet into each test. It still runs.
    """
    monkeypatch.setattr(socket.socket, "connect", _connect_local_only)
