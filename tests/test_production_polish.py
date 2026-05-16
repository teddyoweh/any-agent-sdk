"""Production polish: memory injection, HTTP retries, structured logging."""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
import httpx
import pytest

from any_agent_sdk import (
    Agent,
    MemoryEntry,
    save_memory_entry,
    update_memory_index,
)
from any_agent_sdk.providers.mock import MockProvider
from any_agent_sdk.retry import RetryTransport, _parse_retry_after


# ---------------------------------------------------------------------------
# Memory injection
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_anyagent_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ANYAGENT_HOME", str(tmp_path))
    return tmp_path


def test_agent_loads_memory_index_into_system_prompt(
    tmp_anyagent_home: Path,
) -> None:
    """When include_memory=True (default), MEMORY.md is prepended to system."""

    save_memory_entry(
        MemoryEntry(
            slug="sf_housing",
            name="SF housing search",
            description="Apartment search in SF, $3500 max",
            type="project",
            body="Looking in Mission, Hayes Valley.",
        )
    )
    update_memory_index()

    agent = Agent(
        model="qwen2.5-7b-instruct",
        provider=MockProvider(),
        system="You are a helpful agent.",
    )
    try:
        assert agent.system is not None
        # The memory block is prepended.
        assert "Persistent memory" in agent.system
        assert "SF housing search" in agent.system
        # The user's system is preserved after a separator.
        assert "You are a helpful agent." in agent.system
        assert agent.system.index("Persistent memory") < agent.system.index(
            "You are a helpful agent."
        )
    finally:
        anyio.run(agent.aclose)


def test_agent_include_memory_false_skips_load(tmp_anyagent_home: Path) -> None:
    """Opt out via include_memory=False; system prompt is untouched."""

    save_memory_entry(
        MemoryEntry(
            slug="x",
            name="X",
            description="x",
            type="project",
            body="x",
        )
    )
    update_memory_index()

    agent = Agent(
        model="qwen2.5-7b-instruct",
        provider=MockProvider(),
        system="Pristine system.",
        include_memory=False,
    )
    try:
        assert agent.system == "Pristine system."
    finally:
        anyio.run(agent.aclose)


def test_memory_missing_is_silent_noop(tmp_anyagent_home: Path) -> None:
    """No MEMORY.md → agent constructs without raising; system unchanged."""

    agent = Agent(
        model="qwen2.5-7b-instruct",
        provider=MockProvider(),
        system="Original system.",
    )
    try:
        assert agent.system == "Original system."
    finally:
        anyio.run(agent.aclose)


# ---------------------------------------------------------------------------
# Retry middleware
# ---------------------------------------------------------------------------


def test_retry_after_parsing() -> None:
    assert _parse_retry_after("3") == 3.0
    assert _parse_retry_after("0.5") == 0.5
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("not-a-number") is None
    # Negative values are invalid per spec — ignore.
    assert _parse_retry_after("-1") is None


def test_retry_replays_on_503_then_succeeds() -> None:
    """Inject a flaky transport: first 2 calls 503, then 200. The retry
    middleware should swallow the 503s and surface the 200 to the caller."""

    calls = {"n": 0}

    class FlakyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls["n"] += 1
            if calls["n"] <= 2:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, text="ok")

        async def aclose(self) -> None:
            pass

    rt = RetryTransport(FlakyTransport(), attempts=5, base_s=0.01, max_s=0.05, jitter=False)

    async def main():
        async with httpx.AsyncClient(transport=rt) as client:
            r = await client.get("http://example/x")
            return r

    r = anyio.run(main)
    assert r.status_code == 200
    assert calls["n"] == 3  # 2 failures + 1 success


def test_retry_exhausts_attempts_and_returns_last_failure() -> None:
    """Endless 500s with attempts=2 → return the second 500 (no retries
    left) rather than raising."""

    class AlwaysFails(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(500, text="boom")

        async def aclose(self) -> None:
            pass

    rt = RetryTransport(AlwaysFails(), attempts=2, base_s=0.01, max_s=0.05, jitter=False)

    async def main():
        async with httpx.AsyncClient(transport=rt) as client:
            return await client.get("http://example/x")

    r = anyio.run(main)
    assert r.status_code == 500


def test_retry_on_connect_error() -> None:
    """Two ConnectErrors then a 200 → success."""

    calls = {"n": 0}

    class FlakyNet(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise httpx.ConnectError("kaboom")
            return httpx.Response(200, text="ok")

        async def aclose(self) -> None:
            pass

    rt = RetryTransport(FlakyNet(), attempts=5, base_s=0.01, max_s=0.05, jitter=False)

    async def main():
        async with httpx.AsyncClient(transport=rt) as client:
            return await client.get("http://example/x")

    r = anyio.run(main)
    assert r.status_code == 200
    assert calls["n"] == 3


def test_make_client_uses_retry_by_default() -> None:
    """Smoke test: make_client wires RetryTransport into the AsyncClient."""

    from any_agent_sdk.http import make_client

    client = make_client(base_url="http://x.example/", retries=True)
    try:
        # The transport hierarchy is internal-ish but we can introspect.
        # httpx stores the transport on `_transport` for the default scheme.
        # We just assert it's a RetryTransport instance.
        assert isinstance(
            client._transport, RetryTransport
        ), f"expected RetryTransport, got {type(client._transport).__name__}"
    finally:
        anyio.run(client.aclose)


def test_make_client_can_disable_retries() -> None:
    """retries=False reverts to plain httpx transport."""

    from any_agent_sdk.http import make_client

    client = make_client(base_url="http://x.example/", retries=False)
    try:
        assert not isinstance(client._transport, RetryTransport)
    finally:
        anyio.run(client.aclose)
