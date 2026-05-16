"""Provider protocol + registry.

A provider is a thin async object that:
  1. Takes a list of universal ``Message``s + tools + model params
  2. Calls the underlying API with streaming enabled
  3. Yields normalized ``StreamEvent``s

Everything else (the agent loop, tool dispatch, MCP, sub-agents) is provider-
agnostic.

The registry is a string → factory map. Resolution is lazy — we don't import
``providers.bedrock`` (which pulls boto3) unless someone asks for ``bedrock``.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol, runtime_checkable

import httpx

from ..events import StreamEvent
from ..types import Message


@runtime_checkable
class Provider(Protocol):
    """Adapter interface. Each provider lives in its own module."""

    name: str

    async def stream(
        self,
        *,
        model: str,
        messages: Iterable[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Open a streaming request and yield normalized events."""
        ...

    async def aclose(self) -> None:
        """Release any resources (HTTP client, etc.)."""
        ...


# ---------------------------------------------------------------------------
# Registry — lazy by design
# ---------------------------------------------------------------------------

# Maps provider name -> (module path, class name). Imported on first use.
_LAZY_PROVIDERS: dict[str, tuple[str, str]] = {
    "anthropic": ("any_agent_sdk.providers.anthropic", "AnthropicProvider"),
    "openai": ("any_agent_sdk.providers.openai", "OpenAIProvider"),
    "gemini": ("any_agent_sdk.providers.gemini", "GeminiProvider"),
    "bedrock": ("any_agent_sdk.providers.bedrock", "BedrockProvider"),
    "local": ("any_agent_sdk.providers.local", "LocalProvider"),
}

# Cache of resolved factories.
_RESOLVED: dict[str, type[Provider]] = {}


def register(name: str, factory: type[Provider]) -> None:
    """Register a custom provider. Useful for tests and third-party adapters."""

    _RESOLVED[name] = factory


def resolve(name: str) -> type[Provider]:
    """Get the provider factory for ``name``. Imports on first call."""

    if name in _RESOLVED:
        return _RESOLVED[name]
    if name not in _LAZY_PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(_LAZY_PROVIDERS)}")
    module_path, class_name = _LAZY_PROVIDERS[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    _RESOLVED[name] = cls
    return cls


def detect_provider(model: str) -> str:
    """Best-effort provider detection from the model name.

    Rules
    -----
    * ``claude-*`` → anthropic
    * ``gpt-*``, ``o1-*``, ``o3-*``, ``o4-*`` → openai
    * ``gemini-*`` → gemini
    * ``anthropic.*`` (Bedrock model IDs) → bedrock
    * Anything containing ``/`` → local (Ollama/vLLM-style ``org/model``)
    """

    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("anthropic.", "amazon.", "meta.")):
        return "bedrock"
    if "/" in model:
        return "local"
    raise ValueError(f"cannot detect provider from model name {model!r}; pass provider= explicitly")


# ---------------------------------------------------------------------------
# Shared utilities for adapter authors
# ---------------------------------------------------------------------------


class _HTTPProviderMixin:
    """Convenience base for HTTP-based providers. Stores a client and exposes
    a single ``aclose``. Adapters can inherit and add the ``stream`` method."""

    client: httpx.AsyncClient

    async def aclose(self) -> None:
        await self.client.aclose()
