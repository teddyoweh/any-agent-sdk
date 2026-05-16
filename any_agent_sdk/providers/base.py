"""Provider protocol + lazy registry.

A provider is a backend adapter (an HTTP wire format) — not a model. One
adapter handles N models that speak the same protocol. The OpenAI-compat
adapter alone covers vLLM, Together, Fireworks, Groq, OpenRouter, Cerebras,
DeepInfra, Anyscale, DeepSeek's own API, and any future provider that
implements ``POST /v1/chat/completions``.

The registry is a string → factory map. Resolution is lazy by design — we
don't import ``providers.ollama`` (which doesn't pull anything heavy but
nonetheless costs ~5 ms of parsing on cold start) unless someone asks for
the ``ollama`` adapter.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol, runtime_checkable

import httpx

from ..capabilities import BackendCapability, ModelCapability
from ..events import StreamEvent
from ..types import Message


@runtime_checkable
class Provider(Protocol):
    """Adapter interface. Each provider lives in its own module."""

    name: str
    backend_capability: BackendCapability

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
        model_capability: ModelCapability | None = None,
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
    "openai_compat": ("any_agent_sdk.providers.openai_compat", "OpenAICompatProvider"),
    "ollama": ("any_agent_sdk.providers.ollama", "OllamaProvider"),
    "llamacpp": ("any_agent_sdk.providers.llamacpp", "LlamaCppProvider"),
    "tgi": ("any_agent_sdk.providers.tgi", "TGIProvider"),
    "modal": ("any_agent_sdk.providers.modal_provider", "ModalProvider"),
    "mock": ("any_agent_sdk.providers.mock", "MockProvider"),
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


def detect_provider(model_or_url: str, *, backend_hint: str | None = None) -> str:
    """Best-effort backend detection from a model name OR a backend URL.

    Resolution rules
    ----------------
    * Explicit ``backend_hint`` wins.
    * URL with ``:11434`` or ``ollama`` → ollama.
    * URL with ``llamacpp`` or ``:8080`` → llamacpp.
    * URL with ``tgi`` or ``text-generation-inference`` → tgi.
    * URL containing a known hosted provider name → openai_compat.
    * Bare ``http://`` or ``https://`` URL → openai_compat (the default).
    * ``"mock"`` literal → mock.
    * Otherwise: openai_compat.
    """

    if backend_hint:
        return backend_hint

    s = model_or_url.lower()
    if s == "mock":
        return "mock"
    if "11434" in s or "ollama" in s:
        return "ollama"
    if "llamacpp" in s or "llama.cpp" in s:
        return "llamacpp"
    if "tgi" in s or "text-generation-inference" in s:
        return "tgi"
    if s.startswith(("http://", "https://")):
        return "openai_compat"
    # Bare model name with no URL — default to openai_compat (with localhost vllm).
    return "openai_compat"


# ---------------------------------------------------------------------------
# Shared utilities for adapter authors
# ---------------------------------------------------------------------------


class HTTPProviderMixin:
    """Convenience base for HTTP-based providers. Stores a client and exposes
    a single ``aclose``. Adapters inherit and add the ``stream`` method."""

    client: httpx.AsyncClient

    async def aclose(self) -> None:
        await self.client.aclose()
