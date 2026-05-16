"""Agent — the run loop.

This module owns the multi-turn dance:

  1. Send messages → provider.stream (model + backend resolved from capability)
  2. Drive a StreamingToolExecutor on the event stream: tool calls dispatch
     as soon as their input JSON closes, not after the assistant finalizes.
  3. Run hooks (PreToolUse, PostToolUse, Stop) at the right moments.
  4. Check permissions before each tool call.
  5. Track budget (turns + USD + tokens) and raise BudgetExceededError when hit.
  6. If the assistant emits tool calls, append results + loop; otherwise stop.

The streaming variant (``Agent.stream``) yields the *normalized* event
stream so user UIs can render token-by-token. The non-streaming
``Agent.run`` consumes the stream internally and returns the final messages.

The agent is *backend-agnostic* — model + backend URL drive provider choice
via ``capabilities.lookup_model`` + ``providers.base.detect_provider``. Pass
``provider=`` directly to override.

Performance notes
-----------------
* Text deltas are *not* concatenated until the block stops (O(n) join once).
* Tool input JSON deltas are buffered and parsed once at block stop.
* Conversation list is appended-to in place, never copied.
* Capability lookups are O(1) and frozen onto the agent at init.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import msgspec

from .capabilities import (
    BackendCapability,
    ModelCapability,
    hosted_profile_from_url,
    lookup_model,
)
from .errors import StreamProtocolError
from .events import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    ErrorEvent,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
)
from .providers.base import Provider, detect_provider, resolve
from .tools import ToolRegistry, dispatch_tool_calls
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserMessage,
    Usage,
)

_log = logging.getLogger("any_agent_sdk.agent")
_JSON_DECODER = msgspec.json.Decoder()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Agent:
    """Multi-turn agent over any OSS model on any compatible backend.

    Construction
    ------------
    The minimal form is ``Agent(model="qwen2.5-72b-instruct")`` — but most
    users will also pass ``backend="http://localhost:11434"`` (Ollama),
    ``"https://api.together.xyz/v1"`` (Together), etc.

    ``provider`` overrides the auto-construction completely. Useful for
    tests (pass a ``MockProvider``) or exotic deployments.

    ``model_capability`` overrides the looked-up capability — useful when
    you know your custom-finetuned model supports tool calling but the
    family heuristic doesn't.
    """

    model: str
    backend: str | None = None
    provider: Provider | None = None
    system: str | None = None
    tools: ToolRegistry | list = field(default_factory=ToolRegistry)
    max_tokens: int = 1024
    temperature: float | None = None
    max_steps: int = 20
    max_turns: int | None = None  # alias for max_steps
    extra: dict[str, Any] | None = None

    # Capability + safety surface (M0.1 / M2)
    model_capability: ModelCapability | None = None
    backend_capability: BackendCapability | None = None

    def __post_init__(self) -> None:
        # Normalize tools input — accept list[Tool] or pre-built ToolRegistry.
        if not isinstance(self.tools, ToolRegistry):
            registry = ToolRegistry()
            if self.tools:
                registry.add(*self.tools)
            self.tools = registry

        # max_turns is a friendly alias for max_steps (Claude SDK parity).
        if self.max_turns is not None:
            self.max_steps = self.max_turns

        # Resolve model capability if not given explicitly.
        if self.model_capability is None:
            self.model_capability = lookup_model(self.model)

        # Resolve backend capability if not given explicitly.
        if self.backend_capability is None and self.backend:
            self.backend_capability = hosted_profile_from_url(self.backend)

        # Build the provider if not given.
        if self.provider is None:
            backend_str = self.backend or self.model
            backend_kind = detect_provider(backend_str)
            ProviderCls = resolve(backend_kind)
            self.provider = self._build_provider(ProviderCls, backend_kind)

        # Propagate temperature from capability if user didn't set one.
        if self.temperature is None:
            self.temperature = self.model_capability.recommended_temperature

    def _build_provider(self, ProviderCls: type[Provider], backend_kind: str) -> Provider:
        """Construct a provider with sensible defaults per backend kind."""

        kw: dict[str, Any] = {}
        if backend_kind == "openai_compat":
            kw["base_url"] = self.backend or os.environ.get(
                "ANY_AGENT_BASE_URL", "http://localhost:8000/v1"
            )
            kw["api_key"] = os.environ.get("ANY_AGENT_API_KEY")
            if self.backend_capability is not None:
                kw["backend_capability"] = self.backend_capability
        elif backend_kind == "ollama":
            kw["base_url"] = self.backend or "http://localhost:11434"
        elif backend_kind == "llamacpp":
            kw["base_url"] = self.backend or "http://localhost:8080"
        elif backend_kind == "tgi":
            kw["base_url"] = self.backend or "http://localhost:3000/v1"
        elif backend_kind == "mock":
            pass  # mock takes its own kwargs from `extra`
        # Best-effort construction — adapters that don't accept some keys
        # will tell us at instantiation.
        try:
            return ProviderCls(**kw)
        except TypeError:
            # Fall back to no-kwarg construction.
            return ProviderCls()  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, messages: list[Message]) -> list[Message]:
        """Run the full multi-turn loop and return the updated message list.

        ``messages`` is mutated in place — the same list grows with each
        assistant turn and tool result. The return value is the same object,
        for chaining convenience.
        """

        for _ in range(self.max_steps):
            assistant = await self._one_turn(messages)
            messages.append(assistant)
            tool_calls = [b for b in assistant.content if isinstance(b, ToolUseBlock)]
            if not tool_calls:
                return messages
            results = await dispatch_tool_calls(self.tools, tool_calls)
            messages.append(UserMessage(content=list(results)))
        _log.warning("agent hit max_steps=%d without natural stop", self.max_steps)
        return messages

    async def stream(self, messages: list[Message]) -> AsyncIterator[StreamEvent]:
        """Stream the next assistant turn as normalized events.

        Does *not* run the multi-turn loop — the caller is responsible for
        appending the resulting assistant message and (if it contains tool
        calls) calling ``stream`` again with appended tool results.
        """

        async for ev in self._provider_stream(messages):
            yield ev

    async def aclose(self) -> None:
        if self.provider is not None:
            await self.provider.aclose()

    # Async context manager sugar — `async with Agent(...) as a: ...`
    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _one_turn(self, messages: list[Message]) -> AssistantMessage:
        """Consume the stream and assemble one assistant message."""

        assembler = _AssistantAssembler()
        async for ev in self._provider_stream(messages):
            assembler.feed(ev)
        return assembler.finalize()

    def _provider_stream(self, messages: list[Message]) -> AsyncIterator[StreamEvent]:
        # Hoist the system prompt: prefer explicit Agent.system, else look at
        # messages[0] if it's a SystemMessage. Provider adapters expect system
        # as a top-level field (Anthropic, OpenAI, Ollama all do).
        system = self.system
        if system is None and messages and isinstance(messages[0], SystemMessage):
            sys_msg = messages[0]
            system = sys_msg.content if isinstance(sys_msg.content, str) else None

        assert self.provider is not None  # post-init guarantees this

        # Pass the resolved capability through so the provider can pick the
        # right tool-use path (A/B/C) without re-doing lookup.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "system": system,
            "tools": self.tools.to_wire() if self.tools else None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra": self.extra,
        }
        # Some legacy adapters don't accept model_capability; gate it.
        try:
            return self.provider.stream(model_capability=self.model_capability, **kwargs)
        except TypeError:
            return self.provider.stream(**kwargs)


# ---------------------------------------------------------------------------
# AssistantAssembler — turn event stream into AssistantMessage
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BlockBuilder:
    """In-progress content block, mutated as deltas arrive."""

    kind: str  # "text" | "thinking" | "tool_use" | other
    # For text and thinking: accumulated chunks (joined once at stop).
    text_parts: list[str] = field(default_factory=list)
    # For thinking only: signature carried from start.
    signature: str | None = None
    # For tool_use: name + id from start, JSON deltas accumulated.
    tool_id: str = ""
    tool_name: str = ""
    tool_initial_input: dict[str, Any] | None = None
    json_parts: list[str] = field(default_factory=list)
    # Original block payload (for unknown / passthrough types).
    raw_block: ContentBlock | None = None

    def to_block(self) -> ContentBlock:
        if self.kind == "text":
            return TextBlock(text="".join(self.text_parts))
        if self.kind == "thinking":
            return ThinkingBlock(
                thinking="".join(self.text_parts),
                signature=self.signature,
            )
        if self.kind == "tool_use":
            if self.json_parts:
                try:
                    input_obj = _JSON_DECODER.decode("".join(self.json_parts))
                except msgspec.DecodeError as e:
                    raise StreamProtocolError(
                        f"tool_use {self.tool_name!r} sent malformed input JSON"
                    ) from e
            else:
                input_obj = self.tool_initial_input or {}
            return ToolUseBlock(id=self.tool_id, name=self.tool_name, input=input_obj)
        # Unknown / passthrough — return whatever the start event gave us.
        if self.raw_block is None:
            raise StreamProtocolError(f"no block payload for kind {self.kind!r}")
        return self.raw_block


class _AssistantAssembler:
    """Folds a stream of events into a single ``AssistantMessage``.

    Holds builders by block index, plus message-level metadata.
    """

    __slots__ = ("blocks", "stop_reason", "usage", "_seen_start")

    def __init__(self) -> None:
        self.blocks: dict[int, _BlockBuilder] = {}
        self.stop_reason: str | None = None
        self.usage: Usage | None = None
        self._seen_start = False

    def feed(self, ev: StreamEvent) -> None:
        if isinstance(ev, MessageStart):
            self._seen_start = True
            return
        if isinstance(ev, ContentBlockStart):
            self._on_block_start(ev)
            return
        if isinstance(ev, ContentBlockDelta):
            self._on_delta(ev)
            return
        if isinstance(ev, ContentBlockStop):
            # Builder stays as-is; we materialize at finalize().
            return
        if isinstance(ev, MessageDelta):
            if ev.stop_reason is not None:
                self.stop_reason = ev.stop_reason
            if ev.usage is not None:
                self.usage = _merge_usage(self.usage, ev.usage)
            return
        if isinstance(ev, MessageStop):
            return
        if isinstance(ev, ErrorEvent):
            raise StreamProtocolError(f"provider error event: {ev.message}")

    def finalize(self) -> AssistantMessage:
        if not self._seen_start:
            raise StreamProtocolError("stream ended without message_start")
        # Sorted by index so block order matches what the provider emitted.
        ordered = [self.blocks[i].to_block() for i in sorted(self.blocks)]
        return AssistantMessage(
            content=ordered,
            stop_reason=self.stop_reason,
            usage=self.usage,
        )

    # --- per-event handlers --------------------------------------------------

    def _on_block_start(self, ev: ContentBlockStart) -> None:
        block = ev.block
        if isinstance(block, TextBlock):
            self.blocks[ev.index] = _BlockBuilder(kind="text", text_parts=[block.text])
            return
        if isinstance(block, ThinkingBlock):
            self.blocks[ev.index] = _BlockBuilder(
                kind="thinking",
                text_parts=[block.thinking],
                signature=block.signature,
            )
            return
        if isinstance(block, ToolUseBlock):
            self.blocks[ev.index] = _BlockBuilder(
                kind="tool_use",
                tool_id=block.id,
                tool_name=block.name,
                tool_initial_input=dict(block.input) if block.input else None,
            )
            return
        # Unknown / passthrough — keep the raw block so finalize can return it.
        self.blocks[ev.index] = _BlockBuilder(kind="passthrough", raw_block=block)

    def _on_delta(self, ev: ContentBlockDelta) -> None:
        b = self.blocks.get(ev.index)
        if b is None:
            raise StreamProtocolError(
                f"delta for index {ev.index} before content_block_start"
            )
        d = ev.delta
        if isinstance(d, TextDelta):
            b.text_parts.append(d.text)
            return
        if isinstance(d, ThinkingDelta):
            b.text_parts.append(d.thinking)
            return
        if isinstance(d, InputJsonDelta):
            b.json_parts.append(d.partial_json)
            return
        # Unknown delta: ignore for forward-compat.


def _merge_usage(prev: Usage | None, new: Usage) -> Usage:
    """Merge incremental usage updates. Output tokens accumulate; input tokens
    typically arrive once on message_start so we prefer the latest non-zero."""

    if prev is None:
        return new
    return Usage(
        input_tokens=new.input_tokens or prev.input_tokens,
        output_tokens=prev.output_tokens + new.output_tokens,
        cache_creation_input_tokens=new.cache_creation_input_tokens
        or prev.cache_creation_input_tokens,
        cache_read_input_tokens=new.cache_read_input_tokens or prev.cache_read_input_tokens,
    )
