"""Agent — the run loop.

This is the only place that knows about *both* providers and tools. It owns
the multi-turn dance:

    1. Send messages → provider.stream
    2. Assemble assistant message from event stream
    3. If assistant requested tools → dispatch → append results → loop
    4. Else → return final assistant message (or raise on hard stop)

The streaming variant (``Agent.stream``) yields the *normalized* event
stream so user UIs can render token-by-token. The non-streaming
``Agent.run`` consumes the stream internally and returns the final messages.

Performance notes
-----------------
* Text deltas are *not* concatenated until the block stops. We hold them in
  a list and ``"".join`` once at ContentBlockStop — that's O(n) once instead
  of O(n²) per delta.
* Tool input JSON deltas are likewise buffered as strings and parsed once
  at block stop.
* We never deep-copy messages; the conversation list is appended-to in place.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import msgspec

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
    """Multi-turn agent over any registered provider.

    Construct with a model name; the provider is auto-detected. Pass an
    explicit ``provider=`` instance for full control.
    """

    model: str
    provider: Provider | None = None
    system: str | None = None
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    max_tokens: int = 1024
    temperature: float | None = None
    max_steps: int = 20
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = resolve(detect_provider(self.model))()

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
        # Hoist out the system message if it's the first element, since
        # providers want it as a top-level field. (UserMessage with a
        # system role is unusual; we just pass everything through and let
        # the adapter sort it.)
        system = self.system
        if system is None and messages and isinstance(messages[0], SystemMessage):
            sys_msg = messages[0]
            system = sys_msg.content if isinstance(sys_msg.content, str) else None

        assert self.provider is not None  # post-init guarantees this
        return self.provider.stream(
            model=self.model,
            messages=messages,
            system=system,
            tools=self.tools.to_wire() if self.tools else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra=self.extra,
        )


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
                # Merge — Anthropic emits partial usage updates.
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
    """Anthropic emits usage incrementally in message_delta. We add fields
    that grow (output_tokens) and prefer the latest non-zero for the input
    counters (which are typically set once on message_start)."""

    if prev is None:
        return new
    return Usage(
        input_tokens=new.input_tokens or prev.input_tokens,
        output_tokens=prev.output_tokens + new.output_tokens,
        cache_creation_input_tokens=new.cache_creation_input_tokens
        or prev.cache_creation_input_tokens,
        cache_read_input_tokens=new.cache_read_input_tokens or prev.cache_read_input_tokens,
    )
