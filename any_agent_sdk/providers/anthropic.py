"""Anthropic adapter.

Talks directly to the Messages API over HTTPS + SSE. No anthropic-python
dependency — we own the wire format so we can stream zero-copy and skip a
layer of object construction.

Reference: https://docs.anthropic.com/en/api/messages-streaming

Wire events we accept:
  message_start                  -> MessageStart
  content_block_start            -> ContentBlockStart
  content_block_delta            -> ContentBlockDelta (text_delta | input_json_delta | thinking_delta)
  content_block_stop             -> ContentBlockStop
  message_delta                  -> MessageDelta (stop_reason + usage)
  message_stop                   -> MessageStop
  ping                           -> ignored
  error                          -> ErrorEvent + raise
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

import msgspec

from ..errors import ProviderError, StreamProtocolError
from ..events import (
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
from ..http import iter_sse, make_client, raise_for_status
from ..types import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserMessage,
    Usage,
)
from .base import _HTTPProviderMixin

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"

# msgspec encoder for content blocks (used when serializing messages outbound).
_BLOCK_ENCODER = msgspec.json.Encoder()


class AnthropicProvider(_HTTPProviderMixin):
    """Anthropic Messages API adapter."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        default_headers: dict[str, str] | None = None,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY not set")
        headers = {
            "x-api-key": key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if default_headers:
            headers.update(default_headers)
        self.client = make_client(base_url=base_url, headers=headers)

    # ------------------------------------------------------------------
    # Message serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_messages(
        messages: Iterable[Message],
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Split out system + serialize messages to Anthropic's wire shape.

        Anthropic puts ``system`` as a top-level field (not in ``messages``).
        We hand back ``(system, [messages])`` to the caller.
        """

        system: str | list[dict[str, Any]] | None = None
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                # Last system wins; Anthropic accepts string or block list.
                if isinstance(m.content, str):
                    system = m.content
                else:
                    system = [msgspec.to_builtins(b) for b in m.content]
                continue
            if isinstance(m, (UserMessage, AssistantMessage)):
                out.append(
                    {
                        "role": m.role,
                        "content": (
                            m.content
                            if isinstance(m.content, str)
                            else [msgspec.to_builtins(b) for b in m.content]
                        ),
                    }
                )
                continue
            raise TypeError(f"unsupported message: {type(m).__name__}")
        return system, out

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

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
        sys_field, msgs = self._encode_messages(messages)
        if system is not None and sys_field is None:
            sys_field = system

        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if sys_field is not None:
            payload["system"] = sys_field
        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if extra:
            payload.update(extra)

        async with self.client.stream(
            "POST",
            "/v1/messages",
            content=_BLOCK_ENCODER.encode(payload),
        ) as response:
            if response.status_code >= 400:
                # Drain body for a typed error.
                await response.aread()
                raise_for_status(response)

            async for event_name, data in iter_sse(response):
                ev = _translate_event(event_name, data)
                if ev is None:
                    continue
                yield ev
                if isinstance(ev, ErrorEvent):
                    raise ProviderError(ev.message, raw=ev.raw)


# ---------------------------------------------------------------------------
# Event translation
# ---------------------------------------------------------------------------


def _translate_event(name: str, data: dict[str, Any]) -> StreamEvent | None:
    """Map an Anthropic SSE event to a normalized ``StreamEvent``.

    Returns ``None`` for events we deliberately drop (``ping``).
    """

    t = name or data.get("type")

    if t == "ping":
        return None

    if t == "message_start":
        m = data["message"]
        return MessageStart(
            message_id=m["id"],
            model=m.get("model", ""),
            role=m.get("role", "assistant"),
        )

    if t == "content_block_start":
        return ContentBlockStart(
            index=data["index"],
            block=_decode_block(data["content_block"]),
        )

    if t == "content_block_delta":
        delta = data["delta"]
        dt = delta.get("type")
        if dt == "text_delta":
            return ContentBlockDelta(index=data["index"], delta=TextDelta(text=delta["text"]))
        if dt == "thinking_delta":
            return ContentBlockDelta(
                index=data["index"], delta=ThinkingDelta(thinking=delta["thinking"])
            )
        if dt == "input_json_delta":
            return ContentBlockDelta(
                index=data["index"],
                delta=InputJsonDelta(partial_json=delta["partial_json"]),
            )
        # Unknown delta type — treat as protocol noise rather than crashing.
        return None

    if t == "content_block_stop":
        return ContentBlockStop(index=data["index"])

    if t == "message_delta":
        delta = data.get("delta", {})
        usage_raw = data.get("usage")
        usage = _decode_usage(usage_raw) if usage_raw else None
        return MessageDelta(
            stop_reason=delta.get("stop_reason"),
            stop_sequence=delta.get("stop_sequence"),
            usage=usage,
        )

    if t == "message_stop":
        return MessageStop()

    if t == "error":
        err = data.get("error", {})
        return ErrorEvent(
            error_type=err.get("type", "unknown"),
            message=err.get("message", "unknown error"),
            raw=data,
        )

    # Forward-compatible: unknown event types are dropped, not raised.
    return None


def _decode_block(raw: dict[str, Any]) -> ContentBlock:
    """Decode a content_block_start payload to our typed block.

    We branch on ``type`` so we can construct the right msgspec struct without
    a generic decode (which would have to enumerate every variant).
    """

    bt = raw.get("type")
    if bt == "text":
        return TextBlock(text=raw.get("text", ""))
    if bt == "thinking":
        return ThinkingBlock(thinking=raw.get("thinking", ""), signature=raw.get("signature"))
    if bt == "tool_use":
        return ToolUseBlock(
            id=raw["id"], name=raw["name"], input=raw.get("input", {})
        )
    if bt == "image":
        return ImageBlock(source=raw["source"])
    raise StreamProtocolError(f"unknown content block type {bt!r}")


def _decode_usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
        cache_creation_input_tokens=raw.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=raw.get("cache_read_input_tokens", 0),
    )
