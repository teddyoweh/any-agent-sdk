"""Universal message + content-block types.

Every provider adapter converts its native shapes to/from these. Built on
``msgspec.Struct`` — typed, immutable-by-default, ~5–10× faster to encode/decode
than Pydantic v2 and uses ~3× less memory.

Design notes
------------
* Content blocks are a *tagged union* via ``msgspec.Struct, tag_field="type"``.
  This means msgspec dispatches on a single string compare at decode time —
  no reflection, no isinstance ladders.
* ``frozen=True`` for blocks; that lets us hash, share across tasks, and
  prevents accidental mutation in the streaming hot path.
* ``Message`` is *not* frozen — assistant messages grow during streaming.
  Mutation is confined to streaming assembly; consumers should treat finalized
  messages as immutable.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

import msgspec

# ---------------------------------------------------------------------------
# Content blocks (tagged union)
# ---------------------------------------------------------------------------


class TextBlock(
    msgspec.Struct,
    frozen=True,
    tag="text",
    tag_field="type",
    omit_defaults=True,
):
    """Plain text. Streamed as a sequence of TextDelta events."""

    text: str
    # Anthropic-style cache control. Other providers ignore this field.
    cache_control: dict[str, str] | None = None


class ThinkingBlock(
    msgspec.Struct,
    frozen=True,
    tag="thinking",
    tag_field="type",
    omit_defaults=True,
):
    """Model-internal reasoning. Only emitted by providers that support it."""

    thinking: str
    signature: str | None = None


class ToolUseBlock(
    msgspec.Struct,
    frozen=True,
    tag="tool_use",
    tag_field="type",
    omit_defaults=True,
):
    """Assistant requests a tool call."""

    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(
    msgspec.Struct,
    frozen=True,
    tag="tool_result",
    tag_field="type",
    omit_defaults=True,
):
    """Result of a tool call, sent back as part of a user message."""

    tool_use_id: str
    content: str | list["ContentBlock"]
    is_error: bool = False


class ImageBlock(
    msgspec.Struct,
    frozen=True,
    tag="image",
    tag_field="type",
    omit_defaults=True,
):
    """Image content. ``source`` follows the Anthropic shape; adapters
    convert to OpenAI/Gemini equivalents."""

    source: dict[str, Any]


# Tagged union over all block kinds. msgspec dispatches on the ``type`` field.
ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock],
    msgspec.Meta(description="A single content block in a message."),
]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class SystemMessage(msgspec.Struct, omit_defaults=True):
    """System prompt. Some providers want this in a dedicated field; the
    adapter is responsible for placement. We model it uniformly here."""

    content: str | list[TextBlock]
    role: Literal["system"] = "system"


class UserMessage(msgspec.Struct, omit_defaults=True):
    """User-authored or tool-result-bearing message."""

    content: str | list[ContentBlock]
    role: Literal["user"] = "user"


class AssistantMessage(msgspec.Struct, omit_defaults=True):
    """Assistant turn. Content is a list of blocks; mutated during streaming."""

    content: list[ContentBlock]
    role: Literal["assistant"] = "assistant"
    # Populated when the message finalizes.
    stop_reason: str | None = None
    usage: Usage | None = None


Message = Union[SystemMessage, UserMessage, AssistantMessage]


# ---------------------------------------------------------------------------
# Usage / metadata
# ---------------------------------------------------------------------------


class Usage(msgspec.Struct, frozen=True, omit_defaults=True):
    """Token counts. Fields are optional because not every provider reports
    every metric. ``cache_*`` are Anthropic-specific but kept here for clarity."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# ---------------------------------------------------------------------------
# Encoder / decoder singletons
# ---------------------------------------------------------------------------
# Reusing one encoder/decoder per type beats constructing one per call by ~30%
# for our message sizes. msgspec's encoders are thread-safe.

ENCODE_MESSAGE = msgspec.json.Encoder()
DECODE_ASSISTANT_MESSAGE = msgspec.json.Decoder(AssistantMessage)


def to_json(obj: Any) -> bytes:
    """Fast JSON encode using the shared encoder."""

    return ENCODE_MESSAGE.encode(obj)
