"""Inline ``<think>...</think>`` splitter for reasoning-class OSS models.

DeepSeek-R1, QwQ, R1-Distill-*, Marco-o1 and friends emit chain-of-thought
between ``<think>`` and ``</think>`` tags interleaved with regular content.
This parser slices the text delta stream into two channels:

    ThinkingChunk(text)  — content between <think> and </think>
    TextChunk(text)      — everything else

State machine: ``OUTSIDE`` ↔ ``INSIDE``. Implementation mirrors
``text_tool_parser.py`` but with a single tag pair and no JSON parsing.

This parser is *capability-gated* upstream: it only runs when the model's
``ModelCapability.emits_inline_thinking`` is True, so its cost when the model
doesn't use ``<think>`` is exactly zero (we never instantiate it).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Union

import msgspec

__all__ = [
    "ThinkingParserEvent",
    "TextChunk",
    "ThinkingChunk",
    "ThinkingParser",
]

_OPEN = "<think>"
_CLOSE = "</think>"
# Holdback must cover the longer of the two tags' prefixes.
_HOLDBACK = max(len(_OPEN), len(_CLOSE)) - 1


class TextChunk(msgspec.Struct, frozen=True, tag="text", tag_field="type"):
    """A run of plain text emitted outside any ``<think>`` block."""

    text: str


class ThinkingChunk(msgspec.Struct, frozen=True, tag="thinking", tag_field="type"):
    """A run of model-internal reasoning emitted inside a ``<think>`` block."""

    text: str


ThinkingParserEvent = Union[TextChunk, ThinkingChunk]


class ThinkingParser:
    """Streaming splitter for inline ``<think>...</think>`` content.

    Usage::

        parser = ThinkingParser()
        for delta in stream:
            for event in parser.feed(delta):
                handle(event)
        for event in parser.finalize():
            handle(event)
    """

    __slots__ = ("_inside", "_buf")

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, text: str) -> Iterator[ThinkingParserEvent]:
        if not text:
            return
        self._buf += text
        yield from self._drain()

    def finalize(self) -> Iterator[ThinkingParserEvent]:
        """Flush trailing buffer. Unterminated ``<think>`` is emitted as
        ThinkingChunk — preserving content beats dropping it."""
        if self._buf:
            if self._inside:
                yield ThinkingChunk(text=self._buf)
            else:
                yield TextChunk(text=self._buf)
            self._buf = ""

    def _drain(self) -> Iterator[ThinkingParserEvent]:
        while True:
            target = _CLOSE if self._inside else _OPEN
            idx = self._buf.find(target)
            if idx == -1:
                # No full target tag — flush everything except potential
                # partial-tag tail. When inside <think>, the only tag we look
                # for is </think>; when outside, it's <think>.
                safe = _safe_flush_len(self._buf, target, _HOLDBACK)
                if safe > 0:
                    chunk = self._buf[:safe]
                    self._buf = self._buf[safe:]
                    if self._inside:
                        yield ThinkingChunk(text=chunk)
                    else:
                        yield TextChunk(text=chunk)
                return

            # Emit content before the tag, drop the tag, flip state.
            if idx > 0:
                head = self._buf[:idx]
                if self._inside:
                    yield ThinkingChunk(text=head)
                else:
                    yield TextChunk(text=head)
            self._buf = self._buf[idx + len(target):]
            self._inside = not self._inside
            # Loop — there may be another tag in the remainder.


# Re-use the helper shape from text_tool_parser (kept local to avoid a
# circular import).
def _safe_flush_len(buf: str, tag: str, holdback: int) -> int:
    n = len(buf)
    max_prefix = min(n, len(tag) - 1, holdback)
    for k in range(max_prefix, 0, -1):
        if buf.endswith(tag[:k]):
            return n - k
    return n


