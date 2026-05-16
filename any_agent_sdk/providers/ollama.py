"""Ollama native adapter.

Talks to ``POST /api/chat`` directly. Ollama emits **newline-delimited JSON**
(NDJSON), not SSE — each line is one complete JSON object describing the
next state of the streaming message. We cannot reuse :func:`iter_sse` here.

Wire shape (per line)::

    {
      "model": "qwen2.5",
      "created_at": "2026-...",
      "message": {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": [
          {"function": {"name": "search", "arguments": {"q": "..."}}}
        ]?
      },
      "done": false,
      "done_reason": "stop"?,        // only on the final chunk
      "total_duration": 12345?,
      "prompt_eval_count": 42?,
      "eval_count": 17?
    }

Notes vs OpenAI shape:
* ``tool_calls[].function.arguments`` is a **dict**, not a JSON-encoded string.
* Tool calls usually arrive in a single chunk (Ollama doesn't tokenize them).
* No incremental tool-call streaming — we synthesize Start + InputJsonDelta +
  Stop in one shot the moment we see them.

For models without native tool calling, we inject the Hermes-Pro system
prompt and route content deltas through :class:`ToolCallTextParser`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable
from typing import Any
from uuid import uuid4

import msgspec

from ..capabilities import HOSTED_PROFILES, BackendCapability, ModelCapability
from ..errors import ProviderError, StreamProtocolError
from ..events import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
)
from ..http import make_client, raise_for_status
from ..types import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    Usage,
)
from .base import HTTPProviderMixin

DEFAULT_BASE_URL = "http://localhost:11434"

# NOTE: we render with simple ``%`` substitution rather than ``.format()``
# because the literal example below contains JSON braces — ``str.format``
# would treat them as positional placeholders and KeyError on `'"name"'`.
_HERMES_PROMPT = (
    "You have access to the following tools. To call a tool, emit a single\n"
    "<tool_call> block in your response. You can call multiple tools in one\n"
    "response by emitting multiple <tool_call> blocks back-to-back.\n\n"
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {<JSON object>}}\n'
    "</tool_call>\n\n"
    "Available tools:\n%(tools_json)s\n\n"
    "When you receive <tool_result> messages, continue your response using\n"
    "the new information. If you have completed the user's request, respond\n"
    "without any <tool_call> blocks."
)

_JSON_ENCODER = msgspec.json.Encoder()
_JSON_DECODER = msgspec.json.Decoder()


class OllamaProvider(HTTPProviderMixin):
    """Adapter for Ollama's native ``/api/chat`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        base = base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_BASE_URL
        headers = {"content-type": "application/json", "accept": "application/x-ndjson"}
        # Local Ollama doesn't require auth but remote/proxied deploys may.
        key = api_key or os.environ.get("OLLAMA_API_KEY")
        if key:
            headers["authorization"] = f"Bearer {key}"
        if default_headers:
            headers.update(default_headers)
        self.client = make_client(base_url=base, headers=headers)
        self.backend_capability: BackendCapability = HOSTED_PROFILES["ollama"]

    # ------------------------------------------------------------------
    # Message serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_messages(
        messages: Iterable[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Flatten our typed messages into Ollama's ``messages`` list.

        Ollama accepts a list of ``{role, content, tool_calls?, tool_call_id?}``.
        Content blocks collapse to a single string by concatenating text blocks
        and rendering tool_results as ``<tool_result>...</tool_result>``.
        """

        out: list[dict[str, Any]] = []
        if system is not None:
            out.append({"role": "system", "content": system})

        for m in messages:
            if isinstance(m, SystemMessage):
                content = m.content if isinstance(m.content, str) else _join_text(m.content)
                out.append({"role": "system", "content": content})
                continue
            if isinstance(m, UserMessage):
                if isinstance(m.content, str):
                    out.append({"role": "user", "content": m.content})
                    continue
                # Multi-block user message — may include tool_result blocks.
                text_buf: list[str] = []
                for blk in m.content:
                    if isinstance(blk, TextBlock):
                        text_buf.append(blk.text)
                    else:
                        # tool_result, image, etc. — collapse to a tagged string.
                        text_buf.append(_render_block_as_text(blk))
                out.append({"role": "user", "content": "\n".join(text_buf)})
                continue
            if isinstance(m, AssistantMessage):
                texts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for blk in m.content:
                    if isinstance(blk, TextBlock):
                        texts.append(blk.text)
                    elif isinstance(blk, ToolUseBlock):
                        tool_calls.append(
                            {
                                "function": {
                                    "name": blk.name,
                                    "arguments": blk.input,
                                }
                            }
                        )
                msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
                continue
            raise TypeError(f"unsupported message: {type(m).__name__}")
        return out

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
        model_capability: ModelCapability | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # Decide tool-use path. Native if model + backend both support it.
        use_native_tools = bool(
            tools
            and (model_capability is None or model_capability.supports_native_tools)
            and self.backend_capability.supports_native_tools
        )

        # If tools are present but native path isn't usable, fall back to
        # Hermes-Pro prompt injection. The agent loop then parses content
        # deltas via ToolCallTextParser — we just pipe text through.
        effective_system = system
        if tools and not use_native_tools:
            tools_json = _JSON_ENCODER.encode(tools).decode("utf-8")
            inject = _HERMES_PROMPT % {"tools_json": tools_json}
            effective_system = inject if system is None else f"{system}\n\n{inject}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._encode_messages(messages, effective_system),
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if use_native_tools:
            payload["tools"] = tools
        if extra:
            # Merge ``extra.options`` into payload.options instead of clobbering.
            opts = extra.pop("options", None) if isinstance(extra, dict) else None
            if isinstance(opts, dict):
                payload["options"].update(opts)
            payload.update(extra)

        async for ev in self._stream_ndjson(payload, model=model):
            yield ev

    async def _stream_ndjson(
        self, payload: dict[str, Any], *, model: str
    ) -> AsyncIterator[StreamEvent]:
        message_id = f"ollama-{uuid4().hex[:12]}"
        started = False
        # We use a single virtual text content block (index 0). Each tool call
        # gets its own block at a monotonically increasing index.
        text_open = False
        next_index = 0
        text_index: int | None = None

        async with self.client.stream(
            "POST", "/api/chat", content=_JSON_ENCODER.encode(payload)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response)

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = _JSON_DECODER.decode(line)
                except msgspec.DecodeError as e:
                    raise StreamProtocolError(
                        f"bad NDJSON line from Ollama: {line[:200]}"
                    ) from e
                if not isinstance(chunk, dict):
                    continue

                if not started:
                    started = True
                    yield MessageStart(
                        message_id=message_id,
                        model=chunk.get("model", model),
                    )

                msg = chunk.get("message") or {}
                content = msg.get("content") or ""

                # Text delta — open text block lazily on first non-empty token.
                if content:
                    if not text_open:
                        text_index = next_index
                        next_index += 1
                        yield ContentBlockStart(
                            index=text_index,
                            block=TextBlock(text=""),
                        )
                        text_open = True
                    assert text_index is not None
                    yield ContentBlockDelta(
                        index=text_index,
                        delta=TextDelta(text=content),
                    )

                # Thinking — some Ollama builds expose a ``thinking`` field on
                # the message for R1-class models served in reasoning mode.
                thinking = msg.get("thinking")
                if thinking:
                    # Use a dedicated index for thinking; we don't track it as
                    # an "open" block because Ollama doesn't span it across
                    # chunks — each thinking field is self-contained.
                    yield ContentBlockDelta(
                        index=text_index if text_open else 0,
                        delta=ThinkingDelta(thinking=thinking),
                    )

                # Tool calls — Ollama emits them as a complete list in one
                # chunk (usually the final one). Synthesize Start + Delta + Stop.
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    fn = (tc or {}).get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    # Ollama gives dict args; OpenAI gives string. Tolerate both.
                    if isinstance(args, str):
                        try:
                            args_dict = _JSON_DECODER.decode(args)
                            if not isinstance(args_dict, dict):
                                args_dict = {}
                        except msgspec.DecodeError:
                            args_dict = {}
                    elif isinstance(args, dict):
                        args_dict = args
                    else:
                        args_dict = {}
                    if not isinstance(name, str) or not name:
                        continue
                    tool_id = tc.get("id") or f"call_{uuid4().hex[:12]}"
                    tool_index = next_index
                    next_index += 1
                    yield ContentBlockStart(
                        index=tool_index,
                        block=ToolUseBlock(id=tool_id, name=name, input=args_dict),
                    )
                    yield ContentBlockDelta(
                        index=tool_index,
                        delta=InputJsonDelta(
                            partial_json=_JSON_ENCODER.encode(args_dict).decode("utf-8")
                        ),
                    )
                    yield ContentBlockStop(index=tool_index)

                # Final chunk.
                if chunk.get("done"):
                    if text_open and text_index is not None:
                        yield ContentBlockStop(index=text_index)
                        text_open = False
                    usage = _decode_usage(chunk)
                    stop_reason = _map_stop_reason(
                        chunk.get("done_reason"),
                        has_tool_calls=bool(tool_calls),
                    )
                    yield MessageDelta(stop_reason=stop_reason, usage=usage)
                    yield MessageStop()
                    return

            # Stream ended without a final ``done:true`` chunk.
            if not started:
                raise ProviderError("Ollama stream ended with no chunks")
            if text_open and text_index is not None:
                yield ContentBlockStop(index=text_index)
            yield MessageDelta(stop_reason="end_turn", usage=None)
            yield MessageStop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _join_text(blocks: list[Any]) -> str:
    return "\n".join(b.text for b in blocks if isinstance(b, TextBlock))


def _render_block_as_text(blk: Any) -> str:
    """Collapse a non-text block (tool_result, image) to a tagged string so
    Ollama's plain-text content slot can carry it for prompt-engineered paths."""
    if hasattr(blk, "tool_use_id"):  # ToolResultBlock
        body = blk.content if isinstance(blk.content, str) else _join_text(blk.content)
        err = ' is_error="true"' if getattr(blk, "is_error", False) else ""
        return f'<tool_result tool_call_id="{blk.tool_use_id}"{err}>{body}</tool_result>'
    if hasattr(blk, "source"):  # ImageBlock
        return "[image]"
    return ""


def _decode_usage(chunk: dict[str, Any]) -> Usage | None:
    prompt = chunk.get("prompt_eval_count")
    eval_count = chunk.get("eval_count")
    if prompt is None and eval_count is None:
        return None
    return Usage(
        input_tokens=int(prompt or 0),
        output_tokens=int(eval_count or 0),
    )


def _map_stop_reason(done_reason: str | None, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_use"
    if not done_reason:
        return "end_turn"
    # Ollama uses "stop", "length", "load" — normalize to upstream vocab.
    if done_reason == "length":
        return "max_tokens"
    if done_reason == "stop":
        return "end_turn"
    return done_reason
