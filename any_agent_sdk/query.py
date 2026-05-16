"""``query()`` — the drop-in wrapper.

A thin generator over :class:`Agent` whose signature mirrors the upstream
Claude Agent SDK so callers porting code can swap imports and not much else::

    from any_agent_sdk import query

    async for msg in query(
        prompt="Hello!",
        options={"model": "qwen2.5-7b-instruct", "backend": "http://localhost:11434"},
    ):
        print(msg)

Design notes
------------
* The options dict accepts both ``snake_case`` (idiomatic Python) and
  ``camelCase`` (upstream parity) keys. We normalize via :func:`_to_snake`.
* The yielded values are :class:`SDKMessage` msgspec structs — a tagged union
  of assistant / user / system / result messages. They are *not* the same as
  the internal :mod:`any_agent_sdk.types` messages; the wrapper translates so
  consumers can switch between SDKs without diff churn.
* The wrapper supports two prompt shapes: a plain ``str`` (single user turn),
  or an ``AsyncIterable[SDKUserMessage]`` for streamed multi-turn input.
* We finalize with an :class:`SDKResultMessage` carrying total cost (when
  pricing is known) and number of turns — same fields upstream emits.

What we deliberately do *not* implement here
--------------------------------------------
* No partial-message / delta streaming. ``query`` yields whole messages, same
  as upstream. Token-level streaming lives on ``Agent.stream``.
* No interactive ``can_use_tool`` callback over the wire. ``options`` flows
  through to ``Agent`` and ``Permissions`` directly when those modules land.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, Literal, Union

import msgspec

from .agent import Agent
from .providers.base import Provider, detect_provider, resolve
from .tools import Tool, ToolRegistry
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    SystemMessage,
    TextBlock,
    UserMessage,
)

# ---------------------------------------------------------------------------
# SDK message shapes (the wrapper's own surface — does not leak internal types)
# ---------------------------------------------------------------------------


class SDKAssistantMessage(
    msgspec.Struct,
    tag="assistant",
    tag_field="type",
    omit_defaults=True,
):
    """Final assistant turn. ``content_blocks`` mirrors the upstream shape."""

    content_blocks: list[ContentBlock]
    model: str = ""
    stop_reason: str | None = None


class SDKUserMessage(
    msgspec.Struct,
    tag="user",
    tag_field="type",
    omit_defaults=True,
):
    """User turn — either plain text or a list of content blocks (e.g. for
    multi-modal or tool-result-bearing turns)."""

    content: str | list[ContentBlock]


class SDKSystemMessage(
    msgspec.Struct,
    tag="system",
    tag_field="type",
    omit_defaults=True,
):
    """System prompt echoed back at the top of a streamed conversation. We
    emit it once at the start when ``system`` is set."""

    content: str


class SDKResultMessage(
    msgspec.Struct,
    tag="result",
    tag_field="type",
    omit_defaults=True,
):
    """Terminal message — fired after the last assistant turn. Matches the
    upstream ``SDKResultMessage`` field-for-field."""

    result: str
    is_error: bool = False
    total_cost_usd: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0
    session_id: str = ""


SDKMessage = Union[
    SDKAssistantMessage,
    SDKUserMessage,
    SDKSystemMessage,
    SDKResultMessage,
]


# ---------------------------------------------------------------------------
# Options normalization
# ---------------------------------------------------------------------------


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _to_snake(s: str) -> str:
    """``maxTurns`` → ``max_turns``. Idempotent on already-snake input."""

    return _CAMEL_RE.sub("_", s).lower()


def _normalize_options(opts: dict[str, Any] | None) -> dict[str, Any]:
    """Apply :func:`_to_snake` to every top-level key so the rest of the
    wrapper only ever sees snake_case names. Inner dicts are left alone."""

    if not opts:
        return {}
    return {_to_snake(k): v for k, v in opts.items()}


# Keys we explicitly recognize on options. Anything else is forwarded to the
# Agent's ``extra`` bag so users can pass adapter-specific knobs without us
# growing a constructor parameter every quarter.
_AGENT_KEYS = frozenset(
    {
        "model",
        "backend",
        "system",
        "tools",
        "max_turns",
        "max_tokens",
        "temperature",
        "max_usd",
        "permission_mode",
        "allowed_tools",
        "disallowed_tools",
        "hooks",
        "mcp_servers",
        "fallback_model",
        "session_id",
        "api_key",
    }
)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _build_registry(tools: Any) -> ToolRegistry:
    """Accept a list / ToolRegistry / None and always return a registry."""

    if tools is None:
        return ToolRegistry()
    if isinstance(tools, ToolRegistry):
        return tools
    reg = ToolRegistry()
    for t in tools:
        if isinstance(t, Tool):
            reg.add(t)
        else:
            raise TypeError(
                f"query(options.tools): expected Tool or ToolRegistry, got {type(t).__name__}"
            )
    return reg


def _build_provider(
    *,
    model: str,
    backend: str | None,
    api_key: str | None,
) -> Provider:
    """Instantiate the right provider for ``(model, backend)`` pair.

    Resolution mirrors :func:`providers.base.detect_provider`:
      * explicit URL → matched against URL heuristics
      * bare model name → defaults to ``openai_compat`` (local vLLM-style)
    """

    name = detect_provider(backend or model)
    factory = resolve(name)
    kwargs: dict[str, Any] = {}
    if backend and backend.startswith(("http://", "https://")):
        kwargs["base_url"] = backend
    if api_key is not None:
        kwargs["api_key"] = api_key
    try:
        return factory(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        # Provider didn't accept one of our kwargs (e.g. mock). Fall back to no-arg.
        return factory()  # type: ignore[call-arg]


def _agent_from_options(opts: dict[str, Any]) -> tuple[Agent, dict[str, Any]]:
    """Construct an :class:`Agent` from a normalized options dict.

    Returns the agent plus the leftover keys we couldn't place — those go on
    ``agent.extra`` so sibling features (hooks, permissions, budget) can read
    them once they're built.
    """

    model = opts.get("model")
    if not model:
        raise ValueError("query(options): 'model' is required")

    backend = opts.get("backend")
    api_key = opts.get("api_key")

    provider = _build_provider(model=model, backend=backend, api_key=api_key)
    registry = _build_registry(opts.get("tools"))

    # Map known options onto the Agent's constructor.
    max_turns = opts.get("max_turns", opts.get("max_steps", 20))
    max_tokens = opts.get("max_tokens", 1024)
    temperature = opts.get("temperature")
    system = opts.get("system")

    # Stash anything the Agent doesn't natively understand. Hooks, permissions,
    # mcp_servers, budget knobs etc. live here until those modules wire up.
    extra: dict[str, Any] = {}
    for k, v in opts.items():
        if k in {"model", "backend", "api_key", "tools", "max_turns", "max_steps",
                 "max_tokens", "temperature", "system"}:
            continue
        extra[k] = v

    agent = Agent(
        model=model,
        provider=provider,
        system=system,
        tools=registry,
        max_tokens=max_tokens,
        temperature=temperature,
        max_steps=max_turns,
        extra=extra or None,
    )
    return agent, extra


# ---------------------------------------------------------------------------
# Prompt → seed messages
# ---------------------------------------------------------------------------


async def _collect_prompt(
    prompt: str | AsyncIterable[Any],
) -> list[Message]:
    """Turn the public ``prompt`` argument into a list of internal messages.

    * ``str`` → one ``UserMessage``.
    * ``AsyncIterable[SDKUserMessage | UserMessage | dict]`` → iterated and
      each item converted to a ``UserMessage``. Dicts are interpreted leniently
      so callers can pass plain JSON without a wrapper type.
    """

    if isinstance(prompt, str):
        return [UserMessage(content=prompt)]

    messages: list[Message] = []
    async for item in prompt:
        messages.append(_to_internal_user(item))
    if not messages:
        raise ValueError("query(prompt): async iterable yielded no messages")
    return messages


def _to_internal_user(item: Any) -> UserMessage:
    if isinstance(item, UserMessage):
        return item
    if isinstance(item, SDKUserMessage):
        return UserMessage(content=item.content)
    if isinstance(item, dict):
        # Accept upstream-shape dicts {"type": "user", "content": ...}.
        if "content" in item:
            return UserMessage(content=item["content"])
    if isinstance(item, str):
        return UserMessage(content=item)
    raise TypeError(f"query(prompt): unsupported item type {type(item).__name__}")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def query(
    *,
    prompt: str | AsyncIterable[Any],
    options: dict[str, Any] | None = None,
) -> AsyncIterator[SDKMessage]:
    """Run an agent and yield SDK-shaped messages.

    Yields, in order:
      1. ``SDKSystemMessage`` if ``options['system']`` was set.
      2. ``SDKUserMessage`` for each seed user message.
      3. ``SDKAssistantMessage`` for each assistant turn produced.
      4. ``SDKResultMessage`` once at the end.

    The agent's HTTP client is closed before the final ``SDKResultMessage``
    is yielded — callers don't need to manage lifecycle themselves.
    """

    opts = _normalize_options(options)
    agent, _ = _agent_from_options(opts)

    seeds = await _collect_prompt(prompt)

    # Emit system + seed user messages up front so consumers can render the
    # whole conversation from a single stream.
    if agent.system:
        yield SDKSystemMessage(content=agent.system)
    for seed in seeds:
        if isinstance(seed, UserMessage):
            yield SDKUserMessage(content=seed.content)

    final_text = ""
    is_error = False
    num_turns = 0
    try:
        # Run the agent. ``run`` mutates ``seeds`` in place.
        messages = await agent.run(list(seeds))
        # Skip back over the seeds; yield the assistant turns + any user
        # tool-result turns the loop appended.
        for msg in messages[len(seeds):]:
            if isinstance(msg, AssistantMessage):
                num_turns += 1
                yield SDKAssistantMessage(
                    content_blocks=list(msg.content),
                    model=agent.model,
                    stop_reason=msg.stop_reason,
                )
                # Track the final visible text for the result message.
                final_text = _extract_text(msg.content) or final_text
            elif isinstance(msg, UserMessage):
                # Tool-result-bearing user message produced by the loop.
                yield SDKUserMessage(content=msg.content)
            elif isinstance(msg, SystemMessage):
                # Should not normally appear mid-stream, but pass through.
                content = msg.content if isinstance(msg.content, str) else ""
                yield SDKSystemMessage(content=content)
    except Exception as e:  # noqa: BLE001
        is_error = True
        final_text = f"{type(e).__name__}: {e}"
    finally:
        await agent.aclose()

    yield SDKResultMessage(
        result=final_text,
        is_error=is_error,
        total_cost_usd=0.0,  # populated by budget.py once it lands
        num_turns=num_turns,
    )


def _extract_text(blocks: list[ContentBlock]) -> str:
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            parts.append(b.text)
    return "".join(parts)


__all__ = [
    "SDKAssistantMessage",
    "SDKMessage",
    "SDKResultMessage",
    "SDKSystemMessage",
    "SDKUserMessage",
    "query",
]
