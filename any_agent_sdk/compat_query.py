"""Claude Python SDK-shaped ``query()``.

The original ``any_agent_sdk.query.query`` yields TS-SDK-shape messages
(``SDKAssistantMessage`` with nested ``.message.content``). The Python
SDK ships *flat-shape* messages instead — ``AssistantMessage`` directly
has ``.content``, ``ResultMessage`` directly has ``.total_cost_usd``.

This module exposes a ``query()`` that yields those flat-shape messages,
matching the canonical examples at
https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples
verbatim.

Top-level ``any_agent_sdk.query`` is wired to this when called with a
``ClaudeAgentOptions`` (or no options) so the Claude SDK examples just
work. When called with a dict it falls through to the existing dict-
options behavior.
"""

from __future__ import annotations

import os
import time
import uuid as _uuid
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from .agent import Agent
from .budget import lookup_pricing
from .capabilities import lookup_model
from .claude_compat import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    SystemMessage,
    UserMessage,
)
from .errors import BudgetExceededError
from .tools import Tool, ToolRegistry
from .types import (
    AssistantMessage as _InternalAssistantMessage,
    SystemMessage as _InternalSystemMessage,
    UserMessage as _InternalUserMessage,
    Usage,
)


__all__ = ["query"]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def query(
    *,
    prompt: str | AsyncIterable[Any],
    options: ClaudeAgentOptions | dict[str, Any] | None = None,
) -> AsyncIterator[Message]:
    """Drop-in replacement for ``claude_agent_sdk.query``.

    Accepts either a :class:`ClaudeAgentOptions` instance or a plain
    dict. Yields flat-shape :class:`AssistantMessage`, :class:`UserMessage`,
    :class:`SystemMessage`, and :class:`ResultMessage` matching the
    Claude Python SDK examples 1:1.

    With no ``options``, defaults: model from ``$ANY_AGENT_MODEL`` else
    ``"qwen2.5-7b-instruct"``, backend from ``$ANY_AGENT_BASE_URL`` else
    ``http://localhost:11434``. So ``async for msg in query(prompt="hi"):``
    runs against local Ollama out of the box.
    """

    opts = _normalize_options(options)
    agent = _build_agent(opts)

    session_id = opts.get("session_id") or _new_uuid()

    # 1) Session-init banner. Claude SDK ships ``SystemMessage(subtype="init",
    #    data={"tools": [...], "mcp_servers": [...], "model": "..."})``.
    yield SystemMessage(
        subtype="init",
        data={
            "tools": _system_tools_list(agent, opts),
            "mcp_servers": opts.get("mcp_servers", []),
            "model": agent.model,
            "permissionMode": opts.get("permission_mode", "default"),
            "cwd": opts.get("cwd", ""),
        },
        session_id=session_id,
    )

    # 2) Echo the user prompt as a UserMessage.
    if isinstance(prompt, str):
        yield UserMessage(content=prompt)
        seed_messages = [_InternalUserMessage(content=prompt)]
    else:
        seed_messages = []
        async for item in prompt:
            content = _extract_content(item)
            yield UserMessage(content=content)
            seed_messages.append(_InternalUserMessage(content=content))

    if not seed_messages:
        # Empty prompt — emit a no-op result and bail.
        yield _build_result(session_id=session_id, num_turns=0, usage=Usage(), backend_hint=None, model=agent.model, started_at=time.monotonic(), final_text="", error_subtype="error_during_execution", is_error=True, total_cost_usd=0.0, last_stop_reason=None, errors=["empty prompt"])
        await agent.aclose()
        return

    # 3) Run the agent loop and translate each internal message.
    started_at = time.monotonic()
    num_turns = 0
    last_stop_reason: str | None = None
    agg_usage = Usage()
    final_text = ""
    error_subtype = "success"
    is_error = False
    error_strings: list[str] = []
    backend_hint = (
        agent.backend_capability.provider_hint
        if agent.backend_capability is not None
        else None
    )

    try:
        messages = await agent.run(list(seed_messages))
        for msg in messages[len(seed_messages):]:
            if isinstance(msg, _InternalAssistantMessage):
                num_turns += 1
                last_stop_reason = msg.stop_reason or last_stop_reason
                if msg.usage is not None:
                    agg_usage = _accumulate_usage(agg_usage, msg.usage)
                yield AssistantMessage(
                    content=list(msg.content),
                    stop_reason=msg.stop_reason,
                    usage=msg.usage,
                )
                text = _extract_text(msg.content)
                if text:
                    final_text = text
            elif isinstance(msg, _InternalUserMessage):
                yield UserMessage(content=msg.content, isMeta=getattr(msg, "isMeta", False))
            elif isinstance(msg, _InternalSystemMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                yield SystemMessage(
                    subtype="init",
                    data={"text": content},
                    session_id=session_id,
                )
    except BudgetExceededError as e:
        is_error = True
        error_subtype = (
            "error_max_turns" if e.kind == "max_turns" else "error_max_budget_usd"
        )
        error_strings.append(f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        is_error = True
        error_subtype = "error_during_execution"
        error_strings.append(f"{type(e).__name__}: {e}")
    finally:
        await agent.aclose()

    duration_ms = int((time.monotonic() - started_at) * 1000)

    total_cost_usd = 0.0
    if agent._budget_tracker is not None:
        total_cost_usd = agent._budget_tracker.total_usd
    else:
        pricing = lookup_pricing(agent.model, backend_hint)
        if pricing is not None:
            total_cost_usd = pricing.cost(agg_usage)

    # Snapshot permission denials before _build_result so a closure-
    # mutated list doesn't surprise us. Each entry is a plain dict so
    # it serializes through msgspec cleanly.
    denials = list(getattr(agent, "_permission_denials", []))

    yield _build_result(
        session_id=session_id,
        num_turns=num_turns,
        usage=agg_usage,
        backend_hint=backend_hint,
        model=agent.model,
        started_at=started_at,
        final_text=final_text,
        error_subtype=error_subtype,
        is_error=is_error,
        total_cost_usd=total_cost_usd,
        last_stop_reason=last_stop_reason,
        errors=error_strings,
        permission_denials=denials,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_options(options: ClaudeAgentOptions | dict[str, Any] | None) -> dict[str, Any]:
    """Coerce options to a plain snake_case dict for ``_build_agent``."""

    if options is None:
        return {}
    if isinstance(options, ClaudeAgentOptions):
        return options.to_query_options()
    if isinstance(options, dict):
        return options
    raise TypeError(
        f"query(options): expected ClaudeAgentOptions, dict, or None, got {type(options).__name__}"
    )


def _build_agent(opts: dict[str, Any]) -> Agent:
    """Construct an Agent from the dict-form options with sensible defaults."""

    model = opts.get("model") or os.environ.get(
        "ANY_AGENT_MODEL", "qwen2.5-7b-instruct"
    )
    # Auto-route from model name when no backend was passed. Precedence:
    # explicit ``backend=`` > ``$ANY_AGENT_BASE_URL`` > shape-based
    # inference > Ollama default. This is what makes
    # ``ClaudeAgentOptions(model="qwen2.5:7b")`` work with no extra
    # kwarg — same two-line drop-in story as the Claude SDK.
    from .routing import resolve_backend
    backend = resolve_backend(model, opts.get("backend"))

    # Tool registry from Tool instances; built-in name strings are stashed.
    registry = ToolRegistry()
    raw_tools = opts.get("tools")
    if raw_tools:
        for t in raw_tools:
            if isinstance(t, Tool):
                registry.add(t)
            elif isinstance(t, ToolRegistry):
                for inner in t:
                    registry.add(inner)

    kw: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "system": opts.get("system"),
        "tools": registry,
        "max_tokens": opts.get("max_tokens", 1024),
        "temperature": opts.get("temperature"),
        "max_steps": opts.get("max_turns", opts.get("max_steps", 20)),
        "max_usd": opts.get("max_usd"),
    }
    for key in ("hooks", "permissions", "budget", "include_memory"):
        if key in opts:
            kw[key] = opts[key]

    consumed = set(kw.keys()) | {
        "model", "backend", "tools", "system", "max_tokens", "temperature",
        "max_turns", "max_steps", "max_usd", "api_key",
        "persist", "session_id", "cwd", "permission_mode",
        "mcp_servers", "agents",
    }
    extra = {k: v for k, v in opts.items() if k not in consumed}
    if extra:
        kw["extra"] = extra

    return Agent(**kw)


def _system_tools_list(agent: Agent, opts: dict[str, Any]) -> list[str]:
    """Build the ``data.tools`` list for the init SystemMessage.

    Includes:
      * names of real ``Tool`` instances passed in
      * built-in tool-name strings the user passed (``"Read"``, ``"Glob"``…),
        stashed on ``opts["extra"]["builtin_tool_names"]`` by the compat layer
      * MCP-namespaced names from configured ``mcp_servers``
    """

    names: list[str] = [t.name for t in agent.tools]

    extra = opts.get("extra") or {}
    builtin = extra.get("builtin_tool_names") or []
    names.extend(builtin)

    # If user supplied allowed_tools, it overrides everything (Claude SDK
    # surfaces only allowed tool names in the init message when set).
    allowed = extra.get("allowed_tools")
    if allowed:
        return list(allowed)
    return names


def _accumulate_usage(agg: Usage, msg: Usage) -> Usage:
    return Usage(
        input_tokens=agg.input_tokens + msg.input_tokens,
        output_tokens=agg.output_tokens + msg.output_tokens,
        cache_creation_input_tokens=msg.cache_creation_input_tokens or agg.cache_creation_input_tokens,
        cache_read_input_tokens=msg.cache_read_input_tokens or agg.cache_read_input_tokens,
    )


def _extract_text(blocks: Any) -> str:
    from .types import TextBlock as _TB
    if not isinstance(blocks, list):
        return ""
    return "".join(b.text for b in blocks if isinstance(b, _TB))


def _extract_content(item: Any) -> Any:
    if isinstance(item, str):
        return item
    if isinstance(item, UserMessage):
        return item.content
    if isinstance(item, _InternalUserMessage):
        return item.content
    if isinstance(item, dict):
        return item.get("content", "")
    return str(item)


def _new_uuid() -> str:
    return str(_uuid.uuid4())


def _build_result(
    *,
    session_id: str,
    num_turns: int,
    usage: Usage,
    backend_hint: str | None,
    model: str,
    started_at: float,
    final_text: str,
    error_subtype: str,
    is_error: bool,
    total_cost_usd: float,
    last_stop_reason: str | None,
    errors: list[str],
    permission_denials: list | None = None,
) -> ResultMessage:
    duration_ms = int((time.monotonic() - started_at) * 1000)
    # Flatten Usage to a dict for the wire — keeps ResultMessage
    # decoupled from the internal Usage struct and matches Claude SDK.
    usage_dict = (
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
        }
        if (usage.input_tokens or usage.output_tokens)
        else {}
    )
    model_usage_dict = (
        {model: dict(usage_dict, costUSD=total_cost_usd)} if usage_dict else {}
    )
    return ResultMessage(
        subtype=error_subtype if is_error else "success",
        duration_ms=duration_ms,
        duration_api_ms=duration_ms,
        is_error=is_error,
        num_turns=num_turns,
        result=final_text if not is_error else None,
        stop_reason=last_stop_reason,
        total_cost_usd=total_cost_usd,
        session_id=session_id,
        permission_denials=list(permission_denials or []),
        usage=usage_dict,
        modelUsage=model_usage_dict,
    )
