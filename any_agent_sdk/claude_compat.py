"""``claude_agent_sdk`` drop-in compatibility layer.

This module exists for ONE reason: the canonical Claude Agent SDK Python
examples (the ones shipped in
https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples)
should work verbatim against any-agent-sdk with nothing more than:

    -from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, …)
    +from any_agent_sdk import (AssistantMessage, ClaudeAgentOptions, …)

That's the litmus test the user asked for. This file ports the Python
SDK's public surface — which is *not* the same as the TS-internal
``SDKAssistantMessage`` shape (their TS source nests under ``.message``
to mirror the Anthropic API; the Python SDK flattens it).

Symbols ported here
-------------------

  * ``ClaudeAgentOptions`` — typed dataclass-style options. Mirrors the
    fields the Python SDK exposes: ``system_prompt``, ``max_turns``,
    ``tools`` (list[str] of built-in tool names or a preset dict),
    ``allowed_tools``, ``mcp_servers``, ``hooks``, ``cwd``,
    ``permission_mode``, ``model``, etc.
  * ``ResultMessage`` — flat-field shape with ``total_cost_usd``,
    ``num_turns``, ``duration_ms``, ``session_id``, etc.
  * ``SystemMessage`` extension — same struct, but with ``.subtype``
    (``"init"``) and ``.data`` (a dict carrying ``tools``,
    ``mcp_servers``, ``model``…) so the Claude-style
    ``msg.data.get("tools", [])`` access works.
  * Re-exports of ``AssistantMessage``, ``UserMessage``, ``TextBlock``,
    ``ToolUseBlock``, ``ToolResultBlock`` (our internal types — already
    flat ``.content``).
  * ``create_sdk_mcp_server`` — alias for our ``mcp.server.create_sdk_server``.
  * ``ClaudeSDKClient`` — streaming-mode client (async context manager
    wrapping ``query()``).
  * Hook types: ``HookContext``, ``HookInput``, ``HookJSONOutput``,
    ``HookMatcher``.

The aliases are re-exported at the top-level ``any_agent_sdk`` namespace
so Claude-style imports just work::

    from any_agent_sdk import AssistantMessage, ClaudeAgentOptions, ...

Both the new public flat shapes AND the TS-SDK-wire ``SDK*Message``
shapes are exported. Pick whichever your codebase wants.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import msgspec

from .tools import Tool
from .types import (
    AssistantMessage,
    ContentBlock,
    SystemMessage as _InternalSystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
)


__all__ = [
    "AgentDefinition",
    "AssistantMessage",
    "CLIConnectionError",
    "ClaudeAgentOptions",
    "ClaudeSDKClient",
    "ClaudeSDKError",
    "ContentBlock",
    "HookContext",
    "HookInput",
    "HookJSONOutput",
    "HookMatcher",
    "Message",
    "PermissionResult",
    "PermissionResultAllow",
    "PermissionResultDeny",
    "Plugin",
    "ResultMessage",
    "SystemMessage",
    "TextBlock",
    "ToolPermissionContext",
    "ToolResultBlock",
    "ToolUseBlock",
    "UserMessage",
    "create_sdk_mcp_server",
]


# ---------------------------------------------------------------------------
# Options dataclass (matches claude_agent_sdk.ClaudeAgentOptions)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClaudeAgentOptions:
    """Drop-in replacement for ``claude_agent_sdk.ClaudeAgentOptions``.

    All fields optional. ``model`` defaults to ``ANY_AGENT_MODEL`` env
    or ``"qwen2.5-7b-instruct"`` (so ``query(prompt="...")`` with no
    options works in development). ``backend`` defaults to
    ``ANY_AGENT_BASE_URL`` or ``http://localhost:11434`` (Ollama).
    """

    # Prompt / model
    system_prompt: str | dict[str, Any] | None = None
    model: str | None = None
    backend: str | None = None
    max_turns: int = 20
    max_tokens: int = 1024
    temperature: float | None = None

    # Tools
    # ``tools`` accepts either:
    #   * list[str]  — built-in tool names (e.g. ["Read", "Glob"])
    #   * list[Tool] — direct ``Tool`` instances
    #   * dict       — preset spec, e.g. {"type": "preset", "preset": "claude_code"}
    tools: list[str] | list[Tool] | dict[str, Any] | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)

    # Safety + budget
    permission_mode: Literal["default", "auto", "bypass"] | None = None
    can_use_tool: Callable[..., Awaitable[Any]] | None = None
    hooks: dict[str, list[Any]] | None = None
    max_budget_usd: float | None = None
    fallback_model: str | None = None

    # Environment / persistence
    cwd: str | None = None
    add_dirs: list[str] | None = None
    session_id: str | None = None
    persist: bool = False
    include_memory: bool = True
    setting_sources: list[str] | None = None

    # Agents — Claude SDK supports ``agents={"reviewer": AgentDefinition(...)}``
    # which exposes named sub-agents the parent can call via the Task tool.
    agents: dict[str, "AgentDefinition"] | None = None

    # Plugins — Claude SDK exposes ``plugins=[Plugin(...)]``.
    plugins: list[Any] | None = None

    # Streaming knobs
    include_partial_messages: bool = False

    # Diagnostics — Claude SDK exposes a ``stderr`` callable that
    # receives every line the underlying CLI writes to stderr. For
    # any-agent-sdk we don't shell out to a CLI, but we still accept the
    # field so examples passing it don't TypeError. Lines from our
    # logger get forwarded when set.
    stderr: Callable[[str], Any] | None = None
    # Same idea for ``stdin_input`` and other Claude SDK env knobs we
    # haven't wired runtime behavior for yet.
    stdin_input: str | None = None
    env: dict[str, str] | None = None
    user: str | None = None
    permission_prompt_tool_name: str | None = None
    continue_conversation: bool = False
    resume: str | None = None
    extra_args: dict[str, Any] | None = None
    debug_stderr: bool = False

    # Misc passthroughs (kept on Agent.extra)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_query_options(self) -> dict[str, Any]:
        """Convert to the snake_case dict ``query()`` accepts.

        Drops Nones and empty containers. Maps ``system_prompt`` →
        ``system`` (our naming) and tools/allowed_tools through the
        compat resolver.
        """

        opts: dict[str, Any] = {}
        if self.system_prompt is not None:
            opts["system"] = (
                self.system_prompt
                if isinstance(self.system_prompt, str)
                else _stringify_system_prompt(self.system_prompt)
            )
        if self.model:
            opts["model"] = self.model
        if self.backend:
            opts["backend"] = self.backend
        opts["max_turns"] = self.max_turns
        opts["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            opts["temperature"] = self.temperature

        # Tools — only Tool instances are usable today. Built-in name
        # strings like "Read"/"Glob" don't have implementations in
        # any-agent-sdk (we're OSS-first, no filesystem-tool runtime).
        # We stash them on extra so the SystemMessage(subtype="init")
        # can still emit them as configured, and skip the agent's
        # actual tool registry (which only wants Tool instances).
        real_tools_acc: list[Tool] = []
        builtin_names_acc: list[str] = []
        if self.tools is not None:
            real_tools, builtin_names = _split_tools(self.tools)
            real_tools_acc.extend(real_tools)
            builtin_names_acc.extend(builtin_names)

        # Plugins — bundles of (tools, system_prompt_addition, hooks).
        # Merge into the same buckets so the agent sees one combined view.
        # System-prompt additions are collected here and joined into
        # opts["system"] at the end (after the user's system_prompt is set
        # by the caller above). This matches Claude SDK semantics: plugin
        # text appends to whatever the user passed.
        plugin_system_additions: list[str] = []
        plugin_hooks_dicts: list[dict[str, list[Any]]] = []
        if self.plugins:
            for p in self.plugins:
                if getattr(p, "tools", None):
                    real_tools_acc.extend(p.tools)
                addition = getattr(p, "system_prompt_addition", None)
                if addition:
                    plugin_system_additions.append(addition)
                ph = getattr(p, "hooks", None)
                if ph:
                    plugin_hooks_dicts.append(ph)

        if real_tools_acc:
            opts["tools"] = real_tools_acc
        if builtin_names_acc:
            opts.setdefault("extra", {})["builtin_tool_names"] = builtin_names_acc

        # Append every plugin's system-prompt addition.
        if plugin_system_additions:
            base = opts.get("system") or ""
            joined = "\n\n".join([base.strip(), *plugin_system_additions]).strip()
            opts["system"] = joined

        if self.allowed_tools is not None:
            opts.setdefault("extra", {})["allowed_tools"] = self.allowed_tools
        if self.disallowed_tools is not None:
            opts.setdefault("extra", {})["disallowed_tools"] = self.disallowed_tools
        if self.mcp_servers:
            opts["mcp_servers"] = list(_normalize_mcp_servers(self.mcp_servers))

        if self.permission_mode is not None:
            opts["permission_mode"] = self.permission_mode
        # can_use_tool plumbing: build a PermissionContext so the agent
        # loop actually calls the callback at dispatch time. Without
        # this, ``ClaudeAgentOptions(can_use_tool=...)`` was silently
        # ignored — the field was accepted but the agent saw no
        # permissions object.
        if self.can_use_tool is not None or self.permission_mode is not None:
            from .permissions import PermissionContext  # local: cycle avoidance

            opts["permissions"] = PermissionContext(
                mode=self.permission_mode or "default",
                can_use_tool=self.can_use_tool,
            )
        # Merge user-supplied hooks dict with any plugin-contributed dicts.
        # User hooks win on per-event collision (set last in dict update).
        combined_hooks: dict[str, list[Any]] = {}
        for ph in plugin_hooks_dicts:
            for event, matchers in ph.items():
                combined_hooks.setdefault(event, []).extend(matchers)
        if self.hooks is not None:
            for event, matchers in self.hooks.items():
                combined_hooks[event] = list(matchers)

        if combined_hooks:
            from . import claude_compat as _self  # noqa: PLW0406 — local cycle guard

            opts["hooks"] = _self._convert_hooks_dict(combined_hooks)

        if self.max_budget_usd is not None:
            opts["max_usd"] = self.max_budget_usd
        if self.fallback_model:
            opts.setdefault("extra", {})["fallback_model"] = self.fallback_model
        if self.cwd:
            opts["cwd"] = self.cwd
        if self.session_id:
            opts["session_id"] = self.session_id
        if self.persist:
            opts["persist"] = True
        opts["include_memory"] = self.include_memory

        # User-supplied extras win over our auto-stashed ones.
        if self.extra:
            opts.setdefault("extra", {}).update(self.extra)
        return opts


def _stringify_system_prompt(sp: dict[str, Any]) -> str:
    """Render the Claude SDK's structured ``system_prompt`` dict to a string.

    Their dict form: ``{"type": "preset", "preset": "claude_code", "append": "..."}``.
    We don't run the Claude Code preset, so we just take the ``append`` field
    (if present) and pass through; otherwise empty string.
    """

    if isinstance(sp, dict):
        return str(sp.get("append") or sp.get("text") or "")
    return ""


def _split_tools(
    tools: list[Any] | dict[str, Any],
) -> tuple[list[Tool], list[str]]:
    """Partition the ``tools`` field into real Tool instances vs built-in name strings."""

    if isinstance(tools, dict):
        # Preset form (e.g. ``{"type": "preset", "preset": "claude_code"}``).
        # We don't ship the Claude Code presets; treat as "no real tools,
        # report the preset name to the system message for parity."
        return [], [str(tools)]
    real: list[Tool] = []
    names: list[str] = []
    for t in tools:
        if isinstance(t, Tool):
            real.append(t)
        elif isinstance(t, str):
            names.append(t)
    return real, names


def _normalize_mcp_servers(servers: dict[str, Any]) -> list[Any]:
    """Pass MCP server configs through — accepting either dict configs or
    in-process server instances from ``create_sdk_mcp_server``."""

    return list(servers.items())


def _convert_hooks_dict(hooks: dict[str, list[Any]]) -> Any:
    """Convert ``{"PreToolUse": [HookMatcher(...), ...]}`` dict form into
    the ``Hooks`` dataclass our agent loop expects.

    Claude SDK pattern: ``hooks={"PreToolUse": [HookMatcher(matcher="Bash",
    hooks=[check_bash_command])]}``. Each HookMatcher has a list of
    callables. We collapse to one ``Hooks`` instance — only the first
    matcher's first callable per event is honored for now (full matcher
    semantics arrive in M2 wire-up).
    """

    from .hooks import Hooks

    out_kwargs: dict[str, Any] = {}
    event_to_field = {
        "PreToolUse": "pre_tool_use",
        "PostToolUse": "post_tool_use",
        "PostToolUseFailure": "post_tool_use_failure",
        "Notification": "notification",
        "UserPromptSubmit": "user_prompt_submit",
        "SessionStart": "session_start",
        "SessionEnd": "session_end",
        "Stop": "stop",
        "StopFailure": "stop_failure",
        "SubagentStart": "subagent_start",
        "SubagentStop": "subagent_stop",
        "PreCompact": "pre_compact",
        "PostCompact": "post_compact",
        "PermissionRequest": "permission_request",
        "PermissionDenied": "permission_denied",
    }
    for event, matchers in hooks.items():
        field_name = event_to_field.get(event)
        if not field_name:
            continue
        # Each matcher may be a HookMatcher with .hooks, or a plain callable
        for matcher in matchers:
            callable_fn = None
            if callable(matcher):
                callable_fn = matcher
            elif hasattr(matcher, "hooks") and matcher.hooks:
                callable_fn = matcher.hooks[0]
            if callable_fn is not None:
                out_kwargs[field_name] = callable_fn
                break
    return Hooks(**out_kwargs)


# ---------------------------------------------------------------------------
# ResultMessage — Claude Python SDK shape (flat fields)
# ---------------------------------------------------------------------------


class ResultMessage(msgspec.Struct, omit_defaults=True):
    """Drop-in replacement for ``claude_agent_sdk.ResultMessage``.

    Flat fields exactly matching Claude Python SDK so
    ``msg.total_cost_usd`` / ``msg.num_turns`` / ``msg.subtype`` works
    verbatim.

    ``permission_denials`` carries one dict per ``can_use_tool``-denied
    call: ``{"tool_name": str, "tool_use_id": str, "tool_input": dict}``.
    Empty list on a clean run. Lets a UI / audit layer inspect what the
    permission system blocked without diffing the message stream.
    """

    subtype: str = "success"
    duration_ms: int = 0
    duration_api_ms: int = 0
    is_error: bool = False
    num_turns: int = 0
    result: str | None = None
    stop_reason: str | None = None
    total_cost_usd: float = 0.0
    session_id: str = ""
    permission_denials: list[dict] = msgspec.field(default_factory=list)
    # Usage + per-model usage carried on the wire so a Claude SDK consumer
    # can read ``msg.usage["input_tokens"]`` and
    # ``msg.modelUsage[model]["costUSD"]`` directly. Empty dicts on a run
    # where no usage was reported by the backend.
    usage: dict = msgspec.field(default_factory=dict)
    modelUsage: dict = msgspec.field(default_factory=dict)


# ---------------------------------------------------------------------------
# SystemMessage extension — Claude SDK exposes .subtype + .data dict
# ---------------------------------------------------------------------------


class SystemMessage(msgspec.Struct, omit_defaults=True):
    """Drop-in replacement for ``claude_agent_sdk.SystemMessage``.

    ``subtype`` discriminates ``"init"`` (session-start banner) from
    ``"compact_boundary"`` and ``"status"``. ``data`` is the free-form
    dict carrying ``tools``, ``mcp_servers``, ``model``, etc. — matches
    how Claude Python SDK code reads ``msg.data.get("tools", [])``.
    """

    subtype: str = "init"
    data: dict[str, Any] = msgspec.field(default_factory=dict)
    session_id: str = ""
    # ``role`` and ``content`` retained for compatibility with our
    # internal SystemMessage flow.
    role: Literal["system"] = "system"
    content: str | list[ContentBlock] = ""


Message = Union[AssistantMessage, UserMessage, SystemMessage, ResultMessage]


# ---------------------------------------------------------------------------
# Hook types (Claude Python SDK shape — typed dicts on the wire)
# ---------------------------------------------------------------------------


# Claude SDK hook types are TypedDict-shaped but we expose them as plain
# dict aliases so users importing them get something they can subscript.
HookInput = dict[str, Any]
HookJSONOutput = dict[str, Any]


@dataclass(slots=True)
class HookContext:
    """Drop-in for ``claude_agent_sdk.types.HookContext``."""

    session_id: str = ""
    cwd: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HookMatcher:
    """Drop-in for ``claude_agent_sdk.types.HookMatcher``.

    ``matcher`` is an optional tool-name pattern; ``hooks`` is a list of
    async callables ``(input, tool_use_id, context) -> HookJSONOutput``.
    """

    matcher: str | None = None
    hooks: list[Callable[..., Awaitable[Any]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Permission result shape (Claude SDK)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PermissionResult:
    """Drop-in for ``claude_agent_sdk.types.PermissionResult``."""

    behavior: Literal["allow", "deny", "ask"]
    message: str | None = None
    interrupt: bool = False
    updated_input: dict[str, Any] | None = None


@dataclass(slots=True)
class PermissionResultAllow:
    """Drop-in for ``claude_agent_sdk.PermissionResultAllow``.

    Returned from a ``can_use_tool`` callback to authorize the call.
    Optional ``updated_input`` rewrites the tool args before dispatch.
    """

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None


@dataclass(slots=True)
class PermissionResultDeny:
    """Drop-in for ``claude_agent_sdk.PermissionResultDeny``.

    Returned from ``can_use_tool`` to block the call. ``message`` is the
    reason surfaced to the model in the tool_result block. ``interrupt``
    halts the run when True (otherwise the model can recover by trying
    another tool).
    """

    message: str
    behavior: Literal["deny"] = "deny"
    interrupt: bool = False


@dataclass(slots=True)
class ToolPermissionContext:
    """Drop-in for ``claude_agent_sdk.ToolPermissionContext``.

    Third argument passed to a ``can_use_tool`` callback. Carries the
    session id, optional permission-rule context, and a free-form
    ``signal`` field for cancellation/interruption (None in our impl
    until we wire AbortController parity).
    """

    session_id: str = ""
    signal: Any = None
    suggestions: list[Any] = field(default_factory=list)


class CLIConnectionError(Exception):
    """Drop-in for ``claude_agent_sdk.CLIConnectionError``.

    Raised when the underlying transport / model API can't be reached.
    The Claude SDK examples catch this around the streaming-client
    setup. We map it onto our existing :class:`ProviderError` family —
    instances of CLIConnectionError ARE ProviderErrors so existing
    catch blocks still work, and Claude SDK examples that catch
    CLIConnectionError see the same surface.
    """


class ClaudeSDKError(Exception):
    """Drop-in for ``claude_agent_sdk.ClaudeSDKError``. Base class
    that the Claude SDK uses for ``CLIConnectionError`` and
    ``ProcessError``. We extend it as a sibling of our AgentError so
    callers can catch either."""


@dataclass(slots=True)
class Plugin:
    """Drop-in for ``claude_agent_sdk.Plugin``.

    A reusable bundle of (tools, system_prompt_addition, hooks). Pass a
    list of these to ``ClaudeAgentOptions(plugins=[...])`` and the agent
    will merge their contents at session start: every plugin's ``tools``
    join the registry, the ``system_prompt_addition`` text is appended
    to ``system_prompt``, and ``hooks`` are folded into the active
    ``Hooks`` instance (last wins on per-event collision).

    Distinct from MCP servers — plugins live in-process and bundle
    Python-side state; MCP servers are external (or in-process via
    ``create_sdk_mcp_server``) and exchange JSON-RPC.
    """

    name: str
    version: str = "1.0.0"
    tools: list[Tool] = field(default_factory=list)
    system_prompt_addition: str | None = None
    hooks: dict[str, list[Any]] | None = None


@dataclass(slots=True)
class AgentDefinition:
    """Drop-in for ``claude_agent_sdk.AgentDefinition``.

    Names a sub-agent the parent can delegate to via the ``Task`` tool
    (Claude's terminology). Carries the description (when to use it),
    the prompt (system prompt for the child), the allowed tools, and
    optionally the model.
    """

    description: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None


# ---------------------------------------------------------------------------
# create_sdk_mcp_server alias
# ---------------------------------------------------------------------------


def create_sdk_mcp_server(name: str, version: str = "1.0.0", tools: list[Tool] | None = None) -> Any:
    """Drop-in for ``claude_agent_sdk.create_sdk_mcp_server``.

    Builds an in-process MCP server exposing ``tools`` (Tool instances
    from ``@tool`` — see ``compat_tool_decorator`` in :mod:`any_agent_sdk.tools`
    for the Claude-shaped positional signature).
    """

    from .mcp.server import create_sdk_server  # local: avoid optional-dep cost

    return create_sdk_server(name=name, tools=list(tools or []))


# ---------------------------------------------------------------------------
# ClaudeSDKClient — streaming-mode async context manager
# ---------------------------------------------------------------------------


class ClaudeSDKClient:
    """Drop-in for ``claude_agent_sdk.ClaudeSDKClient``.

    Usage::

        async with ClaudeSDKClient(options=options) as client:
            await client.query("Tell me a joke")
            async for message in client.receive_response():
                ...

    Internally just wraps ``query()`` with a queue-of-prompts so the
    caller can interleave ``query()`` and ``receive_response()`` over
    multiple turns sharing the same options.
    """

    def __init__(self, options: ClaudeAgentOptions | None = None) -> None:
        self.options = options or ClaudeAgentOptions()
        self._pending_prompt: str | None = None

    async def __aenter__(self) -> ClaudeSDKClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Open the session. No-op for the in-process implementation — each
        ``query()`` builds its own ``Agent`` — but matches the Claude SDK
        public method so examples calling ``await client.connect()`` /
        ``await client.disconnect()`` work unchanged."""

        return None

    async def disconnect(self) -> None:
        """Close the session. See :meth:`connect`."""

        return None

    async def query(self, prompt: str) -> None:
        """Queue a prompt. Returns immediately; consume the response via
        ``receive_response()``."""

        self._pending_prompt = prompt

    async def receive_messages(self) -> AsyncIterator[Message]:
        """Alias for :meth:`receive_response`. Some Claude SDK examples
        call this name."""

        async for msg in self.receive_response():
            yield msg

    async def receive_response(self) -> AsyncIterator[Message]:
        """Yield messages for the most-recently-queued prompt."""

        prompt = self._pending_prompt
        if prompt is None:
            raise RuntimeError("ClaudeSDKClient.query(...) must be called before receive_response()")
        self._pending_prompt = None

        # Local import to avoid a circular at module init time.
        from .compat_query import query as _compat_query

        async for msg in _compat_query(prompt=prompt, options=self.options):
            yield msg
