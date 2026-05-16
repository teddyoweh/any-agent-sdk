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

from .budget import Budget, BudgetTracker
from .capabilities import (
    BackendCapability,
    ModelCapability,
    hosted_profile_from_url,
    lookup_model,
)
from .errors import BudgetExceededError, StreamProtocolError
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
from .hooks import HookContext, HookDispatcher, Hooks
from .permissions import (
    Allow,
    Deny,
    PermissionContext,
    check_permission,
)
from .providers.base import Provider, detect_provider, resolve
from .streaming.executor import StreamingToolExecutor
from .tools import Tool, ToolRegistry, dispatch_tool_calls
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
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

    # Safety + budget knobs — None means "no enforcement".
    hooks: Hooks | None = None
    permissions: PermissionContext | None = None
    budget: Budget | None = None
    max_usd: float | None = None  # shortcut: sets budget.max_usd if budget is None

    # Memory — auto-loads ``~/.anyagent/MEMORY.md`` and prepends it to the
    # system prompt. Matches Claude Code's behavior (the index is always in
    # context; individual entries are loaded on demand via the memory tool).
    # Set to False for tests / containerized runs where you don't want disk I/O.
    include_memory: bool = True

    # Internal state populated in __post_init__ / run loop.
    _dispatcher: HookDispatcher | None = field(default=None, init=False)
    _budget_tracker: BudgetTracker | None = field(default=None, init=False)
    _provider_hint: str | None = field(default=None, init=False)
    # Permission denials accumulated across the run. Each entry is a
    # ``{tool_name, tool_use_id, tool_input}`` dict matching the Claude
    # SDK's ``SDKPermissionDenial`` shape. ``query()`` reads this after
    # ``run()`` returns and populates ``SDKResultMessage.permission_denials``.
    _permission_denials: list = field(default_factory=list, init=False)

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

        # Memory + user-context will be injected at run() time as a
        # synthetic <system-reminder>-wrapped UserMessage with
        # isMeta=True, NOT prepended to the system prompt. This matches
        # Claude Code's mechanism (see system_reminder.py for the audit).
        # The actual content is resolved lazily so a session that doesn't
        # call run() pays nothing for it.

        # Wire safety surface.
        self._dispatcher = HookDispatcher(self.hooks or Hooks())

        # Resolve budget — accept either an explicit Budget or shortcut kwargs.
        if self.budget is None and self.max_usd is not None:
            self.budget = Budget(max_usd=self.max_usd, max_turns=self.max_steps)
        if self.budget is not None:
            self._budget_tracker = BudgetTracker(budget=self.budget)

        # Provider hint for pricing lookups (e.g. "together", "fireworks").
        if self.backend_capability is not None:
            self._provider_hint = self.backend_capability.provider_hint or None

    @staticmethod
    def _has_user_context_message(messages: list[Message]) -> bool:
        """True if the first message is already a meta user-context message
        — so re-calling ``run()`` doesn't double-inject."""

        if not messages:
            return False
        first = messages[0]
        if not isinstance(first, UserMessage):
            return False
        return getattr(first, "isMeta", False) is True

    def _build_user_context(self) -> dict[str, str]:
        """Resolve the user-context dict that gets wrapped in a
        ``<system-reminder>`` and prepended to the conversation.

        Matches Claude SDK's ``getUserContext()`` shape — a flat
        ``{key: value}`` dict. Currently populates one key:

          * ``memory`` — contents of ``~/.anyagent/MEMORY.md`` if
            ``include_memory`` is True

        Extension point for future keys: ``claudeMd`` (project-local
        ``CLAUDE.md`` walk), ``skills`` (always-loaded skills bundle),
        custom keys via ``extra={"user_context": {...}}``.
        """

        ctx: dict[str, str] = {}

        if self.include_memory:
            try:
                from .memory import load_memory_index
                index = load_memory_index().strip()
                if index:
                    ctx["memory"] = index
            except Exception:  # noqa: BLE001 — disk I/O can fail in containers
                _log.debug("memory load skipped (I/O error)", exc_info=True)

        # User-supplied extras flow through ``extra={"user_context": {...}}``.
        if isinstance(self.extra, dict):
            extra_ctx = self.extra.get("user_context")
            if isinstance(extra_ctx, dict):
                for k, v in extra_ctx.items():
                    if isinstance(v, str) and v:
                        ctx[k] = v

        return ctx

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

        Drives the full integration:
          * Streams the provider; assembles assistant message.
          * For each ``tool_use`` in the assistant message, runs ``PreToolUse``
            hook → checks permission → dispatches via
            ``StreamingToolExecutor`` (parallel-safe partitioned).
          * After the batch finishes, runs ``PostToolUse`` hook per call.
          * Tracks token + USD usage on a ``BudgetTracker``; raises
            ``BudgetExceededError`` when limits are crossed.
          * Fires ``Stop`` when the assistant emits a turn with no tool calls.

        ``messages`` is mutated in place; the same list is returned for chaining.
        """

        assert self._dispatcher is not None  # set in __post_init__

        # Inject persistent user-context (memory + custom) as a synthetic
        # ``<system-reminder>``-wrapped UserMessage at the head of the
        # conversation. Matches Claude SDK 1:1. No-op when context is empty.
        # We do this once per run() call; subsequent turns reuse the
        # already-injected message.
        if not self._has_user_context_message(messages):
            user_ctx = self._build_user_context()
            if user_ctx:
                from .system_reminder import prepend_user_context  # local import
                prepend_user_context(messages, user_ctx, in_place=True)

        for _ in range(self.max_steps):
            assistant = await self._one_turn(messages)
            messages.append(assistant)

            # Bump turn + cost AFTER the assistant message materializes,
            # so we don't strand a budget overrun mid-stream.
            if self._budget_tracker is not None:
                self._budget_tracker.add_turn()
                if assistant.usage is not None:
                    self._budget_tracker.add_usage(
                        assistant.usage, self.model, backend_hint=self._provider_hint
                    )
                # Enforce — raises BudgetExceededError if any limit is over.
                self._budget_tracker.check()

            tool_calls = [b for b in assistant.content if isinstance(b, ToolUseBlock)]
            if not tool_calls:
                # Fire the Stop hook on natural turn-end (no tool follow-up).
                await self._dispatcher.dispatch(
                    "Stop",
                    HookContext(event="Stop", messages_snapshot=messages),
                )
                return messages

            # Run the tool batch through the streaming executor + safety net.
            results = await self._run_tool_batch(tool_calls, messages)
            messages.append(UserMessage(content=list(results)))

        _log.warning("agent hit max_steps=%d without natural stop", self.max_steps)
        return messages

    async def _run_tool_batch(
        self,
        calls: list[ToolUseBlock],
        messages: list[Message],
    ) -> list[ToolResultBlock]:
        """Dispatch a single batch of tool calls under hooks + permissions."""

        assert self._dispatcher is not None
        registry: ToolRegistry = self.tools  # type: ignore[assignment]

        # Pre-flight every call: PreToolUse hook + permission check.
        # Each call is either approved (and possibly input-mutated) or
        # short-circuited to an is_error result block.
        approved: list[ToolUseBlock] = []
        short_circuit: dict[str, ToolResultBlock] = {}

        for call in calls:
            tool = registry.get(call.name)
            if tool is None:
                # Unknown tool. Let the executor surface this — it has the
                # canonical "not found" path. Skip pre-checks.
                approved.append(call)
                continue

            # PreToolUse hook
            ctx = HookContext(
                event="PreToolUse",
                tool=tool,
                input=call.input,
                messages_snapshot=messages,
            )
            hr = await self._dispatcher.dispatch("PreToolUse", ctx)
            if hr.block:
                short_circuit[call.id] = ToolResultBlock(
                    tool_use_id=call.id,
                    content=f"blocked by hook: {hr.note or 'no reason given'}",
                    is_error=True,
                )
                continue

            # Apply hook-mutated input if any.
            payload = hr.mutated_input if hr.mutated_input is not None else call.input
            if payload is not call.input:
                # ToolUseBlock is frozen — rebuild with new input.
                call = ToolUseBlock(id=call.id, name=call.name, input=payload)

            # Permission check.
            if self.permissions is not None:
                decision = await check_permission(tool, call.input, self.permissions)
                # Bridge Claude-shape PermissionResultAllow/Deny returned
                # from a user-supplied can_use_tool callback to our internal
                # Allow/Deny structs. Duck-type on .behavior since the Claude
                # variants are plain dataclasses, not msgspec Structs.
                decision = _normalize_permission_decision(decision)

                if isinstance(decision, Deny):
                    await self._dispatcher.dispatch(
                        "PermissionDenied",
                        HookContext(
                            event="PermissionDenied",
                            tool=tool,
                            input=call.input,
                            arbitrary={"reason": decision.reason},
                        ),
                    )
                    # Record for the SDKResultMessage.permission_denials
                    # surface. Shape matches Claude SDK's SDKPermissionDenial
                    # so query() can pass straight through without
                    # transforming.
                    self._permission_denials.append(
                        {
                            "tool_name": call.name,
                            "tool_use_id": call.id,
                            "tool_input": dict(call.input or {}),
                        }
                    )
                    short_circuit[call.id] = ToolResultBlock(
                        tool_use_id=call.id,
                        content=f"permission denied: {decision.reason}",
                        is_error=True,
                    )
                    continue

                # Allow.updated_input rewrites the tool args before dispatch
                # — matches Claude SDK PermissionResultAllow semantics. The
                # ToolUseBlock is frozen, so rebuild with the patched input.
                if isinstance(decision, Allow) and decision.updated_input is not None:
                    call = ToolUseBlock(
                        id=call.id,
                        name=call.name,
                        input=decision.updated_input,
                    )
                # Ask → in v0 we treat Ask as Allow at the loop layer.
                # Integrators can convert Ask to Deny via the can_use_tool callback.

            approved.append(call)

        # Dispatch approved calls under the StreamingToolExecutor.
        async with StreamingToolExecutor(registry) as executor:
            for call in approved:
                executor.add_tool_call(call)
            executor_results = await executor.wait_all()

        # PostToolUse hooks for each executed call.
        for call, result in zip(approved, executor_results):
            tool = registry.get(call.name)
            if tool is None:
                continue
            event_name = "PostToolUseFailure" if result.is_error else "PostToolUse"
            await self._dispatcher.dispatch(
                event_name,
                HookContext(
                    event=event_name,
                    tool=tool,
                    input=call.input,
                    output=result.content,
                ),
            )

        # Merge short-circuited results with executor results, preserving
        # original call order.
        by_id: dict[str, ToolResultBlock] = {r.tool_use_id: r for r in executor_results}
        by_id.update(short_circuit)
        return [by_id[c.id] for c in calls]

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
        # Built-in tools (WebSearch/WebFetch) hold a long-lived HTTP client;
        # close it when the agent shuts down. Safe to call when unused —
        # lazily-constructed.
        try:
            from .builtin_tools import aclose_builtin_clients
            await aclose_builtin_clients()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass

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


def _normalize_permission_decision(decision: Any) -> Any:
    """Bridge Claude-shape PermissionResultAllow/Deny → internal Allow/Deny.

    Users passing a ``can_use_tool`` callback via ``ClaudeAgentOptions``
    typically return Claude SDK's dataclasses (``PermissionResultAllow``
    /``PermissionResultDeny``). Our ``check_permission`` returns the
    internal msgspec ``Allow``/``Deny``/``Ask``. This normalizer makes
    the agent loop indifferent to which shape it got.

    Duck-types on ``.behavior`` so we don't have to import the Claude
    compat classes here and create a cycle.
    """

    if isinstance(decision, (Allow, Deny)):
        return decision
    behavior = getattr(decision, "behavior", None)
    if behavior == "allow":
        return Allow(updated_input=getattr(decision, "updated_input", None))
    if behavior == "deny":
        return Deny(reason=getattr(decision, "message", "denied"))
    return decision


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
