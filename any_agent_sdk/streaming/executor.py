"""``StreamingToolExecutor`` — kick off tool calls as soon as their JSON
finalizes mid-stream, run them with the right concurrency rules, and produce
``ToolResultBlock``s in insertion order when the turn is done.

This is the *speed unlock* described in ``docs/plan.md`` §5. The naive
"wait until message_stop, then dispatch" loop pays a per-tool serialization
tax even when tools could overlap. By starting each tool the moment its
``</tool_call>`` / native ``tool_use.stop`` arrives, multi-tool turns finish
in roughly ``max(tool_durations)`` rather than ``sum(tool_durations)``.

Behavior summary
----------------
* **Concurrency-safe tools** run in parallel, bounded by
  ``ANY_AGENT_MAX_TOOL_CONCURRENCY`` (env, default 10) or the
  ``max_concurrency`` constructor arg.
* **Non-concurrency-safe tools** run serially — one at a time, never
  overlapping with another non-safe tool (they *can* overlap with safe
  tools, mirroring upstream's behavior).
* ``Tool.is_concurrency_safe`` may be a bool *or* a callable
  ``(input) -> bool``. Callable form lets ``bash`` say "safe iff paths
  don't conflict" etc.
* ``Tool.abort_siblings_on_error`` — when a tool with this flag errors,
  every in-flight sibling under the executor is cancelled via a shared
  ``CancelScope`` and produces a ``ToolResultBlock(is_error=True)``.
* ``Tool.timeout_s`` — wraps the call in ``anyio.fail_after``; on timeout
  the result is ``is_error=True``.
* User-tool exceptions never propagate. They become ``is_error=True``
  result blocks. The executor is bulletproof against bad tool code.
* Missing tools become ``is_error=True`` immediately.
* ``can_use_tool(tool, input, ctx)`` — if supplied, gates every call.
  Denials become ``is_error=True`` with the reason as the body.

Anyio, not asyncio
------------------
We use ``anyio`` primitives throughout: ``create_task_group``,
``CancelScope``, ``Semaphore``, ``Lock``, ``fail_after``. The agent loop
must be driven by an anyio backend (default trio or asyncio under
``anyio.run``). Never reach for ``asyncio.*`` here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import anyio
import msgspec

from ..errors import ToolExecutionError
from ..tools import Tool, ToolRegistry
from ..types import ToolResultBlock, ToolUseBlock

__all__ = ["CanUseToolFn", "StreamingToolExecutor"]

_LOG = logging.getLogger("any_agent_sdk.streaming.executor")
_ENC = msgspec.json.Encoder()
_DEFAULT_MAX_CONCURRENCY = 10
_ENV_VAR = "ANY_AGENT_MAX_TOOL_CONCURRENCY"


# Signature: (tool, input, ctx) -> (allowed, reason_if_denied)
CanUseToolFn = Callable[
    [Tool, dict[str, Any], dict[str, Any]],
    Awaitable[tuple[bool, Optional[str]]],
]


def _resolve_max_concurrency(override: int | None) -> int:
    if override is not None and override > 0:
        return override
    raw = os.environ.get(_ENV_VAR)
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            _LOG.warning("ignoring invalid %s=%r", _ENV_VAR, raw)
    return _DEFAULT_MAX_CONCURRENCY


def _is_safe(tool: Tool, tool_input: dict[str, Any]) -> bool:
    """Resolve ``Tool.is_concurrency_safe`` against this input.

    Accepts:
      * a static ``bool`` (the common case)
      * a callable ``(input) -> bool``
      * the legacy ``parallel_safe: bool`` attribute (current tools.py)

    The callable path lets ``bash``-class tools decide per-input —
    e.g. two writes to disjoint paths can parallelize."""
    flag = getattr(tool, "is_concurrency_safe", None)
    if flag is None:
        # Fall back to the older static attribute on Tool. v0 of tools.py uses
        # ``parallel_safe: bool``; the M0.1 refactor renames + extends it.
        flag = getattr(tool, "parallel_safe", True)
    if callable(flag):
        try:
            return bool(flag(tool_input))
        except Exception:  # noqa: BLE001 — predicate must never crash dispatch
            _LOG.exception("is_concurrency_safe predicate raised; assuming UNSAFE")
            return False
    return bool(flag)


def _stringify(out: Any) -> str:
    if isinstance(out, str):
        return out
    try:
        return _ENC.encode(out).decode()
    except (TypeError, msgspec.EncodeError):
        return str(out)


def _flatten_exception_group(eg: BaseExceptionGroup) -> list[BaseException]:
    """Walk a (possibly nested) ``BaseExceptionGroup`` and return the leaf
    exceptions. anyio 4.x can nest groups arbitrarily — a single-leaf
    group nested two deep should still unwrap cleanly."""
    out: list[BaseException] = []
    for sub in eg.exceptions:
        if isinstance(sub, BaseExceptionGroup):
            out.extend(_flatten_exception_group(sub))
        else:
            out.append(sub)
    return out


class StreamingToolExecutor:
    """Dispatch tool calls as they arrive on the stream.

    Lifecycle::

        async with StreamingToolExecutor(registry) as ex:
            # in the stream consumer, for every ToolUseBlock that finalizes:
            ex.add_tool_call(block)
            # ...after message_stop:
            results = await ex.wait_all()

    The context-manager exit awaits every still-in-flight task (so leaving
    the ``async with`` block is a safe join point even if you forget to
    call ``wait_all``)."""

    __slots__ = (
        "_registry",
        "_can_use_tool",
        "_max_concurrency",
        "_sem",
        "_serial_lock",
        "_calls",
        "_results",
        "_pending",
        "_idle_event",
        "_tg",
        "_scopes",
        "_aborted",
        "_closed",
    )

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_concurrency: int | None = None,
        can_use_tool: CanUseToolFn | None = None,
    ) -> None:
        self._registry = registry
        self._can_use_tool = can_use_tool
        self._max_concurrency = _resolve_max_concurrency(max_concurrency)
        # Parallel slots for concurrency-safe tools.
        self._sem = anyio.Semaphore(self._max_concurrency)
        # One-at-a-time lock for non-concurrency-safe tools.
        self._serial_lock = anyio.Lock()
        # Track calls in insertion order.
        self._calls: list[ToolUseBlock] = []
        self._results: list[ToolResultBlock | None] = []
        # Number of dispatched-but-not-yet-finished tasks. ``wait_all`` blocks
        # on ``_idle_event`` until this reaches zero. Reset whenever a new
        # call is added so callers can interleave dispatch and waiting.
        self._pending = 0
        self._idle_event = anyio.Event()
        self._idle_event.set()  # start "idle" — no pending work
        # Set in __aenter__.
        self._tg: anyio.abc.TaskGroup | None = None
        # Per-task cancel scopes, kept so sibling-abort can cancel every
        # in-flight task at once. Indexed in insertion order.
        self._scopes: list[anyio.CancelScope] = []
        self._aborted = False
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "StreamingToolExecutor":
        # We do NOT enter the task group as ``async with`` here because that
        # would block on every task at exit. Instead we manage it manually so
        # ``add_tool_call`` can ``start_soon`` into it across the lifetime.
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._closed = True
        # Closing the task group joins all in-flight tasks.
        assert self._tg is not None
        unwrap_exc: BaseException | None = None
        try:
            await self._tg.__aexit__(exc_type, exc, tb)
        except BaseExceptionGroup as eg:
            # anyio 4.x ALWAYS wraps body-raised exceptions in a
            # ``BaseExceptionGroup`` at task-group exit, even when no child
            # task raised. That makes plain exception flow (e.g. a
            # ``BudgetExceededError`` raised in the body of
            # ``async with StreamingToolExecutor(...)``) bubble up as a
            # group to our caller — which is brittle and not what callers
            # expect from a "dispatch tools and run them" helper. Unwrap
            # singleton groups so callers see the original exception.
            flat = _flatten_exception_group(eg)
            if len(flat) == 1:
                unwrap_exc = flat[0]
            else:
                # Multiple distinct exceptions — keep the group so the
                # caller can inspect every failure.
                raise
        finally:
            # Fill any leftover None slots — shouldn't happen if the TG joined
            # cleanly, but a stray cancellation could leave a hole.
            for i, r in enumerate(self._results):
                if r is None:
                    call = self._calls[i]
                    self._results[i] = ToolResultBlock(
                        tool_use_id=call.id,
                        content="tool execution cancelled",
                        is_error=True,
                    )
            # Releases any wait_all() blocked on us.
            self._pending = 0
            if not self._idle_event.is_set():
                self._idle_event.set()
        if unwrap_exc is not None:
            raise unwrap_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tool_call(self, block: ToolUseBlock) -> None:
        """Register a freshly-finalized tool_use block and dispatch it.

        Non-blocking — returns as soon as the task is scheduled. Order of
        ``add_tool_call`` calls is the order of the returned results."""
        if self._closed:
            raise RuntimeError("add_tool_call after executor exit")
        if self._tg is None:
            raise RuntimeError("StreamingToolExecutor used outside async with")
        idx = len(self._calls)
        self._calls.append(block)
        self._results.append(None)
        self._scopes.append(anyio.CancelScope())
        # Fast-fail if already aborted by a sibling error.
        if self._aborted:
            self._results[idx] = ToolResultBlock(
                tool_use_id=block.id,
                content="aborted by sibling tool error",
                is_error=True,
            )
            return
        # Bump pending and clear the idle gate.
        self._pending += 1
        if self._idle_event.is_set():
            self._idle_event = anyio.Event()
        self._tg.start_soon(self._run_one, idx, block)

    async def wait_all(self) -> list[ToolResultBlock]:
        """Block until every dispatched tool finishes and return result blocks
        in the order ``add_tool_call`` was called.

        Safe to call from inside ``async with``. The context exit will also
        flush — but most callers will await this explicitly to get the list
        before leaving the block."""
        if self._tg is None:
            raise RuntimeError("StreamingToolExecutor used outside async with")
        while self._pending > 0:
            await self._idle_event.wait()
        # Fill any leftover Nones (cancellation races).
        out: list[ToolResultBlock] = []
        for i, r in enumerate(self._results):
            if r is None:
                call = self._calls[i]
                r = ToolResultBlock(
                    tool_use_id=call.id,
                    content="tool execution cancelled",
                    is_error=True,
                )
                self._results[i] = r
            out.append(r)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_one(self, idx: int, block: ToolUseBlock) -> None:
        """One tool's lifetime: lookup → permission → concurrency gate →
        invoke → record result. Never raises out (all errors become
        ``is_error=True`` result blocks).

        The body runs under its own ``CancelScope`` so a sibling that calls
        ``_maybe_abort`` can yank in-flight work via ``scope.cancel()`` —
        this is how ``abort_siblings_on_error`` kills subprocess-bearing
        bash tools the moment a peer fails."""
        scope = self._scopes[idx]
        try:
            with scope:
                await self._run_one_inner(idx, block)
            if scope.cancelled_caught and self._results[idx] is None:
                self._results[idx] = self._cancelled_result(block)
        finally:
            self._pending -= 1
            if self._pending <= 0:
                self._pending = 0
                if not self._idle_event.is_set():
                    self._idle_event.set()

    async def _run_one_inner(self, idx: int, block: ToolUseBlock) -> None:
        tool = self._registry.get(block.name)
        if tool is None:
            self._results[idx] = ToolResultBlock(
                tool_use_id=block.id,
                content=f"tool {block.name!r} not found",
                is_error=True,
            )
            return

        # Permission gate first — cheap and may short-circuit.
        if self._can_use_tool is not None:
            try:
                allowed, reason = await self._can_use_tool(tool, block.input, {})
            except Exception as e:  # noqa: BLE001 — caller code must not crash us
                _LOG.exception("can_use_tool raised")
                self._results[idx] = ToolResultBlock(
                    tool_use_id=block.id,
                    content=f"permission check error: {e!r}",
                    is_error=True,
                )
                return
            if not allowed:
                self._results[idx] = ToolResultBlock(
                    tool_use_id=block.id,
                    content=f"permission denied: {reason or 'no reason given'}",
                    is_error=True,
                )
                return

        safe = _is_safe(tool, block.input)
        try:
            if safe:
                async with self._sem:
                    if self._aborted:
                        self._results[idx] = self._cancelled_result(block)
                        return
                    await self._invoke(idx, tool, block)
            else:
                async with self._serial_lock:
                    if self._aborted:
                        self._results[idx] = self._cancelled_result(block)
                        return
                    await self._invoke(idx, tool, block)
        except anyio.get_cancelled_exc_class():
            # Cooperatively cancelled by sibling abort. Surface as error.
            if self._results[idx] is None:
                self._results[idx] = self._cancelled_result(block)
            raise  # propagate so the TG records the cancellation

    async def _invoke(self, idx: int, tool: Tool, block: ToolUseBlock) -> None:
        """Run the tool body with optional timeout. Captures every exception
        and stores a ``ToolResultBlock``. May trigger sibling abort."""
        timeout = getattr(tool, "timeout_s", None)
        try:
            if timeout is not None:
                with anyio.fail_after(timeout):
                    out = await tool.fn(**block.input)
            else:
                out = await tool.fn(**block.input)
        except TimeoutError:
            self._results[idx] = ToolResultBlock(
                tool_use_id=block.id,
                content=f"tool {tool.name!r} timed out after {timeout}s",
                is_error=True,
            )
            self._maybe_abort(tool)
            return
        except anyio.get_cancelled_exc_class():
            # Bubbled up — let _run_one handle it.
            raise
        except Exception as e:  # noqa: BLE001 — user code must not crash us
            err = ToolExecutionError(tool.name, block.id, e)
            self._results[idx] = ToolResultBlock(
                tool_use_id=block.id,
                content=str(err),
                is_error=True,
            )
            self._maybe_abort(tool)
            return

        self._results[idx] = ToolResultBlock(
            tool_use_id=block.id,
            content=_stringify(out),
        )

    def _maybe_abort(self, tool: Tool) -> None:
        """Trigger sibling-abort if this tool requested it.

        Cancels every per-task ``CancelScope`` so in-flight peers die fast
        (this is the whole point — kill subprocesses, abort file writes).
        Already-recorded results are kept; the rest become cancellation
        error blocks. Subsequent ``add_tool_call`` invocations short-circuit
        via the ``_aborted`` flag."""
        if not getattr(tool, "abort_siblings_on_error", False):
            return
        if self._aborted:
            return
        self._aborted = True
        # Cancel every in-flight task; their scope's cancelled_caught path
        # will write a cancelled result block.
        for sc in self._scopes:
            sc.cancel()

    @staticmethod
    def _cancelled_result(block: ToolUseBlock) -> ToolResultBlock:
        return ToolResultBlock(
            tool_use_id=block.id,
            content="aborted by sibling tool error",
            is_error=True,
        )
