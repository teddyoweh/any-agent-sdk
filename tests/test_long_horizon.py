"""Long-horizon agent loop tests — 100+ turns without a hard cap.

Verifies the agent loop can drive a multi-hundred-turn conversation
cleanly when ``max_steps`` is unset (the new default). Pre-PR default was
``max_steps=20``, which silently truncated any long-running agent at turn
20 with a warning log instead of raising — a serious foot-gun for users
following the Claude Code SDK mental model where long horizons are the
common case, not the exception.

Coverage:

  * Default ``max_steps`` is ``None`` (unlimited) — surface check on the
    Agent dataclass field.
  * 120-turn counter-increment loop drives the full ``run()`` path. The
    test ensures conversation length, message ordering, and tool-result
    pairing stay consistent across many iterations.
  * Same 120-turn task driven through ``run_iter()`` — exercises the
    streaming-yield path independently.
  * ``max_turns=10`` back-compat: explicit ceilings still cap the loop
    and surface the "hit ceiling without natural stop" warning.
  * ``max_turns=0`` edge case: zero-turn loop returns immediately
    without invoking the provider.
  * Memory-shape sanity: messages list grows linearly, no duplicated
    refs or dropped tool-result messages.

The tests use a deterministic ``CounterMock`` so there's no network or
LLM-flakiness — the loop's *infrastructure* is what's under test, not
any particular model's compliance.
"""

from __future__ import annotations

import gc
import logging
import tracemalloc

import anyio
import pytest

from any_agent_sdk import (
    Agent,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    tool,
)
from any_agent_sdk.events import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    TextDelta,
)
from any_agent_sdk.providers.mock import MockProvider
from any_agent_sdk.types import Usage


# ---------------------------------------------------------------------------
# Helpers (mirror the conventions in test_run_loop_integration.py so the
# event-stream shape stays consistent across tests).
# ---------------------------------------------------------------------------


def _msg(model: str = "mock-7b") -> list:
    return [MessageStart(message_id="mock-long", model=model)]


def _text_block(idx: int, text: str) -> list:
    return [
        ContentBlockStart(index=idx, block=TextBlock(text="")),
        ContentBlockDelta(index=idx, delta=TextDelta(text=text)),
        ContentBlockStop(index=idx),
    ]


def _tool_use_block(idx: int, call_id: str, name: str, input_json: str) -> list:
    return [
        ContentBlockStart(
            index=idx,
            block=ToolUseBlock(id=call_id, name=name, input={}),
        ),
        ContentBlockDelta(
            index=idx, delta=InputJsonDelta(partial_json=input_json)
        ),
        ContentBlockStop(index=idx),
    ]


def _stop(stop_reason: str = "end_turn", usage: Usage | None = None) -> list:
    return [
        MessageDelta(
            stop_reason=stop_reason,
            usage=usage or Usage(input_tokens=1, output_tokens=1),
        ),
        MessageStop(),
    ]


# ---------------------------------------------------------------------------
# The long-horizon test tool + mock provider.
# ---------------------------------------------------------------------------


@tool
async def increment_counter(by: int) -> str:
    """Increment a counter by N. (Body irrelevant — the test just needs
    a tool the model can call so we exercise the tool-dispatch loop.)"""

    return f"+{by}"


class CounterMock(MockProvider):
    """Drives a deterministic N-turn loop.

    On call ``k < target``: emit a single ``increment_counter`` tool_use
    with stop_reason=``tool_use`` → agent dispatches the tool, appends
    the result, and re-enters the loop.

    On call ``k == target``: emit a final text turn with stop_reason=
    ``end_turn`` → agent's Stop hook fires and the loop exits cleanly.

    Each tool_use gets a unique call id so the streaming executor's
    insertion-order bookkeeping stays valid across hundreds of turns.
    """

    def __init__(self, target: int) -> None:
        super().__init__()
        self._call = 0
        self._target = target

    async def stream(self, **kw):  # type: ignore[override]
        self._call += 1
        if self._call < self._target:
            script = (
                _msg()
                + _tool_use_block(
                    idx=0,
                    call_id=f"c{self._call}",
                    name="increment_counter",
                    input_json='{"by": 1}',
                )
                + _stop(stop_reason="tool_use")
            )
        else:
            script = (
                _msg()
                + _text_block(0, f"Completed {self._target} turns.")
                + _stop(stop_reason="end_turn")
            )
        for ev in script:
            yield ev


# ---------------------------------------------------------------------------
# Surface check — the default really is unlimited.
# ---------------------------------------------------------------------------


def test_default_max_steps_is_unlimited():
    """``Agent(model=...)`` with nothing else should expose
    ``max_steps is None`` — the agent has no implicit turn ceiling, only
    the budget tracker (if configured) or a caller-supplied ``max_turns``
    can stop the loop short of natural-stop."""

    agent = Agent(model="mock-7b", provider=MockProvider())
    try:
        assert agent.max_steps is None
        assert agent.max_turns is None
    finally:
        # No coroutine state was created — aclose is still safe.
        anyio.run(agent.aclose)


# ---------------------------------------------------------------------------
# The headline test: 120 turns through run().
# ---------------------------------------------------------------------------


_TARGET_TURNS = 120  # > 100, well past the pre-PR 20-turn cap


def test_long_horizon_120_turns_via_run():
    """Drive the agent through 120 turns. The pre-PR default would have
    silently truncated this at turn 20 (logged a warning, returned a
    half-finished conversation). New default of ``max_steps=None`` should
    run the full 120 and natural-stop."""

    async def main():
        agent = Agent(
            model="mock-7b",
            provider=CounterMock(_TARGET_TURNS),
            tools=[increment_counter],
            # No max_turns / max_steps — relies on the new unlimited default.
        )
        try:
            msgs = await agent.run([UserMessage(content="count for me")])
        finally:
            await agent.aclose()

        assistant_msgs = [m for m in msgs if isinstance(m, AssistantMessage)]
        assert len(assistant_msgs) == _TARGET_TURNS, (
            f"expected {_TARGET_TURNS} assistant turns, got {len(assistant_msgs)}"
        )

        # Each non-final assistant message must carry exactly one tool_use;
        # the final one must natural-stop with text.
        for i, am in enumerate(assistant_msgs[:-1]):
            tool_uses = [b for b in am.content if isinstance(b, ToolUseBlock)]
            assert len(tool_uses) == 1, f"turn {i}: expected 1 tool_use, got {len(tool_uses)}"
            assert tool_uses[0].name == "increment_counter"
            assert tool_uses[0].id == f"c{i + 1}"

        last = assistant_msgs[-1]
        last_text = "".join(b.text for b in last.content if isinstance(b, TextBlock))
        assert f"Completed {_TARGET_TURNS} turns." in last_text
        assert not [b for b in last.content if isinstance(b, ToolUseBlock)]

        # Conversation shape:
        #   1 user prompt
        # + TARGET assistant messages
        # + (TARGET - 1) tool-result user messages (one per tool turn)
        # = 2 * TARGET
        assert len(msgs) == 2 * _TARGET_TURNS, (
            f"expected {2 * _TARGET_TURNS} total messages, got {len(msgs)}"
        )

        # Every assistant tool_use must be immediately followed by a
        # UserMessage containing the matching tool_result. This is the
        # canonical pairing the streaming executor guarantees.
        for i, am in enumerate(assistant_msgs[:-1]):
            am_idx = msgs.index(am)
            nxt = msgs[am_idx + 1]
            assert isinstance(nxt, UserMessage)
            assert len(nxt.content) == 1
            result = nxt.content[0]
            assert result.tool_use_id == f"c{i + 1}"
            assert not result.is_error
            assert result.content == "+1"

    anyio.run(main)


# ---------------------------------------------------------------------------
# Streaming-mode coverage — same task, different consumer surface.
# ---------------------------------------------------------------------------


def test_long_horizon_120_turns_via_run_iter():
    """``run_iter()`` yields each turn as it finalizes. Drain 120 of them
    and verify the cumulative shape matches what ``run()`` produced."""

    async def main():
        agent = Agent(
            model="mock-7b",
            provider=CounterMock(_TARGET_TURNS),
            tools=[increment_counter],
        )

        messages: list = [UserMessage(content="count for me")]
        assistant_count = 0
        tool_result_count = 0
        try:
            async for yielded in agent.run_iter(messages):
                if isinstance(yielded, AssistantMessage):
                    assistant_count += 1
                elif isinstance(yielded, UserMessage):
                    # The meta user-context message is allowed; tool-result
                    # user messages count for the tool-result tally.
                    if getattr(yielded, "isMeta", False):
                        continue
                    tool_result_count += 1
        finally:
            await agent.aclose()

        assert assistant_count == _TARGET_TURNS
        # One tool-result message per tool-using turn; the final turn is
        # text-only so it doesn't produce one.
        assert tool_result_count == _TARGET_TURNS - 1

    anyio.run(main)


# ---------------------------------------------------------------------------
# Back-compat: explicit ceiling still works.
# ---------------------------------------------------------------------------


def test_explicit_max_turns_caps_loop(caplog):
    """Passing ``max_turns=10`` on a task that would run 120 turns should
    truncate at turn 10 and emit the "hit ceiling without natural stop"
    warning. Confirms we didn't break callers that *want* a cap."""

    async def main():
        agent = Agent(
            model="mock-7b",
            provider=CounterMock(_TARGET_TURNS),
            tools=[increment_counter],
            max_turns=10,
        )
        try:
            msgs = await agent.run([UserMessage(content="count")])
        finally:
            await agent.aclose()

        assistant_msgs = [m for m in msgs if isinstance(m, AssistantMessage)]
        # Loop ran exactly 10 turns; counter mock would have kept going.
        assert len(assistant_msgs) == 10
        # All 10 are tool-calling turns — natural stop never reached.
        for am in assistant_msgs:
            assert any(isinstance(b, ToolUseBlock) for b in am.content)

    with caplog.at_level(logging.WARNING, logger="any_agent_sdk.agent"):
        anyio.run(main)

    assert any(
        "hit max_steps" in rec.message for rec in caplog.records
    ), "expected the 'hit max_steps without natural stop' warning"


# ---------------------------------------------------------------------------
# Edge case: zero turns means "don't loop at all".
# ---------------------------------------------------------------------------


def test_max_turns_zero_no_provider_call():
    """``max_turns=0`` should skip the provider entirely and return the
    input messages untouched. A real foot-gun if it ever silently looped
    once."""

    call_count = {"n": 0}

    class CountingMock(MockProvider):
        async def stream(self, **kw):  # type: ignore[override]
            call_count["n"] += 1
            async for ev in super().stream(**kw):
                yield ev

    async def main():
        agent = Agent(
            model="mock-7b",
            provider=CountingMock(),
            tools=[increment_counter],
            max_turns=0,
        )
        try:
            msgs = await agent.run([UserMessage(content="hi")])
        finally:
            await agent.aclose()

        # Just the original user message back (possibly preceded by a
        # meta user-context message if memory injection fired).
        non_meta = [m for m in msgs if not getattr(m, "isMeta", False)]
        assert len(non_meta) == 1
        assert isinstance(non_meta[0], UserMessage)
        assert call_count["n"] == 0

    anyio.run(main)


# ---------------------------------------------------------------------------
# Memory-growth leak detector.
# ---------------------------------------------------------------------------


# Per-turn allocation ceiling. The agent loop *legitimately* allocates per
# turn (one new AssistantMessage + ~one UserMessage with a ToolResultBlock
# + transient stream events + executor task state). On CPython 3.12 this
# settles around 2-6 KB / turn on the CounterMock script — well under the
# ceiling. The ceiling exists to fail loudly if a future change starts
# retaining transient per-turn state (closed-stream events, finished
# executor results, dead hook contexts, etc.) inside the agent or its
# subsystems. Bump the ceiling only if a *deliberate* change widens the
# per-turn working set and the new value is still O(1) in turn count.
_PER_TURN_BYTES_CEILING = 32 * 1024  # 32 KB / turn

# We measure across the *second half* of the run, after JIT-like warmups
# (anyio task scheduler, msgspec decoder caches, logger handler buffers)
# have settled. Comparing the per-turn delta across two windows in the
# stable region is what actually catches O(n^2) growth — comparing
# absolute total memory at one snapshot doesn't distinguish "constant
# per-turn cost" from "growing per-turn cost".
_LEAK_TURNS = 120
_WARMUP_TURNS = 20  # ignore the first N turns; only assert on steady state


def test_per_turn_memory_growth_is_bounded():
    """Drive a 120-turn loop and snapshot allocations every 20 turns. The
    per-turn average across the warmed-up window must stay under
    ``_PER_TURN_BYTES_CEILING`` AND must not be drifting upward turn over
    turn — both signals of a real leak.

    Why this test exists
    --------------------
    The pre-PR ``max_steps=20`` cap meant nobody actually ran the loop
    long enough for accumulating per-turn state to matter. Now that the
    default is unlimited, anyone using the agent for a real long-horizon
    task (research swarms, multi-day pipelines, autonomous bounty
    hunters) needs assurance that turn 1,000 doesn't cost 50× turn 1 in
    RAM. This test fails loudly the moment a regression starts retaining
    transient per-turn state — closed stream events, finished executor
    coroutines, dead hook contexts, etc.

    What we measure
    ---------------
    ``tracemalloc.get_traced_memory()[0]`` between turn checkpoints. We
    compare the average per-turn growth in two contiguous windows of the
    steady-state region. If the second window's per-turn growth is more
    than 2× the first window's, the loop is allocating more per turn the
    longer it runs — a leak. Some noise is expected (anyio task pool
    churn, dict resizing), hence the 2× headroom.
    """

    async def main():
        # Force a clean baseline so unrelated allocations from other tests
        # don't show up in the windows we compare.
        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        agent = Agent(
            model="mock-7b",
            provider=CounterMock(_LEAK_TURNS),
            tools=[increment_counter],
        )

        messages: list = [UserMessage(content="count")]
        # (turn_number, traced_memory_bytes) checkpoints every 20 turns.
        # Captured AFTER gc.collect so we measure live retained bytes,
        # not transient cycles waiting to be reclaimed.
        snapshots: list[tuple[int, int]] = []
        turn = 0

        try:
            async for yielded in agent.run_iter(messages):
                if isinstance(yielded, AssistantMessage):
                    turn += 1
                    if turn % 20 == 0:
                        gc.collect()
                        snapshots.append((turn, tracemalloc.get_traced_memory()[0]))
        finally:
            await agent.aclose()
            tracemalloc.stop()

        assert turn == _LEAK_TURNS, f"expected {_LEAK_TURNS} turns, got {turn}"
        # Six checkpoints: 20, 40, 60, 80, 100, 120.
        assert len(snapshots) == _LEAK_TURNS // 20

        # Sanity: agent's own internal retention should be empty / minimal
        # after the run. These are the obvious leak buckets to audit
        # first if this test ever fires in CI.
        assert agent._permission_denials == [], (
            "permission_denials accumulated despite no denials: "
            f"{agent._permission_denials}"
        )
        assert not agent.cancellation_signal.is_set(), (
            "cancellation_signal fired unexpectedly during clean run"
        )

        # Conversation list grew linearly: every assistant turn produces
        # 1 (final) or 2 (tool-using) messages. For our script that's
        # 2 * _LEAK_TURNS total + 1 original user prompt - 1 (the final
        # turn has no tool_result), or simply 2 * _LEAK_TURNS.
        non_meta = [m for m in messages if not getattr(m, "isMeta", False)]
        assert len(non_meta) == 2 * _LEAK_TURNS, (
            f"messages list shape wrong: got {len(non_meta)}, expected {2 * _LEAK_TURNS}"
        )

        # ------------------------------------------------------------------
        # Memory growth analysis — the actual leak check.
        # ------------------------------------------------------------------
        warmup_idx = _WARMUP_TURNS // 20  # snapshot index where warmup ends
        # First post-warmup window: snapshots[warmup_idx] → middle
        # Second window: middle → end
        post_warmup = snapshots[warmup_idx:]
        mid = len(post_warmup) // 2
        assert mid >= 1, (
            "not enough post-warmup snapshots to split into two windows — "
            "increase _LEAK_TURNS or lower _WARMUP_TURNS"
        )

        def per_turn_growth(window: list[tuple[int, int]]) -> float:
            """Bytes allocated per turn across the window, averaged."""

            turns_span = window[-1][0] - window[0][0]
            bytes_span = window[-1][1] - window[0][1]
            return bytes_span / turns_span

        first_window = post_warmup[: mid + 1]
        second_window = post_warmup[mid:]
        first = per_turn_growth(first_window)
        second = per_turn_growth(second_window)

        # Absolute ceiling — even the first window must be sane.
        assert first < _PER_TURN_BYTES_CEILING, (
            f"per-turn allocation already over ceiling in first window: "
            f"{first:.0f} B/turn > {_PER_TURN_BYTES_CEILING} B/turn"
        )
        assert second < _PER_TURN_BYTES_CEILING, (
            f"per-turn allocation over ceiling in second window: "
            f"{second:.0f} B/turn > {_PER_TURN_BYTES_CEILING} B/turn"
        )

        # The leak signal — second window must not be meaningfully larger
        # than the first. 2× headroom for measurement noise + GC timing.
        # On a real leak (O(n^2) retention) second/first balloons quickly
        # because the second window's *turns* are accumulating against an
        # ever-growing baseline.
        ratio = second / max(first, 1.0)  # guard div-by-zero on tiny first
        assert ratio < 2.0, (
            f"per-turn growth drifting upward (leak suspect): "
            f"first window {first:.0f} B/turn, second window {second:.0f} B/turn, "
            f"ratio {ratio:.2f}x (ceiling 2.0x). Full snapshots: {snapshots}"
        )

    anyio.run(main)
