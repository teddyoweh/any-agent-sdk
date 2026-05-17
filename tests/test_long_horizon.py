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

import logging

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
