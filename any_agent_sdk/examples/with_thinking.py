"""Reasoning-model example — render thinking and final answer differently.

QwQ-32B-Preview emits inline ``<think>...</think>`` blocks. The streaming
pipeline splits these into ``ThinkingBlock`` deltas before the agent loop
sees them; consumers can render thinking dimmed/collapsed while keeping the
final assistant text loud.

Run::

    python -m any_agent_sdk.examples.with_thinking
"""

from __future__ import annotations

import asyncio
import sys

from any_agent_sdk import Agent, UserMessage
from any_agent_sdk.events import ContentBlockDelta, TextDelta, ThinkingDelta
from any_agent_sdk.providers.openai_compat import OpenAICompatProvider


# ANSI dim/normal — keep the thinking visually distinct without depending on a
# real TUI library. Falls back gracefully on terminals that don't support it.
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


async def main() -> None:
    agent = Agent(
        model="qwq-32b-preview",
        provider=OpenAICompatProvider(base_url="http://localhost:8000/v1"),
        system="Think carefully, then answer.",
        max_tokens=1024,
        temperature=0.6,
    )
    try:
        messages = [
            UserMessage(content="A train leaves at 9 going 60 mph. Where is it at noon?")
        ]
        in_thinking = False
        async for ev in agent.stream(messages):
            if not isinstance(ev, ContentBlockDelta):
                continue
            d = ev.delta
            if isinstance(d, ThinkingDelta):
                if not in_thinking:
                    sys.stdout.write(_DIM + "[thinking] ")
                    in_thinking = True
                sys.stdout.write(d.thinking)
            elif isinstance(d, TextDelta):
                if in_thinking:
                    sys.stdout.write(_RESET + "\n")
                    in_thinking = False
                sys.stdout.write(d.text)
            sys.stdout.flush()
        if in_thinking:
            sys.stdout.write(_RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
