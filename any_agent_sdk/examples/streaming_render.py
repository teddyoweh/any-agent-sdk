"""Streaming render example.

Demonstrates ``Agent.stream`` — token-by-token output. Useful as a starting
point for any UI that wants to render the assistant's text as it arrives
(TUIs, web sockets, etc.).

Run::

    python -m any_agent_sdk.examples.streaming_render
"""

from __future__ import annotations

import asyncio
import sys

from any_agent_sdk import Agent, UserMessage
from any_agent_sdk.events import ContentBlockDelta, TextDelta
from any_agent_sdk.providers.openai_compat import OpenAICompatProvider


async def main() -> None:
    agent = Agent(
        model="qwen2.5-7b-instruct",
        provider=OpenAICompatProvider(base_url="http://localhost:8000/v1"),
        system="Reply in one short paragraph.",
        max_tokens=200,
    )
    try:
        messages = [UserMessage(content="Tell me one interesting fact about the moon.")]
        # Stream the next assistant turn. We use carriage-return + flush to
        # paint each new chunk on top of the same line; for multi-line output
        # you'd want ``end=""`` and let the terminal scroll.
        async for ev in agent.stream(messages):
            if isinstance(ev, ContentBlockDelta) and isinstance(ev.delta, TextDelta):
                sys.stdout.write(ev.delta.text)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
