"""Minimal end-to-end example.

Run with:

    ANTHROPIC_API_KEY=sk-... python -m any_agent_sdk.examples.quickstart

Demonstrates:
  * Defining a tool with @tool
  * Building an Agent (provider auto-detected from model name)
  * Multi-turn run() with tool dispatch
  * Streaming variant for token-by-token UI rendering
"""

from __future__ import annotations

import asyncio

from any_agent_sdk import (
    Agent,
    TextDelta,
    UserMessage,
    tool,
)
from any_agent_sdk.events import ContentBlockDelta
from any_agent_sdk.tools import ToolRegistry


# ---------------------------------------------------------------------------
# A tiny tool
# ---------------------------------------------------------------------------


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns a one-line summary."""
    # Real impl would hit an API. Stubbed here.
    return f"{city}: 67°F, partly cloudy, wind 8 mph NW"


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


async def run_blocking() -> None:
    registry = ToolRegistry()
    registry.add(get_weather)

    agent = Agent(
        model="claude-opus-4-7",
        system="You are a concise weather assistant. Use the tool when asked.",
        tools=registry,
        max_tokens=512,
    )
    try:
        messages = await agent.run(
            [UserMessage(content="What's the weather in San Francisco?")]
        )
        final = messages[-1]
        # The last message is an AssistantMessage; print its text blocks.
        for block in final.content:
            if hasattr(block, "text"):
                print(block.text)
    finally:
        await agent.aclose()


async def run_streaming() -> None:
    """Same task, but render tokens as they arrive."""

    agent = Agent(
        model="claude-opus-4-7",
        system="Reply in one short sentence.",
        max_tokens=128,
    )
    try:
        messages = [UserMessage(content="Say hi in five words.")]
        async for ev in agent.stream(messages):
            if isinstance(ev, ContentBlockDelta) and isinstance(ev.delta, TextDelta):
                print(ev.delta.text, end="", flush=True)
        print()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(run_blocking())
    asyncio.run(run_streaming())
