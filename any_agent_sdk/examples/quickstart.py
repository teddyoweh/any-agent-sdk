"""Minimal end-to-end example.

Run with a local Ollama (default — pull a model first: ``ollama pull qwen2.5:7b``):

    python -m any_agent_sdk.examples.quickstart

Or point at Together / Fireworks / vLLM with env vars:

    ANY_AGENT_BASE_URL=https://api.together.xyz/v1 \\
    ANY_AGENT_API_KEY=$TOGETHER_API_KEY \\
    ANY_AGENT_MODEL=qwen2.5-72b-instruct \\
    python -m any_agent_sdk.examples.quickstart
"""

from __future__ import annotations

import asyncio
import os

from any_agent_sdk import Agent, UserMessage, tool


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


async def main() -> None:
    model = os.environ.get("ANY_AGENT_MODEL", "qwen2.5-7b-instruct")
    backend = os.environ.get("ANY_AGENT_BASE_URL", "http://localhost:11434")

    async with Agent(
        model=model,
        backend=backend,
        system="You are a concise weather assistant. Use the get_weather tool when asked about weather.",
        tools=[get_weather],
        max_tokens=512,
        max_turns=5,
    ) as agent:
        messages = await agent.run(
            [UserMessage(content="What's the weather in San Francisco?")]
        )
        final = messages[-1]
        # Print each text block of the final assistant message.
        for block in final.content:
            text = getattr(block, "text", None)
            if text:
                print(text)


if __name__ == "__main__":
    asyncio.run(main())
