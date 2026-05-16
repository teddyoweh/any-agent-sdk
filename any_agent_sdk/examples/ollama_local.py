"""Local Ollama example.

Prereqs::

    ollama pull qwen2.5:7b-instruct
    ollama serve   # default: http://localhost:11434

Run::

    python -m any_agent_sdk.examples.ollama_local

Demonstrates: ``Agent`` against a fully-local backend, with one tool the model
will call before answering.
"""

from __future__ import annotations

import asyncio

from any_agent_sdk import Agent, UserMessage, tool
from any_agent_sdk.providers.ollama import OllamaProvider
from any_agent_sdk.tools import ToolRegistry


@tool
async def get_weather(city: str) -> str:
    """Return a one-line weather summary for the given city."""

    # Stubbed — a real implementation would call a weather API. We hard-code so
    # the example runs offline.
    return f"{city}: 67°F, partly cloudy."


async def main() -> None:
    registry = ToolRegistry()
    registry.add(get_weather)

    agent = Agent(
        model="qwen2.5-7b-instruct",
        provider=OllamaProvider(base_url="http://localhost:11434"),
        tools=registry,
        system="Reply in one sentence.",
        max_tokens=256,
    )
    try:
        messages = await agent.run(
            [UserMessage(content="Weather in SF?")],
        )
        # The final assistant message is what the user would see.
        print(messages[-1])
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
