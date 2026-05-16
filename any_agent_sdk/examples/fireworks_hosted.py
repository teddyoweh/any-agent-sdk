"""Hosted Fireworks example — DeepSeek-V3.

Prereqs::

    export FIREWORKS_API_KEY=fw_...

Run::

    python -m any_agent_sdk.examples.fireworks_hosted

Fireworks speaks OpenAI-compat, so we use the same adapter as vLLM with a
different base URL. DeepSeek-V3 has native tool calling, so this falls on
Path A and the prompt-engineered fallback parser never fires.
"""

from __future__ import annotations

import asyncio
import os

from any_agent_sdk import Agent, UserMessage, tool
from any_agent_sdk.providers.openai_compat import OpenAICompatProvider
from any_agent_sdk.tools import ToolRegistry


@tool
async def lookup_company(name: str) -> str:
    """Return a one-line description of a company by name."""

    return f"{name}: stub description; wire this to a real API."


async def main() -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY before running this example.")

    registry = ToolRegistry()
    registry.add(lookup_company)

    agent = Agent(
        model="accounts/fireworks/models/deepseek-v3",
        provider=OpenAICompatProvider(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
        ),
        tools=registry,
        system="You are a research assistant. Use the tool when asked about companies.",
        max_tokens=512,
    )
    try:
        messages = await agent.run(
            [UserMessage(content="Tell me about Spawn Labs in one sentence.")]
        )
        print(messages[-1])
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
