"""Multi-agent example — parent + research sub-agent.

A small ``research_agent`` is exposed to the parent as a tool via
``as_subagent_tool`` (from :mod:`any_agent_sdk.subagent`, built by a sibling
module). The parent decides when to delegate; the sub-agent runs with its own
budget and tool set.

Run::

    python -m any_agent_sdk.examples.multi_agent_research

If the ``subagent`` module hasn't landed in your checkout yet, this example
prints a friendly message and exits cleanly — it is wired up to gracefully
degrade rather than crash on import.
"""

from __future__ import annotations

import asyncio

from any_agent_sdk import Agent, UserMessage, tool
from any_agent_sdk.providers.openai_compat import OpenAICompatProvider
from any_agent_sdk.tools import ToolRegistry


@tool
async def fetch_url(url: str) -> str:
    """Stub fetch — returns a short summary instead of a real GET."""

    return f"<fetched {url}: 200 OK, 1.2KB stub body>"


async def main() -> None:
    try:
        from any_agent_sdk.subagent import as_subagent_tool
    except ImportError:
        print("subagent module not present yet — skipping multi-agent example.")
        return

    # The research sub-agent: web-shaped tools, its own system prompt.
    research_registry = ToolRegistry()
    research_registry.add(fetch_url)
    research = Agent(
        model="qwen2.5-7b-instruct",
        provider=OpenAICompatProvider(base_url="http://localhost:8000/v1"),
        tools=research_registry,
        system="You are a research assistant. Fetch URLs and summarize.",
        max_tokens=512,
    )

    # Parent: only knows about the sub-agent as a callable tool.
    parent_registry = ToolRegistry()
    parent_registry.add(
        as_subagent_tool(
            research,
            name="research",
            description="Delegate a research question to a specialist sub-agent.",
        )
    )

    parent = Agent(
        model="qwen2.5-7b-instruct",
        provider=OpenAICompatProvider(base_url="http://localhost:8000/v1"),
        tools=parent_registry,
        system=(
            "You are an orchestrator. For research-heavy questions, call the "
            "'research' tool. For simple questions, answer directly."
        ),
        max_tokens=512,
    )
    try:
        messages = await parent.run(
            [UserMessage(content="What does Spawn Labs build? Use research if needed.")]
        )
        print(messages[-1])
    finally:
        await parent.aclose()
        await research.aclose()


if __name__ == "__main__":
    asyncio.run(main())
