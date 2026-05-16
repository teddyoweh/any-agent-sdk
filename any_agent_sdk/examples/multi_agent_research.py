"""Multi-agent example — parent uses ``query()``; sub-agent is exposed as a tool.

A small research sub-agent is wrapped via :func:`as_subagent_tool` and
handed to the parent as a regular tool. The parent (driven by ``query()``)
decides when to delegate. The sub-agent runs with its own budget + tool
set and shares the parent's HTTP client pool by default.

Run::

    python -m any_agent_sdk.examples.multi_agent_research

The parent's session transcript persists to
``~/.anyagent/sessions/<session_id>.jsonl`` via ``persist=True``.
"""

from __future__ import annotations

import asyncio
import os

from any_agent_sdk import Agent, query, tool


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

    backend = os.environ.get("ANY_AGENT_BASE_URL", "http://localhost:11434")
    model = os.environ.get("ANY_AGENT_MODEL", "qwen2.5-7b-instruct")

    # The research sub-agent: web-shaped tools, its own system prompt.
    # Sub-agents still use the lower-level Agent class because they're
    # invoked as tools — query() is for top-level conversations.
    research = Agent(
        model=model,
        backend=backend,
        tools=[fetch_url],
        system="You are a research assistant. Fetch URLs and summarize.",
        max_tokens=512,
    )

    research_tool = as_subagent_tool(
        research,
        name="research",
        description="Delegate a research question to a specialist sub-agent.",
    )

    # Parent driven by query() — Claude SDK-compatible.
    async for msg in query(
        prompt="What does Spawn Labs build? Use the research tool if useful.",
        options={
            "model": model,
            "backend": backend,
            "tools": [research_tool],
            "system": (
                "You are an orchestrator. For research-heavy questions, call the "
                "'research' tool. For simple questions, answer directly."
            ),
            "max_tokens": 512,
            "max_turns": 5,
            "persist": True,
        },
    ):
        if msg.type == "assistant":
            for block in msg.message.content:
                if hasattr(block, "text") and block.text:
                    print(f"[parent] {block.text}")
        elif msg.type == "result":
            print(
                f"\n[result] {msg.subtype} · {msg.num_turns} parent turns · "
                f"{msg.duration_ms} ms"
            )

    await research.aclose()


if __name__ == "__main__":
    asyncio.run(main())
