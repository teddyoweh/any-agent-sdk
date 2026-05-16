"""Research agent — give a model two built-in tools (WebSearch + WebFetch)
and ask it to research a subject.

Uses the SDK's built-in tools that are 1:1 with Claude Code's named tools.
When ``EXA_API_KEY`` is set (recommended), search routes through Exa
(neural, real URLs, ~$0.005/query). Without a key, falls back to scraping
DuckDuckGo HTML.

Run with local Llama 3.2 on Ollama (the acceptance-test path):

    ollama serve &
    ollama pull llama3.2:3b
    export EXA_API_KEY=...      # optional; better quality
    python -m any_agent_sdk.examples.research_agent "Subject Name"

Or any hosted backend:

    ANY_AGENT_BASE_URL=https://api.together.xyz/v1 \\
    ANY_AGENT_API_KEY=$TOGETHER_API_KEY \\
    ANY_AGENT_MODEL=qwen2.5-72b-instruct \\
    python -m any_agent_sdk.examples.research_agent "Subject Name"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from any_agent_sdk import (
    Agent,
    AssistantMessage,
    TextBlock,
    UserMessage,
    WebFetch,
    WebSearch,
)
from any_agent_sdk.hooks import HookContext, HookResult, Hooks


logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Live transcript hooks
# ---------------------------------------------------------------------------


async def trace_pre(ctx: HookContext) -> HookResult:
    print(f"\n  🔧 {ctx.tool.name}({_short(ctx.input)})", flush=True)
    return HookResult()


async def trace_post(ctx: HookContext) -> HookResult:
    body = _short(ctx.output, 220)
    print(f"  ← {len(str(ctx.output))} chars · {body}", flush=True)
    return HookResult()


def _short(obj: Any, n: int = 120) -> str:
    s = str(obj).replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


# ---------------------------------------------------------------------------
# Research loop
# ---------------------------------------------------------------------------


async def research(
    subject: str,
    model: str | None = None,
    backend: str | None = None,
    max_turns: int = 8,
) -> str:
    model = model or os.environ.get("ANY_AGENT_MODEL", "llama3.2:3b")
    backend = backend or os.environ.get("ANY_AGENT_BASE_URL", "http://localhost:11434")

    provider = "Exa" if os.environ.get("EXA_API_KEY") else (
        "Brave" if os.environ.get("BRAVE_API_KEY") else (
            "Tavily" if os.environ.get("TAVILY_API_KEY") else "DuckDuckGo (fallback)"
        )
    )
    print(f"\n=== Researching {subject!r} ===")
    print(f"    model:   {model}")
    print(f"    backend: {backend}")
    print(f"    search:  {provider}")

    agent = Agent(
        model=model,
        backend=backend,
        system=(
            "You are a research assistant. The user names a subject — a "
            "person, project, or topic. You have two tools:\n"
            "  - WebSearch(query: str) — returns ranked results as "
            "    TITLE — URL — SNIPPET lines. Pass ONLY the `query` argument; "
            "    do NOT pass allowed_domains or blocked_domains unless the "
            "    user explicitly asked for a domain filter.\n"
            "  - WebFetch(url: str) — fetches a URL and returns its text. "
            "    Pass ONLY the `url` argument.\n\n"
            "Process:\n"
            "  1. Call WebSearch with the subject name. Read every result line.\n"
            "  2. If results look thin or off-topic, refine the query with an "
            "     additional keyword (e.g. 'subject X profession' or 'subject "
            "     site:github.com'). Search again.\n"
            "  3. Pick the 1-2 most relevant URLs and WebFetch them.\n"
            "  4. Write a factual 3-6 sentence summary citing the URLs you "
            "     ACTUALLY fetched. Never invent facts. Never fabricate tool "
            "     results — wait for the real tool output.\n"
        ),
        tools=[WebSearch, WebFetch],
        max_tokens=512,
        max_turns=max_turns,
        temperature=0.2,
        hooks=Hooks(pre_tool_use=trace_pre, post_tool_use=trace_post),
    )

    try:
        msgs = await agent.run(
            [UserMessage(content=f"Research {subject!r} and summarize what you find.")]
        )
    finally:
        await agent.aclose()

    final = msgs[-1]
    if isinstance(final, AssistantMessage):
        return "".join(b.text for b in final.content if isinstance(b, TextBlock))
    return f"(no final assistant text; last message type: {type(final).__name__})"


def main() -> None:
    subject = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Teddy Oweh"
    summary = asyncio.run(research(subject))
    print("\n\n=== FINAL SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()
