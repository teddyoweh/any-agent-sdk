"""Research agent — give Llama 3.2 3B running on local Ollama two tools
(``web_search`` and ``fetch_url``) and ask it to research a subject.

This is the closest thing to a real end-to-end smoke test of the SDK with
an OSS model. It exercises:

  * Multi-turn agent loop (assistant → tool → assistant → tool → ...)
  * Native tool calling on a small open model (Llama 3.2 3B)
  * Real HTTP I/O in the tools (DuckDuckGo search + URL fetch)
  * Hook + permission + budget hot paths

Run:

    ollama serve &
    ollama pull llama3.2:3b
    python -m any_agent_sdk.examples.research_agent "Subject Name"

The transcript is printed as the agent works.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any
from urllib.parse import quote_plus

import httpx

# Lazy: BeautifulSoup is only used inside the tools.
from any_agent_sdk import (
    Agent,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    tool,
)
from any_agent_sdk.hooks import HookContext, HookResult, Hooks


logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


# A single shared HTTP client for both tools — cheaper than per-call.
_HTTP = httpx.AsyncClient(
    headers={"User-Agent": "any-agent-sdk-research/0.1 (+github.com/teddyoweh/any-agent-sdk)"},
    timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=10.0),
    follow_redirects=True,
)


@tool(is_read_only=True)
async def web_search(query: str) -> str:
    """Search the web for a query. Returns top results as plain text.

    Each result is one line: ``TITLE — URL — SNIPPET``. Use this to find
    candidate URLs to fetch with fetch_url for details.
    """

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        r = await _HTTP.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"search error: {e!r}"

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(r.text, "lxml")
    rows: list[str] = []
    for result in soup.select("div.result")[:8]:
        title_el = result.select_one("a.result__a")
        snippet_el = result.select_one("a.result__snippet, .result__snippet")
        if title_el is None:
            continue
        title = title_el.get_text(strip=True)
        href = title_el.get("href", "")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        rows.append(f"{title} — {href} — {snippet[:200]}")

    if not rows:
        return "no results"
    return "\n".join(rows)


@tool(is_read_only=True, timeout_s=20.0)
async def fetch_url(url: str) -> str:
    """Fetch a URL and return its readable text content.

    Truncated to 2500 characters. Strips HTML markup. Use this after
    web_search to read a specific result.
    """

    try:
        r = await _HTTP.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"fetch error: {e!r}"

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(r.text, "lxml")
    # Strip script + style noise.
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return text[:2500] if text else "(empty page)"


# ---------------------------------------------------------------------------
# Live transcript hook
# ---------------------------------------------------------------------------


async def trace_pre(ctx: HookContext) -> HookResult:
    """Show every tool call as the agent makes it."""

    print(f"\n  🔧 tool call: {ctx.tool.name}({_short(ctx.input)})", flush=True)
    return HookResult()


async def trace_post(ctx: HookContext) -> HookResult:
    """Show every tool result."""

    print(f"  ← result ({len(str(ctx.output))} chars): {_short(ctx.output, 220)}", flush=True)
    return HookResult()


def _short(obj: Any, n: int = 120) -> str:
    s = str(obj).replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def research(subject: str, model: str = "llama3.2:3b", max_turns: int = 8) -> str:
    """Run a multi-turn research loop on ``subject``. Returns the final
    assistant message text."""

    print(f"\n=== Researching '{subject}' with {model} on local Ollama ===")

    agent = Agent(
        model=model,
        backend="http://localhost:11434",
        system=(
            "You are a research assistant. The user will name a subject — "
            "a person, project, or topic. You have two tools: web_search "
            "(returns search results) and fetch_url (returns a webpage's "
            "text). USE THE TOOLS. Do NOT make up facts. Search first, "
            "then fetch the most relevant 1-2 URLs to dig deeper. After "
            "gathering enough information, write a short factual summary "
            "(3-6 sentences) of who/what they are, based ONLY on what the "
            "tools returned. Cite the source URLs you used."
        ),
        tools=[web_search, fetch_url],
        max_tokens=512,
        max_turns=max_turns,
        temperature=0.2,
        hooks=Hooks(pre_tool_use=trace_pre, post_tool_use=trace_post),
    )

    try:
        msgs = await agent.run(
            [UserMessage(content=f"Please research {subject!r} and summarize what you find.")]
        )
    finally:
        await agent.aclose()
        await _HTTP.aclose()

    # Stitch the final assistant text.
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
