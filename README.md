# any-agent-sdk

**Claude Code for open-source models.** Production-grade agent runtime — streaming, tools, MCP, sub-agents, hooks, permissions, compaction — that runs on Llama, Qwen, DeepSeek, Mixtral, Phi, Gemma, and anything else you serve through Ollama, vLLM, llama.cpp, TGI, Together, Fireworks, Groq, or OpenRouter.

> **Status:** v0 skeleton on `main`. Pivoting to OSS-first per [`docs/plan.md`](docs/plan.md). The Anthropic adapter currently in `providers/anthropic.py` will be **deleted in M0.1** — it was a stand-in to validate the streaming + tool-dispatch path against a known-good wire format.

## Mission

There is no production-grade agent runtime that works across the OSS model + serving matrix. LangGraph and smolagents lean too heavy and miss MCP. llama-stack is too tightly scoped. The Claude Agent SDK and OpenAI Agents SDK are bound to their respective hosted APIs. That's the gap any-agent-sdk fills.

We benchmark against the actual Claude Code source (1,902 TS files we read end-to-end — see `docs/upstream-comparison.md`). The bar is:

- Streaming tool execution (start tools mid-stream, not after the message finalizes)
- 28-event hook system
- Permission system with allow/deny/ask rules per source
- MCP across four transports (stdio, sse, http, in-process)
- Auto-compaction
- Sub-agent orchestration
- Sessions with fork/resume
- Budget tracking with per-model pricing

Plus, the OSS-specific bits:

- **Universal tool use** — Path A (native via OpenAI-compat tools[]) when supported; Path B (prompt-engineered `<tool_call>` XML) when not; Path C (grammar-constrained) when the server can enforce JSON. Capability-table-driven, automatic.
- **Universal thinking** — handles inline `<think>` tags (R1, QwQ, Marco-o1, R1-Distill) and out-of-band thinking blocks (DeepSeek API). Zero cost when the model doesn't emit thinking.
- **Backend agnosticism** — same agent code switches between Ollama at `localhost:11434` and Fireworks at `api.fireworks.ai` with one env var.

## What's on `main` right now

```
any_agent_sdk/
  __init__.py            public surface
  types.py               universal Message + ContentBlock (msgspec, tagged unions)
  events.py              normalized StreamEvent variants
  errors.py              typed exceptions
  http.py                shared httpx.AsyncClient + SSE parser
  agent.py               Agent — multi-turn loop (will be rewritten for streaming dispatch)
  tools.py               @tool, ToolRegistry, parallel dispatcher
  providers/
    base.py              Provider protocol + lazy registry
    anthropic.py         (TO BE DELETED — see plan.md §M0.1)
  examples/
    quickstart.py
docs/
  plan.md                authoritative full plan (v2, OSS-first)
  plan-v1.md             v1 plan for history (hosted-multi-provider, deprecated)
  upstream-comparison.md what we learned from reading the Claude Code source
```

## Quick start (M0.2 target, not yet on `main`)

```python
import asyncio
from any_agent_sdk import Agent, UserMessage, tool

@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 67°F"

async def main():
    agent = Agent(
        model="qwen2.5-72b-instruct",
        backend="https://api.together.xyz/v1",   # or "http://localhost:11434" for Ollama
        tools=[get_weather],
        max_turns=10,
    )
    try:
        messages = await agent.run([UserMessage(content="Weather in SF?")])
        print(messages[-1])
    finally:
        await agent.aclose()

asyncio.run(main())
```

## The acceptance test

We're done with v1.0 when this is true:

> A new user can `pip install any-agent-sdk[ollama]`, `ollama pull qwen2.5:72b`, and run a 10-line script that does a 5-turn agent task with two tools — and it Just Works on the first try.

Then we add Together, Fireworks, vLLM, llama.cpp until that sentence holds across the whole model + backend matrix.

## Milestones (see `docs/plan.md` for the full version)

| | Week | Deliverable |
|---|---|---|
| M0 | done | Skeleton on `main` |
| M0.1 | 1 | Pivot: delete Anthropic adapter, streaming tool dispatch, `<tool_call>` parser, `<think>` parser, concurrency cap, sibling-abort |
| M0.2 | 2 | OpenAI-compat adapter (vLLM, Together, Fireworks, Groq, OpenRouter, Cerebras) |
| M1 | 3 | Ollama, llama.cpp, TGI, Modal adapters + bundled chat templates |
| M2 | 4 | Hooks (28 events), permissions, budget (tokens + USD), fallback model |
| M3 | 5 | MCP client + all four transports + elicitation |
| M4 | 6 | Sub-agents, sessions, `query()` drop-in wrapper |
| M5 | 7 | Compaction, skills, CLI, docs site |
| M6 | 8 | GA 1.0 on PyPI |

## License

Apache-2.0. See `LICENSE`.
