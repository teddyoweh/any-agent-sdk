# any-agent-sdk

Drop-in open-source analog to the Claude Agent SDK. Multi-model, streaming, MCP, sub-agents.

This directory holds the v0 skeleton. The plan lives at [`docs/plan.md`](docs/plan.md).

## Install

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...
python -m any_agent_sdk.examples.quickstart
```

## What's in here

```
any_agent_sdk/
  __init__.py            public surface
  types.py               universal Message + ContentBlock (msgspec)
  events.py              normalized StreamEvent variants
  errors.py              typed exceptions
  http.py                shared httpx client + SSE parser
  agent.py               Agent — the multi-turn loop
  tools.py               @tool decorator, ToolRegistry, dispatcher
  providers/
    base.py              Provider protocol + lazy registry
    anthropic.py         reference adapter (streaming + tool use)
  examples/
    quickstart.py        run + stream end-to-end
```

## Why this is fast

1. **msgspec everywhere.** 5–10× faster than Pydantic v2, ~3× less memory.
2. **One shared `httpx.AsyncClient`** per Agent — HTTP/2 + connection pool, no per-request TLS handshakes.
3. **Streaming SSE parser** that walks `aiter_lines()` and yields events as they arrive. Never buffers a full response body.
4. **Lazy provider imports.** Importing the package doesn't load boto3 or anything else you don't use.
5. **Zero-copy text deltas** — chunks are stored in a list and joined once at `content_block_stop`, not concatenated per delta.
6. **Parallel tool dispatch** by default via `anyio.create_task_group`, with per-name locks for tools declared `parallel_safe=False`.
7. **No global state.** Every long-lived resource hangs off an `Agent` instance.

## Quick start

```python
import asyncio
from any_agent_sdk import Agent, UserMessage, tool
from any_agent_sdk.tools import ToolRegistry


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 67°F"


async def main():
    registry = ToolRegistry()
    registry.add(get_weather)
    agent = Agent(model="claude-opus-4-7", tools=registry)
    try:
        messages = await agent.run(
            [UserMessage(content="Weather in SF?")]
        )
        print(messages[-1])
    finally:
        await agent.aclose()


asyncio.run(main())
```

## What's NOT done yet

This is v0 skeleton. M1–M3 still to come:

- OpenAI, Gemini, Bedrock, local adapters
- MCP client (stdio + HTTP)
- Sub-agent orchestration
- SessionStore (SQLite + in-memory)
- CLI
- Batch API
- Tests + recorded fixtures (none committed yet — next step)

The architecture is set up to absorb all of the above without touching `agent.py` or the event stream.
