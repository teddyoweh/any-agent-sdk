# any-agent-sdk

**Claude Agent SDK for open-source models.** Drop-in compatible with `claude-agent-sdk` — swap the import, keep your code — but the agent loop runs against Llama, Qwen, DeepSeek, Mixtral, Phi, Gemma, or anything you serve through Ollama, vLLM, llama.cpp, TGI, Together, Fireworks, Groq, or OpenRouter.

```python
# Before
from claude_agent_sdk import query, ClaudeAgentOptions, tool

# After
from any_agent_sdk import query, ClaudeAgentOptions, tool
```

That's it. Every canonical Claude SDK example runs verbatim. The wire format underneath is OpenAI-compat or Ollama; the surface above is Anthropic-shaped.

---

## Status

**Pre-1.0, but the surface is real.** 173 tests passing. Six of Anthropic's own canonical SDK examples run verbatim against DeepSeek-R1 1.5B on local Ollama. The drop-in compatibility layer (`ClaudeAgentOptions`, `ClaudeSDKClient`, `query()`, `AgentDefinition`, `Plugin`, `PermissionResultAllow/Deny`, `HookMatcher`, `ToolPermissionContext`, `create_sdk_mcp_server`) is wired through to the agent loop — not a stub.

What's left before 1.0 is polish, more provider matrix coverage, and the streaming-tool-dispatch rewrite of the loop. Tracking in `docs/plan.md`.

---

## Quick start

```bash
pip install any-agent-sdk
ollama pull qwen2.5:7b
```

```python
import asyncio
from any_agent_sdk import query, ClaudeAgentOptions, tool, AssistantMessage

@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 67°F"

async def main():
    async for msg in query(
        prompt="What's the weather in SF?",
        options=ClaudeAgentOptions(
            model="qwen2.5:7b",
            backend="http://localhost:11434",
            tools=[get_weather],
            max_turns=5,
        ),
    ):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text"):
                    print(block.text)

asyncio.run(main())
```

Same script against Together AI:

```python
options = ClaudeAgentOptions(
    model="Qwen/Qwen2.5-72B-Instruct-Turbo",
    backend="https://api.together.xyz/v1",
    tools=[get_weather],
    max_turns=5,
)
```

Same script against Fireworks, vLLM, llama.cpp, Groq — change `backend` and `model`. The agent code doesn't move.

---

## What's shipped

```
any_agent_sdk/
  agent.py               Multi-turn loop with tool dispatch, hooks, permissions, budget
  claude_compat.py       Drop-in surface: ClaudeAgentOptions, ClaudeSDKClient, AgentDefinition,
                         Plugin, PermissionResult{Allow,Deny}, HookMatcher, ToolPermissionContext
  compat_query.py        query() yielding Claude-flat-shape messages (AssistantMessage,
                         ResultMessage, SystemMessage, UserMessage)
  query.py               TS-SDK-shape query() (SDKAssistantMessage with nested .message.content)
  types.py               msgspec tagged unions — TextBlock, ToolUseBlock, ThinkingBlock, ...
  events.py              StreamEvent variants — MessageStart, ContentBlockDelta, ...
  tools.py               @tool decorator, ToolRegistry, parallel dispatcher
  hooks.py               28 hook events (PreToolUse, PostToolUse, SessionStart, ...)
  permissions.py         can_use_tool callback wiring, PermissionResultAllow.updated_input
                         rewrites tool args before dispatch
  budget.py              Per-model pricing, max_usd ceiling, BudgetExceededError
  memory.py              ~/.any-agent/memory/ entries + index; isMeta system reminders
  session.py             JSONL transcripts, fork/resume
  compact.py             Auto-compaction at token threshold
  subagent.py            AgentDefinition execution
  skills.py              Skill loading + activation
  system_reminder.py     <system-reminder> wrapping + live context injection
  retry.py               Provider-error retry with backoff
  http.py                Shared httpx.AsyncClient + SSE parser
  cli.py                 `any-agent` CLI
  capabilities.py        ModelCapability tables — picks tool-use path A/B/C per model
  builtin_tools/         WebFetch (Exa), WebSearch (Exa), file ops
  mcp/                   Client + server (stdio, sse, http, in-process)
  streaming/             ToolCallTextParser, ThinkingParser stacks for Path B/C
  providers/
    base.py              Provider protocol + lazy registry
    openai_compat.py     vLLM, Together, Fireworks, Groq, OpenRouter, Cerebras
    ollama.py            Ollama native API
    llamacpp.py          llama.cpp server
    tgi.py               HuggingFace TGI
    mock.py              Scripted provider for tests
  examples/
    quickstart.py, quick_start.py, ollama_local.py, fireworks_hosted.py,
    vllm_self_hosted.py, tools_option.py, mcp_calculator.py, mcp_filesystem.py,
    multi_agent_research.py, research_agent.py, max_budget_usd.py,
    stderr_callback_example.py, streaming_mode_ipython.py, streaming_render.py,
    system_prompt.py, with_thinking.py
tests/                   105 test functions across 21 files
docs/
  plan.md                Full plan
  upstream-comparison.md What we learned reading 1,902 TS files of Claude Code
```

---

## Why this exists

The Claude Agent SDK is the best-designed agent runtime in the open. Streaming tool dispatch, 28-event hook system, permission rules per source, MCP across four transports, sub-agents, sessions with fork/resume, auto-compaction — none of the OSS alternatives ship the whole set. LangGraph is too heavy and skips MCP. smolagents is too small. llama-stack is tightly scoped. The Anthropic and OpenAI agent SDKs are bound to their hosted APIs.

any-agent-sdk is the same surface, model-agnostic underneath. You write to Anthropic's design; you run it on whatever you can serve.

Plus the OSS-specific bits the hosted SDKs don't need to think about:

- **Universal tool use** — Path A (native via OpenAI-compat `tools[]`) when supported; Path B (prompt-engineered `<tool_call>` XML) when not; Path C (grammar-constrained JSON) when the server can enforce it. Capability-table-driven, automatic per model.
- **Universal thinking** — handles inline `<think>` tags (R1, QwQ, Marco-o1, R1-Distill) and out-of-band thinking blocks. Zero cost when the model doesn't emit thinking.
- **Backend agnosticism** — same agent code, one env var or one kwarg between Ollama at `localhost:11434` and Fireworks at `api.fireworks.ai`.

---

## The acceptance test

v1.0 ships when this is true on a fresh machine:

```bash
pip install any-agent-sdk
ollama pull qwen2.5:7b
# ...10-line script with 2 tools + 5-turn agent task...
python my_agent.py   # Just Works on the first try
```

Then we add Together, Fireworks, vLLM, llama.cpp, Groq until that sentence holds across the whole model + backend matrix.

Today: DeepSeek-R1 1.5B on local Ollama runs six of Anthropic's own canonical examples verbatim. Suite at 173 tests. The acceptance test passes on Ollama; provider matrix expansion is the remaining work.

---

## Drop-in compatibility — what works today

```python
from any_agent_sdk import (
    # Core
    query, ClaudeAgentOptions, ClaudeSDKClient,

    # Messages (flat shape, matches claude_agent_sdk)
    AssistantMessage, UserMessage, SystemMessage, ResultMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock,

    # Tools
    tool, Tool, ToolRegistry, create_sdk_mcp_server,

    # Permissions
    PermissionResultAllow, PermissionResultDeny, ToolPermissionContext,

    # Hooks
    HookMatcher, HookInput, HookJSONOutput, HookContext,

    # Sub-agents
    AgentDefinition,

    # Plugins
    Plugin,

    # Built-in tools
    WebFetch, WebSearch,

    # Errors
    ClaudeSDKError, CLIConnectionError,
)
```

Every name in that import block has a working implementation backed by tests. `ClaudeSDKClient` is a streaming async context manager. `Plugin(tools=..., system_prompt_addition=..., hooks=...)` merges into the agent at session start. `PermissionResultAllow(updated_input={...})` rewrites tool args before dispatch. `ResultMessage.permission_denials` carries every rejected call.

---

## License

Apache-2.0. See `LICENSE`.
