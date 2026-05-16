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

**Pre-1.0, but the surface is real.** 202 tests passing. Six of Anthropic's own canonical SDK examples run verbatim against DeepSeek-R1 1.5B on local Ollama. The drop-in compatibility layer (`ClaudeAgentOptions`, `ClaudeSDKClient`, `query()`, `AgentDefinition`, `Plugin`, `PermissionResultAllow/Deny`, `HookMatcher`, `ToolPermissionContext`, `create_sdk_mcp_server`) is wired through to the agent loop — not a stub.

---

## Quick start

```bash
pip install any-agent-sdk
any-agent setup-local         # installs Ollama if missing, pulls qwen2.5:1.5b, verifies
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
            model="qwen2.5:1.5b",   # routes to local Ollama automatically
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

Same script against Together AI — change one line:

```python
options = ClaudeAgentOptions(
    model="Qwen/Qwen2.5-72B-Instruct-Turbo",  # routes to Together automatically (uses $TOGETHER_API_KEY)
    tools=[get_weather],
    max_turns=5,
)
```

Same script against Fireworks, vLLM, llama.cpp, Groq — just change `model`. The backend URL is inferred from the model name shape; pass `backend=` explicitly to override.

---

## Run locally on CPU (no GPU needed)

The fastest path to a working agent: one command.

```bash
any-agent setup-local
```

What it does: checks for Ollama, installs it if missing (Linux/macOS), pulls a curated CPU-friendly model (default `qwen2.5:1.5b`), and runs a smoke-test request. Pass `--list` to see the catalog, `--model qwen2.5:3b` to pick a different one, `--install-ollama` to auto-install Ollama.

The curated catalog — every entry runs on CPU without a discrete GPU:

| Tag                  | Params | Size  | RAM   | Tool calls | Reasoning | Notes                                              |
|----------------------|-------:|------:|------:|:----------:|:---------:|----------------------------------------------------|
| `smollm2:135m`       | 135M   | 0.3GB | 2 GB+ | no         | no        | Tiny — sanity-check install                        |
| `qwen2.5:0.5b`       | 0.5B   | 0.4GB | 2 GB+ | yes        | no        | Smallest Qwen with tool calls. Fast on anything.   |
| `tinyllama:1.1b`     | 1.1B   | 0.6GB | 2 GB+ | no         | no        | RAM-constrained pick                               |
| `qwen2.5:1.5b`       | 1.5B   | 1.0GB | 4 GB+ | yes        | no        | **Default** — best 1.5B for agent loops            |
| `deepseek-r1:1.5b`   | 1.5B   | 1.1GB | 4 GB+ | yes        | yes       | Emits `<think>` blocks; we parse them              |
| `llama3.2:1b`        | 1.2B   | 1.3GB | 4 GB+ | yes        | no        | Meta's 1B — sharper than 0.5B Qwen                 |
| `gemma2:2b`          | 2B     | 1.6GB | 4 GB+ | no         | no        | Google's 2B — polished prose, no tools             |
| `qwen2.5:3b`         | 3B     | 1.9GB | 6 GB+ | yes        | no        | Same class as Llama 3.2 3B                         |
| `llama3.2:3b`        | 3.2B   | 2.0GB | 6 GB+ | yes        | no        | Solid default if you have 8 GB RAM                 |
| `phi3.5:3.8b`        | 3.8B   | 2.2GB | 6 GB+ | yes        | no        | Strong reasoning for size                          |
| `qwen2.5:7b`         | 7B     | 4.7GB | 8 GB+ | yes        | no        | CPU ceiling — slow without M-series / GPU          |
| `llama3.1:8b`        | 8B     | 4.9GB | 8 GB+ | yes        | no        | Edge of CPU usability                              |

Run any of them by tag — auto-routing sends them to local Ollama:

```python
options = ClaudeAgentOptions(model="qwen2.5:1.5b", tools=[...])
options = ClaudeAgentOptions(model="deepseek-r1:1.5b", tools=[...])   # gets <think> support free
```

---

## Supported models — full catalog

Auto-routing recognizes these shapes (see `any_agent_sdk/routing.py`). Pass `backend=` to override.

**Ollama (local, free, CPU/GPU)** — tag form like `qwen2.5:7b`, `deepseek-r1:1.5b`, `llama3.2:3b`. Routes to `http://localhost:11434`.

- Llama 3, 3.1, 3.2, 3.3 (1B / 3B / 8B / 70B)
- Qwen 2.5, QwQ (0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B)
- DeepSeek-R1 (1.5B / 7B / 8B / 14B / 32B / 70B / 671B), DeepSeek-V3
- Mistral 7B, Mixtral 8x7B / 8x22B, Mistral-Nemo, Codestral
- Phi 3, 3.5, 4 (mini / small / medium)
- Gemma 2, 3 (2B / 9B / 27B)
- Yi, SmolLM2, TinyLlama, Granite, OLMo, command-r, command-r-plus
- Anything else Ollama serves — `ollama pull <name>`, then pass that tag

**Together AI (hosted)** — HuggingFace `org/repo` shape like `Qwen/Qwen2.5-72B-Instruct-Turbo`, `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`. Needs `$TOGETHER_API_KEY`. Routes to `https://api.together.xyz/v1`.

**Fireworks AI (hosted)** — path form `accounts/fireworks/models/<id>`. Needs `$FIREWORKS_API_KEY`.

**Groq (hosted, blazing fast)** — flat model names; set `$ANY_AGENT_BASE_URL=https://api.groq.com/openai/v1` and `$GROQ_API_KEY`. Llama 3.1/3.3, Mixtral, DeepSeek-R1-distill, Gemma 2.

**OpenRouter (hosted aggregator)** — `$ANY_AGENT_BASE_URL=https://openrouter.ai/api/v1`, `$OPENROUTER_API_KEY`. 200+ models.

**OpenAI native** — `gpt-4o`, `gpt-5`, `o1-mini`, `o3-mini`, `o4-mini`. Routes to OpenAI directly.

**Gemini** — `gemini-2.0-flash`, `gemini-1.5-pro`. Routes to Google's OpenAI-compat endpoint.

**Self-hosted** — point `backend=` at any vLLM, llama.cpp (`--jinja`), TGI, or LM Studio server.

**Claude itself** — explicitly refused. Use the real `claude-agent-sdk` for Anthropic models.

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
any-agent setup-local
# ...10-line script with 2 tools + 5-turn agent task...
python my_agent.py   # Just Works on the first try
```

Then the same script works against Together, Fireworks, vLLM, llama.cpp, Groq just by changing `model`. Today: DeepSeek-R1 1.5B on local Ollama runs six of Anthropic's own canonical examples verbatim. Suite at 202 tests. The acceptance test passes on Ollama; provider matrix expansion is the remaining work.

---

## Roadmap

What's shipped — and what's still ahead. Check our progress.

**Drop-in surface (Claude SDK parity)**
- [x] `query()` yielding flat-shape `AssistantMessage` / `UserMessage` / `SystemMessage` / `ResultMessage`
- [x] `ClaudeAgentOptions` with model, backend, tools, system_prompt, max_turns, max_tokens, temperature, hooks, can_use_tool, permissions, mcp_servers, plugins, agents, max_budget_usd, setting_sources, allowed_tools, disallowed_tools, cwd, session_id, persist, stderr
- [x] `ClaudeSDKClient` — streaming async context manager
- [x] `@tool` decorator (Claude-shaped positional signature)
- [x] `AgentDefinition` for sub-agents
- [x] `Plugin(tools=, system_prompt_addition=, hooks=)` — merges at session start
- [x] `PermissionResultAllow(updated_input=...)` rewriting tool args before dispatch
- [x] `PermissionResultDeny` surfacing through `ResultMessage.permission_denials`
- [x] `HookMatcher` for 28 hook events (PreToolUse, PostToolUse, SessionStart, SessionEnd, Stop, ...)
- [x] `ToolPermissionContext` passed to `can_use_tool`
- [x] `create_sdk_mcp_server(name, version, tools=)`
- [x] `WebFetch` / `WebSearch` built-in tools (Exa-backed)
- [x] `CLIConnectionError`, `ClaudeSDKError`
- [ ] `ToolPermissionContext.signal` for cancellation
- [ ] `setting_sources` actually loading and persisting per source
- [ ] Streaming-mode `client.query()` with mid-stream tool dispatch

**Backends**
- [x] Ollama (native API + auto-routing from tag form)
- [x] OpenAI-compat (vLLM, Together, Fireworks, Groq, OpenRouter, Cerebras)
- [x] llama.cpp (via `--jinja`)
- [x] TGI (HuggingFace text-generation-inference)
- [x] OpenAI native (`gpt-*`, `o1`/`o3`/`o4`)
- [x] Gemini OpenAI-compat endpoint
- [x] Mock provider for tests
- [x] Auto-route from model name shape — no `backend=` needed
- [ ] Modal serverless adapter
- [ ] Anthropic via separate `anthropic_passthrough` (for parity testing only)

**Tool use**
- [x] Path A: native via OpenAI-compat `tools[]`
- [x] Path B: prompt-engineered `<tool_call>` XML (for Llama 2, Mistral 7B, older Qwens)
- [x] Path C: grammar-constrained JSON
- [x] Capability-table-driven path selection (30+ models)
- [x] Parallel tool dispatch
- [x] Tool result threading
- [ ] Streaming tool dispatch (start tool execution mid-stream, not after `MessageStop`)

**Thinking / reasoning**
- [x] Inline `<think>` blocks (DeepSeek-R1, QwQ, Marco-o1, R1-distill family)
- [x] Out-of-band thinking blocks (DeepSeek API)
- [x] `ThinkingBlock` in `AssistantMessage.content`

**MCP**
- [x] In-process MCP server via `create_sdk_mcp_server`
- [x] stdio transport
- [x] sse transport
- [x] http transport
- [ ] Elicitation (server prompts user mid-session)
- [ ] Sampling (server calls back into the agent's model)

**Sessions + state**
- [x] JSONL transcript persistence
- [x] `~/.any-agent/` directory + per-session paths
- [x] Memory entries + index
- [x] `<system-reminder>` + `isMeta` injection
- [x] Auto-compaction at token threshold
- [ ] Session fork
- [ ] Session resume from arbitrary checkpoint

**Budget**
- [x] Per-model pricing table
- [x] `max_usd` ceiling → `BudgetExceededError`
- [x] `total_cost_usd` on `ResultMessage`
- [x] `modelUsage` per-model breakdown
- [x] `max_turns` ceiling

**Local install**
- [x] `any-agent setup-local` — installs Ollama if missing, pulls a CPU-friendly model, smoke tests
- [x] 12-entry CPU-friendly catalog (135M → 8B params)
- [x] Auto-install of Ollama on Linux/macOS via official script
- [ ] Windows installer wrapper
- [ ] llama.cpp `setup-local` alternative for users who prefer it

**Examples (run verbatim against DeepSeek-R1 1.5B on local Ollama)**
- [x] `quickstart.py`
- [x] `ollama_local.py`
- [x] `with_thinking.py`
- [x] `tools_option.py`
- [x] `mcp_calculator.py`
- [x] `system_prompt.py`
- [ ] `fireworks_hosted.py` runs against live Fireworks
- [ ] `vllm_self_hosted.py` runs against live vLLM
- [ ] `multi_agent_research.py` end-to-end with sub-agents

**1.0 prerequisites**
- [ ] Streaming tool dispatch rewrite
- [ ] Mid-stream cancellation via `ToolPermissionContext.signal`
- [ ] All 16 examples verified against ≥ 3 backends
- [ ] Docs site (mkdocs-material)
- [ ] PyPI 1.0 release with semver guarantee

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
