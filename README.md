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

## Custom backend — point at any OpenAI-compatible server

Auto-routing covers the well-known providers from the model name. For everything else — your own vLLM on a private GPU box, LM Studio on a custom port, a corporate proxy, OpenRouter, Groq, an internal inference cluster — pass `backend=` explicitly. The URL wins over inference.

```python
# Self-hosted vLLM on a private GPU box
options = ClaudeAgentOptions(
    model="Qwen/Qwen2.5-72B-Instruct",
    backend="https://gpu-box.internal:8000/v1",
    api_key=os.environ["INTERNAL_KEY"],
    tools=[get_weather],
)

# LM Studio on a non-standard port
options = ClaudeAgentOptions(
    model="qwen2.5:7b",
    backend="http://localhost:1234/v1",
    tools=[get_weather],
)

# Groq (blazing fast llama / mixtral)
options = ClaudeAgentOptions(
    model="llama-3.3-70b-versatile",
    backend="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

# OpenRouter aggregator (200+ models behind one API)
options = ClaudeAgentOptions(
    model="anthropic/claude-3.5-sonnet",  # OpenRouter proxies even Anthropic
    backend="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

Or set it once for the whole process via env:

```bash
export ANY_AGENT_BASE_URL=https://gpu-box.internal:8000/v1
export ANY_AGENT_API_KEY=...
python my_agent.py
```

Precedence: explicit `backend=` > `$ANY_AGENT_BASE_URL` > model-name inference > Ollama default.

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

### 🏠 Local — runs on your machine

**Ollama** — tag form like `qwen3:8b`, `deepseek-r1:1.5b`, `gpt-oss:20b`. Auto-routed to `http://localhost:11434`. No API key, no network egress. CPU works for the small end; GPU / Apple Silicon for the larger ones.

Frontier open-weight models that actually run locally:

| Model                          | Tag                          | Min RAM | Notes                                                                 |
|--------------------------------|------------------------------|--------:|-----------------------------------------------------------------------|
| **OpenAI gpt-oss 20B**         | `gpt-oss:20b`                | 16 GB   | OpenAI's open-weight, MXFP4 native. Roughly o3-mini class.            |
| **DeepSeek-R1 distill (1.5B–70B)** | `deepseek-r1:1.5b` … `:70b` | 4–48 GB | Reasoning model, emits `<think>` blocks — we parse them.              |
| **Qwen3 (0.6B–32B)**           | `qwen3:0.6b` … `qwen3:32b`   | 2–24 GB | Strong all-rounder, tool use solid even at 8B.                        |
| **Qwen3 235B-A22B (MoE)**      | `qwen3:235b-a22b`            | 64+ GB  | Top OSS on broad benchmarks; only viable on big rigs locally.         |
| **Llama 4 Scout / Maverick**   | `llama4:scout`, `llama4:maverick` | 24/72 GB | Meta's 2025 line — long-context Scout fits a 24 GB GPU.       |
| **Llama 3.1/3.2/3.3 (1B–70B)** | `llama3.2:1b` … `llama3.3:70b` | 4–48 GB | Stable, well-supported, native tool calls.                          |
| **Mistral / Mixtral / Magistral** | `mistral:7b`, `mixtral:8x7b`, `magistral:24b` | 6–48 GB | Mixtral MoE, Magistral reasoning. |
| **Phi 4 mini / small / medium** | `phi4:mini` / `:small` / `:medium` | 6–24 GB | MS Phi-4 — punches above its weight on reasoning.                |
| **Gemma 3 (2B–27B)**           | `gemma3:2b` … `gemma3:27b`   | 4–24 GB | Google's latest. 27B variant has strong agentic perf.                 |
| **Hermes 4 (8B–70B)**          | `hermes4:8b` … `hermes4:70b` | 8–48 GB | Nous Research — tool use + reasoning emphasis.                        |
| **GLM-4.6 / GLM-5 (smaller)**  | `glm4.6:9b`, `glm5:32b`      | 8–24 GB | THUDM — strong agentic perf, smaller variants run locally.            |
| **DeepSeek-V3 / V3.2 distills**| `deepseek-v3:7b` etc.        | 6+ GB   | Distilled coding-focused variants.                                    |
| **TinyLlama, SmolLM2, OLMo, Granite, Yi, command-r** | various          | 1–24 GB | The long tail — see `any-agent setup-local --list`.        |

Run `any-agent setup-local` to get a CPU-runnable model in two minutes. See the **Run locally on CPU** section above for the curated catalog of CPU-friendly picks.

**Self-hosted (your own GPU / cluster)** — point `backend=` at any vLLM, llama.cpp (with `--jinja`), TGI, or LM Studio. Useful for running gpt-oss-120b, Qwen3 235B, Kimi K2, Llama 4 Maverick, or anything else too big for a laptop.

### ☁️ Cloud — hosted APIs

**Together AI** — HuggingFace `org/repo` shape, auto-routed. Needs `$TOGETHER_API_KEY`. The widest OSS catalog: Qwen3, Llama 4, DeepSeek-V3/R1, Mixtral, Gemma 3, gpt-oss-120b, Kimi-K2 family.

```python
model="Qwen/Qwen3-235B-A22B-Instruct-Turbo"
model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
model="deepseek-ai/DeepSeek-V3.2"
```

**Fireworks AI** — path form `accounts/fireworks/models/<id>`, auto-routed. Needs `$FIREWORKS_API_KEY`. Strong for low-latency Llama 4, DeepSeek-R1, gpt-oss-120b, Qwen3.

**OpenRouter (aggregator)** — set `backend=https://openrouter.ai/api/v1` + `$OPENROUTER_API_KEY`. 300+ models behind one API: Kimi K2.6, Qwen3, DeepSeek, GLM-5, MiniMax M2.5, Hermes 4, gpt-oss, anything you can name.

**Groq (lowest latency)** — set `backend=https://api.groq.com/openai/v1` + `$GROQ_API_KEY`. Llama 3.3/4, Mixtral, DeepSeek-R1-distill, Gemma 3, gpt-oss-20b. Ridiculously fast.

**Moonshot (Kimi)** — set `backend=https://api.moonshot.cn/v1` + `$MOONSHOT_API_KEY`. Kimi K2 / K2.6 — currently #1 on several OSS leaderboards (90.5% GPQA).

**DeepSeek native** — set `backend=https://api.deepseek.com/v1` + `$DEEPSEEK_API_KEY`. DeepSeek-V3.2, DeepSeek-R1 from the source.

**Cerebras / DeepInfra / Anyscale** — same pattern, set `backend=` + the respective key.

**Frontier OSS shortlist (May 2026 leaderboards)**

| Model                  | Where to get it             | Notable                                             |
|------------------------|-----------------------------|-----------------------------------------------------|
| **Kimi K2.6**          | Moonshot, OpenRouter        | #1 open-weights on GPQA (90.5%)                     |
| **Qwen3 235B-A22B**    | Together, Fireworks, local  | Broadest benchmark leader, Apache 2.0               |
| **GLM-5**              | OpenRouter, Z.ai            | Best Arena Elo among open models (1451)             |
| **MiniMax M2.5**       | OpenRouter, MiniMax         | 80.2% SWE-bench, ties Claude Opus 4.6 on coding     |
| **DeepSeek-R1**        | DeepSeek, Together, local   | Reasoning specialist (emits `<think>`)              |
| **DeepSeek-V3.2**      | DeepSeek, Together          | Top general-purpose OSS                             |
| **Llama 4 Scout**      | Together, Fireworks, local  | 10M context window, fits 24 GB GPU                  |
| **gpt-oss-120b**       | OpenAI weights → any host   | OpenAI's open release, ~o4-mini class               |
| **gpt-oss-20b**        | Ollama (local!), any host   | Runs on 16 GB locally                               |
| **Hermes 4 70B**       | OpenRouter, Together, local | Nous — tool-use + reasoning tuned                   |

### Proprietary

**OpenAI native** — `gpt-4o`, `gpt-5`, `o1`, `o3-mini`, `o4-mini`. Auto-routed.
**Gemini** — `gemini-2.0-flash`, `gemini-1.5-pro`. Auto-routed.
**Claude** — explicitly refused. Use the real `claude-agent-sdk` for Anthropic models.

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
