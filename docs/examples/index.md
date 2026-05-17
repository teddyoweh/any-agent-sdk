# Examples

All examples live under `any_agent_sdk/examples/` in the repo. Each one
is runnable directly:

```bash
python -m any_agent_sdk.examples.quickstart
```

## Basics

| Example | What it shows |
|---|---|
| [`quickstart.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/quickstart.py) | `query()` with one tool, byte-for-byte equivalent to the Claude Agent SDK pattern. |
| [`ollama_local.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/ollama_local.py) | Pointing at a local Ollama daemon. |
| [`with_thinking.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/with_thinking.py) | Rendering thinking blocks separately from final answer. |
| [`tools_option.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/tools_option.py) | Multiple tools, parallel dispatch. |
| [`system_prompt.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/system_prompt.py) | Setting `system_prompt`. |

## MCP

| Example | What it shows |
|---|---|
| [`mcp_calculator.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/mcp_calculator.py) | In-process MCP server via `create_sdk_mcp_server`. |
| [`mcp_filesystem.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/mcp_filesystem.py) | External stdio MCP server (`uvx mcp-server-fetch`). |

## Streaming

| Example | What it shows |
|---|---|
| [`streaming_render.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/streaming_render.py) | Token-by-token rendering with `Agent.run_iter`. |
| [`streaming_mode_ipython.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/streaming_mode_ipython.py) | Streaming in an IPython notebook. |
| [`stderr_callback_example.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/stderr_callback_example.py) | Using the `stderr` callback for debug logging. |

## Sub-agents

| Example | What it shows |
|---|---|
| [`multi_agent_research.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/multi_agent_research.py) | Parent agent fanning out to researcher + drafter + reviewer sub-agents. Runs in real-backend mode or `ANY_AGENT_MOCK=1` smoke mode. |
| [`research_agent.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/research_agent.py) | A single research agent with web tools. |

## Budget

| Example | What it shows |
|---|---|
| [`max_budget_usd.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/max_budget_usd.py) | Hitting `BudgetExceededError` deliberately, recovering from a checkpoint. |

## Hosted backends

| Example | What it shows |
|---|---|
| [`fireworks_hosted.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/fireworks_hosted.py) | Running against live Fireworks (or `ANY_AGENT_MOCK=1`). |
| [`vllm_self_hosted.py`](https://github.com/teddyoweh/any-agent-sdk/blob/main/any_agent_sdk/examples/vllm_self_hosted.py) | Running against a self-hosted vLLM (or `ANY_AGENT_MOCK=1`). |

## Running offline

Most examples auto-detect `ANY_AGENT_MOCK=1` and run against the mock
provider so CI works without API keys:

```bash
ANY_AGENT_MOCK=1 python -m any_agent_sdk.examples.quickstart
```

In mock mode the assistant emits canned but well-shaped responses;
useful for verifying integration plumbing.
