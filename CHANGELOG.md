# Changelog

All notable changes to `any-agent-sdk` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and from 1.0.0 on the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The full versioning policy is in [SEMVER.md](SEMVER.md).

## [Unreleased]

## [1.0.0] — 2026-05-17

First stable release. The public API — the set of names in
`any_agent_sdk.__all__` — is now covered by the SemVer guarantee
documented in [SEMVER.md](SEMVER.md).

### Added

- **Locked public API surface.** `any_agent_sdk.__all__` is now the
  single source of truth for what is covered by SemVer. A new test
  (`tests/test_public_api_surface.py`) snapshots the set and fails on
  unintentional drift.
- **`__version__` from package metadata.** `any_agent_sdk.__version__`
  now reads from `importlib.metadata.version("any-agent-sdk")` when the
  package is installed, so it always tracks `pyproject.toml`.
- **`SEMVER.md`** — the versioning policy.
- **`RELEASING.md`** — the release runbook.
- **`.github/workflows/release.yml`** — tag-driven PyPI publish using
  trusted publishing (OIDC). No long-lived API token required.
- **`.github/workflows/test.yml`** — CI matrix on Python 3.11, 3.12, 3.13.
- Expanded `__all__` to include every Claude SDK parity symbol that was
  previously imported at the top of `any_agent_sdk` but only
  conventionally public: `ClaudeAgentOptions`, `ClaudeSDKClient`,
  `ClaudeSDKError`, `CLIConnectionError`, `AgentDefinition`,
  `HookMatcher`, `HookInput`, `HookJSONOutput`, `ClaudeHookContext`,
  `ClaudePermissionResult`, `PermissionResultAllow`,
  `PermissionResultDeny`, `ToolPermissionContext`, `ResultMessage`,
  `IsolationMode`, `create_sdk_mcp_server`.
- PyPI metadata: `readme`, `authors`, `keywords`, `classifiers`,
  `project.urls`, and explicit hatchling `wheel`/`sdist` targets so the
  built sdist contains tests, docs, and policy files.

### Highlights of the road to 1.0

The pre-1.0 series shipped the building blocks the 1.0 surface relies on.
A non-exhaustive summary:

- **Multi-model**: Ollama (native + auto-routing), OpenAI-compat
  (vLLM, Together, Fireworks, Groq, OpenRouter, Cerebras), llama.cpp,
  TGI, OpenAI native (`gpt-*`, `o1`/`o3`/`o4`), Gemini OpenAI-compat,
  Modal serverless adapter, `anthropic_passthrough` for parity testing.
- **Tool use**: three paths (native, prompt-engineered `<tool_call>`,
  grammar-constrained JSON), capability-table-driven selection across
  30+ models, parallel dispatch, mid-stream dispatch, mid-stream
  cancellation via `ToolPermissionContext.signal`.
- **Streaming**: full `ContentBlockStart`/`Delta`/`Stop` plus
  `MessageStart`/`Delta`/`Stop` event surface; tools fire on
  `ContentBlockStop`, not after `MessageStop`.
- **Thinking**: inline `<think>` for DeepSeek-R1, QwQ, Marco-o1, R1-distill;
  out-of-band thinking blocks for the DeepSeek API; `ThinkingBlock`
  in `AssistantMessage.content`.
- **MCP**: stdio / sse / http transports, in-process server via
  `create_sdk_mcp_server`, elicitation, sampling.
- **Sessions**: JSONL transcript persistence, `~/.any-agent/` layout,
  fork + resume from arbitrary checkpoint, memory entries + index,
  `<system-reminder>` and `isMeta` injection, auto-compaction.
- **Budget**: per-model pricing, `max_usd` ceiling →
  `BudgetExceededError`, `total_cost_usd` and `modelUsage` on
  `ResultMessage`, `max_turns` ceiling.
- **Local setup**: `any-agent setup-local` for Ollama (Linux/macOS/Windows)
  and `any-agent setup-local-llamacpp` for llama.cpp.
- **Examples**: 16 verified examples across ≥3 backends, including
  `quickstart`, `ollama_local`, `with_thinking`, `tools_option`,
  `mcp_calculator`, `system_prompt`, `fireworks_hosted`,
  `vllm_self_hosted`, `multi_agent_research`.
- **Docs site**: mkdocs-material at `docs/`.

[Unreleased]: https://github.com/teddyoweh/any-agent-sdk/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/teddyoweh/any-agent-sdk/releases/tag/v1.0.0
