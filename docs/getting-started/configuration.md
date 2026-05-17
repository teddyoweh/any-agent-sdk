# Configuration

There are three layers of configuration, in priority order from highest to
lowest:

1. **Constructor arguments** — `ClaudeAgentOptions(...)` or the `options=`
   dict passed to `query()`. Always wins.
2. **Setting sources** — JSON files on disk loaded via `setting_sources=`.
   Multiple sources merge in declaration order; later sources override
   earlier ones for the same key.
3. **Environment variables** — read at process start.

## Environment variables

| Variable | Effect |
|---|---|
| `ANY_AGENT_HOME` | Override `~/.any-agent/` root. |
| `ANY_AGENT_MODEL` | Default model name if `options.model` is unset. |
| `ANY_AGENT_BACKEND` | Force a backend (`ollama`, `openai_compat`, `openai`, `anthropic_passthrough`, `gemini`, `mock`, …). Skip auto-routing. |
| `ANY_AGENT_BASE_URL` | Base URL for HTTP backends. Most useful with OpenAI-compat (vLLM, Together, Fireworks, …). |
| `ANY_AGENT_API_KEY` | API key for HTTP backends. |
| `OPENAI_API_KEY` | Used by the `openai` backend. |
| `ANTHROPIC_API_KEY` | Used by `anthropic_passthrough` and built-in tools (`WebFetch`). |
| `EXA_API_KEY` | Used by `WebSearch` / `WebFetch` for live web results. |
| `ANY_AGENT_MOCK` | Set to `1` to force the mock provider — useful in CI without API keys. |

## Setting sources

```python
from any_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    setting_sources=[
        "~/.any-agent/settings.json",
        "./.any-agent/project.json",
    ],
)
```

Both files are loaded; values from the second override the first. Writes
back via `save_setting_source(source_path, settings_dict)` go to a single
specified file — settings don't smear across sources on write.

The schema is the same as the constructor arguments — `model`, `tools`,
`system_prompt`, `max_turns`, `max_tokens`, `temperature`, `permissions`,
etc. See [ClaudeAgentOptions](../api/options.md) for the full list.

## `~/.any-agent/settings.json` (default user source)

If you don't pass `setting_sources=`, the SDK loads
`~/.any-agent/settings.json` as a single default source. Edit it to set
your model and backend once and forget about it:

```json
{
  "model": "qwen2.5:7b",
  "max_usd": 1.0,
  "max_turns": 20,
  "permissions": {
    "default_mode": "ask"
  }
}
```

## Programmatic loading

```python
from any_agent_sdk import (
    apply_settings_to_options,
    load_setting_source,
    save_setting_source,
)

s = load_setting_source("~/.any-agent/settings.json")
# mutate
s["model"] = "qwen2.5:7b"
save_setting_source("~/.any-agent/settings.json", s)
```

Or merge several sources into a fresh `ClaudeAgentOptions`:

```python
from any_agent_sdk import load_settings, ClaudeAgentOptions

merged = load_settings(["~/.any-agent/settings.json", "./.any-agent.json"])
options = ClaudeAgentOptions(**merged)
```

## Where state lives

```
~/.any-agent/
├── settings.json           merged user settings
├── memory/                 persistent memory entries + INDEX.md
│   ├── INDEX.md
│   └── *.md
├── sessions/               JSONL transcripts
│   └── {session_id}.jsonl
└── models/                 GGUF cache (setup-local-llamacpp only)
```

The exact paths come from the `paths` module:

```python
from any_agent_sdk import (
    get_anyagent_dir, get_memory_dir, get_memory_index,
    get_sessions_dir, get_session_path,
)
```

All paths honour `ANY_AGENT_HOME` if it's set.
