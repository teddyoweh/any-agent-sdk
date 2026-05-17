# API reference

The public surface — what you can import from `any_agent_sdk` and rely on
across minor versions.

## Top-level imports

```python
from any_agent_sdk import (
    # Core
    query,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    Agent,

    # Tools
    tool,
    Tool,
    ToolRegistry,
    WebFetch,
    WebSearch,

    # Sub-agents
    SubAgentSpec,
    SubAgentTool,
    WrappedAgentTool,
    as_subagent_tool,
    IsolationMode,

    # MCP
    create_sdk_mcp_server,

    # Permissions / hooks
    HookMatcher,
    HookInput,
    HookJSONOutput,
    ToolPermissionContext,
    PermissionResultAllow,
    PermissionResultDeny,

    # Plugins
    Plugin,

    # Sessions
    Session,
    SessionInfo,
    SessionStore,
    InMemorySessionStore,
    SqliteSessionStore,
    Checkpoint,
    make_checkpoints,
    fork_session,
    resume_session,

    # Messages
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,

    # Capabilities
    BackendCapability,
    ModelCapability,
    ToolUsePath,
    lookup_model,
    resolve_tool_use_path,

    # Errors
    AgentError,
    AuthError,
    BudgetExceededError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    StreamProtocolError,
    ToolExecutionError,
    CLIConnectionError,
    ClaudeSDKError,
)
```

## By topic

- [query / ClaudeSDKClient](client.md) — the two entry points.
- [ClaudeAgentOptions](options.md) — every option, with defaults.
- [Message types](messages.md) — flat-shape vs. internal shape.
- [Tools](tools.md) — `@tool`, registries, built-ins.
- [Errors](errors.md) — full error hierarchy.
- [Sessions](sessions.md) — `Session`, `Checkpoint`, fork, resume, stores.

## Versioning

Public surface follows semver from 1.0. Until then:

- Names listed in `any_agent_sdk.__all__` are stable across patch
  versions.
- Anything imported from submodules (`any_agent_sdk.streaming.*`,
  `any_agent_sdk.providers.*`) is implementation detail and may move
  freely.
- Settings file schema and JSONL transcript format are stable across
  minor versions even today.
