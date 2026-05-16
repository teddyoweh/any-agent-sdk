"""any-agent-sdk — Claude Code for open-source models.

Public surface is intentionally tiny. Everything else is implementation
detail and may move between minor versions.
"""

from .agent import Agent
from .builtin_tools import WebFetch, WebSearch, web_fetch, web_search
from .capabilities import (
    BackendCapability,
    ModelCapability,
    ToolUsePath,
    lookup_model,
    resolve_tool_use_path,
)
from .errors import (
    AgentError,
    AuthError,
    BudgetExceededError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    StreamProtocolError,
    ToolExecutionError,
)
from .events import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
)
from .claude_compat import (
    AgentDefinition,
    CLIConnectionError,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    HookContext as ClaudeHookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    PermissionResult as ClaudePermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    Plugin,
    ResultMessage,
    ToolPermissionContext,
    create_sdk_mcp_server,
)
# Pull SystemMessage from claude_compat as the canonical top-level name —
# this is the flat-shape version with .subtype + .data that Claude SDK
# examples use. Our internal SystemMessage (no subtype) is still
# available via `from any_agent_sdk.types import SystemMessage as
# InternalSystemMessage` if needed.
from .claude_compat import SystemMessage  # type: ignore[assignment]  # overrides earlier import
from .memory import (
    MemoryEntry,
    list_memory_entries,
    load_memory_entry,
    load_memory_index,
    save_memory_entry,
    update_memory_index,
)
from .paths import (
    get_anyagent_dir,
    get_memory_dir,
    get_memory_index,
    get_sessions_dir,
    get_session_path,
)
from .settings import (
    KNOWN_SETTING_KEYS,
    SETTING_SOURCES,
    apply_settings_to_options,
    load_setting_source,
    load_settings,
    merge_settings,
    resolve_setting_path,
    save_setting_source,
    update_setting_source,
)
from .query import (
    APIAssistantMessage,
    APIUserMessage,
    SDKAssistantMessage,
    SDKCompactBoundaryMessage,
    SDKMessage,
    SDKPermissionDenial,
    SDKResultMessage,
    SDKStatusMessage,
    SDKSystemMessage,
    SDKUserMessage,
    query,
)
from .system_reminder import (
    build_live_context_block,
    is_system_reminder,
    prepend_user_context,
    render_user_context,
    strip_system_reminders,
    wrap_system_reminder,
)
from .session import (
    Checkpoint,
    InMemorySessionStore,
    Session,
    SessionInfo,
    SessionNotFoundError,
    SessionStore,
    SqliteSessionStore,
    fork_session,
    make_checkpoints,
    resume_session,
)
from .transcripts import (
    JsonlTranscript,
    iter_transcripts,
    read_transcript,
)
from .tools import Tool, ToolRegistry, tool
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ModelUsage,
    SystemMessage as InternalSystemMessage,  # keep accessible for advanced use
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
)

__all__ = [
    "Agent",
    "AgentError",
    "AssistantMessage",
    "AuthError",
    "BackendCapability",
    "BudgetExceededError",
    "ContentBlock",
    "ContentBlockDelta",
    "ContentBlockStart",
    "ContentBlockStop",
    "InputJsonDelta",
    "Message",
    "MessageDelta",
    "MessageStart",
    "MessageStop",
    "ModelCapability",
    "ModelUsage",
    "PermissionDeniedError",
    "Plugin",
    "ProviderError",
    "RateLimitError",
    "Checkpoint",
    "InMemorySessionStore",
    "Session",
    "SessionInfo",
    "SessionNotFoundError",
    "SessionStore",
    "SqliteSessionStore",
    "fork_session",
    "make_checkpoints",
    "resume_session",
    "SDKAssistantMessage",
    "SDKCompactBoundaryMessage",
    "SDKMessage",
    "SDKPermissionDenial",
    "SDKResultMessage",
    "SDKStatusMessage",
    "SDKSystemMessage",
    "SDKUserMessage",
    "StreamEvent",
    "StreamProtocolError",
    "SystemMessage",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "Tool",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUsePath",
    "APIAssistantMessage",
    "APIUserMessage",
    "Usage",
    "UserMessage",
    "WebFetch",
    "WebSearch",
    "lookup_model",
    "query",
    "resolve_tool_use_path",
    "tool",
    "web_fetch",
    "web_search",
]

__version__ = "0.1.0"
