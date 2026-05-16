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
    SystemMessage,
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
    "ProviderError",
    "RateLimitError",
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
