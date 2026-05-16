"""any-agent-sdk — Claude Code for open-source models.

Public surface is intentionally tiny. Everything else is implementation
detail and may move between minor versions.
"""

from .agent import Agent
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
from .tools import Tool, ToolRegistry, tool
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
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
    "PermissionDeniedError",
    "ProviderError",
    "RateLimitError",
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
    "Usage",
    "UserMessage",
    "lookup_model",
    "resolve_tool_use_path",
    "tool",
]

__version__ = "0.1.0"
