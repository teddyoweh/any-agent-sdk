"""any-agent-sdk — drop-in multi-model agent SDK.

Public surface is intentionally tiny. Everything else is implementation detail
and may move between minor versions.
"""

from .agent import Agent
from .errors import (
    AgentError,
    ProviderError,
    RateLimitError,
    ToolExecutionError,
)
from .events import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamEvent,
)
from .tools import Tool, tool
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
)

__all__ = [
    "Agent",
    "AgentError",
    "AssistantMessage",
    "ContentBlock",
    "ContentBlockDelta",
    "ContentBlockStart",
    "ContentBlockStop",
    "Message",
    "MessageDelta",
    "MessageStart",
    "MessageStop",
    "ProviderError",
    "RateLimitError",
    "StreamEvent",
    "SystemMessage",
    "TextBlock",
    "Tool",
    "ToolExecutionError",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
    "UserMessage",
    "tool",
]

__version__ = "0.0.1"
