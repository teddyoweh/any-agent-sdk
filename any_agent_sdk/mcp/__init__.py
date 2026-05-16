"""MCP (Model Context Protocol) client + in-process server.

Public surface
--------------
* ``ServerConfig`` — tagged union of stdio/sse/http/sdk configs.
* ``MCPClient`` — async-context-managed client for one MCP server.
* ``MCPTool`` — remote tool description; ``.to_any_agent_tool(client)``
  drops it into a regular ``any_agent_sdk.Tool`` registry.
* ``create_sdk_server(name, tools)`` — build an in-process MCP server
  exposing local ``@tool`` functions through MCP wire format.

Everything else (transports, JSON-RPC plumbing, elicitation exceptions)
is implementation detail; reach into ``.client`` / ``.transports`` if you
need it but expect those paths to shift.
"""

from .client import (
    MCPClient,
    MCPElicitationRequest,
    MCPError,
    MCPProtocolError,
)
from .server import SdkServer, create_sdk_server
from .types import (
    CallToolResult,
    HttpServerConfig,
    MCPTool,
    SdkServerConfig,
    ServerConfig,
    SseServerConfig,
    StdioServerConfig,
)

__all__ = [
    "CallToolResult",
    "HttpServerConfig",
    "MCPClient",
    "MCPElicitationRequest",
    "MCPError",
    "MCPProtocolError",
    "MCPTool",
    "SdkServer",
    "SdkServerConfig",
    "ServerConfig",
    "SseServerConfig",
    "StdioServerConfig",
    "create_sdk_server",
]
