"""MCP server-config tagged union, tool wrapper, and call result.

Design
------
* ``ServerConfig`` is a tagged union of four transport flavors. msgspec
  dispatches on the ``type`` field at decode time — same trick we use for
  ``ContentBlock`` in ``any_agent_sdk.types``.
* ``MCPTool`` is the *MCP-side* representation of a remote tool. It knows
  which server it came from (``server_id``) and how to invoke itself back
  through an ``MCPClient``. ``.to_any_agent_tool()`` wraps it in a regular
  ``any_agent_sdk.Tool`` so the agent loop can dispatch MCP tools and local
  ``@tool``-decorated tools through the same registry.
* ``CallToolResult`` mirrors the MCP wire shape (``content: list[block]``,
  ``is_error: bool``). Helper ``.to_string()`` flattens text/image blocks to
  a single string suitable for ``ToolResultBlock.content`` — the agent loop
  is unaware of MCP block structure.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import msgspec

from ..tools import Tool

if TYPE_CHECKING:
    from .client import MCPClient
    from .server import SdkServer


# ---------------------------------------------------------------------------
# Server configs (tagged union)
# ---------------------------------------------------------------------------


class StdioServerConfig(
    msgspec.Struct,
    tag="stdio",
    tag_field="type",
    omit_defaults=True,
):
    """Spawn a local subprocess speaking line-delimited JSON-RPC over stdio."""

    command: str
    args: list[str] = []
    env: dict[str, str] = {}


class SseServerConfig(
    msgspec.Struct,
    tag="sse",
    tag_field="type",
    omit_defaults=True,
):
    """Connect to a remote MCP server using the legacy SSE+POST transport."""

    url: str
    headers: dict[str, str] = {}


class HttpServerConfig(
    msgspec.Struct,
    tag="http",
    tag_field="type",
    omit_defaults=True,
):
    """Streamable-HTTP transport (newer MCP spec). Single bidirectional URL."""

    url: str
    headers: dict[str, str] = {}


class SdkServerConfig(
    msgspec.Struct,
    tag="sdk",
    tag_field="type",
    omit_defaults=True,
):
    """In-process MCP server. ``server`` is held by reference, not serialized.

    Constructed via ``mcp.server.create_sdk_server(name, tools)``. Wiring runs
    entirely through ``InProcessTransport`` — no sockets, no subprocess.
    """

    name: str
    # SdkServer instance held opaquely. ``Any`` so msgspec doesn't try to
    # serialize it; we never round-trip an SdkServerConfig over the wire.
    server: Any = None


ServerConfig = Annotated[
    Union[StdioServerConfig, SseServerConfig, HttpServerConfig, SdkServerConfig],
    msgspec.Meta(description="MCP server connection config."),
]


# ---------------------------------------------------------------------------
# MCPTool — remote tool exposed by an MCP server
# ---------------------------------------------------------------------------


class MCPTool(msgspec.Struct, frozen=True, omit_defaults=True):
    """A tool discovered via ``tools/list`` on an MCP server.

    Wrap with ``.to_any_agent_tool(client)`` to drop it into a normal
    ``ToolRegistry`` alongside local tools. The wrapped tool's ``fn``
    calls back into the originating ``MCPClient.call_tool``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str

    def to_any_agent_tool(self, client: "MCPClient") -> Tool:
        """Wrap this MCP tool as a local ``Tool``.

        The returned tool's ``fn`` is an async closure that round-trips the
        call through ``client.call_tool`` and stringifies the result. Errors
        from the server surface as the string representation of
        ``CallToolResult`` with ``is_error=True``; ``Tool`` dispatch in
        ``any_agent_sdk.tools`` handles wrapping it into a ``ToolResultBlock``.
        """

        tool_name = self.name

        async def _invoke(**kwargs: Any) -> str:
            result = await client.call_tool(tool_name, kwargs)
            text = result.to_string()
            if result.is_error:
                # Surface as exception so dispatch_tool_calls flags it
                # as a tool error result.
                raise RuntimeError(text or f"MCP tool {tool_name!r} returned an error")
            return text

        # MCP tool names can include separators MCP servers consider valid
        # but the agent loop's registry should still see a stable key.
        return Tool(
            name=tool_name,
            description=self.description,
            input_schema=self.input_schema,
            fn=_invoke,
            parallel_safe=True,
        )


# ---------------------------------------------------------------------------
# CallToolResult — wire shape of tools/call response
# ---------------------------------------------------------------------------


class CallToolResult(msgspec.Struct, omit_defaults=True):
    """Result of ``tools/call``.

    ``content`` is a list of MCP content blocks. Each block is a dict with
    at least a ``type`` field — usually ``"text"`` (with ``text``), but the
    spec also allows ``"image"`` (``data``, ``mimeType``) and
    ``"resource"`` (``resource`` link). We keep the raw dicts here and
    flatten in ``.to_string()``.
    """

    content: list[dict[str, Any]] = []
    is_error: bool = False

    def to_string(self) -> str:
        """Flatten content blocks into a single string for the agent loop.

        * ``text`` blocks contribute their text.
        * ``image`` blocks contribute a placeholder ``[image:<mime>]``.
        * ``resource`` blocks contribute the resource ``uri`` (if present).
        * Unknown block types fall back to the JSON-encoded block.
        """

        parts: list[str] = []
        for block in self.content:
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "image":
                mime = block.get("mimeType") or block.get("mime_type") or "image"
                parts.append(f"[image:{mime}]")
            elif btype == "resource":
                resource = block.get("resource") or {}
                uri = resource.get("uri") if isinstance(resource, dict) else None
                parts.append(f"[resource:{uri or '?'}]")
            else:
                # Best-effort: stringify the whole block.
                parts.append(str(block))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Elicitation — server-initiated mid-tool prompts to the user
# ---------------------------------------------------------------------------


ElicitAction = Literal["accept", "decline", "cancel"]


class ElicitationRequest(msgspec.Struct, omit_defaults=True):
    """Server is asking the user (via the client) for input mid-tool-call.

    Mirrors the MCP ``elicitation/create`` request shape. ``message`` is the
    human-readable prompt. ``requested_schema`` is an optional JSON Schema
    describing the shape of the answer the server expects — clients with a
    UI render a form from it; headless clients can ignore it and return
    free-form content.

    ``raw`` is the full params payload from the server, kept so handlers
    can inspect MCP fields we haven't surfaced as first-class attributes
    yet (e.g. ``progressToken``, future spec additions).
    """

    message: str
    requested_schema: dict[str, Any] = {}
    raw: dict[str, Any] = {}


class ElicitationResult(msgspec.Struct, omit_defaults=True):
    """Client's response to an ``ElicitationRequest``.

    ``action``:
      * ``"accept"`` — user filled in ``content`` and submitted.
      * ``"decline"`` — user actively declined (e.g. "no, don't ask me that").
      * ``"cancel"``  — user dismissed without answering (e.g. closed dialog).

    ``content`` is the structured answer when ``action == "accept"``; for
    decline/cancel it's left empty. The server interprets ``content`` against
    its own ``requested_schema``.
    """

    action: ElicitAction = "accept"
    content: dict[str, Any] = {}

    @classmethod
    def accept(cls, content: dict[str, Any] | None = None) -> "ElicitationResult":
        """Convenience: ``ElicitationResult.accept({"name": "Ada"})``."""
        return cls(action="accept", content=content or {})

    @classmethod
    def decline(cls) -> "ElicitationResult":
        """User actively declined to answer."""
        return cls(action="decline", content={})

    @classmethod
    def cancel(cls) -> "ElicitationResult":
        """User dismissed the prompt without answering."""
        return cls(action="cancel", content={})


# An async callback the MCPClient invokes when the server sends an
# elicitation/create request. The handler decides how to surface the
# request to the user (CLI prompt, GUI dialog, web form, …) and returns
# the answer.
ElicitationHandler = Callable[[ElicitationRequest], Awaitable[ElicitationResult]]
