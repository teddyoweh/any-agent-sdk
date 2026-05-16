"""In-process MCP server — surface local ``Tool``s through the MCP protocol.

Used to expose a curated set of tools to a sub-agent (or to anything that
speaks MCP) without spinning up a subprocess. Speaks the same JSON-RPC
methods a real MCP server would over stdio/sse/http, but the wire is just
a pair of ``anyio`` memory streams owned by ``InProcessTransport``.

We implement the v0-relevant subset:

* ``initialize`` — capability negotiation
* ``notifications/initialized`` — handshake completion notice
* ``tools/list`` — emit each tool's name + description + input schema
* ``tools/call`` — dispatch through the underlying ``Tool.fn``
* ``notifications/cancelled`` — recorded but not actionable in v0

Anything else gets a ``-32601 Method not found`` error response.

Public entry point is ``create_sdk_server(name, tools)`` which returns an
``SdkServerConfig`` ready to drop into ``MCPClient``.
"""

from __future__ import annotations

from typing import Any

import anyio

from ..tools import Tool, ToolRegistry, _stringify_result
from .types import SdkServerConfig


_PROTOCOL_VERSION = "2025-03-26"


class SdkServer:
    """JSON-RPC dispatcher that exposes a ``ToolRegistry`` over MCP wire format.

    Run is driven externally by ``InProcessTransport``: it hands us a pair
    of memory streams (``inbox`` for client→server, ``outbox`` for
    server→client) and we loop until the inbox closes.
    """

    __slots__ = ("name", "registry")

    def __init__(self, name: str, tools: list[Tool]) -> None:
        self.name = name
        self.registry = ToolRegistry()
        self.registry.add(*tools)

    # -- runtime ------------------------------------------------------------

    async def run(
        self,
        inbox: anyio.streams.memory.MemoryObjectReceiveStream[dict[str, Any]],
        outbox: anyio.streams.memory.MemoryObjectSendStream[dict[str, Any]],
    ) -> None:
        """Drain ``inbox``, dispatch, write responses to ``outbox``.

        Each request is handled in its own task so a slow tool doesn't
        block the next request. Notifications produce no response.
        """

        async with anyio.create_task_group() as tg:
            try:
                async for message in inbox:
                    if not isinstance(message, dict):
                        continue
                    tg.start_soon(self._handle, message, outbox)
            except anyio.EndOfStream:
                pass
            except anyio.ClosedResourceError:
                pass

    async def _handle(
        self,
        message: dict[str, Any],
        outbox: anyio.streams.memory.MemoryObjectSendStream[dict[str, Any]],
    ) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        mid = message.get("id")
        is_notification = mid is None

        # Notifications: no response, just side effects.
        if is_notification:
            # notifications/initialized, notifications/cancelled — nothing
            # to do in v0. Future hook integration goes here.
            return

        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "tools/list":
                result = self._tools_list(params)
            elif method == "tools/call":
                result = await self._tools_call(params)
            else:
                await self._send_error(
                    outbox, mid, -32601, f"method not found: {method!r}"
                )
                return
        except Exception as exc:  # noqa: BLE001 — server errors must not crash
            await self._send_error(outbox, mid, -32603, f"internal error: {exc!r}")
            return

        await self._send_result(outbox, mid, result)

    # -- methods ------------------------------------------------------------

    def _initialize(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {
                # We support tools (listChanged not yet implemented).
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": self.name,
                "version": "0.1.0",
            },
        }

    def _tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        tools = [
            {
                "name": t.name,
                "description": t.description,
                # MCP wire shape uses ``inputSchema`` (camelCase).
                "inputSchema": t.input_schema,
            }
            for t in self.registry
        ]
        return {"tools": tools}

    async def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            return {
                "content": [{"type": "text", "text": "missing tool name"}],
                "isError": True,
            }
        tool = self.registry.get(name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"tool {name!r} not found"}],
                "isError": True,
            }
        try:
            out = await tool.fn(**arguments)
        except Exception as exc:  # noqa: BLE001 — tool errors are isError, not protocol errors
            return {
                "content": [{"type": "text", "text": f"tool error: {exc!r}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": _stringify_result(out)}],
            "isError": False,
        }

    # -- wire helpers -------------------------------------------------------

    async def _send_result(
        self,
        outbox: anyio.streams.memory.MemoryObjectSendStream[dict[str, Any]],
        mid: Any,
        result: dict[str, Any],
    ) -> None:
        try:
            await outbox.send({"jsonrpc": "2.0", "id": mid, "result": result})
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass

    async def _send_error(
        self,
        outbox: anyio.streams.memory.MemoryObjectSendStream[dict[str, Any]],
        mid: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": mid,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            payload["error"]["data"] = data
        try:
            await outbox.send(payload)
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def create_sdk_server(name: str, tools: list[Tool]) -> SdkServerConfig:
    """Build an in-process MCP server exposing ``tools`` under ``name``.

    Drop the returned ``SdkServerConfig`` into ``MCPClient(...)`` (or into
    an Agent's ``mcp_servers=[...]`` list, once that exists) to use it.
    """

    server = SdkServer(name, tools)
    return SdkServerConfig(name=name, server=server)
