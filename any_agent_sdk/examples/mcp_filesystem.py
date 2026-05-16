"""MCP example — filesystem server over stdio.

Spawns the official ``@modelcontextprotocol/server-filesystem`` MCP server as
a subprocess, wires its tools into the agent's registry, then runs a single
prompt that exercises one of those tools (``read_file``).

Prereqs::

    npm install -g @modelcontextprotocol/server-filesystem

Run::

    python -m any_agent_sdk.examples.mcp_filesystem

Falls back to a friendly skip message if the MCP client module hasn't landed
in your checkout yet (sibling agent territory).
"""

from __future__ import annotations

import asyncio
import os

from any_agent_sdk import Agent, UserMessage
from any_agent_sdk.providers.openai_compat import OpenAICompatProvider
from any_agent_sdk.tools import ToolRegistry


async def main() -> None:
    try:
        from any_agent_sdk.mcp.client import MCPClient
        from any_agent_sdk.mcp.types import StdioServerConfig
    except ImportError:
        print("MCP client module not present yet — skipping mcp_filesystem example.")
        return

    cwd = os.getcwd()
    server_cfg = StdioServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", cwd],
    )

    client = MCPClient(server_cfg)
    try:
        # Open the transport and discover tools. (API names match the plan;
        # sibling code may rename these — adjust here if so.)
        await client.connect()
        mcp_tools = await client.list_tools()

        registry = ToolRegistry()
        for mt in mcp_tools:
            registry.add(mt.to_any_agent_tool(client))

        agent = Agent(
            model="qwen2.5-7b-instruct",
            provider=OpenAICompatProvider(base_url="http://localhost:8000/v1"),
            tools=registry,
            system="You can read files via the filesystem tool. Be concise.",
            max_tokens=512,
        )
        try:
            messages = await agent.run(
                [UserMessage(content=f"List the files in {cwd}.")]
            )
            print(messages[-1])
        finally:
            await agent.aclose()
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
