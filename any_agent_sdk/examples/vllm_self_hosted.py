"""Self-hosted vLLM example.

Prereqs::

    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-7B-Instruct --port 8000

Run::

    python -m any_agent_sdk.examples.vllm_self_hosted

vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so the
``OpenAICompatProvider`` adapter is the right call. Grammar is available via
``guided_json`` — capability resolution will pick Path A (native tools).
"""

from __future__ import annotations

import asyncio

from any_agent_sdk import Agent, UserMessage, tool
from any_agent_sdk.providers.openai_compat import OpenAICompatProvider
from any_agent_sdk.tools import ToolRegistry


@tool
async def search_docs(query: str) -> str:
    """Pretend to search internal docs and return a one-line snippet."""

    return f"Docs result for {query!r}: see /docs/playbooks/index.md (stub)."


async def main() -> None:
    registry = ToolRegistry()
    registry.add(search_docs)

    agent = Agent(
        model="qwen2.5-7b-instruct",
        provider=OpenAICompatProvider(base_url="http://localhost:8000/v1"),
        tools=registry,
        system="Be concise. Use the tool when the user asks about docs.",
        max_tokens=512,
    )
    try:
        messages = await agent.run(
            [UserMessage(content="What's in our docs about retries?")]
        )
        print(messages[-1])
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
