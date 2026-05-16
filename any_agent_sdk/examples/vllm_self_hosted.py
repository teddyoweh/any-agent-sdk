"""Self-hosted vLLM example via ``query()``.

Prereqs::

    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-7B-Instruct --port 8000

Run::

    python -m any_agent_sdk.examples.vllm_self_hosted

vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so
the OpenAI-compat adapter is the right call. Capability resolution picks
Path A (native tools).
"""

from __future__ import annotations

import asyncio

from any_agent_sdk import query, tool


@tool
async def search_docs(query_str: str) -> str:
    """Pretend to search internal docs and return a one-line snippet."""

    return f"Docs result for {query_str!r}: see /docs/playbooks/index.md (stub)."


async def main() -> None:
    async for msg in query(
        prompt="What's in our docs about retries?",
        options={
            "model": "qwen2.5-7b-instruct",
            "backend": "http://localhost:8000/v1",
            "tools": [search_docs],
            "system": "Be concise. Use the tool when the user asks about docs.",
            "max_tokens": 512,
            "max_turns": 3,
        },
    ):
        if msg.type == "assistant":
            for block in msg.message.content:
                if hasattr(block, "text") and block.text:
                    print(f"[assistant] {block.text}")
        elif msg.type == "result":
            print(
                f"\n[result] {msg.subtype} · {msg.num_turns} turns · "
                f"{msg.duration_ms} ms"
            )


if __name__ == "__main__":
    asyncio.run(main())
