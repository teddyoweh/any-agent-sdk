"""Hosted Fireworks example — DeepSeek-V3 via ``query()``.

Prereqs::

    export FIREWORKS_API_KEY=fw_...

Run::

    python -m any_agent_sdk.examples.fireworks_hosted

Fireworks speaks OpenAI-compat, so we use the same adapter as vLLM with a
different base URL. DeepSeek-V3 has native tool calling, so this hits
Path A — the prompt-engineered fallback parser never fires.
"""

from __future__ import annotations

import asyncio
import os

from any_agent_sdk import query, tool


@tool
async def lookup_company(name: str) -> str:
    """Return a one-line description of a company by name."""

    return f"{name}: stub description; wire this to a real API."


async def main() -> None:
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise SystemExit("Set FIREWORKS_API_KEY before running this example.")

    async for msg in query(
        prompt="Tell me about Spawn Labs in one sentence.",
        options={
            "model": "accounts/fireworks/models/deepseek-v3",
            "backend": "https://api.fireworks.ai/inference/v1",
            "tools": [lookup_company],
            "system": (
                "You are a research assistant. Use the lookup_company tool "
                "when asked about companies."
            ),
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
                f"${msg.total_cost_usd:.4f} (in={msg.usage.input_tokens}, "
                f"out={msg.usage.output_tokens})"
            )


if __name__ == "__main__":
    asyncio.run(main())
