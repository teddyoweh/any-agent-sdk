"""Sub-agents — orchestration via the tool channel.

A sub-agent is just another tool from the parent's point of view. When the
parent's model decides to call it, we instantiate a child ``Agent`` with the
sub-agent's system prompt + tool kit, run it to completion, and surface the
child's final assistant text as the tool result.

This means the parent's agent loop in ``agent.py`` does **not** need to know
sub-agents exist — they look like any other ``Tool``. All the orchestration
lives in this file.

Isolation modes
---------------
* ``asyncio_task`` (v0 default) — child runs in the same event loop, shares
  the parent's HTTP client pool via the inherited provider. Cheap. The only
  one fully implemented in M3.
* ``subprocess`` — fork a Python child, talk to it over stdio. Hard isolation;
  ~80 ms spawn tax. Stubbed (raises ``NotImplementedError``); lands in M4.
* ``remote`` — submit to a worker node via the SDK's own protocol. Used in
  distributed deployments. Stubbed too; lands in M4.

The parent passes its provider into the child by default so the child reuses
the open HTTP connection pool — that's the single biggest perf win for the
common asyncio_task case, and it's why we don't make people wire it up by
hand.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .agent import Agent
from .providers.base import Provider
from .tools import Tool, ToolRegistry
from .types import AssistantMessage, Message, TextBlock, UserMessage

IsolationMode = Literal["asyncio_task", "subprocess", "remote"]


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubAgentSpec:
    """Describes a sub-agent the parent can invoke as a tool.

    ``name`` is what the model sees and calls.
    ``system_prompt`` defines the sub-agent's persona / scope.
    ``model`` overrides the parent's model (often a cheaper / faster one).
    ``tools`` is the kit the sub-agent has access to — usually a *subset* of
        the parent's, locked down to its responsibility.
    ``max_turns`` caps the child's loop independently of the parent's.
    ``isolation`` picks the execution mode (see module docstring).
    ``description`` is shown to the parent model; defaults to a generic line.
    """

    name: str
    system_prompt: str
    model: str
    tools: list[Tool] = field(default_factory=list)
    max_turns: int = 10
    isolation: IsolationMode = "asyncio_task"
    description: str | None = None


# ---------------------------------------------------------------------------
# SubAgentTool — the bridge object
# ---------------------------------------------------------------------------


def _spec_to_input_schema() -> dict[str, Any]:
    """Every sub-agent takes a single ``prompt`` string. Keep it dumb on purpose
    — richer arg shapes are easier to design once we have real usage."""

    return {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Task the sub-agent should accomplish.",
            },
        },
        "required": ["prompt"],
    }


class SubAgentTool(Tool):
    """A ``Tool`` whose body spawns a child ``Agent`` and runs it to completion.

    The result returned to the parent is the child's final assistant text,
    concatenated from any ``TextBlock``s in the last message. Tool calls
    happen inside the child loop; the parent never sees them.

    We extend ``Tool`` so this drops straight into a ``ToolRegistry`` without
    any special-casing in the agent loop — that's the whole design.
    """

    __slots__ = ("_spec", "_parent_provider")

    def __init__(
        self,
        spec: SubAgentSpec,
        *,
        parent_provider: Provider | None = None,
    ) -> None:
        # Tool is a slotted dataclass — initialize via its fields. We then
        # capture our spec/provider on the instance.
        super().__init__(
            name=spec.name,
            description=spec.description or f"Delegate to the {spec.name} sub-agent.",
            input_schema=_spec_to_input_schema(),
            fn=self._invoke,  # type: ignore[arg-type]
            # Sub-agents involve LLM calls; serializing same-name invocations
            # is safer than racing them through the parent's tool dispatcher.
            parallel_safe=False,
        )
        self._spec = spec
        self._parent_provider = parent_provider

    # ------------------------------------------------------------------
    # The tool body — called by ``dispatch_tool_calls``.
    # ------------------------------------------------------------------

    async def _invoke(self, prompt: str) -> str:
        if self._spec.isolation == "asyncio_task":
            return await self._run_inproc(prompt)
        if self._spec.isolation == "subprocess":
            raise NotImplementedError(
                "subprocess isolation lands in M4 — see plan.md §4.10"
            )
        if self._spec.isolation == "remote":
            raise NotImplementedError(
                "remote isolation lands in M4 — see plan.md §4.10"
            )
        raise ValueError(f"unknown isolation mode: {self._spec.isolation!r}")

    async def _run_inproc(self, prompt: str) -> str:
        """asyncio_task mode: instantiate a child Agent in this loop."""

        registry = ToolRegistry()
        if self._spec.tools:
            registry.add(*self._spec.tools)

        child = Agent(
            model=self._spec.model,
            provider=self._parent_provider,  # share the parent's HTTP pool
            system=self._spec.system_prompt,
            tools=registry,
            max_steps=self._spec.max_turns,
        )

        # The child sees a single user turn: the prompt the parent passed in.
        messages: list[Message] = [UserMessage(content=prompt)]
        await child.run(messages)
        return _extract_final_text(messages)


def as_subagent_tool(
    spec: SubAgentSpec,
    *,
    parent_provider: Provider | None = None,
) -> Tool:
    """Turn a ``SubAgentSpec`` into a ``Tool`` the parent agent can register.

    ``parent_provider`` is optional but recommended in asyncio_task mode —
    passing the parent's provider lets the child reuse the open HTTP client
    pool, which is the dominant perf win for in-process sub-agents.
    """

    return SubAgentTool(spec, parent_provider=parent_provider)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_final_text(messages: list[Message]) -> str:
    """Pick the last assistant message and stitch its text blocks together.

    If the child ran out of turns without producing text (e.g. last turn was
    all tool calls), fall back to a stable marker so the parent model can
    still reason about what happened.
    """

    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            parts: list[str] = []
            for blk in msg.content:
                if isinstance(blk, TextBlock):
                    parts.append(blk.text)
            text = "".join(parts).strip()
            if text:
                return text
            # No text in the final assistant turn — surface stop_reason.
            return f"<sub-agent finished with stop_reason={msg.stop_reason!r} and no text>"
    return "<sub-agent produced no assistant message>"


# ---------------------------------------------------------------------------
# Subprocess / remote interface stubs
# ---------------------------------------------------------------------------
# These are documented here so M4 implementers know the contract. The stubs
# raise from ``_invoke`` above; we keep the protocols here as design notes.

SubprocessLauncher = Callable[[SubAgentSpec, str], Awaitable[str]]
"""Future M4 hook: spawn a Python subprocess, hand it the spec + prompt over
stdio, return the child's final text. Implementations will live in
``any_agent_sdk.runtime.subprocess`` and be wired in via a registry."""

RemoteLauncher = Callable[[SubAgentSpec, str], Awaitable[str]]
"""Future M4 hook: submit to a worker node over the SDK's wire protocol.
Implementations will live in ``any_agent_sdk.runtime.remote``."""
