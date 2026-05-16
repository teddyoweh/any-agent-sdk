"""Tools — declaration, registry, dispatch.

Design
------
* ``@tool`` decorator turns an async function into a ``Tool``. Schema is
  derived from the function's type hints + docstring; we don't ship a
  separate schema-building layer.
* ``ToolRegistry`` holds the live set. Lookup is O(1) by name.
* Dispatch runs tool calls **in parallel by default** via ``anyio.create_task_group``,
  with single-flight enforcement on tools declared ``parallel_safe=False``.
* Exceptions in tool bodies are caught and surfaced as ``ToolResultBlock``
  with ``is_error=True`` — the agent loop never crashes because of a
  user tool throwing.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

import anyio
import msgspec

from .errors import ToolExecutionError
from .types import ToolResultBlock, ToolUseBlock

ToolFn = Callable[..., Awaitable[Any]]


# ---------------------------------------------------------------------------
# Tool model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Tool:
    """Runtime representation of a tool the agent can call."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn
    parallel_safe: bool = True

    def to_wire(self) -> dict[str, Any]:
        """Anthropic-format tool definition. Other providers convert from this."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def tool(
    fn: ToolFn | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parallel_safe: bool = True,
    input_schema: dict[str, Any] | None = None,
) -> Tool | Callable[[ToolFn], Tool]:
    """Decorator: turn an async function into a ``Tool``.

    Usage::

        @tool
        async def get_weather(city: str) -> str:
            \"\"\"Get current weather for a city.\"\"\"
            ...

    Schema is auto-derived from type hints + docstring. Pass ``input_schema``
    explicitly to override (useful for complex shapes).
    """

    def _wrap(fn: ToolFn) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@tool requires async def, got {fn!r}")
        return Tool(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").strip(),
            input_schema=input_schema or _derive_schema(fn),
            fn=fn,
            parallel_safe=parallel_safe,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap


def _derive_schema(fn: ToolFn) -> dict[str, Any]:
    """Build a JSON Schema from the function's annotations.

    Keeps it simple: supports str/int/float/bool/list/dict primitives and
    treats anything else as a generic object. For richer schemas, pass
    ``input_schema=`` explicitly to ``@tool``.
    """

    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        ann = hints.get(param_name, str)
        props[param_name] = _type_to_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


_SCALAR_MAP = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _type_to_schema(t: Any) -> dict[str, Any]:
    if t in _SCALAR_MAP:
        return _SCALAR_MAP[t]
    origin = getattr(t, "__origin__", None)
    if origin is list:
        args = getattr(t, "__args__", ())
        inner = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": inner}
    if origin is dict:
        return {"type": "object"}
    return {"type": "object"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolRegistry:
    """Holds tools by name. Cheap to construct per ``Agent``."""

    _by_name: dict[str, Tool] = field(default_factory=dict)

    def add(self, *tools: Tool) -> None:
        for t in tools:
            if t.name in self._by_name:
                raise ValueError(f"duplicate tool name {t.name!r}")
            self._by_name[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def to_wire(self) -> list[dict[str, Any]]:
        return [t.to_wire() for t in self._by_name.values()]

    def __bool__(self) -> bool:
        return bool(self._by_name)

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ENC = msgspec.json.Encoder()


async def dispatch_tool_calls(
    registry: ToolRegistry,
    calls: list[ToolUseBlock],
) -> list[ToolResultBlock]:
    """Run every tool call in ``calls`` and return result blocks in order.

    Parallel by default; non-parallel-safe tools are serialized by tool name.

    Order of returned results matches the order of ``calls`` — callers can
    pair them by ``tool_use_id``, but stable ordering keeps debugging sane.
    """

    if not calls:
        return []

    results: list[ToolResultBlock | None] = [None] * len(calls)
    # Per-name locks for non-parallel-safe tools.
    locks: dict[str, anyio.Lock] = {}

    async def run_one(idx: int, call: ToolUseBlock) -> None:
        t = registry.get(call.name)
        if t is None:
            results[idx] = ToolResultBlock(
                tool_use_id=call.id,
                content=f"tool {call.name!r} not found",
                is_error=True,
            )
            return

        lock = None
        if not t.parallel_safe:
            lock = locks.setdefault(t.name, anyio.Lock())

        try:
            if lock is not None:
                async with lock:
                    out = await t.fn(**call.input)
            else:
                out = await t.fn(**call.input)
        except Exception as e:  # noqa: BLE001 — user code, must not crash the loop
            # Wrap for typed handling upstream, but still produce a result block.
            err = ToolExecutionError(call.name, call.id, e)
            results[idx] = ToolResultBlock(
                tool_use_id=call.id,
                content=str(err),
                is_error=True,
            )
            return

        results[idx] = ToolResultBlock(
            tool_use_id=call.id,
            content=_stringify_result(out),
        )

    async with anyio.create_task_group() as tg:
        for i, c in enumerate(calls):
            tg.start_soon(run_one, i, c)

    # All slots are guaranteed filled by the task group exit.
    return [r for r in results if r is not None]


def _stringify_result(out: Any) -> str:
    """Coerce a tool return to a string suitable for the wire.

    Strings pass through. msgspec-encodable objects get JSON-encoded.
    Anything else falls back to ``str()``.
    """

    if isinstance(out, str):
        return out
    try:
        return _ENC.encode(out).decode()
    except (TypeError, msgspec.EncodeError):
        return str(out)
