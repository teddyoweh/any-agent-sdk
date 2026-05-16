"""``any-agent`` — diagnostic + quick-chat CLI.

The CLI is *not* meant to be a TUI agent runner. It exists for:

* probing a backend (does it speak OpenAI-compat? does it advertise tool use?)
* listing the bundled model capability table
* one-shot ``run`` for quick smoke tests of (model, backend) pairs
* an interactive ``chat`` loop for poking at a server

We deliberately depend only on ``argparse`` (stdlib) — no ``click`` / ``typer``.
Cold-start matters for ``any-agent --help`` to feel snappy.

Subcommands::

    any-agent version
    any-agent list-models
    any-agent probe   --backend http://localhost:11434
    any-agent run     "prompt..."   --model qwen2.5-7b-instruct --backend http://...
    any-agent chat                  --model qwen2.5-7b-instruct --backend http://...

``run`` and ``chat`` go through :func:`any_agent_sdk.query`. ``probe`` opens
the backend's models endpoint, reads any well-known signal, and prints a
capability summary.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import anyio

from . import __version__
from .capabilities import (
    HOSTED_PROFILES,
    BackendCapability,
    hosted_profile_from_url,
    lookup_model,
    resolve_tool_use_path,
)
from .events import ContentBlockDelta, TextDelta
from .providers.base import detect_provider, resolve

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="any-agent",
        description=(
            "any-agent-sdk CLI — diagnostics + quick chat. "
            "For library use, import any_agent_sdk in Python instead."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    sub.add_parser("version", help="Print the SDK version and exit.")

    sub.add_parser("list-models", help="Print the bundled model capability table.")

    p_probe = sub.add_parser(
        "probe", help="Hit a backend URL and report what we can detect."
    )
    p_probe.add_argument("--backend", required=True, help="Backend base URL.")

    p_run = sub.add_parser(
        "run", help="One-shot: send a prompt, print the final assistant response."
    )
    p_run.add_argument("prompt", help="The user prompt to send.")
    _add_agent_flags(p_run)

    p_chat = sub.add_parser(
        "chat", help="Interactive stdin chat loop. Streams tokens as they arrive."
    )
    _add_agent_flags(p_chat)
    p_chat.add_argument(
        "--system", default=None, help="System prompt for the chat session."
    )

    return p


def _add_agent_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model", required=True, help="Model slug (e.g. qwen2.5-7b-instruct)."
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Backend base URL. Defaults to env-detected.",
    )
    p.add_argument("--api-key", default=None, help="Override API key (else env).")
    p.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens per assistant turn."
    )
    p.add_argument("--temperature", type=float, default=None, help="Sampling temp.")
    p.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum agent turns before stopping (default 10).",
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"any-agent-sdk {__version__}")
    return 0


def _cmd_list_models(_args: argparse.Namespace) -> int:
    # Import lazily so other subcommands skip the table cost.
    from .capabilities import _TABLE  # noqa: PLC0415 — intentional lazy import

    # Column widths sized to fit the longest known model name.
    name_w = max(len(k) for k in _TABLE) + 2
    fmt = f"{{name:<{name_w}}} {{family:<10}} {{ctx:>8}} {{tools:^7}} {{think:^7}}"
    print(fmt.format(name="MODEL", family="FAMILY", ctx="CTX", tools="TOOLS", think="THINK"))
    print("-" * (name_w + 36))
    for key, cap in sorted(_TABLE.items()):
        print(
            fmt.format(
                name=key,
                family=cap.family,
                ctx=str(cap.context_window),
                tools="yes" if cap.supports_native_tools else "no",
                think="yes" if cap.emits_inline_thinking else "no",
            )
        )
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    url = args.backend
    print(f"backend url: {url}")

    name = detect_provider(url)
    print(f"detected provider: {name}")

    profile = hosted_profile_from_url(url)
    if profile is not None:
        _print_profile("matched hosted profile", profile)
    else:
        print("matched hosted profile: <none — using adapter default>")

    # Live probe — only if the backend looks reachable. We don't want to hang
    # on unreachable URLs; httpx.get with a short timeout suffices.
    try:
        import httpx  # noqa: PLC0415

        # Most servers expose /v1/models or /api/tags.
        candidate_paths = ("/v1/models", "/api/tags", "/api/version")
        for path in candidate_paths:
            try:
                with httpx.Client(timeout=2.0) as c:
                    r = c.get(url.rstrip("/") + path)
                if r.status_code < 500:
                    print(f"reachable: {path} -> HTTP {r.status_code}")
                    break
            except httpx.HTTPError:
                continue
        else:
            print("reachable: <no well-known endpoint responded under 2s>")
    except ImportError:  # pragma: no cover — httpx is a hard dep
        print("reachable: <httpx not importable, skipping live probe>")

    return 0


def _print_profile(label: str, profile: BackendCapability) -> None:
    print(f"{label}: {profile.provider_hint or profile.kind}")
    print(f"  native tools:  {'yes' if profile.supports_native_tools else 'no'}")
    print(f"  grammar:       {'yes' if profile.supports_grammar else 'no'}")
    print(f"  logprobs:      {'yes' if profile.supports_logprobs else 'no'}")
    print(f"  prefix cache:  {'yes' if profile.supports_prefix_caching else 'no'}")


async def _cmd_run_async(args: argparse.Namespace) -> int:
    # Imported here so the CLI cold-start path (e.g. ``any-agent version``)
    # doesn't pay for the query module's transitive imports.
    from .query import (  # noqa: PLC0415
        SDKAssistantMessage,
        SDKResultMessage,
        query,
    )

    options = _build_options(args)
    async for msg in query(prompt=args.prompt, options=options):
        if isinstance(msg, SDKAssistantMessage):
            for block in msg.content_blocks:
                text = getattr(block, "text", None)
                if text:
                    print(text)
        elif isinstance(msg, SDKResultMessage):
            if msg.is_error:
                print(f"[error] {msg.result}", file=sys.stderr)
                return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return anyio.run(_cmd_run_async, args)


async def _cmd_chat_async(args: argparse.Namespace) -> int:
    """Interactive REPL. Token-level streaming via ``Agent.stream``."""

    from .agent import Agent  # noqa: PLC0415
    from .tools import ToolRegistry  # noqa: PLC0415
    from .types import (  # noqa: PLC0415
        AssistantMessage,
        UserMessage,
    )

    provider = _build_provider_for_args(args)
    agent = Agent(
        model=args.model,
        provider=provider,
        system=args.system,
        tools=ToolRegistry(),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_steps=args.max_turns,
    )

    print(f"any-agent chat — model={args.model} backend={args.backend or '<default>'}")
    print("type /exit to quit, /reset to clear history, blank line to send")
    messages: list[Any] = []
    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "/exit":
                break
            if line == "/reset":
                messages = []
                print("(history cleared)")
                continue

            messages.append(UserMessage(content=line))
            print("agent> ", end="", flush=True)

            # Token-level stream. Build the final assistant message in parallel
            # so we can append it to history and loop.
            collected: list[str] = []
            async for ev in agent.stream(messages):
                if isinstance(ev, ContentBlockDelta) and isinstance(ev.delta, TextDelta):
                    print(ev.delta.text, end="", flush=True)
                    collected.append(ev.delta.text)
            print()
            # Naive append — we lose tool_use blocks here. The interactive REPL
            # is for smoke testing; complex tool loops should use run() in code.
            messages.append(
                AssistantMessage(content=[_text_block("".join(collected))])
            )
    finally:
        await agent.aclose()
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    return anyio.run(_cmd_chat_async, args)


def _text_block(text: str) -> Any:
    from .types import TextBlock  # noqa: PLC0415

    return TextBlock(text=text)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_options(args: argparse.Namespace) -> dict[str, Any]:
    """Turn argparse output into a ``query()`` options dict."""

    out: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
    }
    if args.backend:
        out["backend"] = args.backend
    if args.temperature is not None:
        out["temperature"] = args.temperature
    if args.api_key:
        out["api_key"] = args.api_key
    return out


def _build_provider_for_args(args: argparse.Namespace) -> Any:
    """Provider constructor for ``chat`` — same logic as ``query._build_provider``
    but inlined so chat can stream without going through ``query()``."""

    name = detect_provider(args.backend or args.model)
    factory = resolve(name)
    kwargs: dict[str, Any] = {}
    if args.backend and args.backend.startswith(("http://", "https://")):
        kwargs["base_url"] = args.backend
    if args.api_key:
        kwargs["api_key"] = args.api_key
    try:
        return factory(**kwargs)
    except TypeError:
        return factory()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_HANDLERS: dict[str, Any] = {
    "version": _cmd_version,
    "list-models": _cmd_list_models,
    "probe": _cmd_probe,
    "run": _cmd_run,
    "chat": _cmd_chat,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.cmd]
    try:
        rc = handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return int(rc or 0)


if __name__ == "__main__":  # pragma: no cover — entry-point shim
    sys.exit(main())


__all__ = ["main"]


# Silence unused import warnings for symbols we re-expose for tests.
_ = (lookup_model, resolve_tool_use_path, HOSTED_PROFILES, os)
