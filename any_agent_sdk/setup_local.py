"""``any-agent setup-local`` — get a working CPU-runnable model in <2 min.

The CLI surface is ``any-agent setup-local`` (see :mod:`any_agent_sdk.cli`).
This module owns the actual logic: installer detection, the curated
CPU-friendly model list, the pull, and a smoke test.

Bar for "CPU-friendly" used here: runs at ≥ 3 tok/s on a 2020-era
laptop with 8 GB RAM and no discrete GPU. Models exceeding 7 B
parameters are excluded — they technically run on CPU but are too slow
to be useful for an agent loop.

The list is intentionally short. Users who want exotic models can run
``ollama pull <name>`` themselves; this helper is for the first-five-
minutes user who just wants something working.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = [
    "LocalModel",
    "CPU_FRIENDLY_MODELS",
    "DEFAULT_RECOMMENDATION",
    "is_ollama_installed",
    "is_ollama_running",
    "install_ollama",
    "pull_model",
    "smoke_test",
    "run_setup_local",
]


# ---------------------------------------------------------------------------
# Curated CPU-friendly model list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalModel:
    """A model we recommend for CPU-only use, with the install one-liner."""

    tag: str                # Ollama tag — what `ollama pull X` accepts
    params: str             # Human-readable param count ("1.5B", "3B", ...)
    size_gb: float          # On-disk size after Q4 quantization
    min_ram_gb: int         # Minimum RAM to run without swapping
    family: str             # Llama / Qwen / DeepSeek / Phi / Gemma / TinyLlama
    tools: bool             # Does the model emit usable tool calls?
    thinking: bool          # Does it emit <think>...</think> reasoning blocks?
    notes: str              # One-line "when to use this"


# Ordered from smallest → largest. The default recommendation is the
# smallest that still has tool-use chops; the user can pick larger if
# their machine handles it.
CPU_FRIENDLY_MODELS: list[LocalModel] = [
    LocalModel(
        tag="smollm2:135m",
        params="135M",
        size_gb=0.27,
        min_ram_gb=2,
        family="SmolLM2",
        tools=False,
        thinking=False,
        notes="Tiny — chat only, no tool use. Sanity-check installs.",
    ),
    LocalModel(
        tag="qwen2.5:0.5b",
        params="0.5B",
        size_gb=0.4,
        min_ram_gb=2,
        family="Qwen",
        tools=True,
        thinking=False,
        notes="Smallest Qwen that does tool calls. Fast on anything.",
    ),
    LocalModel(
        tag="tinyllama:1.1b",
        params="1.1B",
        size_gb=0.64,
        min_ram_gb=2,
        family="TinyLlama",
        tools=False,
        thinking=False,
        notes="Famously light — chat only. Good RAM-constrained pick.",
    ),
    LocalModel(
        tag="qwen2.5:1.5b",
        params="1.5B",
        size_gb=1.0,
        min_ram_gb=4,
        family="Qwen",
        tools=True,
        thinking=False,
        notes="Best 1.5B all-rounder for agent loops.",
    ),
    LocalModel(
        tag="deepseek-r1:1.5b",
        params="1.5B",
        size_gb=1.1,
        min_ram_gb=4,
        family="DeepSeek",
        tools=True,
        thinking=True,
        notes="Reasoning model — emits <think> blocks. Slowest of the 1.5B class.",
    ),
    LocalModel(
        tag="llama3.2:1b",
        params="1.2B",
        size_gb=1.3,
        min_ram_gb=4,
        family="Llama",
        tools=True,
        thinking=False,
        notes="Meta's 1B — native tool calls, sharper than 0.5B Qwen.",
    ),
    LocalModel(
        tag="gemma2:2b",
        params="2B",
        size_gb=1.6,
        min_ram_gb=4,
        family="Gemma",
        tools=False,
        thinking=False,
        notes="Google's small one — chat only, very polished prose.",
    ),
    LocalModel(
        tag="qwen2.5:3b",
        params="3B",
        size_gb=1.9,
        min_ram_gb=6,
        family="Qwen",
        tools=True,
        thinking=False,
        notes="Same class as Llama 3.2 3B — pick whichever feels better.",
    ),
    LocalModel(
        tag="llama3.2:3b",
        params="3.2B",
        size_gb=2.0,
        min_ram_gb=6,
        family="Llama",
        tools=True,
        thinking=False,
        notes="Meta's 3B — solid default if you have 8 GB RAM.",
    ),
    LocalModel(
        tag="phi3.5:3.8b",
        params="3.8B",
        size_gb=2.2,
        min_ram_gb=6,
        family="Phi",
        tools=True,
        thinking=False,
        notes="MS Phi-3.5 — strong reasoning for size; tool calls work.",
    ),
    LocalModel(
        tag="qwen2.5:7b",
        params="7B",
        size_gb=4.7,
        min_ram_gb=8,
        family="Qwen",
        tools=True,
        thinking=False,
        notes="Ceiling for CPU — works, but slow without GPU/Apple Silicon.",
    ),
    LocalModel(
        tag="llama3.1:8b",
        params="8B",
        size_gb=4.9,
        min_ram_gb=8,
        family="Llama",
        tools=True,
        thinking=False,
        notes="At the edge of CPU usability — fine on M-series, slow elsewhere.",
    ),
]


# The default pick: small enough to run anywhere with 4 GB free RAM,
# big enough to do tool calls in our agent loop. Bias toward speed for
# the first-time user.
DEFAULT_RECOMMENDATION = "qwen2.5:1.5b"


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def is_ollama_installed() -> bool:
    """Is the ``ollama`` binary on PATH?"""
    return shutil.which("ollama") is not None


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Does the local Ollama HTTP server answer ``/api/version``?"""
    try:
        with urllib.request.urlopen(f"{base_url}/api/version", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


# ---------------------------------------------------------------------------
# Install + pull
# ---------------------------------------------------------------------------


def install_ollama(*, auto_confirm: bool = False) -> int:
    """Install Ollama via the official installer.

    Linux/macOS only — Windows users should install via the .exe from
    ollama.com (we print a link instead of trying). Returns the exit
    code of the install script.
    """

    if sys.platform.startswith("win"):
        print(
            "Windows: download the installer from https://ollama.com/download/windows",
            file=sys.stderr,
        )
        return 2

    if not auto_confirm:
        print("This will run: curl -fsSL https://ollama.com/install.sh | sh")
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 1
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    # We don't shell `curl | sh` directly because we want to surface
    # errors cleanly. Two-step: download, then exec.
    try:
        with urllib.request.urlopen(
            "https://ollama.com/install.sh", timeout=30
        ) as r:
            script = r.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"Failed to fetch installer: {e}", file=sys.stderr)
        return 1

    return subprocess.call(["sh", "-c", script])


def pull_model(tag: str) -> int:
    """``ollama pull <tag>`` — stream output to the user's terminal."""

    if not is_ollama_installed():
        print(
            "ollama is not on PATH. Run `any-agent setup-local --install-ollama`.",
            file=sys.stderr,
        )
        return 1

    print(f"Pulling {tag} …")
    return subprocess.call(["ollama", "pull", tag])


def smoke_test(tag: str, base_url: str = "http://localhost:11434") -> bool:
    """Send a one-token prompt to confirm the model loads and responds."""

    body = json.dumps(
        {
            "model": tag,
            "prompt": "Reply with one word: ok",
            "stream": False,
            "options": {"num_predict": 8},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
            text = (payload.get("response") or "").strip()
            print(f"  smoke-test response: {text[:120]!r}")
            return bool(text)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"  smoke-test failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Orchestrator (called by ``any-agent setup-local``)
# ---------------------------------------------------------------------------


def run_setup_local(
    *,
    model: str | None = None,
    install_ollama_if_missing: bool = False,
    skip_smoke_test: bool = False,
    base_url: str = "http://localhost:11434",
) -> int:
    """End-to-end: install Ollama if needed → pull the model → smoke test."""

    if not is_ollama_installed():
        if not install_ollama_if_missing:
            print("Ollama is not installed.")
            print("Run: any-agent setup-local --install-ollama")
            return 1
        rc = install_ollama()
        if rc != 0:
            return rc

    if not is_ollama_running(base_url):
        print(f"Ollama server not responding at {base_url}.")
        print("Start it with: `ollama serve` (or restart the Ollama app).")
        return 1

    tag = model or DEFAULT_RECOMMENDATION
    # Validate the tag against our curated list when the user passed one,
    # but allow anything — we don't want to gate on a hardcoded allowlist.
    known = {m.tag: m for m in CPU_FRIENDLY_MODELS}
    if tag in known:
        m = known[tag]
        print(f"Selected: {m.tag}  ({m.params}, ~{m.size_gb:.1f} GB, RAM ≥ {m.min_ram_gb} GB)")
        print(f"  {m.notes}")
    else:
        print(f"Selected: {tag}  (not in curated list — proceeding anyway)")

    rc = pull_model(tag)
    if rc != 0:
        return rc

    if skip_smoke_test:
        print("Done.")
        return 0

    print("Verifying with a tiny prompt …")
    if not smoke_test(tag, base_url):
        print("Smoke test failed — pull succeeded but the model didn't respond.")
        return 1

    print()
    print(f"All set. Use {tag!r} in your code:")
    print()
    print("    from any_agent_sdk import query, ClaudeAgentOptions, tool")
    print("    async for msg in query(")
    print(f"        prompt=\"hi\", options=ClaudeAgentOptions(model={tag!r}),")
    print("    ):")
    print("        print(msg)")
    print()
    return 0


def print_model_table() -> None:
    """Pretty-print the CPU-friendly model catalog. Used by
    ``any-agent setup-local --list``."""

    name_w = max(len(m.tag) for m in CPU_FRIENDLY_MODELS) + 2
    fmt = (
        f"{{tag:<{name_w}}} {{params:>7}} {{size:>8}} {{ram:>7}} "
        "{tools:^7} {think:^7}  {notes}"
    )
    print(
        fmt.format(
            tag="MODEL", params="PARAMS", size="SIZE", ram="RAM",
            tools="TOOLS", think="THINK", notes="NOTES",
        )
    )
    print("-" * (name_w + 70))
    for m in CPU_FRIENDLY_MODELS:
        print(
            fmt.format(
                tag=m.tag,
                params=m.params,
                size=f"{m.size_gb:.1f}GB",
                ram=f"{m.min_ram_gb}GB+",
                tools="yes" if m.tools else "no",
                think="yes" if m.thinking else "no",
                notes=m.notes,
            )
        )
    print()
    print(f"Default recommendation: {DEFAULT_RECOMMENDATION}")
