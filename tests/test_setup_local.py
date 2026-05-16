"""``any-agent setup-local`` — CPU-friendly model catalog + install helper.

These tests cover the pure / mockable bits: the catalog shape, the
default recommendation, environment detection, and that the CLI
subcommand parses + dispatches to the right handler. The actual
``ollama pull`` and ``ollama serve`` paths are integration territory.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from any_agent_sdk import cli
from any_agent_sdk.setup_local import (
    CPU_FRIENDLY_MODELS,
    DEFAULT_RECOMMENDATION,
    LocalModel,
    is_ollama_installed,
    is_ollama_running,
    print_model_table,
)


# ---------------------------------------------------------------------------
# Catalog invariants
# ---------------------------------------------------------------------------


def test_catalog_is_nonempty_and_well_typed() -> None:
    assert len(CPU_FRIENDLY_MODELS) >= 5
    for m in CPU_FRIENDLY_MODELS:
        assert isinstance(m, LocalModel)
        assert m.tag and ":" in m.tag  # all entries are Ollama tag-form
        assert m.size_gb > 0
        assert m.min_ram_gb > 0
        assert m.family


def test_default_recommendation_is_in_catalog() -> None:
    tags = {m.tag for m in CPU_FRIENDLY_MODELS}
    assert DEFAULT_RECOMMENDATION in tags


def test_catalog_ordered_by_size() -> None:
    """Smaller models first — we want users to see the cheap options
    at the top of the table."""

    sizes = [m.size_gb for m in CPU_FRIENDLY_MODELS]
    assert sizes == sorted(sizes), "CPU_FRIENDLY_MODELS should be size-ascending"


def test_catalog_caps_at_8b() -> None:
    """Nothing > 8 B params — anything bigger is GPU territory."""

    for m in CPU_FRIENDLY_MODELS:
        # `params` is a short string like "1.5B" / "8B" — strip and parse.
        n = float(m.params.rstrip("BMK").rstrip("b").rstrip())
        if m.params.endswith(("M", "m")):
            n /= 1000  # 135M = 0.135B
        assert n <= 8.0, f"{m.tag} ({m.params}) exceeds CPU ceiling"


def test_reasoning_models_flagged_thinking() -> None:
    """If a tag is in the DeepSeek-R1 family, it must be marked
    `thinking=True` so the agent loop knows to parse `<think>` blocks."""

    for m in CPU_FRIENDLY_MODELS:
        if m.family == "DeepSeek":
            assert m.thinking, f"{m.tag} should emit <think> blocks"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def test_is_ollama_installed_returns_bool() -> None:
    """Type-check only — actual installation state varies per machine."""

    assert isinstance(is_ollama_installed(), bool)


def test_is_ollama_running_returns_false_for_unreachable_host() -> None:
    """A host that can't possibly answer should give False, not raise."""

    assert is_ollama_running("http://127.0.0.1:1") is False


# ---------------------------------------------------------------------------
# CLI plumbing — `any-agent setup-local --list`
# ---------------------------------------------------------------------------


def test_setup_local_command_registered() -> None:
    """The CLI parser must accept `setup-local` as a subcommand."""

    parser = cli._build_parser()
    args = parser.parse_args(["setup-local", "--list"])
    assert args.cmd == "setup-local"
    assert args.list_models is True


def test_setup_local_list_prints_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    """`any-agent setup-local --list` should print every model tag."""

    print_model_table()
    out = capsys.readouterr().out
    for m in CPU_FRIENDLY_MODELS:
        assert m.tag in out
    assert "Default recommendation" in out


def test_setup_local_list_dispatches_through_main(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: invoking the CLI with `setup-local --list` runs
    cleanly and returns 0."""

    rc = cli.main(["setup-local", "--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert DEFAULT_RECOMMENDATION in out
