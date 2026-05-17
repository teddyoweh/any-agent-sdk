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
    run_setup_local,
    start_ollama_server,
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


# ---------------------------------------------------------------------------
# Auto-start `ollama serve` when the daemon isn't running
# ---------------------------------------------------------------------------


def test_start_ollama_server_short_circuits_when_already_running() -> None:
    """If the daemon already answers, we must not spawn a second one."""

    with patch(
        "any_agent_sdk.setup_local.is_ollama_running", return_value=True
    ) as running, patch(
        "any_agent_sdk.setup_local.subprocess.Popen"
    ) as popen:
        ok, log_path = start_ollama_server()
    assert ok is True
    assert log_path is None
    assert running.called
    popen.assert_not_called()


def test_start_ollama_server_returns_false_when_not_installed() -> None:
    """No ollama on PATH → can't start it. Don't try."""

    with patch(
        "any_agent_sdk.setup_local.is_ollama_running", return_value=False
    ), patch(
        "any_agent_sdk.setup_local.is_ollama_installed", return_value=False
    ), patch(
        "any_agent_sdk.setup_local.subprocess.Popen"
    ) as popen:
        ok, log_path = start_ollama_server()
    assert ok is False
    assert log_path is None
    popen.assert_not_called()


def test_start_ollama_server_spawns_and_polls_until_ready() -> None:
    """When ollama is installed but down, spawn `ollama serve` and wait."""

    # First two probes return False (still booting), third returns True.
    probes = iter([False, False, True])

    def fake_running(*_a: object, **_k: object) -> bool:
        return next(probes)

    with patch(
        "any_agent_sdk.setup_local.is_ollama_installed", return_value=True
    ), patch(
        "any_agent_sdk.setup_local.is_ollama_running", side_effect=fake_running
    ), patch(
        "any_agent_sdk.setup_local.subprocess.Popen"
    ) as popen, patch(
        "any_agent_sdk.setup_local.time.sleep"  # don't actually sleep in tests
    ):
        ok, log_path = start_ollama_server(timeout_s=5.0)

    assert ok is True
    assert log_path is not None
    assert popen.call_count == 1
    # We MUST detach the daemon — otherwise it dies when the user's
    # `any-agent setup-local` process exits.
    _, kwargs = popen.call_args
    if "start_new_session" in kwargs:
        assert kwargs["start_new_session"] is True
    elif "creationflags" in kwargs:
        # Windows: must request a new process group + detached
        assert kwargs["creationflags"] != 0


def test_start_ollama_server_times_out_cleanly() -> None:
    """If the daemon never answers, return False — don't hang forever."""

    with patch(
        "any_agent_sdk.setup_local.is_ollama_installed", return_value=True
    ), patch(
        "any_agent_sdk.setup_local.is_ollama_running", return_value=False
    ), patch(
        "any_agent_sdk.setup_local.subprocess.Popen"
    ), patch(
        "any_agent_sdk.setup_local.time.sleep"
    ):
        ok, log_path = start_ollama_server(timeout_s=0.1)

    assert ok is False
    assert log_path is not None  # caller can read the captured server log


def test_run_setup_local_auto_starts_server_by_default() -> None:
    """The whole point of `setup-local`: don't punt to the user — start
    `ollama serve` ourselves when it's installed but not running."""

    with patch(
        "any_agent_sdk.setup_local.is_ollama_installed", return_value=True
    ), patch(
        "any_agent_sdk.setup_local.is_ollama_running", return_value=False
    ), patch(
        "any_agent_sdk.setup_local.start_ollama_server",
        return_value=(True, "/tmp/ollama.log"),
    ) as starter, patch(
        "any_agent_sdk.setup_local.pull_model", return_value=0
    ), patch(
        "any_agent_sdk.setup_local.smoke_test", return_value=True
    ):
        rc = run_setup_local(skip_smoke_test=True)

    assert rc == 0
    starter.assert_called_once()


def test_run_setup_local_can_opt_out_of_auto_start(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`auto_start_server=False` preserves the old behavior: just say
    'go run ollama serve' and exit 1."""

    with patch(
        "any_agent_sdk.setup_local.is_ollama_installed", return_value=True
    ), patch(
        "any_agent_sdk.setup_local.is_ollama_running", return_value=False
    ), patch(
        "any_agent_sdk.setup_local.start_ollama_server"
    ) as starter:
        rc = run_setup_local(auto_start_server=False)

    assert rc == 1
    starter.assert_not_called()
    out = capsys.readouterr().out
    assert "ollama serve" in out


def test_run_setup_local_reports_when_auto_start_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If auto-start fails, surface the captured log path so the user can
    actually debug — don't just say 'failed'."""

    with patch(
        "any_agent_sdk.setup_local.is_ollama_installed", return_value=True
    ), patch(
        "any_agent_sdk.setup_local.is_ollama_running", return_value=False
    ), patch(
        "any_agent_sdk.setup_local.start_ollama_server",
        return_value=(False, "/tmp/ollama-fail.log"),
    ):
        rc = run_setup_local()

    assert rc == 1
    err = capsys.readouterr().err
    assert "/tmp/ollama-fail.log" in err


def test_cli_no_auto_start_server_flag_parses() -> None:
    """The CLI must expose the opt-out flag and forward it correctly."""

    parser = cli._build_parser()
    args = parser.parse_args(["setup-local", "--no-auto-start-server"])
    assert args.auto_start_server is False

    args_default = parser.parse_args(["setup-local"])
    assert args_default.auto_start_server is True


def test_cli_start_timeout_flag_parses() -> None:
    """`--start-timeout 30` should land in args as start_timeout_s=30.0."""

    parser = cli._build_parser()
    args = parser.parse_args(["setup-local", "--start-timeout", "30"])
    assert args.start_timeout_s == 30.0
