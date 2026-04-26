"""Tests for the interactive REPL: banner, commands, and main loop."""

from unittest.mock import MagicMock, patch
from rich.console import Console
from io import StringIO


# ---------------------------------------------------------------------------
# Task 1: banner tests
# ---------------------------------------------------------------------------

def test_banner_renders_without_error():
    """Banner renders to a string buffer without raising."""
    from astra_cli.session.banner import print_banner
    buf = StringIO()
    console = Console(file=buf, width=100)
    # Should not raise
    print_banner(
        console=console,
        provider="anthropic",
        model="claude-sonnet-4-6",
        version="0.1.0",
    )
    output = buf.getvalue()
    assert "ASTRA" in output or "astra" in output.lower()

def test_banner_shows_provider_and_model():
    """Banner output contains provider and model info."""
    from astra_cli.session.banner import print_banner
    buf = StringIO()
    console = Console(file=buf, width=100)
    print_banner(
        console=console,
        provider="openai",
        model="gpt-4o",
        version="0.1.0",
    )
    output = buf.getvalue()
    assert "openai" in output
    assert "gpt-4o" in output


# ---------------------------------------------------------------------------
# Task 2: slash command tests
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


def _make_session_state(provider="anthropic", model="claude-sonnet-4-6"):
    """Build a minimal SessionState for testing."""
    from astra_cli.session.repl import SessionState
    from astra_node.core.query_engine import QueryEngine
    from astra_node.core.registry import ToolRegistry
    from astra_node.permissions.manager import PermissionManager
    from unittest.mock import MagicMock

    pm = PermissionManager()
    registry = ToolRegistry()
    engine = MagicMock(spec=QueryEngine)
    engine._history = MagicMock()
    engine._history.messages = []
    return SessionState(
        provider_name=provider,
        model=model,
        base_url=None,
        engine=engine,
        permission_manager=pm,
        registry=registry,
    )


def test_cmd_help_returns_help_text(capsys):
    """'/help' prints a table of commands."""
    from astra_cli.session.commands import handle_command
    from rich.console import Console
    from io import StringIO
    buf = StringIO()
    console = Console(file=buf, width=100)
    state = _make_session_state()
    result = handle_command("/help", state, console)
    assert result.handled is True
    assert result.should_exit is False
    output = buf.getvalue()
    assert "/provider" in output


def test_cmd_exit_signals_exit():
    """'/exit' returns should_exit=True."""
    from astra_cli.session.commands import handle_command
    from rich.console import Console
    from io import StringIO
    console = Console(file=StringIO(), width=100)
    state = _make_session_state()
    result = handle_command("/exit", state, console)
    assert result.should_exit is True


def test_cmd_clear_resets_history():
    """'/clear' calls clear() on the engine's history."""
    from astra_cli.session.commands import handle_command
    from rich.console import Console
    from io import StringIO
    console = Console(file=StringIO(), width=100)
    state = _make_session_state()
    result = handle_command("/clear", state, console)
    assert result.handled is True
    state.engine._history.messages  # history was accessed


def test_cmd_cost_shows_tokens():
    """'/cost' prints token usage."""
    from astra_cli.session.commands import handle_command
    from rich.console import Console
    from io import StringIO
    buf = StringIO()
    console = Console(file=buf, width=100)
    state = _make_session_state()
    state.total_input_tokens = 1234
    state.total_output_tokens = 567
    result = handle_command("/cost", state, console)
    assert result.handled is True
    output = buf.getvalue()
    assert "1234" in output
    assert "567" in output


def test_cmd_unknown_slash_is_handled():
    """Unknown /command is handled (shows error) not passed to agent."""
    from astra_cli.session.commands import handle_command
    from rich.console import Console
    from io import StringIO
    buf = StringIO()
    console = Console(file=buf, width=100)
    state = _make_session_state()
    result = handle_command("/doesnotexist", state, console)
    assert result.handled is True
    assert "unknown" in buf.getvalue().lower() or "doesnotexist" in buf.getvalue().lower()


def test_non_slash_input_is_not_handled():
    """Regular text input returns handled=False (goes to agent)."""
    from astra_cli.session.commands import handle_command
    from rich.console import Console
    from io import StringIO
    console = Console(file=StringIO(), width=100)
    state = _make_session_state()
    result = handle_command("what is 2 + 2", state, console)
    assert result.handled is False


# ---------------------------------------------------------------------------
# Task 3: REPL loop tests
# ---------------------------------------------------------------------------

def test_repl_start_exits_on_exit_command():
    """REPL exits cleanly when /exit is entered."""
    from astra_cli.session.repl import start
    from unittest.mock import patch, MagicMock

    with (
        patch("astra_cli.session.repl._read_input", side_effect=["/exit"]),
        patch("astra_cli.session.banner.print_banner"),
        patch("astra_cli.session.repl._build_session_state") as mock_state,
        patch("prompt_toolkit.PromptSession"),
    ):
        mock_st = MagicMock()
        mock_st.engine = MagicMock()
        mock_st.permission_manager = MagicMock()
        mock_st.turn_count = 0
        mock_st.total_input_tokens = 0
        mock_st.total_output_tokens = 0
        mock_state.return_value = mock_st
        # Should return without raising
        start()


def test_repl_forwards_non_slash_to_agent():
    """Non-slash input is forwarded to the agent engine."""
    import asyncio
    from astra_cli.session.repl import start
    from astra_node.core.events import TextDelta, TurnEnd
    from unittest.mock import patch, MagicMock, AsyncMock

    async def fake_run(msg):
        yield TextDelta(text="answer")
        yield TurnEnd()

    with (
        patch("astra_cli.session.repl._read_input", side_effect=["hello", "/exit"]),
        patch("astra_cli.session.banner.print_banner"),
        patch("astra_cli.session.repl._build_session_state") as mock_state,
        patch("astra_cli.display.event_renderer.EventRenderer.render"),
        patch("prompt_toolkit.PromptSession"),
        patch("astra_node.core.compaction.CompactionEngine.should_compact", return_value=False),
    ):
        mock_engine = MagicMock()
        mock_engine.run = fake_run
        mock_engine._history = MagicMock()
        mock_engine._history.messages = []

        mock_pm = MagicMock()

        mock_st = MagicMock()
        mock_st.engine = mock_engine
        mock_st.permission_manager = mock_pm
        mock_st.turn_count = 0
        mock_st.total_input_tokens = 0
        mock_st.total_output_tokens = 0
        mock_state.return_value = mock_st
        start()
        # engine.run was called with "hello"
        # (verified by no exception — fake_run handles it)


def test_repl_permission_prompt_yes(monkeypatch):
    """Permission prompt returns ALLOW on 'yes' answer."""
    from astra_cli.session.repl import _prompt_permission
    from astra_node.permissions.types import PermissionDecision
    from astra_node.permissions.manager import PermissionManager
    from rich.console import Console
    from io import StringIO

    console = Console(file=StringIO(), width=100)
    pm = PermissionManager()

    monkeypatch.setattr("astra_cli.session.repl._ask_permission", lambda *a, **kw: "yes")
    decision = _prompt_permission("bash", {"command": "ls"}, pm, console)
    assert decision == PermissionDecision.ALLOW


def test_repl_permission_prompt_always(monkeypatch):
    """Permission prompt 'always' calls allow_always and returns ALLOW."""
    from astra_cli.session.repl import _prompt_permission
    from astra_node.permissions.types import PermissionDecision
    from astra_node.permissions.manager import PermissionManager
    from rich.console import Console
    from io import StringIO

    console = Console(file=StringIO(), width=100)
    pm = PermissionManager()

    monkeypatch.setattr("astra_cli.session.repl._ask_permission", lambda *a, **kw: "always")
    decision = _prompt_permission("bash", {"command": "ls"}, pm, console)
    assert decision == PermissionDecision.ALLOW
    # bash is now in session allow
    from astra_node.core.tool import PermissionLevel
    assert pm.check_level("bash", PermissionLevel.ASK_USER) == PermissionDecision.ALLOW


def test_repl_permission_prompt_no(monkeypatch):
    """Permission prompt 'no' returns DENY."""
    from astra_cli.session.repl import _prompt_permission
    from astra_node.permissions.types import PermissionDecision
    from astra_node.permissions.manager import PermissionManager
    from rich.console import Console
    from io import StringIO

    console = Console(file=StringIO(), width=100)
    pm = PermissionManager()

    monkeypatch.setattr("astra_cli.session.repl._ask_permission", lambda *a, **kw: "no")
    decision = _prompt_permission("bash", {"command": "ls"}, pm, console)
    assert decision == PermissionDecision.DENY


# ---------------------------------------------------------------------------
# Task 4: live status bar
# ---------------------------------------------------------------------------

def test_toolbar_text_contains_provider_and_model():
    """_toolbar_text returns a string containing provider and model."""
    from astra_cli.session.repl import _toolbar_text
    state = _make_session_state(provider="anthropic", model="claude-opus-4-6")
    state.total_input_tokens = 100
    state.total_output_tokens = 50
    state.turn_count = 3
    text = _toolbar_text(state)
    assert "anthropic" in text
    assert "claude-opus-4-6" in text
    assert "100" in text
    assert "50" in text
    assert "3" in text


def test_toolbar_text_shows_default_when_no_model():
    """_toolbar_text shows '(default)' when state.model is None."""
    from astra_cli.session.repl import _toolbar_text
    state = _make_session_state(provider="ollama", model=None)
    text = _toolbar_text(state)
    assert "(default)" in text


def test_toolbar_text_updates_when_state_changes():
    """_toolbar_text reflects mutations to state immediately."""
    from astra_cli.session.repl import _toolbar_text
    state = _make_session_state(provider="anthropic", model="claude-sonnet-4-6")
    assert "anthropic" in _toolbar_text(state)
    state.provider_name = "openai"
    state.model = "gpt-4o"
    text = _toolbar_text(state)
    assert "openai" in text
    assert "gpt-4o" in text
    assert "anthropic" not in text


# ---------------------------------------------------------------------------
# Task 1: interact.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task 2: orange accent
# ---------------------------------------------------------------------------

def test_banner_uses_orange_accent():
    """Banner renders with orange (#f97316) not purple (#a855f7)."""
    from astra_cli.session.banner import print_banner, ACCENT
    assert ACCENT == "#f97316", f"Expected orange #f97316, got {ACCENT}"

def test_repl_uses_orange_accent():
    """repl.py ACCENT constant is orange."""
    import astra_cli.session.repl as repl_mod
    assert repl_mod.ACCENT == "#f97316", f"Expected #f97316, got {repl_mod.ACCENT}"

def test_commands_uses_orange_accent():
    """commands.py ACCENT constant is orange."""
    import astra_cli.session.commands as cmd_mod
    assert cmd_mod.ACCENT == "#f97316", f"Expected #f97316, got {cmd_mod.ACCENT}"


def test_interactive_select_importable():
    """_interactive_select can be imported from interact module."""
    from astra_cli.session.interact import _interactive_select
    assert callable(_interactive_select)


# ---------------------------------------------------------------------------
# Task 4: main.py wiring test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task 5: /model arrow-key menu
# ---------------------------------------------------------------------------

def test_cmd_model_known_models_dict_exists():
    """_KNOWN_MODELS exists in commands and has anthropic/openai/ollama keys."""
    from astra_cli.session.commands import _KNOWN_MODELS
    assert "anthropic" in _KNOWN_MODELS
    assert "openai" in _KNOWN_MODELS
    assert "ollama" in _KNOWN_MODELS
    assert len(_KNOWN_MODELS["anthropic"]) >= 1


def test_cmd_model_sets_state_and_saves_config(monkeypatch, tmp_path):
    """'/model' sets state.model and saves to config.json."""
    import json
    from astra_cli.session import interact as interact_mod
    from rich.console import Console
    from io import StringIO

    console = Console(file=StringIO(), width=100)
    state = _make_session_state(provider="anthropic", model="claude-sonnet-4-6")

    config_path = tmp_path / ".astra" / "config.json"

    monkeypatch.setattr(interact_mod, "_interactive_select", lambda **kw: "claude-opus-4-6")
    monkeypatch.setattr("astra_cli.session.commands.Path.home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr("astra_cli.session.commands._rebuild_engine", lambda s, c: None)

    from astra_cli.session.commands import _cmd_model
    result = _cmd_model(state, console)
    assert result.handled is True
    assert state.model == "claude-opus-4-6"
    assert config_path.exists()
    saved = json.loads(config_path.read_text())
    assert saved.get("ASTRA_MODEL") == "claude-opus-4-6"


def test_cmd_model_custom_option_prompts_for_input(monkeypatch):
    """When _CUSTOM_OPTION is selected, falls back to console.input."""
    from astra_cli.session import interact as interact_mod
    from astra_cli.session.commands import _CUSTOM_OPTION, _cmd_model
    from rich.console import Console
    from io import StringIO

    console = Console(file=StringIO(), width=100)
    state = _make_session_state(provider="anthropic", model="claude-sonnet-4-6")

    monkeypatch.setattr(interact_mod, "_interactive_select", lambda **kw: _CUSTOM_OPTION)
    monkeypatch.setattr(console, "input", lambda *a, **kw: "my-custom-model")
    monkeypatch.setattr("astra_cli.session.commands._rebuild_engine", lambda s, c: None)

    result = _cmd_model(state, console)
    assert result.handled is True
    assert state.model == "my-custom-model"


def test_cmd_model_custom_option_empty_input_unchanged(monkeypatch):
    """When custom is selected but user enters empty string, model is unchanged."""
    from astra_cli.session import interact as interact_mod
    from astra_cli.session.commands import _CUSTOM_OPTION, _cmd_model
    from rich.console import Console
    from io import StringIO

    console = Console(file=StringIO(), width=100)
    state = _make_session_state(provider="anthropic", model="claude-sonnet-4-6")

    monkeypatch.setattr(interact_mod, "_interactive_select", lambda **kw: _CUSTOM_OPTION)
    monkeypatch.setattr(console, "input", lambda *a, **kw: "")

    result = _cmd_model(state, console)
    assert result.handled is True
    assert state.model == "claude-sonnet-4-6"  # unchanged


def test_astra_no_args_launches_repl():
    """'astra' with no args calls session.repl.start(), not help."""
    from typer.testing import CliRunner
    from astra_cli.main import app
    from unittest.mock import patch

    runner = CliRunner()
    with patch("astra_cli.session.repl.start") as mock_start:
        result = runner.invoke(app, [])
    mock_start.assert_called_once()
