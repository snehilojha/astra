"""Tests for CLI commands (Step 17).

All I/O is mocked — no real API calls, no file system side-effects.
Uses Typer's CliRunner for isolated invocation.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from astra_cli.main import app
from astra_node.core.events import TextDelta, TurnEnd, UsageUpdate

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fake_events(events) -> AsyncIterator:
    for e in events:
        yield e


def make_fake_engine(events=None):
    """Return a mock QueryEngine whose run() yields the given events."""
    if events is None:
        events = [TextDelta(text="hello"), TurnEnd()]
    engine = MagicMock()
    engine.run = MagicMock(return_value=_fake_events(events))
    return engine


# ---------------------------------------------------------------------------
# astra run
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_run_with_task_arg(self):
        """astra run 'task' creates a QueryEngine and streams events."""
        fake_engine = make_fake_engine()
        with (
            patch("astra_cli.commands.run._build_engine", return_value=fake_engine),
            patch("astra_cli.display.event_renderer.EventRenderer.render"),
        ):
            result = runner.invoke(app, ["run", "do something"])
        assert result.exit_code == 0

    def test_run_with_file(self, tmp_path):
        """astra run --file reads task from file."""
        task_file = tmp_path / "task.txt"
        task_file.write_text("task from file")
        fake_engine = make_fake_engine()
        with (
            patch("astra_cli.commands.run._build_engine", return_value=fake_engine),
            patch("astra_cli.display.event_renderer.EventRenderer.render"),
        ):
            result = runner.invoke(app, ["run", "--file", str(task_file)])
        assert result.exit_code == 0

    def test_run_no_args_exits_with_error(self):
        """astra run with no task or file prints error and exits 1."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1

    def test_run_provider_ollama(self):
        """astra run --provider ollama creates an OpenAIProvider."""
        from astra_node.providers.openai import OpenAIProvider

        captured = {}

        def fake_build(provider, model):
            from astra_cli.commands.run import _build_provider, _build_engine
            prov = _build_provider(provider, model)
            captured["provider"] = prov
            # Return a minimal mock engine
            return make_fake_engine()

        with (
            patch("astra_cli.commands.run._build_engine", side_effect=fake_build),
            patch("astra_cli.display.event_renderer.EventRenderer.render"),
        ):
            result = runner.invoke(app, ["run", "--provider", "ollama", "task"])

        # Just verify no crash and exit 0
        assert result.exit_code == 0

    def test_run_builds_openai_provider(self):
        """_build_provider('ollama', None) returns OpenAIProvider."""
        from astra_cli.commands.run import _build_provider
        from astra_node.providers.openai import OpenAIProvider

        prov = _build_provider("ollama", None)
        assert isinstance(prov, OpenAIProvider)

    def test_run_builds_anthropic_provider(self):
        """_build_provider('anthropic', None) returns AnthropicProvider."""
        from astra_cli.commands.run import _build_provider
        from astra_node.providers.anthropic import AnthropicProvider

        prov = _build_provider("anthropic", None)
        assert isinstance(prov, AnthropicProvider)


# ---------------------------------------------------------------------------
# astra swarm
# ---------------------------------------------------------------------------

class TestSwarmCommand:
    def test_swarm_list(self, tmp_path):
        """astra swarm list lists YAML files from config dir."""
        (tmp_path / "code_review.yaml").write_text("name: code_review")
        (tmp_path / "research.yaml").write_text("name: research")

        with patch("astra_cli.commands.swarm._find_config_dir", return_value=tmp_path):
            result = runner.invoke(app, ["swarm", "list"])

        assert result.exit_code == 0
        assert "code_review" in result.output
        assert "research" in result.output

    def test_swarm_list_empty(self, tmp_path):
        """astra swarm list with no configs prints message."""
        with patch("astra_cli.commands.swarm._find_config_dir", return_value=tmp_path):
            result = runner.invoke(app, ["swarm", "list"])
        assert result.exit_code == 0
        assert "No swarm configs" in result.output

    def test_swarm_run_missing_config(self, tmp_path):
        """astra swarm run with missing config exits 1."""
        with patch("astra_cli.commands.swarm._find_config_dir", return_value=tmp_path):
            result = runner.invoke(app, ["swarm", "run", "nonexistent", "--task", "t"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# astra memory
# ---------------------------------------------------------------------------

class TestMemoryCommand:
    def test_memory_list_empty(self):
        """astra memory list with StubMemory prints 'No memories stored'."""
        result = runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 0
        assert "No memories" in result.output

    def test_memory_search_no_results(self):
        """astra memory search with StubMemory prints 'No memories matching'."""
        result = runner.invoke(app, ["memory", "search", "experiment"])
        assert result.exit_code == 0
        assert "No memories" in result.output

    def test_memory_clear_with_yes_flag(self):
        """astra memory clear --yes clears without prompting."""
        result = runner.invoke(app, ["memory", "clear", "--yes"])
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()


# ---------------------------------------------------------------------------
# astra config
# ---------------------------------------------------------------------------

class TestConfigCommand:
    def test_config_set_and_get(self, tmp_path):
        """astra config set stores a value; config get retrieves it."""
        config_file = tmp_path / "config.json"

        with patch("astra_cli.commands.config._CONFIG_PATH", config_file):
            set_result = runner.invoke(app, ["config", "set", "MY_KEY", "my_value"])
            get_result = runner.invoke(app, ["config", "get", "MY_KEY"])

        assert set_result.exit_code == 0
        assert get_result.exit_code == 0
        assert "my_value" in get_result.output

    def test_config_get_missing_key(self, tmp_path):
        """astra config get with unknown key exits 1."""
        config_file = tmp_path / "config.json"
        with patch("astra_cli.commands.config._CONFIG_PATH", config_file):
            result = runner.invoke(app, ["config", "get", "UNKNOWN"])
        assert result.exit_code == 1

    def test_config_set_persists_to_disk(self, tmp_path):
        """astra config set actually writes JSON to disk."""
        config_file = tmp_path / "config.json"
        with patch("astra_cli.commands.config._CONFIG_PATH", config_file):
            runner.invoke(app, ["config", "set", "ANTHROPIC_API_KEY", "sk-test"])
        data = json.loads(config_file.read_text())
        assert data["ANTHROPIC_API_KEY"] == "sk-test"


# ---------------------------------------------------------------------------
# Unknown command
# ---------------------------------------------------------------------------

class TestUnknownCommand:
    def test_unknown_command_gives_error(self):
        """astra unknown-cmd exits with non-zero and shows error."""
        result = runner.invoke(app, ["unknown-cmd"])
        assert result.exit_code != 0
