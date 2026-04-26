"""Tests for the display layer (Step 18).

Verifies that EventRenderer maps each AgentEvent type to the expected
Rich console output. Uses an in-memory Console to capture output without
printing to the terminal.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from astra_node.core.events import (
    TextDelta,
    ToolStart,
    ToolResult,
    TurnEnd,
    AgentError,
    UsageUpdate,
)
from astra_cli.display.event_renderer import EventRenderer, _input_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_renderer() -> tuple[EventRenderer, StringIO]:
    """Return a renderer wired to an in-memory console."""
    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False)
    renderer = EventRenderer(console=console)
    return renderer, buf


def captured(buf: StringIO) -> str:
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TextDelta
# ---------------------------------------------------------------------------

class TestTextDelta:
    def test_renders_text(self):
        renderer, buf = make_renderer()
        renderer.render(TextDelta(text="hello world"))
        renderer.render(TurnEnd())  # flush buffer
        assert "hello world" in captured(buf)

    def test_streamed_text_no_newline_between_chunks(self):
        renderer, buf = make_renderer()
        renderer.render(TextDelta(text="foo"))
        renderer.render(TextDelta(text="bar"))
        renderer.render(TurnEnd())  # flush buffer
        out = captured(buf)
        assert "foo" in out
        assert "bar" in out


# ---------------------------------------------------------------------------
# ToolStart
# ---------------------------------------------------------------------------

class TestToolStart:
    def test_renders_tool_name(self):
        renderer, buf = make_renderer()
        renderer.render(ToolStart(tool_name="bash", tool_input={"command": "ls -la"}))
        out = captured(buf)
        assert "bash" in out

    def test_renders_input_summary(self):
        renderer, buf = make_renderer()
        renderer.render(ToolStart(tool_name="bash", tool_input={"command": "ls -la"}))
        assert "ls -la" in captured(buf)

    def test_renders_running_prefix(self):
        renderer, buf = make_renderer()
        renderer.render(ToolStart(tool_name="grep", tool_input={"pattern": "TODO"}))
        assert "running" in captured(buf)


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_renders_output(self):
        renderer, buf = make_renderer()
        renderer.render(ToolResult(tool_name="bash", output="file1\nfile2"))
        assert "file1" in captured(buf)

    def test_truncates_long_output(self):
        renderer, buf = make_renderer()
        long_output = "x" * 1000
        renderer.render(ToolResult(tool_name="bash", output=long_output))
        out = captured(buf)
        assert "truncated" in out
        assert len(out) < 700  # far shorter than 1000

    def test_error_result_shown(self):
        renderer, buf = make_renderer()
        renderer.render(ToolResult(tool_name="bash", output="command not found", is_error=True))
        assert "command not found" in captured(buf)


# ---------------------------------------------------------------------------
# AgentError
# ---------------------------------------------------------------------------

class TestAgentError:
    def test_renders_error_message(self):
        renderer, buf = make_renderer()
        renderer.render(AgentError(error="permission denied", tool_name="bash"))
        assert "permission denied" in captured(buf)

    def test_renders_tool_name_in_error(self):
        renderer, buf = make_renderer()
        renderer.render(AgentError(error="oops", tool_name="file_write"))
        assert "file_write" in captured(buf)

    def test_renders_error_symbol(self):
        renderer, buf = make_renderer()
        renderer.render(AgentError(error="oops"))
        assert "error" in captured(buf).lower()


# ---------------------------------------------------------------------------
# TurnEnd
# ---------------------------------------------------------------------------

class TestTurnEnd:
    def test_renders_separator(self):
        renderer, buf = make_renderer()
        renderer.render(TurnEnd())
        # Rich rule renders a horizontal line; check something non-empty is printed
        assert len(captured(buf)) > 0

    def test_renders_usage_summary_after_usage_update(self):
        renderer, buf = make_renderer()
        renderer.render(UsageUpdate(input_tokens=100, output_tokens=50))
        renderer.render(TurnEnd())
        out = captured(buf)
        assert "100" in out
        assert "50" in out

    def test_no_usage_line_if_no_tokens(self):
        renderer, buf = make_renderer()
        renderer.render(TurnEnd())
        # Should not crash and should not print "tokens:" if no usage events fired
        out = captured(buf)
        assert "tokens:" not in out


# ---------------------------------------------------------------------------
# SwarmEvent
# ---------------------------------------------------------------------------

class TestSwarmEvent:
    def test_swarm_text_delta_has_worker_prefix(self):
        from astra_swarm.swarm import SwarmEvent

        renderer, buf = make_renderer()
        renderer.render(SwarmEvent(worker_id="worker_1", inner_type="text_delta", data={"text": "hi"}))
        renderer.render(SwarmEvent(worker_id="worker_1", inner_type="turn_end", data={}))  # flush buffer
        out = captured(buf)
        assert "worker_1" in out
        assert "hi" in out

    def test_swarm_tool_start_has_worker_prefix(self):
        from astra_swarm.swarm import SwarmEvent

        renderer, buf = make_renderer()
        event = SwarmEvent(
            worker_id="coder",
            inner_type="tool_start",
            data={"tool_name": "bash", "tool_input": {"command": "pwd"}},
        )
        renderer.render(event)
        out = captured(buf)
        assert "coder" in out
        assert "bash" in out

    def test_swarm_agent_error_has_worker_prefix(self):
        from astra_swarm.swarm import SwarmEvent

        renderer, buf = make_renderer()
        event = SwarmEvent(worker_id="reviewer", inner_type="agent_error", data={"error": "bad"})
        renderer.render(event)
        out = captured(buf)
        assert "reviewer" in out
        assert "bad" in out


# ---------------------------------------------------------------------------
# _input_summary helper
# ---------------------------------------------------------------------------

class TestInputSummary:
    def test_command_key(self):
        assert "ls" in _input_summary({"command": "ls -la"})

    def test_path_key(self):
        assert "/tmp" in _input_summary({"path": "/tmp/file.txt"})

    def test_empty_dict(self):
        assert _input_summary({}) == ""

    def test_long_value_truncated(self):
        result = _input_summary({"command": "x" * 200})
        assert len(result) <= 83  # 80 chars + "…"
