"""EventRenderer — maps AgentEvent / SwarmEvent to Rich terminal output."""

from __future__ import annotations

import itertools
import sys
import threading
import time

from rich.console import Console
from rich.markdown import Heading as _Heading, Markdown
from rich.text import Text


class _LeftHeading(_Heading):
    def __rich_console__(self, console, options):
        self.text.justify = "left"
        yield self.text


Markdown.elements["heading_open"] = _LeftHeading

from astra_node.core.events import (
    AgentEvent,
    TextDelta,
    ToolStart,
    ToolResult,
    TurnEnd,
    AgentError,
    UsageUpdate,
)

_MAX_OUTPUT_CHARS = 500  # truncate long tool output in the terminal
ACCENT = "#f97316"


class EventRenderer:
    """Renders agent events to the terminal using Rich.

    Usage::

        renderer = EventRenderer()
        async for event in engine.run(task):
            renderer.render(event)
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._total_input = 0
        self._total_output = 0
        self._swarm_text_buffers: dict[str, str] = {}  # worker_id -> buffered text
        self._response_buffer: str = ""  # accumulates agent text for markdown rendering
        self._spinner_thread: threading.Thread | None = None
        self._spinner_stop: threading.Event = threading.Event()
        self._last_swarm_worker: str | None = None  # for worker separator headers

    # ------------------------------------------------------------------
    # Spinner control
    # ------------------------------------------------------------------

    def start_thinking(self) -> None:
        """Show a dots spinner on stderr while waiting for the first token."""
        if self._spinner_thread is not None:
            return
        self._spinner_stop.clear()

        def _spin() -> None:
            frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            while not self._spinner_stop.is_set():
                sys.stderr.write(f"\r  {next(frames)} thinking…")
                sys.stderr.flush()
                time.sleep(0.08)
            sys.stderr.write("\r" + " " * 20 + "\r")
            sys.stderr.flush()

        self._spinner_thread = threading.Thread(target=_spin, daemon=True)
        self._spinner_thread.start()

    def stop_thinking(self) -> None:
        """Stop and clear the spinner."""
        if self._spinner_thread is not None:
            self._spinner_stop.set()
            self._spinner_thread.join(timeout=0.5)
            self._spinner_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, event: AgentEvent) -> None:
        """Dispatch event to the appropriate render method."""
        # Try SwarmEvent first (avoids importing at module level)
        try:
            from astra_swarm.swarm import SwarmEvent

            if isinstance(event, SwarmEvent):
                self._render_swarm(event)
                return
        except ImportError:
            pass

        if isinstance(event, TextDelta):
            self._render_text_delta(event)
        elif isinstance(event, ToolStart):
            self._render_tool_start(event)
        elif isinstance(event, ToolResult):
            self._render_tool_result(event)
        elif isinstance(event, AgentError):
            self._render_agent_error(event)
        elif isinstance(event, TurnEnd):
            self._render_turn_end(event)
        elif isinstance(event, UsageUpdate):
            self._accumulate_usage(event)

    # ------------------------------------------------------------------
    # Per-event renderers
    # ------------------------------------------------------------------

    def _render_text_delta(self, event: TextDelta) -> None:
        self._response_buffer += event.text

    def _flush_response_buffer(self) -> None:
        self.stop_thinking()
        if self._response_buffer.strip():
            self._console.print(Text("▌", style=f"dim {ACCENT}"), end=" ")
            self._console.print(Markdown(_strip_latex(self._response_buffer)))
            self._response_buffer = ""

    def _render_tool_start(self, event: ToolStart) -> None:
        self.stop_thinking()
        self._flush_response_buffer()
        summary = _input_summary(event.tool_input)
        line = Text()
        line.append("⚙ running ", style="dim")
        line.append(event.tool_name, style="bold cyan")
        if summary:
            line.append(f": {summary}", style="dim")
        self._console.print(line)

    def _render_tool_result(self, event: ToolResult) -> None:
        output = event.output
        truncated = len(output) > _MAX_OUTPUT_CHARS
        if truncated:
            output = output[:_MAX_OUTPUT_CHARS] + " … (truncated)"
        style = "red" if event.is_error else "dim green"
        self._console.print(output, style=style)
        # Restart spinner — engine is about to call the LLM again with tool results
        self.start_thinking()

    def _render_agent_error(self, event: AgentError) -> None:
        line = Text()
        line.append("✗ error", style="bold red")
        if event.tool_name:
            line.append(f" in {event.tool_name}", style="yellow")
        line.append(f": {event.error}", style="red")
        self._console.print(line)

    def _render_turn_end(self, event: TurnEnd) -> None:
        self._flush_response_buffer()

    def _accumulate_usage(self, event: UsageUpdate) -> None:
        self._total_input += event.input_tokens
        self._total_output += event.output_tokens

    def _render_swarm(self, event) -> None:  # type: ignore[no-untyped-def]
        """Render a SwarmEvent with per-worker section headers and spinner."""
        worker_id = event.worker_id
        inner_type = event.inner_type
        data = event.data

        if inner_type == "text_delta":
            self._swarm_text_buffers.setdefault(worker_id, "")
            self._swarm_text_buffers[worker_id] += data.get("text", "")
            return

        if inner_type == "turn_end":
            self.stop_thinking()
            buffered = self._swarm_text_buffers.pop(worker_id, "")
            if buffered:
                self._print_worker_header(worker_id)
                self._console.print(Markdown(_strip_latex(buffered)))
            self._console.rule(style="dim")
            self.start_thinking()
            return

        # Non-delta event: flush any buffered text first
        self.stop_thinking()
        buffered = self._swarm_text_buffers.pop(worker_id, "")
        if buffered:
            self._print_worker_header(worker_id)
            self._console.print(buffered, markup=False)

        self._print_worker_header(worker_id)

        if inner_type == "tool_start":
            summary = _input_summary(data.get("tool_input", {}))
            line = Text()
            line.append("  ⚙ running ", style="dim")
            line.append(data.get("tool_name", ""), style="bold cyan")
            if summary:
                line.append(f": {summary}", style="dim")
            self._console.print(line)
            # Start spinner — waiting for tool result then LLM response
            self.start_thinking()
        elif inner_type == "tool_result":
            self.stop_thinking()
            output = data.get("output", "")
            if len(output) > _MAX_OUTPUT_CHARS:
                output = output[:_MAX_OUTPUT_CHARS] + " … (truncated)"
            style = "red" if data.get("is_error") else "dim green"
            self._console.print(f"  {output}", style=style)
            # Start spinner — LLM processing tool result
            self.start_thinking()
        elif inner_type == "agent_error":
            line = Text()
            line.append("  ✗ ", style="bold red")
            line.append(data.get("error", ""), style="red")
            self._console.print(line)

    def _print_worker_header(self, worker_id: str) -> None:
        """Print a section rule for a worker if it changed."""
        if self._last_swarm_worker != worker_id:
            self._console.rule(
                f"[bold {ACCENT}]{worker_id}[/bold {ACCENT}]",
                style=f"dim {ACCENT}",
            )
            self._last_swarm_worker = worker_id


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _input_summary(tool_input: dict) -> str:
    """Return a short one-line summary of tool input for display."""
    if not tool_input:
        return ""
    # Try common keys first
    for key in ("command", "path", "pattern", "query", "url"):
        if key in tool_input:
            val = str(tool_input[key])
            return val[:80] + ("…" if len(val) > 80 else "")
    # Fall back to first key
    first_key = next(iter(tool_input))
    val = str(tool_input[first_key])
    return val[:80] + ("…" if len(val) > 80 else "")


def _strip_latex(text: str) -> str:
    """Remove LaTeX math delimiters so terminals render plain readable text."""
    import re
    # Block math: $$...$$ → contents only
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
    # Inline math: $...$ → contents only
    text = re.sub(r'\$(.+?)\$', r'\1', text)
    return text
