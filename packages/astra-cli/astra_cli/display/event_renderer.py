"""EventRenderer — maps AgentEvent / SwarmEvent to Rich terminal output."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

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
        self._console.print(event.text, end="", markup=False)

    def _render_tool_start(self, event: ToolStart) -> None:
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

    def _render_agent_error(self, event: AgentError) -> None:
        line = Text()
        line.append("✗ error", style="bold red")
        if event.tool_name:
            line.append(f" in {event.tool_name}", style="yellow")
        line.append(f": {event.error}", style="red")
        self._console.print(line)

    def _render_turn_end(self, event: TurnEnd) -> None:
        self._console.print()  # blank line after streamed text
        self._console.rule(style="dim")
        if self._total_input or self._total_output:
            self._console.print(
                f"[dim]tokens: {self._total_input} in / {self._total_output} out[/dim]"
            )

    def _accumulate_usage(self, event: UsageUpdate) -> None:
        self._total_input += event.input_tokens
        self._total_output += event.output_tokens

    def _render_swarm(self, event) -> None:  # type: ignore[no-untyped-def]
        """Render a SwarmEvent by prefixing with the worker id."""
        worker_id = event.worker_id
        inner_type = event.inner_type
        data = event.data

        prefix = Text(f"[{worker_id}] ", style="bold magenta")

        if inner_type == "text_delta":
            self._swarm_text_buffers.setdefault(worker_id, "")
            self._swarm_text_buffers[worker_id] += data.get("text", "")
            return
        elif inner_type == "turn_end":
            # Flush any buffered text before printing the turn_end separator
            buffered = self._swarm_text_buffers.pop(worker_id, "")
            if buffered:
                self._console.print(prefix, end="")
                self._console.print(buffered, markup=False)
            self._console.print(prefix, end="")
            self._console.rule(style="dim")
            return
        else:
            # Flush buffered text before any non-delta event (tool calls, errors)
            buffered = self._swarm_text_buffers.pop(worker_id, "")
            if buffered:
                self._console.print(prefix, end="")
                self._console.print(buffered, markup=False)

        if inner_type == "tool_start":
            summary = _input_summary(data.get("tool_input", {}))
            line = prefix.copy()
            line.append("⚙ running ", style="dim")
            line.append(data.get("tool_name", ""), style="bold cyan")
            if summary:
                line.append(f": {summary}", style="dim")
            self._console.print(line)
        elif inner_type == "tool_result":
            output = data.get("output", "")
            if len(output) > _MAX_OUTPUT_CHARS:
                output = output[:_MAX_OUTPUT_CHARS] + " … (truncated)"
            style = "red" if data.get("is_error") else "dim green"
            self._console.print(prefix, end="")
            self._console.print(output, style=style)
        elif inner_type == "agent_error":
            line = prefix.copy()
            line.append("✗ ", style="bold red")
            line.append(data.get("error", ""), style="red")
            self._console.print(line)


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
