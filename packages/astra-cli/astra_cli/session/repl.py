"""Astra interactive REPL — main session loop."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.text import Text

from astra_cli.display.event_renderer import EventRenderer
import astra_cli.session.banner as _banner_mod
from astra_cli.session.commands import handle_command
from astra_cli.session.interact import _interactive_select
from astra_node.core.query_engine import QueryEngine
from astra_node.core.registry import ToolRegistry
from astra_node.permissions.manager import PermissionManager
from astra_node.permissions.types import PermissionDecision

ACCENT = "#f97316"
DANGEROUS_TOOLS = {"bash", "file_write", "file_edit"}
COMPACTION_TURN_THRESHOLD = 20

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("astra-cli")
except Exception:
    _VERSION = "0.1.0"


@dataclass
class SessionState:
    provider_name: str
    model: str | None
    base_url: str | None
    engine: QueryEngine
    permission_manager: PermissionManager
    registry: ToolRegistry
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def start() -> None:
    """Launch the interactive REPL."""
    console = Console()
    state = _build_session_state(console)

    _banner_mod.print_banner(
        console=console,
        provider=state.provider_name,
        model=state.model,
        version=_VERSION,
    )

    asyncio.run(_repl_loop(state, console))


async def _repl_loop(state: SessionState, console: Console) -> None:
    """Main async REPL loop."""
    renderer = EventRenderer(console=console)

    while True:
        try:
            raw = await _read_input(state)
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[{ACCENT}]Goodbye.[/{ACCENT}]")
            break

        raw = raw.strip()
        if not raw:
            continue

        # Slash command dispatch
        result = handle_command(raw, state, console)
        if result.should_exit:
            break
        if result.handled:
            continue

        # Agent turn
        state.turn_count += 1
        await _run_agent_turn(raw, state, renderer, console)

        # Auto-compaction
        if state.turn_count % COMPACTION_TURN_THRESHOLD == 0:
            await _compact(state, console)


async def _run_agent_turn(
    user_input: str,
    state: SessionState,
    renderer: EventRenderer,
    console: Console,
) -> None:
    """Run one user turn through the engine, intercepting ASK permissions."""
    from astra_node.core.events import ToolStart, UsageUpdate

    async for event in state.engine.run(user_input):
        if isinstance(event, ToolStart) and event.tool_name in DANGEROUS_TOOLS:
            tool = state.registry._tools.get(event.tool_name)
            if tool is not None:
                decision = state.permission_manager.check_level(
                    event.tool_name, tool.permission_level, event.tool_input
                )
                if decision == PermissionDecision.ASK:
                    user_decision = _prompt_permission(
                        event.tool_name,
                        event.tool_input,
                        state.permission_manager,
                        console,
                    )
                    if user_decision == PermissionDecision.DENY:
                        renderer.render(event)
                        continue
        if isinstance(event, UsageUpdate):
            state.total_input_tokens += event.input_tokens
            state.total_output_tokens += event.output_tokens
        renderer.render(event)


async def _compact(state: SessionState, console: Console) -> None:
    """Compact conversation history using CompactionEngine."""
    from astra_node.core.compaction import CompactionEngine

    console.print(f"[dim]  compacting history…[/dim]")
    try:
        compactor = CompactionEngine()
        model = state.model or "claude-sonnet-4-6"
        new_history = await compactor.compact(
            history=state.engine._history,
            provider=state.engine._provider,
            model=model,
        )
        state.engine._history = new_history
    except Exception as exc:
        console.print(f"[dim]  compaction failed: {exc}[/dim]")


def _prompt_permission(
    tool_name: str,
    tool_input: dict,
    pm: PermissionManager,
    console: Console,
) -> PermissionDecision:
    """Prompt the user for permission to run a dangerous tool."""
    from astra_cli.display.event_renderer import _input_summary

    summary = _input_summary(tool_input)
    console.print(f"  [{ACCENT}]⚙ {tool_name}[/{ACCENT}][dim]: {summary}[/dim]")
    answer = _ask_permission(console)
    answer = answer.strip().lower()
    if answer == "always":
        pm.allow_always(tool_name)
        return PermissionDecision.ALLOW
    if answer == "yes":
        return PermissionDecision.ALLOW
    return PermissionDecision.DENY


def _ask_permission(console: Console) -> str:
    """Prompt [yes/no/always] with arrow key selection and return the raw answer."""
    return _interactive_select(
        prompt="Allow?",
        options=["yes", "no", "always"],
        console=console,
    )


def _toolbar_text(state: SessionState) -> str:
    """Return prompt_toolkit HTML string for the bottom status toolbar."""
    model = state.model or "(default)"
    return (
        f" <style fg='#f97316'>{state.provider_name}</style>"
        f" <style fg='#666'>·</style>"
        f" <style fg='#f97316'>{model}</style>"
        f" <style fg='#666'>·</style>"
        f" <style fg='#888'>↑{state.total_input_tokens} ↓{state.total_output_tokens}</style>"
        f" <style fg='#666'>·</style>"
        f" <style fg='#888'>turn {state.turn_count}</style> "
    )


async def _read_input(state: SessionState) -> str:
    """Read one line of user input via prompt-toolkit (async) with live status bar."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    style = Style.from_dict({"prompt": "#f97316 bold"})
    session: PromptSession[str] = PromptSession()
    return await session.prompt_async(
        "> ",
        style=style,
        bottom_toolbar=lambda: HTML(_toolbar_text(state)),
    )


def _build_session_state(console: Console) -> SessionState:
    """Build SessionState from config, defaulting to anthropic."""
    from astra_cli.commands.run import _build_provider
    from astra_node.tools.bash import BashTool
    from astra_node.tools.file_read import FileReadTool
    from astra_node.tools.file_write import FileWriteTool
    from astra_node.tools.file_edit import FileEditTool
    from astra_node.tools.grep import GrepTool
    from astra_node.tools.glob_tool import GlobTool

    # Read saved provider/model from config
    config_path = Path.home() / ".astra" / "config.json"
    cfg: dict = {}
    if config_path.exists():
        import json

        cfg = json.loads(config_path.read_text())

    provider_name = cfg.get("ASTRA_PROVIDER", "anthropic")
    model = cfg.get("ASTRA_MODEL") or None
    base_url = cfg.get("ASTRA_BASE_URL") or None

    registry = ToolRegistry()
    for tool in [
        BashTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GrepTool(),
        GlobTool(),
    ]:
        registry.register(tool)

    pm = PermissionManager()
    provider = _build_provider(provider_name, model, base_url)

    engine = QueryEngine(
        provider=provider,
        registry=registry,
        permission_manager=pm,
        system_prompt="You are Astra, a helpful AI agent.",
    )

    return SessionState(
        provider_name=provider_name,
        model=model,
        base_url=base_url,
        engine=engine,
        permission_manager=pm,
        registry=registry,
    )
