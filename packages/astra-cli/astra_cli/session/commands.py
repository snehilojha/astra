"""Slash command handlers for the Astra REPL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

ACCENT = "#f97316"

_HELP_ROWS = [
    ("/provider", "Switch LLM provider (anthropic, openai, openrouter, ollama)"),
    ("/model", "Switch model for current provider"),
    ("/clear", "Clear conversation history"),
    ("/memory", "Show persistent memory (~/.astra/memory/)"),
    ("/tools", "List registered tools and permission levels"),
    ("/cost", "Show token usage for this session"),
    ("/swarm <file>", "Run a swarm YAML file"),
    ("/help", "Show this help"),
    ("/exit", "Exit Astra"),
]

_KNOWN_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3-mini"],
    "openrouter": [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-opus-4",
        "openai/gpt-4o",
        "openai/o1",
        "google/gemini-2.0-flash",
    ],
    "ollama": ["llama3.2", "mistral", "gemma3", "phi4"],
}
_CUSTOM_OPTION = "enter custom model name..."


@dataclass
class CommandResult:
    handled: bool = False
    should_exit: bool = False


def handle_command(raw: str, state, console: Console) -> CommandResult:
    """Dispatch a raw input line to the appropriate handler.

    Returns CommandResult(handled=False) for non-slash input so the
    REPL loop knows to forward it to the agent.
    """
    if not raw.startswith("/"):
        return CommandResult(handled=False)

    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        return _cmd_help(console)
    if cmd == "/exit":
        return _cmd_exit(console)
    if cmd == "/clear":
        return _cmd_clear(state, console)
    if cmd == "/cost":
        return _cmd_cost(state, console)
    if cmd == "/memory":
        return _cmd_memory(console)
    if cmd == "/tools":
        return _cmd_tools(state, console)
    if cmd == "/provider":
        return _cmd_provider(state, console)
    if cmd == "/model":
        return _cmd_model(state, console)
    if cmd == "/swarm":
        return _cmd_swarm(arg, state, console)

    console.print(f"[red]Unknown command:[/red] {cmd}  (type /help for commands)")
    return CommandResult(handled=True)


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------


def _cmd_help(console: Console) -> CommandResult:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style=f"bold {ACCENT}")
    table.add_column(style="dim")
    for cmd, desc in _HELP_ROWS:
        table.add_row(cmd, desc)
    console.print(table)
    return CommandResult(handled=True)


def _cmd_exit(console: Console) -> CommandResult:
    console.print(f"[{ACCENT}]Goodbye.[/{ACCENT}]")
    return CommandResult(handled=True, should_exit=True)


def _cmd_clear(state, console: Console) -> CommandResult:
    state.engine._history.messages.clear()
    state.turn_count = 0
    console.print(f"[{ACCENT}]Conversation history cleared.[/{ACCENT}]")
    return CommandResult(handled=True)


def _cmd_cost(state, console: Console) -> CommandResult:
    console.print(
        f"  tokens in:  [{ACCENT}]{state.total_input_tokens}[/{ACCENT}]\n"
        f"  tokens out: [{ACCENT}]{state.total_output_tokens}[/{ACCENT}]"
    )
    return CommandResult(handled=True)


def _cmd_memory(console: Console) -> CommandResult:
    memory_dir = Path.home() / ".astra" / "memory"
    if not memory_dir.exists() or not any(memory_dir.iterdir()):
        console.print("[dim]No memory entries found.[/dim]")
        return CommandResult(handled=True)
    for f in sorted(memory_dir.glob("*.md")):
        console.print(f"[{ACCENT}]{f.stem}[/{ACCENT}]")
        console.print(f.read_text().strip())
        console.print()
    return CommandResult(handled=True)


def _cmd_tools(state, console: Console) -> CommandResult:
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("tool", style=f"bold {ACCENT}")
    table.add_column("permission", style="dim")
    for name, tool in state.registry._tools.items():
        level = getattr(tool, "permission_level", "unknown")
        level_str = level.name if hasattr(level, "name") else str(level)
        table.add_row(name, level_str)
    console.print(table)
    return CommandResult(handled=True)


def _cmd_provider(state, console: Console) -> CommandResult:
    """Interactive provider selection with optional API key entry."""
    from astra_cli.session.interact import _interactive_select

    providers = ["anthropic", "openai", "openrouter", "ollama"]
    prompt = f"Select provider (current: {state.provider_name}):"
    chosen = _interactive_select(prompt=prompt, options=providers, console=console)

    import os

    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "ollama": None,
    }
    key_name = key_map.get(chosen)
    if key_name and not os.environ.get(key_name):
        api_key = console.input(
            f"  [{ACCENT}]{key_name} not set. Enter API key:[/{ACCENT}] "
        ).strip()
        if api_key:
            os.environ[key_name] = api_key
            config_path = Path.home() / ".astra" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            cfg = _read_config_safe(config_path)
            cfg[key_name] = api_key
            config_path.write_text(json.dumps(cfg, indent=2))
            console.print(f"  [{ACCENT}]Saved {key_name} to config.[/{ACCENT}]")

    state.provider_name = chosen
    config_path = Path.home() / ".astra" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _read_config_safe(config_path)
    cfg["ASTRA_PROVIDER"] = chosen
    config_path.write_text(json.dumps(cfg, indent=2))
    _rebuild_engine(state, console)
    console.print(f"  [{ACCENT}]Provider set to {chosen}.[/{ACCENT}]")
    return CommandResult(handled=True)


def _cmd_model(state, console: Console) -> CommandResult:
    from astra_cli.session.interact import _interactive_select

    known = _KNOWN_MODELS.get(state.provider_name, [])
    options = known + [_CUSTOM_OPTION]

    chosen = _interactive_select(
        prompt=f"Select model ({state.provider_name}):",
        options=options,
        console=console,
    )

    if chosen == _CUSTOM_OPTION:
        chosen = console.input("  Model name: ").strip()
        if not chosen:
            console.print("[dim]Model unchanged.[/dim]")
            return CommandResult(handled=True)

    state.model = chosen
    config_path = Path.home() / ".astra" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _read_config_safe(config_path)
    cfg["ASTRA_MODEL"] = chosen
    config_path.write_text(json.dumps(cfg, indent=2))

    _rebuild_engine(state, console)
    console.print(f"  [{ACCENT}]Model set to {chosen}.[/{ACCENT}]")
    return CommandResult(handled=True)


def _cmd_swarm(arg: str, state, console: Console) -> CommandResult:
    import asyncio
    import threading
    from astra_cli.display.event_renderer import EventRenderer

    if not arg:
        console.print("[red]Usage: /swarm <path/to/config.yaml>[/red]")
        return CommandResult(handled=True)

    yaml_path = Path(arg).expanduser()
    if not yaml_path.exists():
        console.print(f"[red]File not found:[/red] {yaml_path}")
        return CommandResult(handled=True)

    try:
        from astra_swarm.swarm_loader import load_swarm_from_yaml

        _cfg, coordinator = load_swarm_from_yaml(yaml_path)
    except Exception as exc:
        console.print(f"[red]Failed to load swarm:[/red] {exc}")
        return CommandResult(handled=True)

    task = console.input(f"  [{ACCENT}]Task for swarm:[/{ACCENT}] ").strip()
    if not task:
        console.print("[dim]Cancelled.[/dim]")
        return CommandResult(handled=True)

    renderer = EventRenderer(console=console)

    async def _run_swarm() -> None:
        async for event in coordinator.run(task):
            renderer.render(event)

    def _run_in_thread() -> None:
        asyncio.run(_run_swarm())

    try:
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join()
    except RuntimeError:
        asyncio.run(_run_swarm())
    return CommandResult(handled=True)

    yaml_path = Path(arg).expanduser()
    if not yaml_path.exists():
        console.print(f"[red]File not found:[/red] {yaml_path}")
        return CommandResult(handled=True)

    try:
        from astra_swarm.swarm_loader import load_swarm_from_yaml

        _cfg, coordinator = load_swarm_from_yaml(yaml_path)
    except Exception as exc:
        console.print(f"[red]Failed to load swarm:[/red] {exc}")
        return CommandResult(handled=True)

    task = console.input(f"  [{ACCENT}]Task for swarm:[/{ACCENT}] ").strip()
    if not task:
        console.print("[dim]Cancelled.[/dim]")
        return CommandResult(handled=True)

    renderer = EventRenderer(console=console)

    async def _run_swarm() -> None:
        async for event in coordinator.run(task):
            renderer.render(event)

    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio

        nest_asyncio.apply(loop)
        asyncio.run(_run_swarm())
    except RuntimeError:
        asyncio.run(_run_swarm())
    except ImportError:
        try:
            import asyncio as _asyncio

            _asyncio.run(_run_swarm())
        except RuntimeError as e:
            console.print(f"[red]Cannot run swarm from async context:[/red] {e}")
            console.print("[dim]Install nest_asyncio: pip install nest_asyncio[/dim]")
    return CommandResult(handled=True)


# ---------------------------------------------------------------------------
# Engine rebuild (shared by /provider and /model)
# ---------------------------------------------------------------------------


def _rebuild_engine(state, console: Console) -> None:
    """Rebuild the QueryEngine preserving existing history."""
    from astra_cli.commands.run import _build_provider
    from astra_node.core.query_engine import QueryEngine

    old_history = state.engine._history

    new_provider = _build_provider(state.provider_name, state.model, state.base_url)
    new_engine = QueryEngine(
        provider=new_provider,
        registry=state.registry,
        permission_manager=state.permission_manager,
        system_prompt="You are Astra, a helpful AI agent.",
    )
    # Transplant history so conversation context is preserved
    new_engine._history = old_history
    state.engine = new_engine


def _read_config_safe(config_path: Path) -> dict:
    """Read config JSON and tolerate invalid files."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
