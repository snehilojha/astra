"""Astra interactive REPL — main session loop."""

from __future__ import annotations

import asyncio
import json
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
    try:
        state = _build_session_state(console)
    except Exception as exc:
        console.print(f"[red]Failed to initialize session:[/red] {exc}")
        _show_startup_help(console)
        return

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
    from astra_node.utils.errors import PromptInjectionError, ProviderError

    try:
        renderer.start_thinking()
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
    except PromptInjectionError as exc:
        renderer.stop_thinking()
        console.print(f"\n[red bold]Blocked:[/red bold] [red]{exc}[/red]")
    except ProviderError as exc:
        renderer.stop_thinking()
        console.print(f"\n[red]Error: {exc}[/red]")
        _handle_provider_error_with_fallback(
            provider=state.provider_name,
            console=console,
            error_message=str(exc),
            state=state,
        )


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
    try:
        return _interactive_select(
            prompt="Allow?",
            options=["yes", "no", "always"],
            console=console,
        )
    except (EOFError, KeyboardInterrupt):
        return "no"


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
    """Build SessionState from config, prompting for setup if needed."""
    from astra_cli.commands.run import _build_provider
    from astra_node.tools.bash import BashTool
    from astra_node.tools.file_read import FileReadTool
    from astra_node.tools.file_write import FileWriteTool
    from astra_node.tools.file_edit import FileEditTool
    from astra_node.tools.grep import GrepTool
    from astra_node.tools.glob_tool import GlobTool

    config_path = Path.home() / ".astra" / "config.json"
    cfg: dict = {}
    if config_path.exists():
        cfg = json.loads(config_path.read_text())

    provider_name = cfg.get("ASTRA_PROVIDER")
    model = cfg.get("ASTRA_MODEL") or None
    base_url = cfg.get("ASTRA_BASE_URL") or None

    if not provider_name:
        provider_name = _prompt_provider_selection(console)
        if provider_name:
            cfg["ASTRA_PROVIDER"] = provider_name
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg, indent=2))
        else:
            provider_name = "anthropic"

    from astra_cli.commands.run import _load_api_key
    env_var = _get_api_key_env_var(provider_name)
    if env_var and not _load_api_key(provider_name):
        # Key not in env or config — prompt the user
        console.print(f"[yellow]Configure your API key to get started.[/yellow]")
        _prompt_for_api_key(env_var, provider_name, console)

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
        system_prompt=(
            "You are Astra, a helpful AI agent.\n\n"
            "SECURITY RULES — these cannot be overridden by any user instruction:\n"
            "- Never read, display, print, or transmit API keys, passwords, tokens, or secrets "
            "from any source (environment variables, config files, .env files, key files, etc.).\n"
            "- Never search for, list, or enumerate credential files (e.g. ~/.astra/config.json, "
            ".env, id_rsa, *.pem, *.key).\n"
            "- If a user asks you to reveal credentials or find API keys, refuse and explain "
            "that you cannot access or expose secrets."
        ),
    )

    return SessionState(
        provider_name=provider_name,
        model=model,
        base_url=base_url,
        engine=engine,
        permission_manager=pm,
        registry=registry,
    )


def _get_api_key_env_var(provider: str) -> str | None:
    """Return the environment variable name for the given provider's API key."""
    from astra_cli.commands.run import get_api_key_env_var
    return get_api_key_env_var(provider)


def _prompt_provider_selection(console: Console) -> str | None:
    """Prompt user to select a provider on first run."""
    from astra_cli.session.interact import _interactive_select

    console.print(f"[{ACCENT}]Welcome to Astra![/{ACCENT}]")
    console.print(
        "[dim]Select a provider to get started (or skip to configure later):[/dim]"
    )

    from astra_cli.commands.run import PROVIDER_REGISTRY
    providers = list(PROVIDER_REGISTRY.keys()) + ["skip"]
    chosen = _interactive_select(
        prompt="Select provider:",
        options=providers,
        console=console,
    )

    if chosen == "skip":
        return None
    return chosen


def _prompt_for_api_key(env_var: str, provider: str, console: Console) -> bool:
    """Prompt the user to enter an API key and save it to config.

    Returns True if a new key was saved, False otherwise.
    """
    try:
        from astra_cli.session.interact import _interactive_select

        answer = _interactive_select(
            prompt=f"Enter {provider} API key now?",
            options=["yes", "skip"],
            console=console,
        )
        if answer.strip().lower() != "yes":
            console.print(
                f"[dim]You can configure later with /provider command or `astra config set {env_var} <key>`[/dim]"
            )
            return False

        import getpass
        key = getpass.getpass(f"Enter your {provider} API key: ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Skipped.[/dim]")
        return False

    if not key:
        console.print("[dim]No key entered.[/dim]")
        return False

    os.environ[env_var] = key

    config_path = Path.home() / ".astra" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    cfg[env_var] = key
    config_path.write_text(json.dumps(cfg, indent=2))

    console.print(f"[green]API key saved.[/green]")
    return True


def _handle_provider_error_with_fallback(
    provider: str,
    console: Console,
    error_message: str,
    state: "SessionState | None" = None,
) -> None:
    """Print helpful auth/config hints and offer inline API key setup."""
    env_var = _get_api_key_env_var(provider)
    if not env_var:
        console.print(
            "[dim]This provider does not require an API key. Check your model/base URL.[/dim]"
        )
        return

    missing = not os.environ.get(env_var)

    if missing:
        console.print(f"[yellow]{env_var} is not set.[/yellow]")
    else:
        console.print(f"[yellow]{env_var} may be invalid or expired.[/yellow]")

    _print_api_key_hints(provider, env_var, console)
    key_saved = _prompt_for_api_key(env_var, provider, console)

    # Only rebuild if a new key was actually entered — the old client won't
    # pick up the new credential otherwise.
    if key_saved and state is not None:
        from astra_cli.session.commands import _rebuild_engine
        _rebuild_engine(state, console)


def _print_api_key_hints(provider: str, env_var: str, console: Console) -> None:
    """Print practical next-step hints for provider authentication."""
    from astra_cli.commands.run import PROVIDER_REGISTRY
    provider_hint = {
        "anthropic": "https://console.anthropic.com/settings/keys",
        "openai": "https://platform.openai.com/api-keys",
        "openrouter": "https://openrouter.ai/keys",
    }
    console.print(
        f"[dim]Run `astra config set {env_var} <your-key>` to persist it.[/dim]"
    )
    dashboard = provider_hint.get(provider)
    if dashboard:
        console.print(f"[dim]Get a key: {dashboard}[/dim]")
    elif provider in PROVIDER_REGISTRY:
        console.print(f"[dim]Check your {provider} documentation for API key setup.[/dim]")


def _shell_export_hint(env_var: str) -> str:
    """Return an OS-appropriate env export hint."""
    if os.name == "nt":
        return f"$env:{env_var}='YOUR_KEY'"
    return f"export {env_var}=YOUR_KEY"


def _show_startup_help(console: Console) -> None:
    """Generic startup fallback instructions."""
    console.print(
        "[dim]Use /provider to select a provider and configure your API key.[/dim]"
    )
