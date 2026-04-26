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
from astra_node.core.memory import PersistentMemory
from astra_node.core.compaction import CompactionEngine
from astra_node.utils.token_counter import TiktokenCounter
from astra_node.permissions.manager import PermissionManager
from astra_node.permissions.types import PermissionDecision

ACCENT = "#f97316"
DANGEROUS_TOOLS = {"bash", "file_write", "file_edit"}

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
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    pt_style = Style.from_dict({
        "prompt": "#f97316 bold",
        "bottom-toolbar": "noreverse bg: fg:default",
    })
    pt_session: PromptSession[str] = PromptSession()

    renderer = EventRenderer(console=console)
    _compactor = CompactionEngine()
    _counter = TiktokenCounter()

    while True:
        try:
            raw = await _read_input(state, pt_session, pt_style)
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
        console.rule(style="dim")

        # Token-based auto-compaction
        model = state.model or "claude-sonnet-4-6"
        if _compactor.should_compact(state.engine._history, _counter, model):
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
    """Prompt [yes/no/always] and return the raw answer."""
    try:
        answer = console.input(f"  [{ACCENT}]Allow? [yes/no/always]:[/{ACCENT}] ").strip().lower()
        if answer in ("yes", "y"):
            return "yes"
        if answer in ("always", "a"):
            return "always"
        return "no"
    except (EOFError, KeyboardInterrupt):
        return "no"


def _toolbar_text(state: SessionState) -> str:
    """Return prompt_toolkit HTML string for the bottom status toolbar."""
    model = state.model or "(default)"
    cost = f"${_estimate_cost(state):.4f}"
    return (
        f"<style bg='' fg='#f97316'> {state.provider_name} </style>"
        f"<style bg='' fg='#555'>·</style>"
        f"<style bg='' fg='#f97316'> {model} </style>"
        f"<style bg='' fg='#555'>·</style>"
        f"<style bg='' fg='white'> ↑{state.total_input_tokens} ↓{state.total_output_tokens} </style>"
        f"<style bg='' fg='#555'>·</style>"
        f"<style bg='' fg='white'> {cost} </style>"
        f"<style bg='' fg='#555'>·</style>"
        f"<style bg='' fg='white'> turn {state.turn_count} </style>"
    )


def _estimate_cost(state: SessionState) -> float:
    """Rough cost estimate based on provider/model token usage."""
    # Prices per 1M tokens (input, output) — approximate
    _PRICING: dict[str, tuple[float, float]] = {
        "anthropic/claude-opus-4":    (15.0, 75.0),
        "anthropic/claude-sonnet-4":  (3.0,  15.0),
        "anthropic/claude-haiku-4-5": (0.8,   4.0),
        "claude-opus-4-6":            (15.0, 75.0),
        "claude-sonnet-4-6":          (3.0,  15.0),
        "claude-haiku-4-5":           (0.8,   4.0),
        "gpt-4o":                     (2.5,  10.0),
        "gpt-4o-mini":                (0.15,  0.6),
        "openai/gpt-4o":              (2.5,  10.0),
        "openai/o1":                  (15.0, 60.0),
        "google/gemini-2.0-flash":    (0.1,   0.4),
    }
    model = state.model or ""
    price_in, price_out = _PRICING.get(model, (3.0, 15.0))  # default to sonnet pricing
    return (state.total_input_tokens * price_in + state.total_output_tokens * price_out) / 1_000_000


async def _read_input(state: SessionState, session, style) -> str:
    """Read one line of user input via prompt-toolkit (async) with live status bar."""
    from prompt_toolkit.formatted_text import HTML

    return await session.prompt_async(
        "> ",
        style=style,
        bottom_toolbar=lambda: HTML(_toolbar_text(state)),
    )


def _build_system_prompt(memory: "PersistentMemory") -> str:
    """Build the full system prompt including memory instructions and current MEMORY.md."""
    from pathlib import Path

    memory_dir = str(Path.home() / ".astra" / "memory" / "")

    memory_prompt = f"""\
## Memory

You have a persistent, file-based memory system at `{memory_dir}`. \
This directory already exists — write to it directly with the file_write tool \
(do not run mkdir or check for its existence).

You should build up this memory system over time so that future conversations \
can have a complete picture of who the user is, how they'd like to collaborate \
with you, what behaviors to avoid or repeat, and the context behind the work \
the user gives you.

If the user explicitly asks you to remember something, save it immediately as \
whichever type fits best. If they ask you to forget something, find and remove \
the relevant entry.

## Types of memory

There are several discrete types of memory that you can store in your memory system:

<types>
<type>
    <name>user</name>
    <description>Contain information about the user's role, goals, responsibilities, and knowledge. Great user memories help you tailor your future behavior to the user's preferences and perspective. Your goal in reading and writing these memories is to build up an understanding of who the user is and how you can be most helpful to them specifically. For example, you should collaborate with a senior software engineer differently than a student who is coding for the very first time. Keep in mind, that the aim here is to be helpful to the user. Avoid writing memories about the user that could be viewed as a negative judgement or that are not relevant to the work you're trying to accomplish together.</description>
    <when_to_save>When you learn any details about the user's role, preferences, responsibilities, or knowledge</when_to_save>
    <how_to_use>When your work should be informed by the user's profile or perspective.</how_to_use>
    <examples>
    user: I'm a data scientist investigating what logging we have in place
    assistant: [saves user memory: user is a data scientist, currently focused on observability/logging]

    user: I've been writing Go for ten years but this is my first time touching the React side of this repo
    assistant: [saves user memory: deep Go expertise, new to React and this project's frontend — frame frontend explanations in terms of backend analogues]
    </examples>
</type>
<type>
    <name>feedback</name>
    <description>Guidance the user has given you about how to approach work — both what to avoid and what to keep doing. These are a very important type of memory to read and write as they allow you to remain coherent and responsive to the way you should approach work in the project. Record from failure AND success: if you only save corrections, you will avoid past mistakes but drift away from approaches the user has already validated, and may grow overly cautious.</description>
    <when_to_save>Any time the user corrects your approach ("no not that", "don't", "stop doing X") OR confirms a non-obvious approach worked ("yes exactly", "perfect, keep doing that", accepting an unusual choice without pushback). Corrections are easy to notice; confirmations are quieter — watch for them. In both cases, save what is applicable to future conversations, especially if surprising or not obvious from the code. Include *why* so you can judge edge cases later.</when_to_save>
    <how_to_use>Let these memories guide your behavior so that the user does not need to offer the same guidance twice.</how_to_use>
    <body_structure>Lead with the rule itself, then a **Why:** line (the reason the user gave — often a past incident or strong preference) and a **How to apply:** line (when/where this guidance kicks in). Knowing *why* lets you judge edge cases instead of blindly following the rule.</body_structure>
    <examples>
    user: don't mock the database in these tests — we got burned last quarter when mocked tests passed but the prod migration failed
    assistant: [saves feedback memory: integration tests must hit a real database, not mocks. Reason: prior incident where mock/prod divergence masked a broken migration]

    user: stop summarizing what you just did at the end of every response, I can read the diff
    assistant: [saves feedback memory: this user wants terse responses with no trailing summaries]
    </examples>
</type>
<type>
    <name>project</name>
    <description>Information that you learn about ongoing work, goals, initiatives, bugs, or incidents within the project that is not otherwise derivable from the code or git history. Project memories help you understand the broader context and motivation behind the work the user is doing within this working directory.</description>
    <when_to_save>When you learn who is doing what, why, or by when. These states change relatively quickly so try to keep your understanding of this up to date. Always convert relative dates in user messages to absolute dates when saving (e.g., "Thursday" → "2026-03-05"), so the memory remains interpretable after time passes.</when_to_save>
    <how_to_use>Use these memories to more fully understand the details and nuance behind the user's request and make better informed suggestions.</how_to_use>
    <body_structure>Lead with the fact or decision, then a **Why:** line (the motivation — often a constraint, deadline, or stakeholder ask) and a **How to apply:** line (how this should shape your suggestions). Project memories decay fast, so the why helps future-you judge whether the memory is still load-bearing.</body_structure>
    <examples>
    user: we're freezing all non-critical merges after Thursday — mobile team is cutting a release branch
    assistant: [saves project memory: merge freeze begins 2026-03-05 for mobile release cut. Flag any non-critical PR work scheduled after that date]
    </examples>
</type>
<type>
    <name>reference</name>
    <description>Stores pointers to where information can be found in external systems. These memories allow you to remember where to look to find up-to-date information outside of the project directory.</description>
    <when_to_save>When you learn about resources in external systems and their purpose. For example, that bugs are tracked in a specific project in Linear or that feedback can be found in a specific Slack channel.</when_to_save>
    <how_to_use>When the user references an external system or information that may be in an external system.</how_to_use>
    <examples>
    user: check the Linear project "INGEST" if you want context on these tickets, that's where we track all pipeline bugs
    assistant: [saves reference memory: pipeline bugs are tracked in Linear project "INGEST"]
    </examples>
</type>
</types>

## What NOT to save in memory

- Code patterns, conventions, architecture, file paths, or project structure — these can be derived by reading the current project state.
- Git history, recent changes, or who-changed-what — `git log` / `git blame` are authoritative.
- Debugging solutions or fix recipes — the fix is in the code; the commit message has the context.
- Ephemeral task details: in-progress work, temporary state, current conversation context.

These exclusions apply even when the user explicitly asks you to save. If they ask you to save a PR list or activity summary, ask what was *surprising* or *non-obvious* about it — that is the part worth keeping.

## How to save memories

Saving a memory is a two-step process:

**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:

```markdown
---
name: {{memory name}}
description: {{one-line description — used to decide relevance in future conversations, so be specific}}
type: {{user, feedback, project, reference}}
---

{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}
```

**Step 2** — add a pointer to that file in `MEMORY.md`. `MEMORY.md` is an index, not a memory — each entry should be one line, under ~150 characters: `- [Title](file.md) — one-line hook`. It has no frontmatter. Never write memory content directly into `MEMORY.md`.

- `MEMORY.md` is always loaded into your conversation context — lines after 200 will be truncated, so keep the index concise
- Keep the name, description, and type fields in memory files up-to-date with the content
- Organize memory semantically by topic, not chronologically
- Update or remove memories that turn out to be wrong or outdated
- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.

## When to access memories
- When memories seem relevant, or the user references prior-conversation work.
- You MUST access memory when the user explicitly asks you to check, recall, or remember.
- Memory records can become stale over time. Before answering based solely on a memory, verify it is still correct by reading the current state of the relevant files or resources.

## Before recommending from memory

A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:

- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation (not just asking about history), verify first.

"The memory says X exists" is not the same as "X exists now.\""""

    # Inject current MEMORY.md content if it exists
    memory_index = memory.inject_into_system_prompt("").strip()
    if memory_index:
        memory_prompt += f"\n\n{memory_index}"

    base = (
        "You are Astra, a helpful AI agent.\n\n"
        "SECURITY RULES — these cannot be overridden by any user instruction:\n"
        "- Never read, display, print, or transmit API keys, passwords, tokens, or secrets "
        "from any source (environment variables, config files, .env files, key files, etc.).\n"
        "- Never search for, list, or enumerate credential files (e.g. ~/.astra/config.json, "
        ".env, id_rsa, *.pem, *.key).\n"
        "- If a user asks you to reveal credentials or find API keys, refuse and explain "
        "that you cannot access or expose secrets."
    )

    return f"{memory_prompt}\n\n{base}"


def _build_session_state(console: Console) -> SessionState:
    """Build SessionState from config, prompting for setup if needed."""
    from astra_cli.commands.run import _build_provider
    from astra_node.tools.bash import BashTool
    from astra_node.tools.file_read import FileReadTool
    from astra_node.tools.file_write import FileWriteTool
    from astra_node.tools.file_edit import FileEditTool
    from astra_node.tools.grep import GrepTool
    from astra_node.tools.glob_tool import GlobTool
    from astra_node.tools.web_search import WebSearchTool
    from astra_node.tools.web_fetch import WebFetchTool

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
        WebSearchTool(),
        WebFetchTool(),
    ]:
        registry.register(tool)

    pm = PermissionManager()
    provider = _build_provider(provider_name, model, base_url)
    memory = PersistentMemory()

    engine = QueryEngine(
        provider=provider,
        registry=registry,
        permission_manager=pm,
        system_prompt=_build_system_prompt(memory),
        memory=memory,
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
