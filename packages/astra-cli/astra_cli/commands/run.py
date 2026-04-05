"""astra run — single-agent task runner."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run a single agent on a task.")

# ---------------------------------------------------------------------------
# Central provider registry — single source of truth for all provider config.
# Adding a new provider: add one entry here. Nothing else needs changing.
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, dict] = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-6",
        "base_url": None,
        "adapter": "anthropic",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "base_url": None,
        "adapter": "openai",
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4",
        "base_url": "https://openrouter.ai/api/v1",
        "adapter": "openai",
    },
    "ollama": {
        "env_var": None,
        "default_model": "qwen2.5-coder",
        "base_url": "http://localhost:11434/v1",
        "adapter": "openai",
        "api_key_override": "ollama",  # Ollama accepts any non-empty string
    },
}


def get_api_key_env_var(provider: str) -> str | None:
    """Return the environment variable name for the given provider's API key."""
    return PROVIDER_REGISTRY.get(provider, {}).get("env_var")


def _load_api_key(provider: str) -> str | None:
    """Resolve the API key for a provider: config file → env var → None.

    Writes the config value into os.environ so the SDK can find it via its
    own env-var lookup, keeping the provider constructors simple.
    """
    cfg_entry = PROVIDER_REGISTRY.get(provider, {})
    api_key_override = cfg_entry.get("api_key_override")
    if api_key_override:
        return api_key_override  # e.g. Ollama always uses "ollama"

    env_var = cfg_entry.get("env_var")
    if not env_var:
        return None

    # Already in environment — nothing to do
    if os.environ.get(env_var):
        return os.environ[env_var]

    # Try loading from ~/.astra/config.json
    config_path = Path.home() / ".astra" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            key = cfg.get(env_var)
            if key:
                os.environ[env_var] = key  # expose to SDK
                return key
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _build_provider(provider: str, model: str | None, base_url: str | None = None):
    """Build the appropriate LLM provider, loading auth from config if needed."""
    cfg_entry = PROVIDER_REGISTRY.get(provider)
    if cfg_entry is None:
        # Unknown provider — fall back to Anthropic and warn
        typer.echo(
            f"[warning] Unknown provider '{provider}', falling back to anthropic.",
            err=True,
        )
        cfg_entry = PROVIDER_REGISTRY["anthropic"]
        provider = "anthropic"

    resolved_model = model or cfg_entry["default_model"]
    resolved_base_url = base_url or cfg_entry.get("base_url")
    api_key = _load_api_key(provider)

    adapter = cfg_entry["adapter"]
    if adapter == "openai":
        from astra_node.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=api_key,
            provider_name=provider,
        )
    else:
        from astra_node.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=resolved_model, api_key=api_key)


def _build_engine(provider_name: str, model: str | None, base_url: str | None = None):
    """Build a QueryEngine with default tools and permissions."""
    from astra_node.core.query_engine import QueryEngine
    from astra_node.core.registry import ToolRegistry
    from astra_node.permissions.manager import PermissionManager
    from astra_node.tools.bash import BashTool
    from astra_node.tools.file_read import FileReadTool
    from astra_node.tools.file_write import FileWriteTool
    from astra_node.tools.file_edit import FileEditTool
    from astra_node.tools.grep import GrepTool
    from astra_node.tools.glob_tool import GlobTool

    provider = _build_provider(provider_name, model, base_url)

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

    permission_manager = PermissionManager()

    return QueryEngine(
        provider=provider,
        registry=registry,
        permission_manager=permission_manager,
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


@app.callback(invoke_without_command=True)
def run(
    task: Optional[str] = typer.Argument(None, help="Task description to run."),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Read task from file."
    ),
    provider: str = typer.Option(
        "anthropic",
        "--provider",
        "-p",
        help="LLM provider: anthropic, openai, openrouter, ollama.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name override."
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        "-u",
        help="Base URL for OpenAI-compatible API (e.g. OpenRouter, local servers).",
    ),
) -> None:
    """Run a single agent on a task."""
    from astra_cli.display.event_renderer import EventRenderer

    if file is not None:
        task_text = file.read_text()
    elif task is not None:
        task_text = task
    else:
        typer.echo("Error: provide a task argument or --file.", err=True)
        raise typer.Exit(1)

    engine = _build_engine(provider, model, base_url)
    renderer = EventRenderer()

    async def _run() -> None:
        async for event in engine.run(task_text):
            renderer.render(event)

    try:
        asyncio.run(_run())
    except Exception as exc:
        from astra_node.utils.errors import PromptInjectionError
        if isinstance(exc, PromptInjectionError):
            typer.echo(f"\nBlocked: {exc}", err=True)
            raise typer.Exit(1)
        _handle_provider_error(exc, provider)


def _handle_provider_error(exc: Exception, provider: str) -> None:
    """Handle provider errors with a friendly message and optional API key prompt."""
    from astra_node.utils.errors import ProviderError

    if isinstance(exc, ProviderError):
        typer.echo(f"\nError: {exc}", err=True)
        env_var = _get_api_key_env_var(provider)
        if env_var:
            looks_like_auth = (
                "auth" in str(exc).lower()
                or "api key" in str(exc).lower()
                or not os.environ.get(env_var)
            )
            if looks_like_auth:
                _show_api_key_hints(provider, env_var)
                _prompt_for_api_key(env_var, provider)
        raise typer.Exit(1)
    else:
        typer.echo(f"\nUnexpected error: {exc}", err=True)
        raise typer.Exit(1)


def _get_api_key_env_var(provider: str) -> str | None:
    """Return the environment variable name for the given provider's API key."""
    return get_api_key_env_var(provider)


def _prompt_for_api_key(env_var: str, provider: str) -> None:
    """Prompt the user to enter an API key and save it to config."""
    try:
        answer = (
            typer.prompt(
                f"Configure {provider} API key now? [y/N]",
                default="n",
                show_default=False,
            )
            .strip()
            .lower()
        )
        if answer not in {"y", "yes"}:
            typer.echo(
                f"Skipped. You can set it later with: astra config set {env_var} <your-key>",
                err=True,
            )
            return
        key = typer.prompt(f"\nEnter your {provider} API key", hide_input=True)
    except (KeyboardInterrupt, EOFError):
        typer.echo("\nAborted.", err=True)
        return

    if not key.strip():
        typer.echo("No key entered.", err=True)
        return

    os.environ[env_var] = key.strip()

    config_path = Path.home() / ".astra" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    cfg[env_var] = key.strip()
    config_path.write_text(json.dumps(cfg, indent=2))
    # Restrict to owner read/write only (rw-------) to protect stored API keys.
    import stat as _stat
    try:
        os.chmod(config_path, _stat.S_IRUSR | _stat.S_IWUSR)
    except OSError:
        pass  # Windows may not support POSIX permissions; best-effort.

    typer.echo(f"API key saved to {config_path}", err=True)
    typer.echo("You can now retry your command.", err=True)


def _show_api_key_hints(provider: str, env_var: str) -> None:
    """Show helpful hints for API key setup."""
    providers = {
        "anthropic": "https://console.anthropic.com/settings/keys",
        "openai": "https://platform.openai.com/api-keys",
        "openrouter": "https://openrouter.ai/keys",
    }
    if os.environ.get(env_var):
        typer.echo(
            f"{env_var} is set but authentication still failed. It may be invalid or expired.",
            err=True,
        )
    else:
        typer.echo(f"{env_var} is not set.", err=True)

    typer.echo(f"Set it with: astra config set {env_var} <your-key>", err=True)
    if os.name == "nt":
        typer.echo(f"PowerShell (current shell): $env:{env_var}='YOUR_KEY'", err=True)
    else:
        typer.echo(f"Bash/Zsh (current shell): export {env_var}=YOUR_KEY", err=True)
    key_url = providers.get(provider)
    if key_url:
        typer.echo(f"Get a key from: {key_url}", err=True)
