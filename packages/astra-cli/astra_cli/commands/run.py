"""astra run — single-agent task runner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run a single agent on a task.")


def _build_provider(provider: str, model: str | None, base_url: str | None = None):
    """Build the appropriate LLM provider."""
    if provider == "ollama":
        from astra_node.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=model or "qwen2.5-coder",
            base_url=base_url or "http://localhost:11434/v1",
            api_key="ollama",
        )
    elif provider == "openai":
        from astra_node.providers.openai import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-4o-mini", base_url=base_url)
    else:
        # default: anthropic
        from astra_node.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-4-6")


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
        system_prompt="You are Astra, a helpful AI agent.",
    )


@app.callback(invoke_without_command=True)
def run(
    task: Optional[str] = typer.Argument(None, help="Task description to run."),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Read task from file."
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM provider: anthropic, openai, ollama."
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

    asyncio.run(_run())
