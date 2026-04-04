"""Astra CLI entry point — Typer app root."""

import json
import os
from pathlib import Path

import typer

from astra_cli.commands import run, swarm, memory, config


def _load_config_into_env() -> None:
    """Load ~/.astra/config.json and inject keys into os.environ if not already set."""
    config_path = Path.home() / ".astra" / "config.json"
    if not config_path.exists():
        return
    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    for key, value in cfg.items():
        if key not in os.environ:
            os.environ[key] = value


app = typer.Typer(
    name="astra",
    help="Astra — an agentic framework for running AI agents and swarms.",
    no_args_is_help=False,
    invoke_without_command=True,
)

app.add_typer(run.app, name="run")
app.add_typer(swarm.app, name="swarm")
app.add_typer(memory.app, name="memory")
app.add_typer(config.app, name="config")


@app.callback()
def _root(ctx: typer.Context) -> None:
    """Astra — an agentic framework for running AI agents and swarms."""
    _load_config_into_env()
    if ctx.invoked_subcommand is None:
        from astra_cli.session import repl as _repl

        _repl.start()


def main() -> None:
    app()
