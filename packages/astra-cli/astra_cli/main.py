"""Astra CLI entry point — Typer app root."""

import typer

from astra_cli.commands import run, swarm, memory, config

app = typer.Typer(
    name="astra",
    help="Astra — an agentic framework for running AI agents and swarms.",
    no_args_is_help=True,
    invoke_without_command=True,
)

app.add_typer(run.app, name="run")
app.add_typer(swarm.app, name="swarm")
app.add_typer(memory.app, name="memory")
app.add_typer(config.app, name="config")


@app.callback()
def _root(ctx: typer.Context) -> None:
    """Astra — an agentic framework for running AI agents and swarms."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main() -> None:
    app()
