"""astra swarm — run and list swarm configurations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run or list swarm configurations.")

# Default location for swarm YAML configs
_DEFAULT_CONFIG_DIR = Path(__file__).parents[4] / "swarm-configs"


def _find_config_dir() -> Path:
    """Return the swarm config directory, falling back to ~/.astra/swarms."""
    if _DEFAULT_CONFIG_DIR.exists():
        return _DEFAULT_CONFIG_DIR
    fallback = Path.home() / ".astra" / "swarms"
    return fallback


@app.command("list")
def swarm_list() -> None:
    """List available swarm configurations."""
    config_dir = _find_config_dir()
    if not config_dir.exists():
        typer.echo(f"No swarm config directory found (checked {config_dir}).")
        return

    yamls = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not yamls:
        typer.echo("No swarm configs found.")
        return

    typer.echo("Available swarm configs:")
    for path in yamls:
        typer.echo(f"  {path.stem}")


@app.command("run")
def swarm_run(
    name: str = typer.Argument(..., help="Swarm config name (without .yaml)."),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task to run."),
    target: Optional[str] = typer.Option(None, "--target", help="Target path (injected into task)."),
) -> None:
    """Run a named swarm configuration."""
    from astra_swarm.swarm_loader import load_swarm_from_yaml
    from astra_cli.display.event_renderer import EventRenderer

    config_dir = _find_config_dir()
    yaml_path = config_dir / f"{name}.yaml"
    if not yaml_path.exists():
        yaml_path = config_dir / f"{name}.yml"
    if not yaml_path.exists():
        typer.echo(f"Swarm config '{name}' not found in {config_dir}.", err=True)
        raise typer.Exit(1)

    if task is None and target is None:
        typer.echo("Error: provide --task or --target.", err=True)
        raise typer.Exit(1)

    task_text = task or f"Analyze {target}"
    if target:
        task_text = task_text + f" (target: {target})" if task else f"Analyze {target}"

    coordinator = load_swarm_from_yaml(yaml_path)
    renderer = EventRenderer()

    async def _run() -> None:
        async for event in coordinator.run(task_text):
            renderer.render(event)

    asyncio.run(_run())
