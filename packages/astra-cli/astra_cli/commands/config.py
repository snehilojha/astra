"""astra config — get/set configuration values."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Get and set configuration values.")

_CONFIG_PATH = Path.home() / ".astra" / "config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. ANTHROPIC_API_KEY)."),
    value: str = typer.Argument(..., help="Value to store."),
) -> None:
    """Store a configuration value."""
    cfg = _load_config()
    cfg[key] = value
    _save_config(cfg)
    typer.echo(f"Set {key}.")


@app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key to retrieve."),
) -> None:
    """Retrieve a configuration value."""
    cfg = _load_config()
    if key not in cfg:
        typer.echo(f"{key} is not set.", err=True)
        raise typer.Exit(1)
    typer.echo(cfg[key])
