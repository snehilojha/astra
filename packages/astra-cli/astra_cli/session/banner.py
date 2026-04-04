"""Astra REPL startup banner — Rich-rendered splash screen."""

from __future__ import annotations

import os

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ACCENT = "#f97316"

_ASCII_ART = """\
 █████╗ ███████╗████████╗██████╗  █████╗
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
███████║███████╗   ██║   ██████╔╝███████║
██╔══██║╚════██║   ██║   ██╔══██╗██╔══██║
██║  ██║███████║   ██║   ██║  ██║██║  ██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝"""


def print_banner(
    console: Console,
    provider: str,
    model: str | None,
    version: str = "0.1.0",
) -> None:
    """Print the Astra startup banner to the given console."""
    # Title
    title = Text(_ASCII_ART, style=f"bold {ACCENT}")
    console.print(title)
    console.print()

    # Left panel: session info
    info = Table.grid(padding=(0, 2))
    info.add_row(Text("provider", style="dim"), Text(provider, style=f"bold {ACCENT}"))
    info.add_row(
        Text("model", style="dim"),
        Text(model or "(default)", style=f"bold {ACCENT}"),
    )
    info.add_row(Text("dir", style="dim"), Text(os.getcwd(), style="white"))
    info.add_row(Text("memory", style="dim"), Text("enabled", style="green"))

    # Right panel: tips
    tips = Table.grid(padding=(0, 2))
    tips.add_row(Text("/provider", style=f"bold {ACCENT}"), Text("switch provider", style="dim"))
    tips.add_row(Text("/model   ", style=f"bold {ACCENT}"), Text("switch model", style="dim"))
    tips.add_row(Text("/swarm   ", style=f"bold {ACCENT}"), Text("run swarm <file>", style="dim"))
    tips.add_row(Text("/help    ", style=f"bold {ACCENT}"), Text("all commands", style="dim"))

    left = Panel(info, border_style=ACCENT, title="session", title_align="left")
    right = Panel(tips, border_style=ACCENT, title="tips", title_align="left")
    console.print(Columns([left, right]))

    # Version line
    console.print(
        f"  [dim]v{version}[/dim]  [{ACCENT}]●[/{ACCENT}]",
        justify="right",
    )
    console.print()
