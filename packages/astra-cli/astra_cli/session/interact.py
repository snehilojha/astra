"""Shared interactive input helpers for the Astra REPL."""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

ACCENT = "#f97316"


def _interactive_select(
    prompt: str,
    options: list[str],
    console: Console,
) -> str:
    """Show an interactive selector and return the selected option.

    Uses prompt_toolkit when available in an interactive terminal.
    Falls back to a numeric prompt for compatibility with limited terminals.
    """
    if not options:
        raise ValueError("options must not be empty")

    if not sys.stdin.isatty():
        return _numeric_fallback_select(prompt, options, console)

    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style

    selected = 0

    def get_menu_text():
        lines = []
        lines.append(("class:prompt-text", f"  {prompt} "))
        for i, opt in enumerate(options):
            if i == selected:
                lines.append(("", f" ▶ {opt} "))
            else:
                lines.append(("class:dim", f"   {opt} "))
        lines.append(("", "\n  (use arrow keys, enter to select)"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    @kb.add("c-p")
    def _up(event):
        nonlocal selected
        selected = (selected - 1) % len(options)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    @kb.add("c-n")
    def _down(event):
        nonlocal selected
        selected = (selected + 1) % len(options)
        event.app.invalidate()

    @kb.add("left")
    @kb.add("h")
    def _left(event):
        nonlocal selected
        selected = (selected - 1) % len(options)
        event.app.invalidate()

    @kb.add("right")
    @kb.add("l")
    def _right(event):
        nonlocal selected
        selected = (selected + 1) % len(options)
        event.app.invalidate()

    @kb.add("tab")
    def _tab(event):
        nonlocal selected
        selected = (selected + 1) % len(options)
        event.app.invalidate()

    @kb.add("s-tab")
    def _shift_tab(event):
        nonlocal selected
        selected = (selected - 1) % len(options)
        event.app.invalidate()

    @kb.add("enter")
    def _select(event):
        event.app.exit(result=options[selected])

    @kb.add("c-c")
    @kb.add("c-d")
    def _exit(event):
        event.app.exit(result=options[0])

    container = HSplit(
        [
            Window(
                content=FormattedTextControl(get_menu_text),
                width=Dimension(preferred=40),
            )
        ]
    )

    style = Style.from_dict(
        {
            "prompt-text": ACCENT + " bold",
            "dim": "#888888",
        }
    )

    app = Application(
        layout=Layout(container),
        key_bindings=kb,
        full_screen=False,
        style=style,
    )

    try:
        # REPL slash-commands run while an asyncio loop is active.
        # In that case, run the selector in a worker thread so arrow-key UI still works.
        try:
            asyncio.get_running_loop()
            return app.run(in_thread=True)
        except RuntimeError:
            return app.run()
    except Exception:
        return _numeric_fallback_select(prompt, options, console)


def _numeric_fallback_select(prompt: str, options: list[str], console: Console) -> str:
    """Fallback selector for non-interactive or incompatible terminals."""
    console.print(f"  [{ACCENT}]{prompt}[/{ACCENT}]")
    for i, option in enumerate(options, start=1):
        console.print(f"    {i}. {option}")
    raw = console.input("  Select option number: ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    return options[0]
