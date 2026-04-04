"""Shared interactive input helpers for the Astra REPL."""

from __future__ import annotations

from rich.console import Console

ACCENT = "#f97316"


def _interactive_select(
    prompt: str,
    options: list[str],
    console: Console,
) -> str:
    """Show an interactive arrow-key selectable menu and return the selected option."""
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
    def _up(event):
        nonlocal selected
        selected = (selected - 1) % len(options)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        nonlocal selected
        selected = (selected + 1) % len(options)
        event.app.invalidate()

    @kb.add("left")
    def _left(event):
        nonlocal selected
        selected = (selected - 1) % len(options)
        event.app.invalidate()

    @kb.add("right")
    def _right(event):
        nonlocal selected
        selected = (selected + 1) % len(options)
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

    return app.run()
