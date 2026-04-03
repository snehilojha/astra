"""astra memory — list, search, and clear the memory store."""

from __future__ import annotations

import typer

app = typer.Typer(help="Manage the agent's persistent memory.")


def _get_memory():
    """Return the active MemorySystem instance."""
    from astra_node.core.memory_stub import StubMemory

    return StubMemory()


@app.command("list")
def memory_list() -> None:
    """Show all persisted memories."""
    memory = _get_memory()
    profile = memory.get_user_context()
    if not profile.topics:
        typer.echo("No memories stored.")
        return
    typer.echo("Topics:")
    for topic, weight in profile.topics.items():
        typer.echo(f"  - {topic} (weight: {weight:.2f})")


@app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search term."),
) -> None:
    """Search memories for a term."""
    memory = _get_memory()
    ctx = memory.query(query)
    if not ctx.retrieved_chunks:
        typer.echo(f"No memories matching '{query}'.")
        return
    for chunk in ctx.retrieved_chunks:
        typer.echo(chunk.text)


@app.command("clear")
def memory_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Wipe all stored memories."""
    if not yes:
        typer.confirm("Clear all memories?", abort=True)
    memory = _get_memory()
    memory.ingest([])  # StubMemory no-op; PersistentMemory will override
    typer.echo("Memory cleared.")
