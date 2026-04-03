"""SwarmCoordinator and supporting types for multi-agent orchestration.

Three strategies:
  pipeline     — worker[0] output → worker[1] input → … → final output
  parallel     — all workers run the same task concurrently, results merged
  hierarchical — coordinator plans + delegates, workers execute, coordinator aggregates

SwarmEvent wraps any AgentEvent with a worker_id so callers always know
which agent produced each event.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from astra_node.core.events import AgentEvent
from astra_node.core.query_engine import QueryEngine
from astra_node.core.registry import ToolRegistry
from astra_node.permissions.manager import PermissionManager
from astra_node.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# SwarmEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwarmEvent(AgentEvent):
    """An AgentEvent annotated with the worker that produced it.

    inner_type mirrors the wrapped event's .type string so consumers can
    filter without unwrapping (e.g. inner_type == "text_delta").
    data carries the wrapped event's fields as a plain dict for easy
    serialisation.
    """

    worker_id: str = ""
    inner_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """Configuration for a single worker agent within a swarm.

    id:            Unique identifier — appears as worker_id on every SwarmEvent.
    system_prompt: The worker's system prompt (defines its role/specialisation).
    allowed_tools: Subset of the swarm's ToolRegistry the worker may use.
                   Empty set = worker gets no tools.
    provider:      LLMProvider for this worker. Required at construction time
                   (loader sets it from YAML model/provider fields).
    max_turns:     Max LLM calls per task for this worker.
    """

    id: str
    system_prompt: str
    allowed_tools: set[str] = field(default_factory=set)
    provider: LLMProvider | None = None
    max_turns: int = 10


@dataclass
class SwarmConfig:
    """Top-level swarm configuration.

    strategy must be one of: "pipeline", "parallel", "hierarchical".
    coordinator is required for hierarchical strategy.
    workers must be non-empty.
    """

    name: str
    strategy: Literal["hierarchical", "pipeline", "parallel"]
    workers: list[WorkerConfig]
    coordinator: WorkerConfig | None = None

    def __post_init__(self) -> None:
        valid_strategies = {"hierarchical", "pipeline", "parallel"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}."
            )
        if not self.workers:
            raise ValueError("SwarmConfig.workers must not be empty.")
        if self.strategy == "hierarchical" and self.coordinator is None:
            raise ValueError(
                "strategy='hierarchical' requires a coordinator WorkerConfig."
            )


# ---------------------------------------------------------------------------
# WorkerExecutor Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class WorkerExecutor(Protocol):
    """Swappable execution backend for worker agents.

    Default implementation: AsyncioExecutor (coroutines in the current loop).
    Future: SubprocessExecutor, CeleryExecutor for distributed workers.
    """

    async def run(
        self,
        engine: QueryEngine,
        task: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run one task on the given engine, yielding events."""
        ...


class AsyncioExecutor:
    """Default executor — runs worker engines as asyncio coroutines."""

    async def run(
        self,
        engine: QueryEngine,
        task: str,
    ) -> AsyncIterator[AgentEvent]:
        async for event in engine.run(task):
            yield event
