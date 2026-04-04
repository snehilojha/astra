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


# ---------------------------------------------------------------------------
# SwarmCoordinator helpers
# ---------------------------------------------------------------------------


def _event_to_swarm(worker_id: str, event: AgentEvent) -> SwarmEvent:
    """Wrap any AgentEvent as a SwarmEvent, preserving fields as data dict."""
    data = {k: v for k, v in vars(event).items() if k != "type"}
    return SwarmEvent(worker_id=worker_id, inner_type=event.type, data=data)


def _extract_final_text(events: list[SwarmEvent]) -> str:
    """Extract the concatenated text_delta content from a list of SwarmEvents."""
    return "".join(
        e.data.get("text", "") for e in events if e.inner_type == "text_delta"
    )


# ---------------------------------------------------------------------------
# SwarmCoordinator
# ---------------------------------------------------------------------------


class SwarmCoordinator:
    """Orchestrates multiple QueryEngine workers according to a SwarmConfig.

    Strategies:
      pipeline     — sequential; each worker's text output becomes the next's task.
      parallel     — concurrent; all workers run the same task simultaneously.
      hierarchical — coordinator LLM plans and delegates; workers execute; coordinator aggregates.
    """

    def __init__(
        self,
        config: SwarmConfig,
        registry: ToolRegistry,
        executor: WorkerExecutor | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._executor = executor or AsyncioExecutor()

    def _make_engine(self, worker_cfg: WorkerConfig) -> QueryEngine:
        """Build an isolated QueryEngine for one worker."""
        if worker_cfg.provider is None:
            raise ValueError(
                f"Worker '{worker_cfg.id}' has no provider. "
                "Set WorkerConfig.provider before running the swarm."
            )
        worker_registry = (
            self._registry.filter(worker_cfg.allowed_tools)
            if worker_cfg.allowed_tools
            else ToolRegistry()
        )
        return QueryEngine(
            provider=worker_cfg.provider,
            registry=worker_registry,
            permission_manager=PermissionManager(),
            system_prompt=worker_cfg.system_prompt,
            max_turns=worker_cfg.max_turns,
        )

    async def run(self, task: str) -> AsyncIterator[SwarmEvent]:
        """Run the swarm on the given task, yielding SwarmEvents."""
        strategy = self._config.strategy
        if strategy == "pipeline":
            async for event in self._run_pipeline(task):
                yield event
        elif strategy == "parallel":
            async for event in self._run_parallel(task):
                yield event
        elif strategy == "hierarchical":
            async for event in self._run_hierarchical(task):
                yield event

    async def _run_pipeline(self, task: str) -> AsyncIterator[SwarmEvent]:
        """Sequential: each worker's output becomes the next worker's input."""
        current_task = task
        for worker_cfg in self._config.workers:
            engine = self._make_engine(worker_cfg)
            worker_events: list[SwarmEvent] = []
            async for event in self._executor.run(engine, current_task):
                wrapped = _event_to_swarm(worker_cfg.id, event)
                worker_events.append(wrapped)
                yield wrapped
            current_task = _extract_final_text(worker_events) or current_task

    async def _run_parallel(self, task: str) -> AsyncIterator[SwarmEvent]:
        """Concurrent: all workers run the same task simultaneously."""

        async def run_worker(worker_cfg: WorkerConfig) -> list[SwarmEvent]:
            engine = self._make_engine(worker_cfg)
            worker_events = []
            try:
                async for event in self._executor.run(engine, task):
                    worker_events.append(_event_to_swarm(worker_cfg.id, event))
            except Exception as exc:
                from astra_node.core.events import AgentError

                err = AgentError(
                    error=str(exc),
                    tool_name="",
                    tool_use_id="",
                    recoverable=False,
                )
                worker_events.append(_event_to_swarm(worker_cfg.id, err))
            return worker_events

        tasks = [asyncio.create_task(run_worker(wcfg)) for wcfg in self._config.workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                for event in result:
                    yield event
            elif isinstance(result, Exception):
                from astra_node.core.events import AgentError

                err = AgentError(
                    error=f"Worker failed: {result}",
                    tool_name="",
                    tool_use_id="",
                    recoverable=False,
                )
                yield SwarmEvent(
                    worker_id="parallel_executor",
                    inner_type="agent_error",
                    data={"error": str(result), "recoverable": False},
                )

    async def _run_hierarchical(self, task: str) -> AsyncIterator[SwarmEvent]:
        """Coordinator plans, workers execute in parallel, coordinator aggregates."""
        coordinator_cfg = self._config.coordinator  # validated non-None by SwarmConfig
        coord_engine = self._make_engine(coordinator_cfg)

        # Phase 1: coordinator plans
        plan_task = (
            f"You are a coordinator. Break down this task and assign subtasks to workers "
            f"(worker IDs: {[w.id for w in self._config.workers]}). "
            f"For each worker, output exactly one line: WORKER_ID: subtask description.\n\n"
            f"Task: {task}"
        )
        coord_events: list[SwarmEvent] = []
        async for event in self._executor.run(coord_engine, plan_task):
            wrapped = _event_to_swarm(coordinator_cfg.id, event)
            coord_events.append(wrapped)
            yield wrapped

        coord_text = _extract_final_text(coord_events)

        # Parse worker assignments from coordinator output
        worker_tasks: dict[str, str] = {}
        for line in coord_text.splitlines():
            for worker_cfg in self._config.workers:
                prefix = f"{worker_cfg.id}:"
                if line.strip().startswith(prefix):
                    worker_tasks[worker_cfg.id] = line.strip()[len(prefix) :].strip()
                    break

        # Fallback: unmatched workers get the original task
        for worker_cfg in self._config.workers:
            if worker_cfg.id not in worker_tasks:
                worker_tasks[worker_cfg.id] = task

        # Phase 2: workers run in parallel
        async def run_worker(worker_cfg: WorkerConfig) -> list[SwarmEvent]:
            engine = self._make_engine(worker_cfg)
            worker_events = []
            try:
                async for event in self._executor.run(
                    engine, worker_tasks[worker_cfg.id]
                ):
                    worker_events.append(_event_to_swarm(worker_cfg.id, event))
            except Exception as exc:
                from astra_node.core.events import AgentError

                err = AgentError(
                    error=str(exc), tool_name="", tool_use_id="", recoverable=False
                )
                worker_events.append(_event_to_swarm(worker_cfg.id, err))
            return worker_events

        worker_task_objects = [
            asyncio.create_task(run_worker(wcfg)) for wcfg in self._config.workers
        ]
        worker_results = await asyncio.gather(
            *worker_task_objects, return_exceptions=True
        )
        all_worker_events: list[SwarmEvent] = []
        for result in worker_results:
            if isinstance(result, list):
                all_worker_events.extend(result)
                for event in result:
                    yield event
            elif isinstance(result, Exception):
                from astra_node.core.events import AgentError

                yield SwarmEvent(
                    worker_id="hierarchical_worker",
                    inner_type="agent_error",
                    data={"error": str(result), "recoverable": False},
                )

        # Phase 3: coordinator aggregates
        worker_summaries = "\n".join(
            f"{w.id}: {_extract_final_text([e for e in all_worker_events if e.worker_id == w.id])}"
            for w in self._config.workers
        )
        aggregation_task = (
            f"Aggregate these worker results into a final answer:\n{worker_summaries}"
        )
        coord_engine2 = self._make_engine(coordinator_cfg)
        async for event in self._executor.run(coord_engine2, aggregation_task):
            yield _event_to_swarm(coordinator_cfg.id, event)
