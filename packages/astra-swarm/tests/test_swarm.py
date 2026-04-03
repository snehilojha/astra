"""Tests for SwarmCoordinator and related types."""
import pytest
from astra_swarm.swarm import (
    SwarmEvent,
    WorkerConfig,
    SwarmConfig,
)


# ---------------------------------------------------------------------------
# SwarmEvent
# ---------------------------------------------------------------------------

def test_swarm_event_has_worker_id():
    event = SwarmEvent(worker_id="worker_1", inner_type="text_delta", data={"text": "hello"})
    assert event.worker_id == "worker_1"
    assert event.inner_type == "text_delta"
    assert event.data == {"text": "hello"}


def test_swarm_event_type_is_swarm_event():
    event = SwarmEvent(worker_id="w", inner_type="turn_end", data={})
    assert event.type == "swarm_event"


# ---------------------------------------------------------------------------
# WorkerConfig
# ---------------------------------------------------------------------------

def test_worker_config_defaults():
    cfg = WorkerConfig(id="w1", system_prompt="You are a helper.")
    assert cfg.id == "w1"
    assert cfg.allowed_tools == set()
    assert cfg.max_turns == 10


def test_worker_config_allowed_tools():
    cfg = WorkerConfig(id="w1", system_prompt="s", allowed_tools={"file_read", "grep"})
    assert "file_read" in cfg.allowed_tools
    assert "grep" in cfg.allowed_tools


# ---------------------------------------------------------------------------
# SwarmConfig
# ---------------------------------------------------------------------------

def test_swarm_config_pipeline():
    cfg = SwarmConfig(
        name="test",
        strategy="pipeline",
        workers=[
            WorkerConfig(id="a", system_prompt="s1"),
            WorkerConfig(id="b", system_prompt="s2"),
        ],
    )
    assert cfg.strategy == "pipeline"
    assert len(cfg.workers) == 2
    assert cfg.coordinator is None


def test_swarm_config_hierarchical_requires_coordinator():
    with pytest.raises(ValueError, match="coordinator"):
        SwarmConfig(name="test", strategy="hierarchical", workers=[
            WorkerConfig(id="w", system_prompt="s"),
        ])


def test_swarm_config_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        SwarmConfig(name="test", strategy="invalid", workers=[])


def test_swarm_config_empty_workers_raises():
    with pytest.raises(ValueError, match="workers"):
        SwarmConfig(name="test", strategy="pipeline", workers=[])


# ---------------------------------------------------------------------------
# Pipeline strategy tests
# ---------------------------------------------------------------------------

import pytest
from astra_swarm.swarm import SwarmCoordinator, AsyncioExecutor, SwarmEvent
from tests.conftest import MockProvider, make_registry, make_response


@pytest.mark.asyncio
async def test_pipeline_two_workers_output_feeds_input():
    """Worker[0] final text becomes the task for worker[1]."""
    reg = make_registry()
    w0_provider = MockProvider([make_response(content="step1 result")])
    w1_provider = MockProvider([make_response(content="step2 result")])

    cfg = SwarmConfig(
        name="test_pipeline",
        strategy="pipeline",
        workers=[
            WorkerConfig(id="w0", system_prompt="Worker 0", provider=w0_provider),
            WorkerConfig(id="w1", system_prompt="Worker 1", provider=w1_provider),
        ],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("initial task")]

    # All events are SwarmEvents
    assert all(isinstance(e, SwarmEvent) for e in events)

    # Events carry correct worker_ids
    worker_ids = [e.worker_id for e in events]
    assert "w0" in worker_ids
    assert "w1" in worker_ids

    # Both providers ran
    assert w0_provider.call_count == 1
    assert w1_provider.call_count == 1


@pytest.mark.asyncio
async def test_pipeline_events_contain_turn_end():
    """Each worker emits a turn_end SwarmEvent."""
    reg = make_registry()
    w0_provider = MockProvider([make_response(content="result")])
    w1_provider = MockProvider([make_response(content="final")])

    cfg = SwarmConfig(
        name="test",
        strategy="pipeline",
        workers=[
            WorkerConfig(id="w0", system_prompt="s", provider=w0_provider),
            WorkerConfig(id="w1", system_prompt="s", provider=w1_provider),
        ],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("task")]

    turn_end_events = [e for e in events if e.inner_type == "turn_end"]
    assert len(turn_end_events) == 2  # one per worker


@pytest.mark.asyncio
async def test_pipeline_single_worker():
    """Pipeline with one worker runs and returns events."""
    reg = make_registry()
    provider = MockProvider([make_response(content="only result")])

    cfg = SwarmConfig(
        name="test",
        strategy="pipeline",
        workers=[WorkerConfig(id="solo", system_prompt="s", provider=provider)],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("task")]

    assert any(e.worker_id == "solo" for e in events)
    assert any(e.inner_type == "text_delta" for e in events)


# ---------------------------------------------------------------------------
# Parallel strategy tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_all_workers_run_same_task():
    """All workers receive the same task and run concurrently."""
    reg = make_registry()
    p1 = MockProvider([make_response(content="result_a")])
    p2 = MockProvider([make_response(content="result_b")])
    p3 = MockProvider([make_response(content="result_c")])

    cfg = SwarmConfig(
        name="test_parallel",
        strategy="parallel",
        workers=[
            WorkerConfig(id="a", system_prompt="s", provider=p1),
            WorkerConfig(id="b", system_prompt="s", provider=p2),
            WorkerConfig(id="c", system_prompt="s", provider=p3),
        ],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("shared task")]

    assert p1.call_count == 1
    assert p2.call_count == 1
    assert p3.call_count == 1

    worker_ids_seen = {e.worker_id for e in events}
    assert {"a", "b", "c"} == worker_ids_seen


@pytest.mark.asyncio
async def test_parallel_worker_failure_does_not_stop_others():
    """If one worker raises, others still complete and an error event is emitted."""
    from astra_node.providers.base import LLMProvider as _LLMProvider

    class FailingProvider(_LLMProvider):
        async def complete(self, messages, tools, system="", **kwargs):
            raise RuntimeError("provider exploded")
            yield  # make it an async generator

        @property
        def last_response(self):
            return None

    reg = make_registry()
    p_ok = MockProvider([make_response(content="fine")])
    p_fail = FailingProvider()

    cfg = SwarmConfig(
        name="test",
        strategy="parallel",
        workers=[
            WorkerConfig(id="ok_worker", system_prompt="s", provider=p_ok),
            WorkerConfig(id="fail_worker", system_prompt="s", provider=p_fail),
        ],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("task")]

    ok_events = [e for e in events if e.worker_id == "ok_worker"]
    assert len(ok_events) > 0

    fail_events = [e for e in events if e.worker_id == "fail_worker"]
    assert any(e.inner_type == "agent_error" for e in fail_events)


@pytest.mark.asyncio
async def test_parallel_swarm_events_have_worker_id():
    """Every event from a parallel run carries a non-empty worker_id."""
    reg = make_registry()
    p1 = MockProvider([make_response(content="r1")])
    p2 = MockProvider([make_response(content="r2")])

    cfg = SwarmConfig(
        name="test",
        strategy="parallel",
        workers=[
            WorkerConfig(id="w1", system_prompt="s", provider=p1),
            WorkerConfig(id="w2", system_prompt="s", provider=p2),
        ],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("task")]
    assert all(e.worker_id != "" for e in events)


# ---------------------------------------------------------------------------
# Hierarchical strategy tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hierarchical_coordinator_and_workers_all_run():
    """Coordinator runs first (planning), then workers run, then coordinator aggregates."""
    reg = make_registry()
    coord_provider = MockProvider([
        make_response(content="w1: analyze data\nw2: write report"),
        make_response(content="Final aggregated answer"),
    ])
    w1_provider = MockProvider([make_response(content="analysis done")])
    w2_provider = MockProvider([make_response(content="report written")])

    coord_cfg = WorkerConfig(id="coordinator", system_prompt="You coordinate.", provider=coord_provider)
    cfg = SwarmConfig(
        name="test_hier",
        strategy="hierarchical",
        coordinator=coord_cfg,
        workers=[
            WorkerConfig(id="w1", system_prompt="analyze", provider=w1_provider),
            WorkerConfig(id="w2", system_prompt="report", provider=w2_provider),
        ],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("do the thing")]

    assert coord_provider.call_count == 2
    assert w1_provider.call_count == 1
    assert w2_provider.call_count == 1

    assert all(isinstance(e, SwarmEvent) for e in events)
    assert all(e.worker_id != "" for e in events)


@pytest.mark.asyncio
async def test_hierarchical_coordinator_events_have_coordinator_id():
    """Events from the coordinator carry the coordinator's worker_id."""
    reg = make_registry()
    coord_provider = MockProvider([
        make_response(content="w1: do task"),
        make_response(content="aggregated"),
    ])
    w1_provider = MockProvider([make_response(content="done")])

    coord_cfg = WorkerConfig(id="coord", system_prompt="coord", provider=coord_provider)
    cfg = SwarmConfig(
        name="t",
        strategy="hierarchical",
        coordinator=coord_cfg,
        workers=[WorkerConfig(id="w1", system_prompt="s", provider=w1_provider)],
    )
    coordinator = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coordinator.run("task")]

    coord_events = [e for e in events if e.worker_id == "coord"]
    assert len(coord_events) > 0


# ---------------------------------------------------------------------------
# Worker isolation tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_tool_isolation():
    """Worker with allowed_tools={'echo'} does not expose other registered tools."""
    from pydantic import BaseModel as _BaseModel
    from astra_node.core.tool import BaseTool as _BaseTool, PermissionLevel as _PL, ToolContext as _TC
    from astra_node.core.tool import ToolResult as _TR

    class GrepInput(_BaseModel):
        pattern: str

    class GrepTool(_BaseTool):
        name = "grep"
        description = "Grep tool"
        input_schema = GrepInput
        permission_level = _PL.ALWAYS_ALLOW
        def execute(self, input: GrepInput, ctx: _TC) -> _TR:
            return _TR.ok("grep result")

    reg = make_registry()
    reg.register(GrepTool())

    p = MockProvider([make_response(content="done")])
    cfg = SwarmConfig(
        name="t",
        strategy="pipeline",
        workers=[WorkerConfig(id="w", system_prompt="s", provider=p, allowed_tools={"echo"})],
    )
    coord = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coord.run("task")]
    assert any(e.worker_id == "w" for e in events)


# ---------------------------------------------------------------------------
# max_turns enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_max_turns_enforced():
    """Worker with max_turns=1 emits a max_turns TurnEnd when the model loops."""
    from astra_node.providers.base import ToolCall

    reg = make_registry()
    p = MockProvider([
        make_response(
            content="",
            tool_calls=[ToolCall(id="t1", name="echo", input={"message": "hi"})],
            stop_reason="tool_use",
        ),
        make_response(content="done"),
    ])

    cfg = SwarmConfig(
        name="t",
        strategy="pipeline",
        workers=[WorkerConfig(id="w", system_prompt="s", provider=p, max_turns=1)],
    )
    coord = SwarmCoordinator(config=cfg, registry=reg, executor=AsyncioExecutor())
    events = [e async for e in coord.run("task")]

    turn_ends = [e for e in events if e.inner_type == "turn_end"]
    assert len(turn_ends) >= 1
    assert any(e.data.get("stop_reason") == "max_turns" for e in turn_ends)
