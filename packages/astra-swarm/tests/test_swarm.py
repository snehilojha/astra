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
