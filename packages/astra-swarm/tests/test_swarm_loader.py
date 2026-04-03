"""Tests for swarm_loader — YAML → SwarmConfig/SwarmCoordinator."""
import textwrap
import os
import pytest
from pathlib import Path

from astra_swarm.swarm_loader import load_swarm_from_yaml, LoadError
from astra_swarm.swarm import SwarmConfig, SwarmCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, content: str) -> Path:
    """Write YAML content to a temp file and return its path."""
    p = tmp_path / "swarm.yaml"
    p.write_text(textwrap.dedent(content))
    return p


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch):
    """Ensure provider constructors don't fail due to missing API keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Valid YAML loads
# ---------------------------------------------------------------------------

def test_pipeline_yaml_loads(tmp_path):
    path = write_yaml(tmp_path, """
        name: ml_pipeline
        strategy: pipeline
        workers:
          - id: data_validator
            model: claude-haiku-4-5-20251001
            provider: anthropic
            system_prompt: "Validate the dataset."
          - id: feature_engineer
            model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "Engineer features."
    """)
    cfg, coordinator = load_swarm_from_yaml(path)
    assert cfg.name == "ml_pipeline"
    assert cfg.strategy == "pipeline"
    assert len(cfg.workers) == 2
    assert cfg.workers[0].id == "data_validator"
    assert cfg.workers[1].id == "feature_engineer"
    assert isinstance(coordinator, SwarmCoordinator)


def test_hierarchical_yaml_loads(tmp_path):
    path = write_yaml(tmp_path, """
        name: code_review
        strategy: hierarchical
        coordinator:
          id: reviewer_lead
          model: claude-opus-4-6
          provider: anthropic
          system_prompt: "Break down code review tasks."
          max_turns: 10
        workers:
          - id: security_reviewer
            model: claude-sonnet-4-6
            provider: anthropic
            allowed_tools: [file_read, grep]
            system_prompt: "Review for security."
          - id: style_reviewer
            model: claude-haiku-4-5-20251001
            provider: anthropic
            allowed_tools: [file_read]
            system_prompt: "Review for style."
    """)
    cfg, coordinator = load_swarm_from_yaml(path)
    assert cfg.strategy == "hierarchical"
    assert cfg.coordinator is not None
    assert cfg.coordinator.id == "reviewer_lead"
    assert len(cfg.workers) == 2
    assert cfg.workers[0].allowed_tools == {"file_read", "grep"}


def test_allowed_tools_loaded_as_set(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
            allowed_tools: [bash, file_read, grep]
            system_prompt: "s"
    """)
    cfg, _ = load_swarm_from_yaml(path)
    assert cfg.workers[0].allowed_tools == {"bash", "file_read", "grep"}


def test_max_turns_default_is_10(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "s"
    """)
    cfg, _ = load_swarm_from_yaml(path)
    assert cfg.workers[0].max_turns == 10


def test_max_turns_custom(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "s"
            max_turns: 5
    """)
    cfg, _ = load_swarm_from_yaml(path)
    assert cfg.workers[0].max_turns == 5


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_swarm_from_yaml(tmp_path / "nonexistent.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed")
    with pytest.raises(LoadError, match="YAML"):
        load_swarm_from_yaml(path)


def test_missing_name_field_raises(tmp_path):
    path = write_yaml(tmp_path, """
        strategy: pipeline
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "s"
    """)
    with pytest.raises(LoadError, match="name"):
        load_swarm_from_yaml(path)


def test_missing_strategy_raises(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "s"
    """)
    with pytest.raises(LoadError, match="strategy"):
        load_swarm_from_yaml(path)


def test_unknown_strategy_raises(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: unknown
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "s"
    """)
    with pytest.raises((LoadError, ValueError)):
        load_swarm_from_yaml(path)


def test_missing_workers_raises(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
    """)
    with pytest.raises((LoadError, ValueError)):
        load_swarm_from_yaml(path)


def test_unknown_provider_raises(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
        workers:
          - id: w
            model: gpt-4o
            provider: unknown_provider
            system_prompt: "s"
    """)
    with pytest.raises(LoadError, match="provider"):
        load_swarm_from_yaml(path)


def test_missing_worker_id_raises(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
        workers:
          - model: claude-sonnet-4-6
            provider: anthropic
            system_prompt: "s"
    """)
    with pytest.raises(LoadError, match="id"):
        load_swarm_from_yaml(path)


def test_missing_system_prompt_raises(tmp_path):
    path = write_yaml(tmp_path, """
        name: t
        strategy: pipeline
        workers:
          - id: w
            model: claude-sonnet-4-6
            provider: anthropic
    """)
    with pytest.raises(LoadError, match="system_prompt"):
        load_swarm_from_yaml(path)
