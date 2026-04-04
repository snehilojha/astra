"""YAML config loader for SwarmCoordinator.

load_swarm_from_yaml(path) reads a YAML file, validates required fields,
constructs LLMProvider instances for each worker, and returns a
(SwarmConfig, SwarmCoordinator) tuple ready to run.

Supported providers (YAML `provider` field):
  anthropic  — AnthropicProvider
  openai     — OpenAIProvider

Supported models (YAML `model` field):
  Any model string supported by the chosen provider SDK.
  Recommended: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from astra_node.core.registry import ToolRegistry
from astra_swarm.swarm import (
    AsyncioExecutor,
    SwarmCoordinator,
    SwarmConfig,
    WorkerConfig,
)


class LoadError(ValueError):
    """Raised when a swarm YAML file fails validation."""


def _build_provider(worker_data: dict[str, Any], worker_id: str):
    """Construct an LLMProvider from a worker's YAML dict."""
    provider_name = worker_data.get("provider")
    if not provider_name:
        raise LoadError(f"Worker '{worker_id}' is missing required field 'provider'.")

    model = worker_data.get("model")

    if provider_name == "anthropic":
        from astra_node.providers.anthropic import AnthropicProvider
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        return AnthropicProvider(**kwargs)

    if provider_name == "openai":
        from astra_node.providers.openai import OpenAIProvider
        kwargs = {}
        if model:
            kwargs["model"] = model
        return OpenAIProvider(**kwargs)

    if provider_name == "openrouter":
        import os
        from astra_node.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model=model or "anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    if provider_name == "ollama":
        from astra_node.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model=model or "llama3.2",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    raise LoadError(
        f"Worker '{worker_id}' has unknown provider '{provider_name}'. "
        "Supported values: 'anthropic', 'openai', 'openrouter', 'ollama'."
    )


def _parse_worker(data: dict[str, Any], label: str = "worker") -> WorkerConfig:
    """Parse one worker dict into a WorkerConfig."""
    worker_id = data.get("id")
    if not worker_id:
        raise LoadError(f"A {label} entry is missing required field 'id'.")

    system_prompt = data.get("system_prompt")
    if not system_prompt:
        raise LoadError(
            f"{label.capitalize()} '{worker_id}' is missing required field 'system_prompt'."
        )

    allowed_tools_raw = data.get("allowed_tools", [])
    if not isinstance(allowed_tools_raw, list):
        raise LoadError(
            f"{label.capitalize()} '{worker_id}': allowed_tools must be a list."
        )
    allowed_tools = set(str(t) for t in allowed_tools_raw)

    max_turns = data.get("max_turns", 10)
    if not isinstance(max_turns, int) or max_turns < 1:
        raise LoadError(
            f"{label.capitalize()} '{worker_id}': max_turns must be a positive integer."
        )

    provider = _build_provider(data, worker_id)

    return WorkerConfig(
        id=worker_id,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        provider=provider,
        max_turns=max_turns,
    )


def load_swarm_from_yaml(
    path: Path | str,
    registry: ToolRegistry | None = None,
) -> tuple[SwarmConfig, SwarmCoordinator]:
    """Load a SwarmCoordinator from a YAML config file.

    Args:
        path:     Path to the YAML file.
        registry: Optional ToolRegistry to pass to SwarmCoordinator.
                  Defaults to an empty registry.

    Returns:
        (SwarmConfig, SwarmCoordinator) — both ready to use.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        LoadError:         If required fields are missing or values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Swarm config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LoadError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise LoadError(
            f"Invalid swarm config in {path}: expected a YAML mapping at top level."
        )

    name = raw.get("name")
    if not name:
        raise LoadError(f"Swarm config in {path} is missing required field 'name'.")

    strategy = raw.get("strategy")
    if not strategy:
        raise LoadError(f"Swarm config '{name}' is missing required field 'strategy'.")

    workers_raw = raw.get("workers")
    if not workers_raw:
        raise LoadError(f"Swarm config '{name}' is missing required field 'workers'.")
    if not isinstance(workers_raw, list):
        raise LoadError(f"Swarm config '{name}': 'workers' must be a list.")

    workers = [_parse_worker(w, label="worker") for w in workers_raw]

    coordinator: WorkerConfig | None = None
    coord_raw = raw.get("coordinator")
    if coord_raw:
        coordinator = _parse_worker(coord_raw, label="coordinator")

    try:
        cfg = SwarmConfig(
            name=name,
            strategy=strategy,
            workers=workers,
            coordinator=coordinator,
        )
    except ValueError as exc:
        raise LoadError(str(exc)) from exc

    swarm = SwarmCoordinator(
        config=cfg,
        registry=registry or ToolRegistry(),
        executor=AsyncioExecutor(),
    )
    return cfg, swarm
