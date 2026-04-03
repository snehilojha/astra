# Astra

A provider-agnostic framework for building and running LLM agents and swarms from the terminal.

```bash
pip install astra-cli
astra run "refactor the auth module to use JWT"
```

---

## Packages

| Package | Description |
|---------|-------------|
| `astra-node` | Core agent engine — providers, tools, memory, history |
| `astra-swarm` | Swarm orchestration — pipeline, parallel, hierarchical |
| `astra-cli` | Terminal interface — `astra run`, `astra swarm`, `astra memory`, `astra config` |

Installing `astra-cli` pulls in all three.

---

## Quickstart

```bash
pip install astra-cli

# Store your API key
astra config set ANTHROPIC_API_KEY sk-ant-...

# Run a single agent
astra run "explain this codebase"

# Run from a file
astra run --file task.txt

# Use a local Ollama model (no API key needed)
astra run --provider ollama --model qwen2.5-coder "explain this codebase"
```

---

## Swarms

Run multi-agent pipelines defined in YAML:

```bash
astra swarm list
astra swarm run code_review --task "review the auth module"
astra swarm run research --task "summarize recent papers on RAG"
```

Built-in configs: `code_review`, `research`, `ml_pipeline`, `refactor`.

---

## Memory

Astra persists memories across sessions as markdown files under `~/.astra/memory/`.

```bash
astra memory list
astra memory search "JWT"
astra memory clear
```

---

## Providers

| Provider | Flag | Notes |
|----------|------|-------|
| Anthropic (default) | `--provider anthropic` | Requires `ANTHROPIC_API_KEY` |
| OpenAI | `--provider openai` | Requires `OPENAI_API_KEY` |
| Ollama | `--provider ollama --model <name>` | Local, no API key |

---

## Using astra-node directly

```python
import asyncio
from astra_node.core.query_engine import QueryEngine
from astra_node.providers.anthropic import AnthropicProvider
from astra_node.core.registry import ToolRegistry
from astra_node.permissions.manager import PermissionManager
from astra_node.permissions.types import PermissionLevel

async def main():
    engine = QueryEngine(
        provider=AnthropicProvider(),
        registry=ToolRegistry(),
        permission_manager=PermissionManager(default_level=PermissionLevel.ALWAYS_ALLOW),
    )
    async for event in engine.run("list files in the current directory"):
        print(event)

asyncio.run(main())
```

---

## License

MIT
