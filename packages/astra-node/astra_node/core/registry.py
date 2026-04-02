"""Tool registry — stores and looks up BaseTool instances by name.

The registry is the single source of truth for which tools are available
in a session or worker. The query engine uses it to look up tools by name
after the LLM requests them. The swarm coordinator uses filter() to give
each worker a restricted subset.
"""

from astra_node.core.tool import BaseTool


class ToolRegistry:
    """Central store for BaseTool instances.

    Tools are keyed by their name attribute. Duplicate registrations raise
    ValueError so that misconfiguration is caught early rather than silently
    shadowing an existing tool.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry.

        Args:
            tool: A BaseTool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Use a unique name or unregister it first."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name.

        Args:
            name: The tool's name attribute.

        Returns:
            The registered BaseTool instance.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool named '{name}' is registered.")
        return self._tools[name]

    def list_all(self) -> list[BaseTool]:
        """Return all registered tools in insertion order."""
        return list(self._tools.values())

    def filter(self, allowed: set[str]) -> "ToolRegistry":
        """Return a new registry containing only the named tools.

        Used by the swarm coordinator to create per-worker registries that
        expose only the tools each worker is permitted to use.

        Args:
            allowed: Set of tool names to include. Names that are not
                     registered are silently ignored (caller is responsible
                     for validating names at config load time).

        Returns:
            A new ToolRegistry with only the matching tools.
        """
        filtered = ToolRegistry()
        for name in allowed:
            if name in self._tools:
                filtered._tools[name] = self._tools[name]
        return filtered

    def to_api_format(self, provider: str) -> list[dict]:
        """Serialise all tools to the format expected by the given provider.

        Args:
            provider: Either "anthropic" or "openai".

        Returns:
            List of tool schema dicts ready to pass to the LLM API.

        Raises:
            ValueError: If provider is not "anthropic" or "openai".
        """
        if provider == "anthropic":
            return [self._to_anthropic_schema(tool) for tool in self._tools.values()]
        if provider == "openai":
            return [self._to_openai_schema(tool) for tool in self._tools.values()]
        raise ValueError(
            f"Unknown provider '{provider}'. Expected 'anthropic' or 'openai'."
        )

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Private schema builders
    # ------------------------------------------------------------------

    @staticmethod
    def _to_anthropic_schema(tool: BaseTool) -> dict:
        """Build an Anthropic-format tool schema dict from a BaseTool."""
        schema = tool.input_schema.model_json_schema()
        # Remove the Pydantic title from the top-level schema — Anthropic
        # uses the tool name instead.
        schema.pop("title", None)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": schema,
        }

    @staticmethod
    def _to_openai_schema(tool: BaseTool) -> dict:
        """Build an OpenAI-format tool schema dict from a BaseTool."""
        schema = tool.input_schema.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }
