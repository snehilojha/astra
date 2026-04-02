"""Tests for ToolRegistry."""

import pytest
from pydantic import BaseModel

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult
from astra_node.core.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixture tools
# ---------------------------------------------------------------------------

class _AInput(BaseModel):
    x: str


class _BInput(BaseModel):
    y: int


class ToolA(BaseTool):
    name = "tool_a"
    description = "Tool A description."
    input_schema = _AInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: _AInput, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(input.x)


class ToolB(BaseTool):
    name = "tool_b"
    description = "Tool B description."
    input_schema = _BInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: _BInput, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(str(input.y))


class ToolC(BaseTool):
    name = "tool_c"
    description = "Tool C description."
    input_schema = _AInput
    permission_level = PermissionLevel.DENY

    def execute(self, input: _AInput, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("c")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolRegistryRegisterGet:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        tool = reg.get("tool_a")
        assert tool.name == "tool_a"

    def test_get_unknown_raises_key_error(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="tool_a"):
            reg.get("tool_a")

    def test_duplicate_registration_raises_value_error(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ToolA())

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(ToolA())
        assert len(reg) == 1
        reg.register(ToolB())
        assert len(reg) == 2


class TestToolRegistryListAll:
    def test_list_all_returns_all(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        reg.register(ToolB())
        tools = reg.list_all()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    def test_list_all_empty(self):
        reg = ToolRegistry()
        assert reg.list_all() == []

    def test_list_all_insertion_order(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        reg.register(ToolB())
        reg.register(ToolC())
        names = [t.name for t in reg.list_all()]
        assert names == ["tool_a", "tool_b", "tool_c"]


class TestToolRegistryFilter:
    def test_filter_returns_subset(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        reg.register(ToolB())
        reg.register(ToolC())

        filtered = reg.filter({"tool_a", "tool_c"})
        names = {t.name for t in filtered.list_all()}
        assert names == {"tool_a", "tool_c"}
        assert "tool_b" not in names

    def test_filter_empty_set_returns_empty_registry(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        reg.register(ToolB())
        filtered = reg.filter(set())
        assert len(filtered) == 0

    def test_filter_unknown_names_silently_ignored(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        filtered = reg.filter({"tool_a", "nonexistent"})
        assert len(filtered) == 1
        assert filtered.get("tool_a").name == "tool_a"

    def test_filter_returns_new_registry(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        filtered = reg.filter({"tool_a"})
        assert filtered is not reg


class TestToolRegistryToAPIFormat:
    def test_to_api_format_anthropic(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        schemas = reg.to_api_format("anthropic")
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "tool_a"
        assert schema["description"] == "Tool A description."
        assert "input_schema" in schema
        assert "properties" in schema["input_schema"]

    def test_to_api_format_openai(self):
        reg = ToolRegistry()
        reg.register(ToolB())
        schemas = reg.to_api_format("openai")
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "tool_b"
        assert schema["function"]["description"] == "Tool B description."
        assert "parameters" in schema["function"]

    def test_to_api_format_unknown_provider_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown provider"):
            reg.to_api_format("bedrock")

    def test_to_api_format_empty_registry(self):
        reg = ToolRegistry()
        assert reg.to_api_format("anthropic") == []
        assert reg.to_api_format("openai") == []

    def test_anthropic_schema_no_title_at_top_level(self):
        """Pydantic adds a 'title' field to JSON schemas — Anthropic doesn't want it."""
        reg = ToolRegistry()
        reg.register(ToolA())
        schema = reg.to_api_format("anthropic")[0]
        assert "title" not in schema["input_schema"]

    def test_openai_schema_no_title_at_top_level(self):
        reg = ToolRegistry()
        reg.register(ToolA())
        schema = reg.to_api_format("openai")[0]
        assert "title" not in schema["function"]["parameters"]
