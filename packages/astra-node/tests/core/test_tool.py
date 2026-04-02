"""Tests for BaseTool ABC, ToolResult, and ToolContext."""

import pytest
from pathlib import Path
from pydantic import BaseModel, ValidationError

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


# ---------------------------------------------------------------------------
# Concrete tool fixture used across multiple tests
# ---------------------------------------------------------------------------

class EchoInput(BaseModel):
    """Input schema for EchoTool."""
    message: str


class EchoTool(BaseTool):
    """Concrete tool that echoes its input. Used for testing BaseTool."""

    name = "echo"
    description = "Echoes the message back."
    input_schema = EchoInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(input.message)


class FailingInput(BaseModel):
    value: int


class FailingTool(BaseTool):
    """Tool whose execute() always returns an error result."""

    name = "failing"
    description = "Always fails."
    input_schema = FailingInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: FailingInput, ctx: ToolContext) -> ToolResult:
        return ToolResult.err(f"Tool failed for value {input.value}")


# ---------------------------------------------------------------------------
# PermissionLevel tests
# ---------------------------------------------------------------------------

class TestPermissionLevel:
    def test_has_three_values(self):
        assert len(PermissionLevel) == 3

    def test_values(self):
        assert PermissionLevel.ALWAYS_ALLOW.value == "always_allow"
        assert PermissionLevel.ASK_USER.value == "ask_user"
        assert PermissionLevel.DENY.value == "deny"


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_ok_constructor(self):
        result = ToolResult.ok("success output")
        assert result.output == "success output"
        assert result.is_error is False
        assert result.error is None

    def test_err_constructor(self):
        result = ToolResult.err("something broke")
        assert result.output == "something broke"
        assert result.is_error is True
        assert result.error == "something broke"

    def test_direct_construction_success(self):
        result = ToolResult(output="ok", is_error=False)
        assert result.is_error is False

    def test_direct_construction_error(self):
        result = ToolResult(output="bad", is_error=True, error="bad details")
        assert result.is_error is True
        assert result.error == "bad details"


# ---------------------------------------------------------------------------
# ToolContext tests
# ---------------------------------------------------------------------------

class TestToolContext:
    def test_default_cwd_is_path(self):
        ctx = ToolContext()
        assert isinstance(ctx.cwd, Path)

    def test_explicit_cwd(self):
        ctx = ToolContext(cwd=Path("/tmp"))
        assert ctx.cwd == Path("/tmp")

    def test_env_is_dict(self):
        ctx = ToolContext(env={"FOO": "bar"})
        assert ctx.env["FOO"] == "bar"

    def test_env_default_empty(self):
        ctx = ToolContext()
        assert ctx.env == {}

    def test_metadata_is_extensible(self):
        ctx = ToolContext(metadata={"swarm_id": "worker_1", "turn": 3})
        assert ctx.metadata["swarm_id"] == "worker_1"
        assert ctx.metadata["turn"] == 3

    def test_metadata_default_empty(self):
        ctx = ToolContext()
        assert ctx.metadata == {}


# ---------------------------------------------------------------------------
# BaseTool ABC tests
# ---------------------------------------------------------------------------

class TestBaseToolABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_without_execute_raises(self):
        class IncompleteT(BaseTool):
            name = "x"
            description = "y"
            permission_level = PermissionLevel.ALWAYS_ALLOW

            class _Schema(BaseModel):
                pass

            input_schema = _Schema

        with pytest.raises(TypeError):
            IncompleteT()  # type: ignore[abstract]

    def test_concrete_can_instantiate(self):
        tool = EchoTool()
        assert tool is not None

    def test_name_and_description_non_empty(self):
        tool = EchoTool()
        assert tool.name == "echo"
        assert len(tool.description) > 0

    def test_input_schema_is_pydantic_model(self):
        tool = EchoTool()
        assert issubclass(tool.input_schema, BaseModel)

    def test_permission_level_type(self):
        tool = EchoTool()
        assert isinstance(tool.permission_level, PermissionLevel)

    def test_execute_success(self):
        tool = EchoTool()
        ctx = ToolContext()
        inp = EchoInput(message="hello")
        result = tool.execute(inp, ctx)
        assert result.output == "hello"
        assert result.is_error is False

    def test_execute_error_result(self):
        tool = FailingTool()
        ctx = ToolContext()
        inp = FailingInput(value=42)
        result = tool.execute(inp, ctx)
        assert result.is_error is True
        assert "42" in result.output

    def test_invalid_input_rejected_by_pydantic(self):
        """Pydantic raises ValidationError for wrong input types."""
        with pytest.raises(ValidationError):
            EchoInput(message=123)  # type: ignore[arg-type]

    def test_permission_level_always_allow(self):
        tool = EchoTool()
        assert tool.permission_level == PermissionLevel.ALWAYS_ALLOW

    def test_permission_level_ask_user(self):
        tool = FailingTool()
        assert tool.permission_level == PermissionLevel.ASK_USER
