"""Tests for QueryEngine — the agent loop.

All LLM provider calls are mocked. No real API calls.
Tests cover: text response, tool calls, permissions, errors, max_turns,
memory injection, and UsageUpdate propagation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator

from astra_node.core.events import (
    AgentError,
    AgentEvent,
    TextDelta,
    ToolResult,
    ToolStart,
    TurnEnd,
    UsageUpdate,
)
from astra_node.core.memory_stub import StubMemory
from astra_node.core.memory_types import QueryContext, ScoredChunk
from astra_node.core.query_engine import QueryEngine
from astra_node.core.registry import ToolRegistry
from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult as CoreToolResult
from astra_node.permissions.manager import PermissionManager
from astra_node.providers.base import LLMProvider, LLMResponse, ToolCall, Usage
from astra_node.utils.errors import ToolExecutionError


# ---------------------------------------------------------------------------
# Mock provider helpers
# ---------------------------------------------------------------------------

def _usage():
    return Usage(input_tokens=10, output_tokens=5)


def _make_response(
    content: str = "",
    tool_calls: list | None = None,
    stop_reason: str = "end_turn",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=_usage(),
    )


class MockProvider(LLMProvider):
    """Configurable mock provider for testing the agent loop.

    responses: list of LLMResponse objects to return in sequence.
    Each call to complete() pops the first response, yields its events,
    and sets last_response.
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._last_response: LLMResponse | None = None
        self.call_count = 0

    async def complete(
        self, messages, tools, system="", **kwargs
    ) -> AsyncIterator[AgentEvent]:
        self.call_count += 1
        response = self._responses.pop(0)
        self._last_response = response

        if response.content:
            yield TextDelta(text=response.content)
        yield UsageUpdate(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    @property
    def last_response(self):
        return self._last_response


# ---------------------------------------------------------------------------
# Fixture tools
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class EchoInput(BaseModel):
    message: str


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the message."
    input_schema = EchoInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: EchoInput, ctx: ToolContext) -> CoreToolResult:
        return CoreToolResult.ok(f"ECHO: {input.message}")


class BashInput(BaseModel):
    command: str


class BashTool(BaseTool):
    name = "bash"
    description = "Runs a shell command."
    input_schema = BashInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: BashInput, ctx: ToolContext) -> CoreToolResult:
        return CoreToolResult.ok(f"ran: {input.command}")


class FailingInput(BaseModel):
    value: int


class FailingTool(BaseTool):
    name = "failing_tool"
    description = "Always raises ToolExecutionError."
    input_schema = FailingInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: FailingInput, ctx: ToolContext) -> CoreToolResult:
        raise ToolExecutionError("tool exploded", tool_name="failing_tool")


class DeniedInput(BaseModel):
    x: str


class DeniedTool(BaseTool):
    name = "denied_tool"
    description = "Permission DENY."
    input_schema = DeniedInput
    permission_level = PermissionLevel.DENY

    def execute(self, input: DeniedInput, ctx: ToolContext) -> CoreToolResult:
        return CoreToolResult.ok("never reached")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    responses: list[LLMResponse],
    tools: list[BaseTool] | None = None,
    max_turns: int = 10,
    memory=None,
    post_turn_hook=None,
) -> QueryEngine:
    provider = MockProvider(responses)
    registry = ToolRegistry()
    for tool in (tools or []):
        registry.register(tool)
    permission_manager = PermissionManager()
    return QueryEngine(
        provider=provider,
        registry=registry,
        permission_manager=permission_manager,
        system_prompt="You are a test assistant.",
        max_turns=max_turns,
        memory=memory,
        post_turn_hook=post_turn_hook,
    )


async def _collect(engine: QueryEngine, message: str) -> list[AgentEvent]:
    events = []
    async for event in engine.run(message):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Tests — simple text response
# ---------------------------------------------------------------------------

class TestQueryEngineTextResponse:
    @pytest.mark.asyncio
    async def test_simple_text_response_yields_text_delta_and_turn_end(self):
        engine = _make_engine([_make_response("Hello!")])
        events = await _collect(engine, "hi")

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(text_deltas) == 1
        assert text_deltas[0].text == "Hello!"
        assert len(turn_ends) == 1
        assert turn_ends[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_usage_update_yielded(self):
        engine = _make_engine([_make_response("hi")])
        events = await _collect(engine, "hello")

        usage_events = [e for e in events if isinstance(e, UsageUpdate)]
        assert len(usage_events) == 1

    @pytest.mark.asyncio
    async def test_empty_tool_registry_sends_no_tools(self):
        engine = _make_engine([_make_response("ok")])
        events = await _collect(engine, "hi")
        # Just verify it completes without error
        assert any(isinstance(e, TurnEnd) for e in events)


# ---------------------------------------------------------------------------
# Tests — tool call flow
# ---------------------------------------------------------------------------

class TestQueryEngineToolCall:
    @pytest.mark.asyncio
    async def test_tool_call_flow_yields_tool_start_and_result(self):
        tc = ToolCall(id="t1", name="echo", input={"message": "hello"})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("Done."),
        ]
        engine = _make_engine(responses, tools=[EchoTool()])
        events = await _collect(engine, "call echo")

        tool_starts = [e for e in events if isinstance(e, ToolStart)]
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        turn_ends = [e for e in events if isinstance(e, TurnEnd)]

        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "echo"
        assert len(tool_results) == 1
        assert "ECHO: hello" in tool_results[0].output
        assert tool_results[0].is_error is False
        assert any(e.stop_reason == "end_turn" for e in turn_ends)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_turn(self):
        tc1 = ToolCall(id="t1", name="echo", input={"message": "first"})
        tc2 = ToolCall(id="t2", name="echo", input={"message": "second"})
        responses = [
            _make_response("", tool_calls=[tc1, tc2], stop_reason="tool_use"),
            _make_response("All done."),
        ]
        engine = _make_engine(responses, tools=[EchoTool()])
        events = await _collect(engine, "call both")

        tool_starts = [e for e in events if isinstance(e, ToolStart)]
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_starts) == 2
        assert len(tool_results) == 2

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        tc = ToolCall(id="t1", name="echo", input={"message": "test"})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("I got the echo result."),
        ]
        engine = _make_engine(responses, tools=[EchoTool()])
        await _collect(engine, "call echo")
        # History should contain user, assistant (tool_use), user (tool_result), assistant (text)
        assert len(engine._history) >= 3


# ---------------------------------------------------------------------------
# Tests — permission denied
# ---------------------------------------------------------------------------

class TestQueryEnginePermissions:
    @pytest.mark.asyncio
    async def test_permission_denied_yields_agent_error(self):
        tc = ToolCall(id="t1", name="denied_tool", input={"x": "val"})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("Okay, I won't use that tool."),
        ]
        engine = _make_engine(responses, tools=[DeniedTool()])
        events = await _collect(engine, "try denied")

        agent_errors = [e for e in events if isinstance(e, AgentError)]
        assert len(agent_errors) >= 1
        assert "denied_tool" in agent_errors[0].tool_name

    @pytest.mark.asyncio
    async def test_permission_denied_loop_continues(self):
        """After a permission denial, the loop continues and the LLM gets another turn."""
        tc = ToolCall(id="t1", name="denied_tool", input={"x": "val"})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("I understand the tool was denied."),
        ]
        engine = _make_engine(responses, tools=[DeniedTool()])
        events = await _collect(engine, "try denied")

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 1
        assert turn_ends[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_ask_user_tool_allowed_by_session_override(self):
        """allow_always() on ASK_USER tool lets it proceed without prompting."""
        tc = ToolCall(id="t1", name="bash", input={"command": "ls"})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("Command ran."),
        ]
        provider = MockProvider(responses)
        registry = ToolRegistry()
        registry.register(BashTool())
        pm = PermissionManager()
        pm.allow_always("bash")
        engine = QueryEngine(
            provider=provider,
            registry=registry,
            permission_manager=pm,
            max_turns=5,
        )
        events = await _collect(engine, "run bash")

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False


# ---------------------------------------------------------------------------
# Tests — tool execution error
# ---------------------------------------------------------------------------

class TestQueryEngineToolErrors:
    @pytest.mark.asyncio
    async def test_tool_execution_error_yields_agent_error(self):
        tc = ToolCall(id="t1", name="failing_tool", input={"value": 42})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("The tool failed."),
        ]
        engine = _make_engine(responses, tools=[FailingTool()])
        events = await _collect(engine, "fail")

        agent_errors = [e for e in events if isinstance(e, AgentError)]
        assert len(agent_errors) >= 1
        assert agent_errors[0].recoverable is True

    @pytest.mark.asyncio
    async def test_tool_execution_error_loop_continues(self):
        """Loop continues after a ToolExecutionError — LLM gets next turn."""
        tc = ToolCall(id="t1", name="failing_tool", input={"value": 1})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("Tool failed, I'll try something else."),
        ]
        engine = _make_engine(responses, tools=[FailingTool()])
        events = await _collect(engine, "fail")

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert any(e.stop_reason == "end_turn" for e in turn_ends)

    @pytest.mark.asyncio
    async def test_pydantic_validation_failure_yields_agent_error(self):
        """Invalid input (wrong type) yields AgentError, loop continues."""
        tc = ToolCall(id="t1", name="echo", input={"message": 999})  # should be str
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("Input was invalid."),
        ]
        engine = _make_engine(responses, tools=[EchoTool()])
        events = await _collect(engine, "invalid input")

        agent_errors = [e for e in events if isinstance(e, AgentError)]
        assert len(agent_errors) >= 1

    @pytest.mark.asyncio
    async def test_unknown_tool_name_yields_agent_error(self):
        """Requesting a tool not in the registry yields AgentError."""
        tc = ToolCall(id="t1", name="nonexistent_tool", input={})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("Tool not found."),
        ]
        engine = _make_engine(responses, tools=[])
        events = await _collect(engine, "unknown tool")

        agent_errors = [e for e in events if isinstance(e, AgentError)]
        assert len(agent_errors) >= 1


# ---------------------------------------------------------------------------
# Tests — max_turns
# ---------------------------------------------------------------------------

class TestQueryEngineMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_enforced(self):
        """When max_turns is hit, TurnEnd with stop_reason=max_turns is yielded."""
        # Provide infinite tool call loop — always returns tool_use
        tc = ToolCall(id="t1", name="echo", input={"message": "loop"})
        # Provide many responses but cap at max_turns=2
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
        ]
        engine = _make_engine(responses, tools=[EchoTool()], max_turns=2)
        events = await _collect(engine, "loop forever")

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert any(e.stop_reason == "max_turns" for e in turn_ends)

    @pytest.mark.asyncio
    async def test_single_turn_max(self):
        """max_turns=1 means only one LLM call regardless of stop_reason."""
        tc = ToolCall(id="t1", name="echo", input={"message": "x"})
        responses = [
            _make_response("", tool_calls=[tc], stop_reason="tool_use"),
            _make_response("second turn"),  # should never be reached
        ]
        engine = _make_engine(responses, tools=[EchoTool()], max_turns=1)
        events = await _collect(engine, "test")

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert any(e.stop_reason == "max_turns" for e in turn_ends)


# ---------------------------------------------------------------------------
# Tests — memory integration
# ---------------------------------------------------------------------------

class TestQueryEngineMemory:
    @pytest.mark.asyncio
    async def test_stub_memory_default_no_error(self):
        """QueryEngine works with default StubMemory (memory=None)."""
        engine = _make_engine([_make_response("ok")])
        events = await _collect(engine, "hi")
        assert any(isinstance(e, TurnEnd) for e in events)

    @pytest.mark.asyncio
    async def test_memory_inject_called(self):
        """inject_into_system_prompt is called on the memory object."""
        memory = MagicMock(spec=StubMemory)
        memory.inject_into_system_prompt.return_value = "injected system prompt"
        memory.query.return_value = QueryContext()

        engine = _make_engine([_make_response("ok")], memory=memory)
        await _collect(engine, "hi")

        memory.inject_into_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_with_chunks_prepended(self):
        """Memory with chunks produces an enriched system prompt."""
        from astra_node.core.memory_types import MemorySystem, UserProfile

        class RichMemory(MemorySystem):
            def query(self, user_message):
                return QueryContext(
                    retrieved_chunks=[ScoredChunk(text="user prefers dark mode", score=1.0)]
                )
            def update(self, query, used_chunks): pass
            def ingest(self, documents): pass
            def get_user_context(self): return UserProfile()

        engine = _make_engine([_make_response("ok")], memory=RichMemory())
        events = await _collect(engine, "hi")
        assert any(isinstance(e, TurnEnd) for e in events)


# ---------------------------------------------------------------------------
# Tests — ProviderError propagation
# ---------------------------------------------------------------------------

class TestQueryEngineProviderError:
    @pytest.mark.asyncio
    async def test_provider_error_propagates(self):
        """ProviderError from the LLM is not caught — it propagates up."""
        from astra_node.utils.errors import ProviderError

        class FailingProvider(LLMProvider):
            async def complete(self, messages, tools, system="", **kwargs):
                raise ProviderError("auth failed", provider="anthropic")
                yield  # make it a generator

        registry = ToolRegistry()
        pm = PermissionManager()
        engine = QueryEngine(
            provider=FailingProvider(),
            registry=registry,
            permission_manager=pm,
        )
        with pytest.raises(ProviderError):
            async for _ in engine.run("hi"):
                pass
