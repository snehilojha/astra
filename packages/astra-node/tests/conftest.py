"""Shared pytest fixtures for astra-node tests.

These fixtures are available to all test modules under packages/astra-node/tests/
automatically — pytest discovers conftest.py at collection time.

MockProvider and EchoTool are also defined locally in test_query_engine.py
(pre-dates this conftest). New tests should use these fixtures instead.
"""

import pytest
from typing import AsyncIterator

from pydantic import BaseModel

from astra_node.core.events import AgentEvent, TextDelta, UsageUpdate
from astra_node.core.query_engine import QueryEngine
from astra_node.core.registry import ToolRegistry
from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext
from astra_node.core.tool import ToolResult as CoreToolResult
from astra_node.permissions.manager import PermissionManager
from astra_node.providers.base import LLMProvider, LLMResponse, ToolCall, Usage


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

def _usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def make_response(
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
    def __init__(self, responses: list[LLMResponse]) -> None:
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
    def last_response(self) -> LLMResponse | None:
        return self._last_response


# ---------------------------------------------------------------------------
# EchoTool
# ---------------------------------------------------------------------------

class EchoInput(BaseModel):
    message: str


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the message."
    input_schema = EchoInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: EchoInput, ctx: ToolContext) -> CoreToolResult:
        return CoreToolResult.ok(f"ECHO: {input.message}")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_registry(extra_tools: list[BaseTool] | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    for t in (extra_tools or []):
        reg.register(t)
    return reg


def make_engine(
    responses: list[LLMResponse],
    registry: ToolRegistry | None = None,
    system_prompt: str = "",
    max_turns: int = 10,
) -> QueryEngine:
    return QueryEngine(
        provider=MockProvider(responses),
        registry=registry or make_registry(),
        permission_manager=PermissionManager(),
        system_prompt=system_prompt,
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def echo_registry() -> ToolRegistry:
    return make_registry()


@pytest.fixture
def permission_manager() -> PermissionManager:
    return PermissionManager()
