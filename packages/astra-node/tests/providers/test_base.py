"""Tests for the provider abstraction layer (base.py).

Verifies that the ABC contract is correct, data classes are well-formed,
and that concrete implementations must provide complete().
"""

import pytest

from astra_node.providers.base import LLMProvider, LLMResponse, ToolCall, Usage


class TestUsage:
    """Usage dataclass creation and field access."""

    def test_basic_creation(self):
        u = Usage(input_tokens=10, output_tokens=20)
        assert u.input_tokens == 10
        assert u.output_tokens == 20

    def test_cache_fields_default_to_zero(self):
        u = Usage(input_tokens=5, output_tokens=5)
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_cache_fields_explicit(self):
        u = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=30,
            cache_read_input_tokens=10,
        )
        assert u.cache_creation_input_tokens == 30
        assert u.cache_read_input_tokens == 10

    def test_frozen(self):
        u = Usage(input_tokens=1, output_tokens=1)
        with pytest.raises(Exception):
            u.input_tokens = 99  # type: ignore[misc]


class TestToolCall:
    """ToolCall dataclass creation and field access."""

    def test_basic_creation(self):
        tc = ToolCall(id="call_123", name="bash", input={"command": "ls"})
        assert tc.id == "call_123"
        assert tc.name == "bash"
        assert tc.input == {"command": "ls"}

    def test_empty_input(self):
        tc = ToolCall(id="x", name="file_read", input={})
        assert tc.input == {}

    def test_frozen(self):
        tc = ToolCall(id="a", name="b", input={})
        with pytest.raises(Exception):
            tc.name = "changed"  # type: ignore[misc]


class TestLLMResponse:
    """LLMResponse dataclass creation and field access."""

    def test_end_turn_response(self):
        usage = Usage(input_tokens=10, output_tokens=5)
        resp = LLMResponse(
            content="Hello",
            tool_calls=[],
            stop_reason="end_turn",
            usage=usage,
        )
        assert resp.content == "Hello"
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"
        assert resp.usage is usage

    def test_tool_use_response(self):
        usage = Usage(input_tokens=20, output_tokens=10)
        tc = ToolCall(id="t1", name="bash", input={"command": "echo hi"})
        resp = LLMResponse(
            content="",
            tool_calls=[tc],
            stop_reason="tool_use",
            usage=usage,
        )
        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "bash"

    def test_max_tokens_response(self):
        usage = Usage(input_tokens=1000, output_tokens=500)
        resp = LLMResponse(
            content="partial...",
            tool_calls=[],
            stop_reason="max_tokens",
            usage=usage,
        )
        assert resp.stop_reason == "max_tokens"

    def test_frozen(self):
        usage = Usage(input_tokens=1, output_tokens=1)
        resp = LLMResponse(content="x", tool_calls=[], stop_reason="end_turn", usage=usage)
        with pytest.raises(Exception):
            resp.content = "changed"  # type: ignore[misc]


class TestLLMProviderABC:
    """LLMProvider ABC cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_concrete_without_complete_raises(self):
        """A subclass that does not implement complete() cannot be instantiated."""

        class IncompleteProvider(LLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]

    def test_concrete_with_complete_can_instantiate(self):
        """A subclass that implements complete() can be instantiated."""

        class ConcreteProvider(LLMProvider):
            async def complete(self, messages, tools, system="", **kwargs):
                yield  # pragma: no cover

        provider = ConcreteProvider()
        assert provider is not None
