"""Tests for the Anthropic provider adapter.

All Anthropic SDK calls are mocked — no real API keys or network calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic

from astra_node.core.events import TextDelta, UsageUpdate
from astra_node.providers.anthropic import AnthropicProvider
from astra_node.providers.base import LLMResponse, ToolCall, Usage
from astra_node.utils.errors import ProviderError


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic stream objects
# ---------------------------------------------------------------------------

def _make_text_delta_event(text: str):
    """Build a mock content_block_delta event with a text_delta."""
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.type = "text_delta"
    event.delta.text = text
    return event


def _make_tool_use_block(id: str, name: str, input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_usage(input_tokens: int, output_tokens: int, cache_creation=0, cache_read=0):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation
    usage.cache_read_input_tokens = cache_read
    return usage


def _make_final_message(
    content_blocks: list,
    stop_reason: str,
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    msg = MagicMock()
    msg.content = content_blocks
    msg.stop_reason = stop_reason
    msg.usage = _make_usage(input_tokens, output_tokens)
    return msg


class MockStream:
    """Async context manager that yields pre-set streaming events."""

    def __init__(self, events: list, final_message):
        self._events = events
        self._final = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnthropicProviderInit:
    def test_instantiation(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider is not None

    def test_last_response_none_before_call(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.last_response is None


class TestAnthropicProviderComplete:

    @pytest.fixture
    def provider(self):
        return AnthropicProvider(api_key="test-key", model="claude-sonnet-4-5")

    @pytest.mark.asyncio
    async def test_yields_text_delta_events(self, provider):
        """complete() yields TextDelta events from streamed text chunks."""
        final = _make_final_message(
            [_make_text_block("Hello world")],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
        )
        stream = MockStream(
            [_make_text_delta_event("Hello "), _make_text_delta_event("world")],
            final,
        )

        with patch.object(provider._client.messages, "stream", return_value=stream):
            events = []
            async for event in provider.complete([{"role": "user", "content": "hi"}], []):
                events.append(event)

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 2
        assert text_deltas[0].text == "Hello "
        assert text_deltas[1].text == "world"

    @pytest.mark.asyncio
    async def test_yields_usage_update(self, provider):
        """complete() yields a UsageUpdate event at the end of the stream."""
        final = _make_final_message(
            [_make_text_block("Done")],
            stop_reason="end_turn",
            input_tokens=20,
            output_tokens=10,
        )
        stream = MockStream([_make_text_delta_event("Done")], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            events = []
            async for event in provider.complete([{"role": "user", "content": "hi"}], []):
                events.append(event)

        usage_events = [e for e in events if isinstance(e, UsageUpdate)]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 20
        assert usage_events[0].output_tokens == 10

    @pytest.mark.asyncio
    async def test_tool_calls_translated_correctly(self, provider):
        """Tool use blocks in the final message are translated to ToolCall objects."""
        tool_block = _make_tool_use_block("tool_abc", "bash", {"command": "ls"})
        final = _make_final_message(
            [tool_block],
            stop_reason="tool_use",
            input_tokens=30,
            output_tokens=8,
        )
        stream = MockStream([], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            async for _ in provider.complete([{"role": "user", "content": "run ls"}], []):
                pass

        assert provider.last_response is not None
        assert len(provider.last_response.tool_calls) == 1
        tc = provider.last_response.tool_calls[0]
        assert tc.id == "tool_abc"
        assert tc.name == "bash"
        assert tc.input == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_stop_reason_end_turn(self, provider):
        """stop_reason end_turn is mapped correctly."""
        final = _make_final_message([_make_text_block("ok")], stop_reason="end_turn")
        stream = MockStream([_make_text_delta_event("ok")], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            async for _ in provider.complete([], []):
                pass

        assert provider.last_response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stop_reason_tool_use(self, provider):
        """stop_reason tool_use is mapped correctly."""
        tool_block = _make_tool_use_block("t1", "file_read", {"path": "/tmp/f"})
        final = _make_final_message([tool_block], stop_reason="tool_use")
        stream = MockStream([], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            async for _ in provider.complete([], []):
                pass

        assert provider.last_response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_stop_reason_max_tokens(self, provider):
        """stop_reason max_tokens is mapped correctly."""
        final = _make_final_message(
            [_make_text_block("truncated")], stop_reason="max_tokens"
        )
        stream = MockStream([_make_text_delta_event("truncated")], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            async for _ in provider.complete([], []):
                pass

        assert provider.last_response.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_auth_error_raises_provider_error(self, provider):
        """AuthenticationError is wrapped into ProviderError."""
        with patch.object(
            provider._client.messages,
            "stream",
            side_effect=anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={},
            ),
        ):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.complete([], []):
                    pass

        assert exc_info.value.provider == "anthropic"
        assert isinstance(exc_info.value.cause, anthropic.AuthenticationError)

    @pytest.mark.asyncio
    async def test_network_error_raises_provider_error(self, provider):
        """APIConnectionError is wrapped into ProviderError with cause."""
        with patch.object(
            provider._client.messages,
            "stream",
            side_effect=anthropic.APIConnectionError(request=MagicMock()),
        ):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.complete([], []):
                    pass

        assert exc_info.value.provider == "anthropic"
        assert isinstance(exc_info.value.cause, anthropic.APIConnectionError)

    @pytest.mark.asyncio
    async def test_api_status_error_raises_provider_error(self, provider):
        """APIStatusError (e.g. 500) is wrapped into ProviderError."""
        with patch.object(
            provider._client.messages,
            "stream",
            side_effect=anthropic.APIStatusError(
                message="Internal server error",
                response=MagicMock(status_code=500),
                body={},
            ),
        ):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.complete([], []):
                    pass

        assert exc_info.value.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_last_response_populated_after_complete(self, provider):
        """last_response is set and contains the correct LLMResponse after iteration."""
        final = _make_final_message(
            [_make_text_block("response text")],
            stop_reason="end_turn",
            input_tokens=15,
            output_tokens=7,
        )
        stream = MockStream([_make_text_delta_event("response text")], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            async for _ in provider.complete([{"role": "user", "content": "hi"}], []):
                pass

        resp = provider.last_response
        assert isinstance(resp, LLMResponse)
        assert resp.content == "response text"
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 15
        assert resp.usage.output_tokens == 7

    @pytest.mark.asyncio
    async def test_non_text_delta_events_ignored(self, provider):
        """Non-text_delta content_block_delta events do not produce TextDelta yields."""
        non_text_event = MagicMock()
        non_text_event.type = "content_block_delta"
        non_text_event.delta = MagicMock()
        non_text_event.delta.type = "input_json_delta"  # tool input streaming

        final = _make_final_message([_make_text_block("")], stop_reason="tool_use")
        stream = MockStream([non_text_event], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            events = []
            async for event in provider.complete([], []):
                events.append(event)

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 0

    @pytest.mark.asyncio
    async def test_cache_token_fields_populated(self, provider):
        """Cache token fields from usage are propagated to UsageUpdate."""
        final = _make_final_message(
            [_make_text_block("ok")], stop_reason="end_turn", input_tokens=100, output_tokens=10
        )
        final.usage.cache_creation_input_tokens = 50
        final.usage.cache_read_input_tokens = 20
        stream = MockStream([_make_text_delta_event("ok")], final)

        with patch.object(provider._client.messages, "stream", return_value=stream):
            events = []
            async for event in provider.complete([], []):
                events.append(event)

        usage_event = next(e for e in events if isinstance(e, UsageUpdate))
        assert usage_event.cache_creation_input_tokens == 50
        assert usage_event.cache_read_input_tokens == 20
