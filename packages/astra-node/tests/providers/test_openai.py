"""Tests for the OpenAI provider adapter.

All OpenAI SDK calls are mocked — no real API keys or network calls.
Also verifies Ollama compatibility (base_url override).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from astra_node.core.events import TextDelta, UsageUpdate
from astra_node.providers.openai import OpenAIProvider
from astra_node.providers.base import LLMResponse, ToolCall, Usage
from astra_node.utils.errors import ProviderError


# ---------------------------------------------------------------------------
# Helpers to build mock OpenAI streaming chunks
# ---------------------------------------------------------------------------

def _make_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls=None,
    usage=None,
):
    """Build a mock streaming chunk."""
    chunk = MagicMock()
    chunk.usage = usage

    if finish_reason is None and content is None and tool_calls is None and usage is not None:
        # Usage-only final chunk — no choices
        chunk.choices = []
        return chunk

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.delta = MagicMock()
    choice.delta.content = content
    choice.delta.tool_calls = tool_calls
    chunk.choices = [choice]
    return chunk


def _make_usage_obj(prompt_tokens: int, completion_tokens: int):
    u = MagicMock()
    u.prompt_tokens = prompt_tokens
    u.completion_tokens = completion_tokens
    return u


def _make_tool_call_chunk(index: int, id: str = "", name: str = "", arguments: str = ""):
    """Build a mock tool call delta chunk."""
    tc = MagicMock()
    tc.index = index
    tc.id = id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class _AsyncIterWrapper:
    """Async iterable wrapping a list of items. Returned by the mocked create()."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item


def _async_stream(items):
    """Return an AsyncMock whose awaited result is an async iterable over items."""
    mock = AsyncMock(return_value=_AsyncIterWrapper(items))
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOpenAIProviderInit:
    def test_instantiation(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider is not None

    def test_last_response_none_before_call(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.last_response is None

    def test_base_url_override(self):
        """base_url param is stored for Ollama compatibility."""
        provider = OpenAIProvider(
            api_key="ollama",
            model="qwen2.5-coder",
            base_url="http://localhost:11434/v1",
        )
        assert provider._client.base_url is not None


class TestOpenAIProviderComplete:

    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", model="gpt-4o")

    @pytest.mark.asyncio
    async def test_yields_text_delta_events(self, provider):
        """complete() yields TextDelta events for each streamed text chunk."""
        chunks = [
            _make_chunk(content="Hello "),
            _make_chunk(content="world"),
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage_obj(10, 5)),
        ]

        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            events = []
            async for event in provider.complete([{"role": "user", "content": "hi"}], []):
                events.append(event)

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 2
        assert text_deltas[0].text == "Hello "
        assert text_deltas[1].text == "world"

    @pytest.mark.asyncio
    async def test_yields_usage_update(self, provider):
        """complete() yields a UsageUpdate at the end with correct token counts."""
        chunks = [
            _make_chunk(content="Done"),
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage_obj(20, 8)),
        ]

        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            events = []
            async for event in provider.complete([], []):
                events.append(event)

        usage_events = [e for e in events if isinstance(e, UsageUpdate)]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 20
        assert usage_events[0].output_tokens == 8

    @pytest.mark.asyncio
    async def test_tool_calls_translated_correctly(self, provider):
        """Tool call chunks are accumulated and translated to ToolCall objects."""
        tc1 = _make_tool_call_chunk(index=0, id="call_xyz", name="bash", arguments="")
        tc2 = _make_tool_call_chunk(index=0, arguments='{"command": "ls"}')

        chunks = [
            _make_chunk(tool_calls=[tc1]),
            _make_chunk(tool_calls=[tc2]),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(usage=_make_usage_obj(30, 10)),
        ]

        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            async for _ in provider.complete([], [{"type": "function", "function": {"name": "bash"}}]):
                pass

        assert provider.last_response is not None
        assert len(provider.last_response.tool_calls) == 1
        tc = provider.last_response.tool_calls[0]
        assert tc.id == "call_xyz"
        assert tc.name == "bash"
        assert tc.input == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_stop_reason_end_turn(self, provider):
        """finish_reason=stop maps to end_turn."""
        chunks = [
            _make_chunk(content="ok", finish_reason="stop"),
            _make_chunk(usage=_make_usage_obj(5, 2)),
        ]
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            async for _ in provider.complete([], []):
                pass

        assert provider.last_response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stop_reason_tool_use(self, provider):
        """finish_reason=tool_calls maps to tool_use."""
        tc = _make_tool_call_chunk(index=0, id="t1", name="grep", arguments='{"pattern":"x"}')
        chunks = [
            _make_chunk(tool_calls=[tc], finish_reason="tool_calls"),
            _make_chunk(usage=_make_usage_obj(10, 4)),
        ]
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            async for _ in provider.complete([], []):
                pass

        assert provider.last_response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_stop_reason_max_tokens(self, provider):
        """finish_reason=length maps to max_tokens."""
        chunks = [
            _make_chunk(content="truncated...", finish_reason="length"),
            _make_chunk(usage=_make_usage_obj(100, 500)),
        ]
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            async for _ in provider.complete([], []):
                pass

        assert provider.last_response.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_auth_error_raises_provider_error(self, provider):
        """AuthenticationError is wrapped into ProviderError."""
        with patch.object(
            provider._client.chat.completions,
            "create",
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401, headers={}),
                body={},
            ),
        ):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.complete([], []):
                    pass

        assert exc_info.value.provider == "openai"
        assert isinstance(exc_info.value.cause, openai.AuthenticationError)

    @pytest.mark.asyncio
    async def test_network_error_raises_provider_error(self, provider):
        """APIConnectionError is wrapped into ProviderError."""
        with patch.object(
            provider._client.chat.completions,
            "create",
            side_effect=openai.APIConnectionError(request=MagicMock()),
        ):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.complete([], []):
                    pass

        assert exc_info.value.provider == "openai"
        assert isinstance(exc_info.value.cause, openai.APIConnectionError)

    @pytest.mark.asyncio
    async def test_api_status_error_raises_provider_error(self, provider):
        """APIStatusError is wrapped into ProviderError."""
        with patch.object(
            provider._client.chat.completions,
            "create",
            side_effect=openai.APIStatusError(
                message="Server error",
                response=MagicMock(status_code=500, headers={}),
                body={},
            ),
        ):
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.complete([], []):
                    pass

        assert exc_info.value.provider == "openai"

    @pytest.mark.asyncio
    async def test_base_url_override_works(self):
        """Ollama-style base_url override is accepted and used."""
        provider = OpenAIProvider(
            api_key="ollama",
            model="qwen2.5-coder",
            base_url="http://localhost:11434/v1",
        )
        chunks = [
            _make_chunk(content="hi", finish_reason="stop"),
            _make_chunk(usage=_make_usage_obj(3, 2)),
        ]
        mock_create = _async_stream(chunks)
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=mock_create,
        ):
            async for _ in provider.complete([{"role": "user", "content": "hi"}], []):
                pass

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_prompt_prepended_as_system_message(self, provider):
        """System prompt is prepended as a system role message."""
        chunks = [
            _make_chunk(content="ok", finish_reason="stop"),
            _make_chunk(usage=_make_usage_obj(5, 2)),
        ]
        mock_create = _async_stream(chunks)
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=mock_create,
        ):
            async for _ in provider.complete(
                [{"role": "user", "content": "hello"}],
                [],
                system="You are a helpful assistant.",
            ):
                pass

        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_last_response_populated_after_complete(self, provider):
        """last_response contains correct LLMResponse after iteration."""
        chunks = [
            _make_chunk(content="result text", finish_reason="stop"),
            _make_chunk(usage=_make_usage_obj(12, 6)),
        ]
        with patch.object(
            provider._client.chat.completions,
            "create",
            new=_async_stream(chunks),
        ):
            async for _ in provider.complete([], []):
                pass

        resp = provider.last_response
        assert isinstance(resp, LLMResponse)
        assert resp.content == "result text"
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 12
        assert resp.usage.output_tokens == 6
