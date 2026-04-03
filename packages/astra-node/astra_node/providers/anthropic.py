"""Anthropic SDK adapter.

Wraps anthropic.AsyncAnthropic and translates streaming events into the
framework's AgentEvent types. All provider-specific details (event names,
content block structure, stop reason strings) are isolated here.
"""

from typing import AsyncIterator

import anthropic

from astra_node.core.events import AgentEvent, TextDelta, UsageUpdate
from astra_node.providers.base import LLMProvider, LLMResponse, ToolCall, Usage
from astra_node.utils.errors import ProviderError

# Models supported by this adapter
SUPPORTED_MODELS = {
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
}


class AnthropicProvider(LLMProvider):
    """LLMProvider implementation backed by the Anthropic Messages API.

    Translates Anthropic streaming events into framework AgentEvents.
    Handles auth errors, network errors, and stop reason mapping.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8096,
    ) -> None:
        """Initialise the Anthropic adapter.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Model name to use for completions.
            max_tokens: Maximum tokens in the completion response.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        **kwargs,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a completion from the Anthropic Messages API.

        Yields TextDelta events for each text chunk, then a final UsageUpdate.
        Collects the full response internally so the query engine can read
        stop_reason and tool_calls after iteration.

        Args:
            messages: Conversation history in Anthropic format.
            tools: Tool schemas in Anthropic format.
            system: System prompt text.
            **kwargs: Overrides for model or max_tokens.

        Yields:
            TextDelta for streamed text, UsageUpdate at turn end.

        Raises:
            ProviderError: On auth failure (401), network error, or any
                Anthropic APIError.
        """
        model = kwargs.get("model", self._model)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        api_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            api_kwargs["system"] = system
        if tools:
            api_kwargs["tools"] = tools

        try:
            async with self._client.messages.stream(**api_kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextDelta(text=delta.text)

                # Retrieve the final accumulated message for usage + stop reason
                final = await stream.get_final_message()

            usage = Usage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                cache_creation_input_tokens=getattr(
                    final.usage, "cache_creation_input_tokens", 0
                ) or 0,
                cache_read_input_tokens=getattr(
                    final.usage, "cache_read_input_tokens", 0
                ) or 0,
            )
            yield UsageUpdate(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
            )

            # Store the full response on the instance so query_engine can read it
            # after draining the generator.
            tool_calls = []
            for block in final.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            input=dict(block.input),
                        )
                    )

            stop_reason = self._map_stop_reason(final.stop_reason)
            text_content = "".join(
                block.text for block in final.content if block.type == "text"
            )
            self._last_response = LLMResponse(
                content=text_content,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                usage=usage,
            )

        except anthropic.AuthenticationError as exc:
            raise ProviderError(
                "Authentication failed — check your ANTHROPIC_API_KEY",
                provider="anthropic",
                cause=exc,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Network error connecting to Anthropic API: {exc}",
                provider="anthropic",
                cause=exc,
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error {exc.status_code}: {exc.message}",
                provider="anthropic",
                cause=exc,
            ) from exc

    @property
    def last_response(self) -> LLMResponse | None:
        """The most recent complete LLMResponse, available after draining complete()."""
        return getattr(self, "_last_response", None)

    @staticmethod
    def _map_stop_reason(reason: str | None) -> str:
        """Normalise Anthropic stop_reason strings to the framework's contract."""
        mapping = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
        }
        return mapping.get(reason or "", "end_turn")
