"""OpenAI SDK adapter (also compatible with Ollama and other OpenAI-API servers).

Wraps openai.AsyncOpenAI and translates streaming events into the
framework's AgentEvent types. Set base_url to point at a local Ollama
instance (e.g. http://localhost:11434/v1) for zero-code-change local inference.
"""

from typing import AsyncIterator

import openai
from openai import AsyncOpenAI

from astra_node.core.events import AgentEvent, TextDelta, UsageUpdate
from astra_node.providers.base import LLMProvider, LLMResponse, ToolCall, Usage
from astra_node.utils.errors import ProviderError


class OpenAIProvider(LLMProvider):
    """LLMProvider implementation backed by the OpenAI Chat Completions API.

    Also works with any OpenAI-compatible server: Ollama, LiteLLM, vLLM, etc.
    Set base_url to the server's /v1 endpoint and api_key as required by that
    server (Ollama accepts any non-empty string).
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 8096,
        base_url: str | None = None,
    ) -> None:
        """Initialise the OpenAI adapter.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
                     For Ollama, pass any non-empty string (e.g. "ollama").
            model: Model name (e.g. "gpt-4o", "gpt-4o-mini", "qwen2.5-coder").
            max_tokens: Maximum tokens in the completion response.
            base_url: Override the API base URL. Use for Ollama or other
                      OpenAI-compatible servers (e.g. "http://localhost:11434/v1").
        """
        self._model = model
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._last_response: LLMResponse | None = None

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        **kwargs,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a completion from the OpenAI Chat Completions API.

        Prepends the system prompt as a system message if provided.
        Yields TextDelta events for each streamed text chunk, then a
        final UsageUpdate.

        Args:
            messages: Conversation history in OpenAI format.
            tools: Tool schemas in OpenAI format.
            system: System prompt text. Prepended as a {"role": "system"} message.
            **kwargs: Overrides for model or max_tokens.

        Yields:
            TextDelta for streamed text chunks, UsageUpdate at turn end.

        Raises:
            ProviderError: On auth failure (401), network error, or any
                openai.APIError.
        """
        model = kwargs.get("model", self._model)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Build message list: prepend system prompt if provided
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        api_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": full_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = "auto"

        try:
            # Accumulate response state while streaming
            content_parts: list[str] = []
            tool_call_accumulators: dict[int, dict] = {}
            stop_reason: str | None = None
            usage: Usage | None = None

            stream = await self._client.chat.completions.create(**api_kwargs)

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None

                if choice is not None:
                    delta = choice.delta

                    # Accumulate text content
                    if delta.content:
                        content_parts.append(delta.content)
                        yield TextDelta(text=delta.content)

                    # Accumulate tool call fragments
                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if idx not in tool_call_accumulators:
                                tool_call_accumulators[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            acc = tool_call_accumulators[idx]
                            if tc_chunk.id:
                                acc["id"] += tc_chunk.id
                            if tc_chunk.function and tc_chunk.function.name:
                                acc["name"] += tc_chunk.function.name
                            if tc_chunk.function and tc_chunk.function.arguments:
                                acc["arguments"] += tc_chunk.function.arguments

                    if choice.finish_reason:
                        stop_reason = choice.finish_reason

                # Usage arrives on the final chunk (stream_options include_usage)
                if chunk.usage is not None:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )

            # Parse accumulated tool calls from JSON argument strings
            import json

            tool_calls: list[ToolCall] = []
            for acc in tool_call_accumulators.values():
                try:
                    parsed_input = (
                        json.loads(acc["arguments"]) if acc["arguments"] else {}
                    )
                except json.JSONDecodeError:
                    parsed_input = {"_raw": acc["arguments"]}
                tool_calls.append(
                    ToolCall(id=acc["id"], name=acc["name"], input=parsed_input)
                )

            if usage is None:
                usage = Usage(input_tokens=0, output_tokens=0)

            yield UsageUpdate(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

            self._last_response = LLMResponse(
                content="".join(content_parts),
                tool_calls=tool_calls,
                stop_reason=self._map_stop_reason(stop_reason),
                usage=usage,
            )

        except openai.AuthenticationError as exc:
            self._last_response = None
            raise ProviderError(
                "Authentication failed — check your OPENAI_API_KEY",
                provider="openai",
                cause=exc,
            ) from exc
        except openai.APIConnectionError as exc:
            self._last_response = None
            raise ProviderError(
                f"Network error connecting to OpenAI API: {exc}",
                provider="openai",
                cause=exc,
            ) from exc
        except openai.APIStatusError as exc:
            self._last_response = None
            raise ProviderError(
                f"OpenAI API error {exc.status_code}: {exc.message}",
                provider="openai",
                cause=exc,
            ) from exc

    @property
    def last_response(self) -> LLMResponse | None:
        """The most recent complete LLMResponse, available after draining complete()."""
        return self._last_response

    @staticmethod
    def _map_stop_reason(reason: str | None) -> str:
        """Normalise OpenAI finish_reason strings to the framework's contract."""
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        return mapping.get(reason or "", "end_turn")
