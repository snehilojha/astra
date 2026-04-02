"""Provider abstraction layer for LLM backends.

All LLM providers implement the LLMProvider ABC. The query engine only
depends on this interface, never on a concrete SDK. This makes it trivial
to swap Anthropic for OpenAI, or add an Ollama backend, without touching
the agent loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from astra_node.core.events import AgentEvent


@dataclass(frozen=True)
class Usage:
    """Token usage from a single LLM API call.

    cache_creation_input_tokens and cache_read_input_tokens are non-zero
    only for providers that support prompt caching (Anthropic). Both are
    always 0 for OpenAI and Ollama.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the LLM.

    id maps to the provider's tool_use_id (Anthropic) or tool_call.id
    (OpenAI). The query engine uses this id when sending the tool_result
    back to the model.
    """

    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from a completed LLM turn.

    content holds the text portion of the response (may be empty if the
    model only requested tool calls). tool_calls is empty unless
    stop_reason is "tool_use".
    """

    content: str
    tool_calls: list[ToolCall]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens"]
    usage: Usage


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters.

    Concrete implementations translate provider-specific SDK calls and
    streaming events into the framework's AgentEvent types. Callers
    always work against this interface.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        **kwargs,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a completion from the LLM.

        Yields TextDelta events as text arrives, then a final UsageUpdate.
        The caller collects these to build the full response and determine
        whether to execute tool calls.

        Args:
            messages: Conversation history in provider-specific format.
            tools: Tool schemas in provider-specific format.
            system: System prompt text.
            **kwargs: Provider-specific overrides (e.g. temperature, model).

        Yields:
            TextDelta events for streaming text, UsageUpdate at turn end.

        Raises:
            ProviderError: On auth failure, network error, or rate limit.
        """
        # Satisfy type checker for abstract async generator
        raise NotImplementedError
        # This line is unreachable but makes the return type an async generator
        yield  # type: ignore[misc]
