"""Typed event hierarchy for the agent loop.

The query engine yields these events as an AsyncGenerator. Callers (CLI, web
UI, tests) consume them and decide how to render or react. This is the public
contract — all downstream code depends on these types.

Design: events are frozen dataclasses (immutable after creation). Each carries
only the data its consumer needs, nothing more.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    """Base class for all events yielded by the query engine.

    Every event has a `type` string so consumers can switch on it without
    isinstance checks if they prefer (useful for serialization).
    """

    type: str = field(init=False)

    def __post_init__(self) -> None:
        # Set type to the class name in snake_case for easy matching
        # e.g., TextDelta -> "text_delta", ToolStart -> "tool_start"
        class_name = self.__class__.__name__
        snake = ""
        for i, ch in enumerate(class_name):
            if ch.isupper() and i > 0:
                snake += "_"
            snake += ch.lower()
        # frozen=True means we can't do self.type = ..., so use object.__setattr__
        object.__setattr__(self, "type", snake)


@dataclass(frozen=True)
class TextDelta(AgentEvent):
    """Incremental text chunk from the LLM's response.

    In V1 (non-streaming), the entire response text arrives as a single
    TextDelta. In V2 (streaming), multiple TextDeltas arrive as the LLM
    generates tokens.
    """

    text: str = ""


@dataclass(frozen=True)
class ToolStart(AgentEvent):
    """Emitted when the LLM requests a tool call and execution is about to begin.

    This fires BEFORE permission checks and execution. The consumer can use
    this to show a "running..." indicator.
    """

    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""


@dataclass(frozen=True)
class ToolResult(AgentEvent):
    """Emitted after a tool finishes executing (success or error).

    If is_error is True, the output contains the error message that was
    sent back to the LLM as a tool_result with is_error=True.
    """

    tool_use_id: str = ""
    tool_name: str = ""
    output: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class TurnEnd(AgentEvent):
    """Emitted when the LLM's turn is complete — no more tool calls.

    stop_reason indicates WHY the turn ended:
    - "end_turn"  — LLM finished naturally (gave a final text response)
    - "max_turns" — hit the max_turns limit, forced stop
    """

    stop_reason: str = "end_turn"


@dataclass(frozen=True)
class AgentError(AgentEvent):
    """Non-fatal error during the agent loop.

    This covers tool execution failures and permission denials. These are
    NOT crashes — the error was sent back to the LLM as is_error=True
    and the loop continues. The model decides what to do next.

    recoverable is always True for now. It exists so V2 can distinguish
    between errors the model can recover from and those it can't.
    """

    error: str = ""
    tool_name: str = ""
    tool_use_id: str = ""
    recoverable: bool = True


@dataclass(frozen=True)
class UsageUpdate(AgentEvent):
    """Token usage after each LLM API call.

    Emitted so the consumer can track cumulative cost. Not rendered
    in the CLI by default — just accumulated silently.

    cache_creation_input_tokens: tokens written to the prompt cache this call.
    cache_read_input_tokens: tokens read from the prompt cache (billed at ~10% rate).
    Both are 0 for providers that don't support prompt caching (OpenAI, Ollama).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
