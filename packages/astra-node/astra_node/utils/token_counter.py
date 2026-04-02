"""Token counting utilities for conversation history.

Uses tiktoken for OpenAI-compatible models (local, no network call).
For Claude models, uses a tiktoken-based estimate with a scaling factor
(~1.1x) to avoid real HTTP calls — this is the fallback/offline path.
A real Anthropic token counting implementation can be injected via the
TokenCounter protocol for production use.

Design: the TokenCounter protocol allows tests to inject a mock counter
without any network calls. The concrete implementations are separate.
"""

from typing import Protocol, runtime_checkable

import tiktoken


# ---------------------------------------------------------------------------
# Protocol — injectable interface
# ---------------------------------------------------------------------------

@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting implementations.

    Anything that implements count_messages() satisfies this protocol.
    Use this type hint in MessageHistory and CompactionEngine so tests
    can inject mock counters without depending on real API calls.
    """

    def count_messages(self, messages: list[dict], model: str) -> int:
        """Count the tokens in a list of messages for the given model.

        Args:
            messages: List of message dicts (role + content).
            model: Model name string (e.g. "gpt-4o", "claude-sonnet-4-5").

        Returns:
            Estimated token count as an integer >= 0.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI / tiktoken-based counter
# ---------------------------------------------------------------------------

# Tiktoken encoding name by model family
_OPENAI_ENCODING_MAP: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

# Claude models use cl100k_base as a proxy (close enough for budget tracking)
_CLAUDE_ENCODING = "cl100k_base"
_CLAUDE_SCALE_FACTOR = 1.1  # Anthropic tokens run ~10% higher than cl100k estimates


class TiktokenCounter:
    """Token counter backed by tiktoken — works fully offline.

    For OpenAI models: uses the correct encoding for each model family.
    For Claude models: uses cl100k_base + 1.1x scaling factor as an
    estimate. Accurate enough for budget tracking and history truncation.
    Inject a real Anthropic counter for billing-critical applications.
    """

    def count_messages(self, messages: list[dict], model: str) -> int:
        """Count tokens in a message list using tiktoken.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            model: Model name. Determines encoding.

        Returns:
            Token count. Returns 0 for an empty list.

        Raises:
            ValueError: If the model name is not recognised.
        """
        if not messages:
            return 0

        encoding = self._get_encoding(model)
        total = 0

        for message in messages:
            # Per-message overhead (matches OpenAI's counting: 3 tokens/message)
            total += 3
            role = message.get("role", "")
            content = message.get("content", "")

            if role:
                total += len(encoding.encode(str(role)))

            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                # Content blocks (Anthropic format: list of {"type", "text"})
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(encoding.encode(block.get("text", "")))

        # Reply priming: 3 tokens per OpenAI convention
        total += 3

        if self._is_claude_model(model):
            total = int(total * _CLAUDE_SCALE_FACTOR)

        return total

    def _get_encoding(self, model: str):
        """Return the tiktoken encoding for the given model."""
        if self._is_claude_model(model):
            return tiktoken.get_encoding(_CLAUDE_ENCODING)

        # Exact match first
        for prefix, encoding_name in _OPENAI_ENCODING_MAP.items():
            if model.startswith(prefix):
                return tiktoken.get_encoding(encoding_name)

        # Fallback for unknown models: try tiktoken's model lookup
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            raise ValueError(
                f"Unknown model '{model}'. Cannot determine token encoding. "
                "Add it to _OPENAI_ENCODING_MAP or use a Claude model prefix."
            )

    @staticmethod
    def _is_claude_model(model: str) -> bool:
        """Return True if model is a Claude/Anthropic model."""
        return model.startswith("claude")


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_counter = TiktokenCounter()


def count_messages(messages: list[dict], model: str) -> int:
    """Count tokens in a message list using the default TiktokenCounter.

    Convenience wrapper — callers that need dependency injection should
    use TiktokenCounter directly or accept a TokenCounter protocol parameter.

    Args:
        messages: List of message dicts.
        model: Model name.

    Returns:
        Estimated token count.
    """
    return _default_counter.count_messages(messages, model)
