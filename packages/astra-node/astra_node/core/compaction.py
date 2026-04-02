"""CompactionEngine — context window management via LLM summarisation.

When a conversation grows large enough to approach the model's context
window, the compaction engine makes a secondary LLM call to summarise the
history into a shorter form. This lets long sessions continue without
hitting the context limit.

Trigger: ~80% of the model's context window (configurable).
Strategy: summarise, preserving decisions, code snippets, errors, and
          pending tasks. Optionally prepend a SessionSummary as an anchor.

Reference: src/services/compact/ in Claude Code.
"""

from astra_node.core.history import MessageHistory
from astra_node.core.events import TextDelta
from astra_node.providers.base import LLMProvider
from astra_node.utils.token_counter import TokenCounter


# Context window sizes by model family (conservative estimates)
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4": 128_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000
_COMPACT_THRESHOLD = 0.80  # Trigger at 80% of context window


class CompactionEngine:
    """Manages context window usage via LLM-based history summarisation.

    Call should_compact() at the start of each turn to decide whether to
    compact. If True, call compact() to replace the history with a
    condensed version before the next LLM call.
    """

    def __init__(
        self,
        threshold: float = _COMPACT_THRESHOLD,
    ) -> None:
        """Initialise the compaction engine.

        Args:
            threshold: Fraction of the context window at which to trigger
                       compaction. Default 0.80 (80%).
        """
        self._threshold = threshold

    def should_compact(
        self,
        history: MessageHistory,
        counter: TokenCounter,
        model: str,
        budget: int | None = None,
    ) -> bool:
        """Decide whether the history should be compacted.

        Args:
            history: Current conversation history.
            counter: TokenCounter to measure history size.
            model: Model name (determines context window size).
            budget: Override the context window size. Useful for testing.

        Returns:
            True if the history exceeds the threshold fraction of the
            context window.
        """
        context_window = budget if budget is not None else self._get_context_window(model)
        current_tokens = history.token_count(counter, model)
        return current_tokens >= int(context_window * self._threshold)

    async def compact(
        self,
        history: MessageHistory,
        provider: LLMProvider,
        model: str = "gpt-4o",
        session_summary: str = "",
    ) -> MessageHistory:
        """Summarise the history into a shorter MessageHistory.

        Makes a secondary LLM call with a summarisation prompt. Returns a
        new MessageHistory containing only the summary as a system-style
        user message, preserving the most important content.

        Args:
            history: The full current history to compact.
            provider: LLMProvider to use for the summarisation call.
            model: Model name (for provider kwargs if needed).
            session_summary: Optional running summary from SessionSummary.
                             Prepended to the summarisation prompt as context.

        Returns:
            A new, shorter MessageHistory containing the compacted summary.
        """
        summary_prefix = ""
        if session_summary:
            summary_prefix = f"## Session context\n{session_summary}\n\n"

        compaction_prompt = (
            f"{summary_prefix}"
            "Summarise this conversation history concisely. Preserve:\n"
            "- Decisions made and their rationale\n"
            "- Code snippets and file paths referenced\n"
            "- Errors encountered and how they were resolved\n"
            "- Pending tasks and open questions\n\n"
            "Write a structured summary that captures everything a developer "
            "needs to continue working on this task. Be specific, not vague."
        )

        # Build a flat text representation of the history for the summarisation prompt
        flat_history = self._flatten_history(history)

        summarisation_messages = [
            {
                "role": "user",
                "content": (
                    f"Please summarise the following conversation history:\n\n"
                    f"{flat_history}"
                ),
            }
        ]

        summary_text = ""
        try:
            async for event in provider.complete(
                messages=summarisation_messages,
                tools=[],
                system=compaction_prompt,
            ):
                if isinstance(event, TextDelta):
                    summary_text += event.text
        except Exception as exc:
            # On provider error, return a minimal history with error note
            summary_text = f"[Compaction failed: {exc}. Original history truncated.]"

        summary_text = summary_text.strip() or "[History compacted — no summary generated.]"

        # Build a new minimal history with the summary as a user message
        # followed by a synthetic assistant acknowledgement
        compacted = MessageHistory()
        compacted.add_user(
            f"[Conversation history was compacted. Summary follows:]\n\n{summary_text}"
        )
        compacted.add_assistant(
            [{"type": "text", "text": "Understood. I have the context from the previous conversation."}]
        )
        return compacted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_context_window(model: str) -> int:
        """Return the context window size for the given model."""
        for prefix, window in _MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(prefix) or model == prefix:
                return window
        return _DEFAULT_CONTEXT_WINDOW

    @staticmethod
    def _flatten_history(history: MessageHistory) -> str:
        """Flatten the history to a human-readable text block for summarisation."""
        lines: list[str] = []
        for msg in history.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role.upper()}: {content}")
            elif isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            parts.append(f"[tool_use: {block.get('name')}({block.get('input', {})})]")
                        elif block.get("type") == "tool_result":
                            parts.append(f"[tool_result: {block.get('content', '')}]")
                lines.append(f"{role.upper()}: {''.join(parts)}")
        return "\n\n".join(lines)
