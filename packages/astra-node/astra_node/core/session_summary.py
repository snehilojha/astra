"""SessionSummary — per-session running summary for compaction context.

Maintains a single markdown file updated at turn end. Fed into
CompactionEngine so compaction has running context about what has happened
in the session — without this, compaction summaries lose continuity across
multiple compactions.

Storage: ~/.astra/sessions/<session_id>.md
"""

import re
from pathlib import Path

from astra_node.core.events import TextDelta


class SessionSummary:
    """Per-session rolling summary. Updated at turn end, read by CompactionEngine.

    The summary is not append-only — each update overwrites the previous
    summary to keep it concise. The LLM summarises what has happened so far,
    using the previous summary as a starting anchor if one exists.
    """

    def __init__(self, session_id: str) -> None:
        """Initialise the session summary.

        Args:
            session_id: Unique identifier for this session (used as filename).
        """
        self.path = Path(f"~/.astra/sessions/{session_id}.md").expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def update(self, messages: list[dict], provider) -> None:
        """Summarise the conversation so far.

        Makes a secondary LLM call to generate a 1-3 sentence summary of
        what has happened. Overwrites the previous summary.

        Triggered at turn end when message count crosses a threshold.
        At Step 11 the QueryEngine does NOT call this automatically —
        the CLI or caller is responsible for triggering it as needed.

        Args:
            messages: Current conversation history.
            provider: LLMProvider instance for the summarisation call.
        """
        previous = self.read()
        previous_section = ""
        if previous:
            previous_section = f"\nPrevious summary:\n{previous}\n\n"

        system_prompt = (
            "You are a session summariser. "
            "Write a concise 1-3 sentence summary of what has happened in this conversation. "
            "Focus on decisions made, tasks completed, and pending work. "
            "Be factual and brief."
        )

        from astra_node.core.memory import _extract_text
        convo_text = "\n".join(
            f"{m.get('role', 'user')}: {_extract_text(m)}"
            for m in messages[-20:]  # Last 20 messages only
        )
        user_message = (
            f"{previous_section}"
            f"Conversation so far:\n{convo_text}\n\n"
            "Write a concise summary of what has happened."
        )

        summary_text = ""
        try:
            async for event in provider.complete(
                messages=[{"role": "user", "content": user_message}],
                tools=[],
                system=system_prompt,
            ):
                if isinstance(event, TextDelta):
                    summary_text += event.text
        except Exception:
            return  # Silently skip on provider error — summary is non-critical

        summary_text = summary_text.strip()
        if summary_text:
            self.path.write_text(summary_text, encoding="utf-8")

    def read(self) -> str:
        """Return the current summary text, or '' if not yet written.

        Returns:
            Summary text string, or empty string if no summary exists.
        """
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
