"""Tests for CompactionEngine — context window management."""

import pytest
from unittest.mock import MagicMock

from astra_node.core.compaction import CompactionEngine
from astra_node.core.events import TextDelta, UsageUpdate
from astra_node.core.history import MessageHistory
from astra_node.providers.base import LLMProvider, LLMResponse, Usage


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_counter(return_value: int):
    counter = MagicMock()
    counter.count_messages = MagicMock(return_value=return_value)
    return counter


class SummaryProvider(LLMProvider):
    """Provider that yields a fixed summary text."""

    def __init__(self, summary: str = "Compacted summary text."):
        self._summary = summary
        self._last_response = LLMResponse(
            content=summary,
            tool_calls=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=20),
        )

    async def complete(self, messages, tools, system="", **kwargs):
        yield TextDelta(text=self._summary)
        yield UsageUpdate(input_tokens=50, output_tokens=20)

    @property
    def last_response(self):
        return self._last_response


class ErrorProvider(LLMProvider):
    """Provider that always raises on complete()."""

    async def complete(self, messages, tools, system="", **kwargs):
        raise RuntimeError("provider failed")
        yield  # make it a generator

    @property
    def last_response(self):
        return None


# ---------------------------------------------------------------------------
# should_compact()
# ---------------------------------------------------------------------------

class TestShouldCompact:
    def setup_method(self):
        self.engine = CompactionEngine(threshold=0.80)

    def test_returns_false_when_under_budget(self):
        history = MessageHistory()
        history.add_user("hi")
        counter = _mock_counter(100)  # 100 tokens
        result = self.engine.should_compact(history, counter, "gpt-4o", budget=10_000)
        assert result is False

    def test_returns_true_at_threshold(self):
        history = MessageHistory()
        history.add_user("hi")
        counter = _mock_counter(8_000)  # 8000/10000 = 80%
        result = self.engine.should_compact(history, counter, "gpt-4o", budget=10_000)
        assert result is True

    def test_returns_true_above_threshold(self):
        history = MessageHistory()
        history.add_user("hi")
        counter = _mock_counter(9_500)  # 95% of budget
        result = self.engine.should_compact(history, counter, "gpt-4o", budget=10_000)
        assert result is True

    def test_returns_false_just_below_threshold(self):
        history = MessageHistory()
        history.add_user("hi")
        counter = _mock_counter(7_999)  # just under 80%
        result = self.engine.should_compact(history, counter, "gpt-4o", budget=10_000)
        assert result is False

    def test_uses_model_context_window_when_no_budget(self):
        """When budget is not passed, uses model's known context window."""
        history = MessageHistory()
        history.add_user("hi")
        # gpt-4o has 128_000 context, 80% = 102_400
        counter_under = _mock_counter(100_000)  # under threshold
        counter_over = _mock_counter(110_000)   # over threshold
        assert self.engine.should_compact(history, counter_under, "gpt-4o") is False
        assert self.engine.should_compact(history, counter_over, "gpt-4o") is True


# ---------------------------------------------------------------------------
# compact()
# ---------------------------------------------------------------------------

class TestCompact:
    @pytest.mark.asyncio
    async def test_returns_shorter_history(self):
        history = MessageHistory()
        for i in range(10):
            history.add_user(f"user message {i}")
            history.add_assistant([{"type": "text", "text": f"response {i}"}])

        engine = CompactionEngine()
        provider = SummaryProvider("Summary of the conversation.")
        compacted = await engine.compact(history, provider)

        assert isinstance(compacted, MessageHistory)
        assert len(compacted) < len(history)

    @pytest.mark.asyncio
    async def test_compacted_history_contains_summary(self):
        history = MessageHistory()
        history.add_user("What is the capital of France?")
        history.add_assistant([{"type": "text", "text": "Paris."}])

        engine = CompactionEngine()
        provider = SummaryProvider("Paris is the capital of France.")
        compacted = await engine.compact(history, provider)

        # The summary should appear in the compacted history
        full_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in compacted.messages
        )
        assert "Paris" in full_text or "compacted" in full_text.lower()

    @pytest.mark.asyncio
    async def test_compact_with_session_summary_prepends_it(self):
        """Session summary is included in the compaction prompt."""
        history = MessageHistory()
        history.add_user("Continue the work.")

        received_systems: list[str] = []

        class RecordingProvider(LLMProvider):
            async def complete(self, messages, tools, system="", **kwargs):
                received_systems.append(system)
                yield TextDelta(text="Compacted.")
                yield UsageUpdate(input_tokens=10, output_tokens=5)

            @property
            def last_response(self):
                return LLMResponse(content="Compacted.", tool_calls=[], stop_reason="end_turn",
                                   usage=Usage(input_tokens=10, output_tokens=5))

        engine = CompactionEngine()
        await engine.compact(
            history,
            RecordingProvider(),
            session_summary="User was working on a Python project.",
        )

        assert len(received_systems) == 1
        assert "Session context" in received_systems[0]
        assert "Python project" in received_systems[0]

    @pytest.mark.asyncio
    async def test_compact_without_session_summary(self):
        """Compaction works with empty session_summary."""
        history = MessageHistory()
        history.add_user("Hello.")
        history.add_assistant([{"type": "text", "text": "Hi."}])

        engine = CompactionEngine()
        provider = SummaryProvider("Brief summary.")
        compacted = await engine.compact(history, provider, session_summary="")

        assert isinstance(compacted, MessageHistory)
        assert len(compacted) > 0

    @pytest.mark.asyncio
    async def test_compact_preserves_most_recent_turn(self):
        """After compaction, the compacted history has at least 2 messages."""
        history = MessageHistory()
        for i in range(5):
            history.add_user(f"msg {i}")

        engine = CompactionEngine()
        provider = SummaryProvider("Session summary here.")
        compacted = await engine.compact(history, provider)

        assert len(compacted) >= 2

    @pytest.mark.asyncio
    async def test_compact_provider_error_handled_gracefully(self):
        """ProviderError during compaction returns a minimal history with error note."""
        history = MessageHistory()
        history.add_user("Some message.")

        engine = CompactionEngine()
        compacted = await engine.compact(history, ErrorProvider())

        assert isinstance(compacted, MessageHistory)
        assert len(compacted) > 0
        # Should contain an error note
        content = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in compacted.messages
        )
        assert "compacted" in content.lower() or "failed" in content.lower()

    @pytest.mark.asyncio
    async def test_compact_with_session_summary_empty_behaves_same_as_no_summary(self):
        """Empty session_summary produces identical behaviour to no summary."""
        history = MessageHistory()
        history.add_user("A message.")

        engine = CompactionEngine()
        p1 = SummaryProvider("result A")
        p2 = SummaryProvider("result A")

        c1 = await engine.compact(history, p1, session_summary="")
        c2 = await engine.compact(history, p2)

        assert len(c1) == len(c2)
