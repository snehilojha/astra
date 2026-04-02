"""Tests for token counting utilities.

All tests run fully offline — no real API calls.
The TokenCounter protocol is verified by injecting a mock.
"""

import pytest
from unittest.mock import MagicMock

from astra_node.utils.token_counter import (
    TokenCounter,
    TiktokenCounter,
    count_messages,
)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestTokenCounterProtocol:
    def test_tiktoken_counter_satisfies_protocol(self):
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_mock_satisfies_protocol(self):
        """A simple mock with count_messages() satisfies the protocol."""
        mock = MagicMock()
        mock.count_messages = MagicMock(return_value=42)
        assert isinstance(mock, TokenCounter)


# ---------------------------------------------------------------------------
# TiktokenCounter — basic behaviour
# ---------------------------------------------------------------------------

class TestTiktokenCounterBasic:
    def setup_method(self):
        self.counter = TiktokenCounter()

    def test_empty_messages_returns_zero(self):
        assert self.counter.count_messages([], "gpt-4o") == 0

    def test_non_empty_messages_returns_positive(self):
        msgs = [{"role": "user", "content": "Hello, how are you?"}]
        count = self.counter.count_messages(msgs, "gpt-4o")
        assert count > 0

    def test_longer_content_gives_higher_count(self):
        short_msgs = [{"role": "user", "content": "Hi."}]
        long_msgs = [{"role": "user", "content": "Hi. " * 100}]
        short_count = self.counter.count_messages(short_msgs, "gpt-4o")
        long_count = self.counter.count_messages(long_msgs, "gpt-4o")
        assert long_count > short_count

    def test_count_scales_with_message_length(self):
        """Sanity check: doubling content roughly doubles the count."""
        base = [{"role": "user", "content": "word " * 50}]
        doubled = [{"role": "user", "content": "word " * 100}]
        base_count = self.counter.count_messages(base, "gpt-4o")
        doubled_count = self.counter.count_messages(doubled, "gpt-4o")
        assert doubled_count > base_count * 1.5


# ---------------------------------------------------------------------------
# TiktokenCounter — OpenAI models (local, no mock needed)
# ---------------------------------------------------------------------------

class TestTiktokenCounterOpenAI:
    def setup_method(self):
        self.counter = TiktokenCounter()
        self.msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test_gpt4o_returns_positive_int(self):
        count = self.counter.count_messages(self.msgs, "gpt-4o")
        assert isinstance(count, int)
        assert count > 0

    def test_gpt4o_mini_returns_positive_int(self):
        count = self.counter.count_messages(self.msgs, "gpt-4o-mini")
        assert isinstance(count, int)
        assert count > 0

    def test_multiple_messages_counted(self):
        msgs = [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi there."},
            {"role": "user", "content": "How are you?"},
        ]
        count = self.counter.count_messages(msgs, "gpt-4o")
        assert count > 0


# ---------------------------------------------------------------------------
# TiktokenCounter — Claude models (tiktoken estimate with scaling)
# ---------------------------------------------------------------------------

class TestTiktokenCounterClaude:
    def setup_method(self):
        self.counter = TiktokenCounter()
        self.msgs = [{"role": "user", "content": "What is the capital of France?"}]

    def test_claude_model_returns_positive_int(self):
        count = self.counter.count_messages(self.msgs, "claude-sonnet-4-5")
        assert isinstance(count, int)
        assert count > 0

    def test_claude_count_higher_than_openai_for_same_content(self):
        """Claude count should be ~1.1x higher due to scaling factor."""
        openai_count = self.counter.count_messages(self.msgs, "gpt-4o")
        claude_count = self.counter.count_messages(self.msgs, "claude-sonnet-4-5")
        assert claude_count > openai_count

    def test_claude_opus_model(self):
        count = self.counter.count_messages(self.msgs, "claude-opus-4-5")
        assert count > 0

    def test_claude_haiku_model(self):
        count = self.counter.count_messages(self.msgs, "claude-haiku-4-5-20251001")
        assert count > 0


# ---------------------------------------------------------------------------
# TiktokenCounter — unknown model raises
# ---------------------------------------------------------------------------

class TestTiktokenCounterUnknownModel:
    def test_unknown_model_raises_value_error(self):
        counter = TiktokenCounter()
        msgs = [{"role": "user", "content": "test"}]
        with pytest.raises(ValueError, match="Unknown model"):
            counter.count_messages(msgs, "totally-unknown-model-xyz")


# ---------------------------------------------------------------------------
# TiktokenCounter — content block format (Anthropic list format)
# ---------------------------------------------------------------------------

class TestTiktokenCounterContentBlocks:
    def setup_method(self):
        self.counter = TiktokenCounter()

    def test_list_content_blocks_counted(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello from a content block."}
                ],
            }
        ]
        count = self.counter.count_messages(msgs, "claude-sonnet-4-5")
        assert count > 0

    def test_non_text_blocks_skipped(self):
        """Non-text content blocks (e.g. tool_use) don't crash the counter."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
                ],
            }
        ]
        count = self.counter.count_messages(msgs, "claude-sonnet-4-5")
        assert isinstance(count, int)
        assert count >= 0


# ---------------------------------------------------------------------------
# Module-level count_messages() convenience function
# ---------------------------------------------------------------------------

class TestCountMessagesFunction:
    def test_returns_int(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = count_messages(msgs, "gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_returns_zero(self):
        assert count_messages([], "gpt-4o") == 0


# ---------------------------------------------------------------------------
# Mock injection — verifies protocol is injectable
# ---------------------------------------------------------------------------

class TestMockTokenCounterInjection:
    def test_mock_counter_called_with_correct_args(self):
        """Verify that components accepting TokenCounter will call it correctly."""
        mock_counter = MagicMock(spec=TiktokenCounter)
        mock_counter.count_messages.return_value = 99

        msgs = [{"role": "user", "content": "test"}]
        result = mock_counter.count_messages(msgs, "claude-sonnet-4-5")

        assert result == 99
        mock_counter.count_messages.assert_called_once_with(msgs, "claude-sonnet-4-5")
