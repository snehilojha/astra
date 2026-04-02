"""Tests for MessageHistory."""

import pytest
from unittest.mock import MagicMock

from astra_node.core.history import MessageHistory
from astra_node.utils.token_counter import TiktokenCounter


def _mock_counter(return_value: int):
    """Return a mock TokenCounter that always returns return_value."""
    counter = MagicMock()
    counter.count_messages = MagicMock(return_value=return_value)
    return counter


class TestMessageHistoryAddUser:
    def test_add_user_appends_message(self):
        h = MessageHistory()
        h.add_user("Hello")
        assert len(h) == 1
        assert h.messages[0]["role"] == "user"
        assert h.messages[0]["content"] == "Hello"

    def test_add_user_multiple(self):
        h = MessageHistory()
        h.add_user("first")
        h.add_user("second")
        assert len(h) == 2
        assert h.messages[1]["content"] == "second"


class TestMessageHistoryAddAssistant:
    def test_add_assistant_appends_message(self):
        h = MessageHistory()
        h.add_assistant([{"type": "text", "text": "Hi there"}])
        assert len(h) == 1
        assert h.messages[0]["role"] == "assistant"
        assert h.messages[0]["content"][0]["type"] == "text"

    def test_add_assistant_with_tool_use_block(self):
        h = MessageHistory()
        h.add_assistant([
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}},
        ])
        block = h.messages[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["name"] == "bash"


class TestMessageHistoryAddToolResult:
    def test_add_tool_result_appends_user_message(self):
        h = MessageHistory()
        h.add_tool_result("t1", "output text")
        assert h.messages[0]["role"] == "user"
        block = h.messages[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"
        assert block["content"] == "output text"

    def test_add_tool_result_error_flag(self):
        h = MessageHistory()
        h.add_tool_result("t1", "error occurred", is_error=True)
        block = h.messages[0]["content"][0]
        assert block["is_error"] is True

    def test_add_tool_result_success_no_error_flag(self):
        h = MessageHistory()
        h.add_tool_result("t1", "success", is_error=False)
        block = h.messages[0]["content"][0]
        assert "is_error" not in block


class TestMessageHistoryToAPIFormatAnthropic:
    def test_plain_user_message(self):
        h = MessageHistory()
        h.add_user("hello")
        msgs = h.to_api_format("anthropic")
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_assistant_with_text(self):
        h = MessageHistory()
        h.add_assistant([{"type": "text", "text": "response"}])
        msgs = h.to_api_format("anthropic")
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"][0]["text"] == "response"

    def test_tool_result_block_preserved(self):
        h = MessageHistory()
        h.add_tool_result("t1", "output")
        msgs = h.to_api_format("anthropic")
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"

    def test_empty_history(self):
        h = MessageHistory()
        assert h.to_api_format("anthropic") == []

    def test_unknown_provider_raises(self):
        h = MessageHistory()
        with pytest.raises(ValueError, match="Unknown provider"):
            h.to_api_format("bedrock")


class TestMessageHistoryToAPIFormatOpenAI:
    def test_plain_user_message_converted(self):
        h = MessageHistory()
        h.add_user("hello")
        msgs = h.to_api_format("openai")
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_assistant_text_block_flattened(self):
        h = MessageHistory()
        h.add_assistant([{"type": "text", "text": "I can help with that."}])
        msgs = h.to_api_format("openai")
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "I can help with that."

    def test_tool_result_becomes_tool_role_message(self):
        h = MessageHistory()
        h.add_tool_result("call_123", "file content here")
        msgs = h.to_api_format("openai")
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_123"
        assert msgs[0]["content"] == "file content here"

    def test_assistant_tool_use_becomes_tool_calls(self):
        import json
        h = MessageHistory()
        h.add_assistant([
            {"type": "tool_use", "id": "call_xyz", "name": "bash", "input": {"command": "ls"}},
        ])
        msgs = h.to_api_format("openai")
        assert msgs[0]["role"] == "assistant"
        assert "tool_calls" in msgs[0]
        tc = msgs[0]["tool_calls"][0]
        assert tc["id"] == "call_xyz"
        assert tc["function"]["name"] == "bash"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    def test_empty_history_openai(self):
        h = MessageHistory()
        assert h.to_api_format("openai") == []


class TestMessageHistoryTokenCount:
    def test_token_count_calls_counter(self):
        h = MessageHistory()
        h.add_user("hello")
        counter = _mock_counter(42)
        count = h.token_count(counter, "gpt-4o")
        assert count == 42
        counter.count_messages.assert_called_once_with(h.messages, "gpt-4o")

    def test_token_count_empty_history(self):
        h = MessageHistory()
        counter = _mock_counter(0)
        assert h.token_count(counter, "gpt-4o") == 0


class TestMessageHistoryTruncate:
    def test_truncate_no_op_when_under_budget(self):
        h = MessageHistory()
        h.add_user("hello")
        h.add_assistant([{"type": "text", "text": "hi"}])
        counter = _mock_counter(10)
        h.truncate(max_tokens=100, counter=counter, model="gpt-4o")
        assert len(h) == 2

    def test_truncate_removes_oldest_first(self):
        h = MessageHistory()
        h.add_user("first")
        h.add_user("second")
        h.add_user("third")

        # Counter returns > budget on first call, then fits after removals
        call_count = [0]
        original_msgs = [m.copy() for m in h.messages]

        def side_effect(msgs, model):
            call_count[0] += 1
            # Simulate: too big until only 1 message remains
            return 100 if len(msgs) > 1 else 5

        counter = MagicMock()
        counter.count_messages.side_effect = side_effect
        h.truncate(max_tokens=10, counter=counter, model="gpt-4o")
        # Should have removed messages until only 1 remains
        assert len(h) == 1
        assert h.messages[0]["content"] == "third"

    def test_truncate_preserves_most_recent_turn(self):
        h = MessageHistory()
        for i in range(5):
            h.add_user(f"message {i}")
        last_content = h.messages[-1]["content"]

        def side_effect(msgs, model):
            return 50 if len(msgs) > 1 else 5

        counter = MagicMock()
        counter.count_messages.side_effect = side_effect
        h.truncate(max_tokens=10, counter=counter, model="gpt-4o")
        assert h.messages[-1]["content"] == last_content

    def test_truncate_budget_larger_than_history_is_noop(self):
        h = MessageHistory()
        h.add_user("hi")
        counter = _mock_counter(5)
        original_len = len(h)
        h.truncate(max_tokens=1000, counter=counter, model="gpt-4o")
        assert len(h) == original_len
