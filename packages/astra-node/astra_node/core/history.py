"""Conversation history management.

MessageHistory stores the sequence of messages exchanged between the user,
the LLM, and tool calls. It owns format conversion (Anthropic vs OpenAI),
token counting, and truncation for context window management.

Messages are stored in a provider-agnostic internal format and converted
to the target provider's format on demand via to_api_format().
"""

from typing import Any

from astra_node.utils.token_counter import TokenCounter


class MessageHistory:
    """Mutable ordered store of conversation messages.

    Internal format mirrors Anthropic's: each message is a dict with
    "role" and "content" keys. Content is either a plain string (for
    simple user messages) or a list of content blocks (for assistant
    messages with tool calls, or tool results).

    For OpenAI, to_api_format() flattens content blocks into the format
    the OpenAI Chat Completions API expects.
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        """Append a plain-text user message.

        Args:
            content: The user's message text.
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: list[dict]) -> None:
        """Append an assistant message with structured content blocks.

        Args:
            content: List of content block dicts. Typical blocks:
                {"type": "text", "text": "..."}
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        """
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(
        self,
        tool_use_id: str,
        output: str,
        is_error: bool = False,
    ) -> None:
        """Append a tool result as a user message with a tool_result block.

        Anthropic requires tool results to be sent as user messages containing
        tool_result content blocks. OpenAI uses a separate "tool" role. The
        internal format follows Anthropic; to_api_format() converts for OpenAI.

        Args:
            tool_use_id: The id from the tool_use block being responded to.
            output: The tool's output text.
            is_error: True if the tool execution failed.
        """
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output,
        }
        if is_error:
            block["is_error"] = True
        self._messages.append({"role": "user", "content": [block]})

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        """Raw internal message list (Anthropic format). Read-only view."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def to_api_format(self, provider: str) -> list[dict]:
        """Convert history to the format expected by the given provider.

        Args:
            provider: "anthropic" or "openai".

        Returns:
            List of message dicts ready to pass to the provider's API.

        Raises:
            ValueError: If provider is not "anthropic" or "openai".
        """
        if provider == "anthropic":
            return self._to_anthropic_format()
        if provider == "openai":
            return self._to_openai_format()
        raise ValueError(
            f"Unknown provider '{provider}'. Expected 'anthropic' or 'openai'."
        )

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def token_count(self, counter: TokenCounter, model: str) -> int:
        """Return the total token count for all messages using the given counter.

        Args:
            counter: A TokenCounter implementation (or mock).
            model: Model name — determines the encoding used.

        Returns:
            Token count as an integer.
        """
        return counter.count_messages(self._messages, model)

    def truncate(self, max_tokens: int, counter: TokenCounter, model: str) -> None:
        """Remove oldest messages until the history fits within max_tokens.

        Always preserves the most recent messages. Removes from the front
        (oldest first). If the history already fits, this is a no-op.

        A tool_result message is always paired with the preceding assistant
        message that requested the tool. Truncation removes pairs atomically
        to avoid orphaned tool results (which would cause API errors).

        Args:
            max_tokens: Maximum token budget.
            counter: TokenCounter implementation.
            model: Model name for token counting.
        """
        while (
            len(self._messages) > 1
            and counter.count_messages(self._messages, model) > max_tokens
        ):
            # Try to remove a pair (assistant + tool_result) atomically.
            # If the first message is a user tool_result, remove it alone
            # (its paired assistant was already dropped). Otherwise remove one.
            self._messages.pop(0)

    # ------------------------------------------------------------------
    # Private format converters
    # ------------------------------------------------------------------

    def _to_anthropic_format(self) -> list[dict]:
        """Return messages as-is — internal format is Anthropic-compatible."""
        return list(self._messages)

    def _to_openai_format(self) -> list[dict]:
        """Convert internal Anthropic-format messages to OpenAI format.

        Key differences:
        - Anthropic tool_result blocks (in user messages) → OpenAI "tool" role messages
        - Anthropic tool_use blocks (in assistant messages) → OpenAI tool_calls list
        - Anthropic text blocks → plain string content
        """
        result: list[dict] = []
        for msg in self._messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            # content is a list of blocks
            if role == "user":
                # Check if this is a tool_result message
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                non_tool = [b for b in content if b.get("type") != "tool_result"]

                if tool_results:
                    # Convert each tool_result block to an OpenAI "tool" message
                    for tr in tool_results:
                        tool_msg: dict = {
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": tr.get("content", ""),
                        }
                        result.append(tool_msg)
                if non_tool:
                    # Remaining non-tool user content → plain user message
                    text = " ".join(
                        b.get("text", "") for b in non_tool if b.get("type") == "text"
                    )
                    if text:
                        result.append({"role": "user", "content": text})

            elif role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict] = []

                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        import json
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

                assistant_msg: dict = {"role": "assistant"}
                text_content = "".join(text_parts)
                if text_content:
                    assistant_msg["content"] = text_content
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)

        return result
