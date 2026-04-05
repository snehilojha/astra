"""QueryEngine — the agent loop.

The agent loop orchestrates all framework components: provider, history,
tool registry, permission manager, and memory. It drives the LLM through
multiple turns until the model produces a final text response or hits the
max_turns limit.

Loop sequence per turn:
  1. Append user message to history
  2. Enrich system prompt via memory.inject_into_system_prompt()
  3. Call provider.complete(history, tools, system)
  4. Yield TextDelta events as text arrives
  5. On stop_reason == "tool_use": execute each tool call sequentially
       - Yield ToolStart
       - Check permissions → yield AgentError + continue if denied
       - Validate input via Pydantic → yield AgentError + continue if invalid
       - Execute tool → yield AgentError + continue on ToolExecutionError
       - Append tool_result to history
       - Yield ToolResult
     Then loop back to step 3
  6. On stop_reason == "end_turn": yield TurnEnd, fire post_turn_hook, stop
  7. Yield UsageUpdate after each API call
  8. Enforce max_turns — yield TurnEnd(stop_reason="max_turns") if hit
"""

import asyncio
from typing import AsyncIterator, Callable, Awaitable

from pydantic import ValidationError

from astra_node.core.events import (
    AgentError,
    AgentEvent,
    TextDelta,
    ToolResult,
    ToolStart,
    TurnEnd,
    UsageUpdate,
)
from astra_node.core.history import MessageHistory
from astra_node.core.memory_types import MemorySystem
from astra_node.core.registry import ToolRegistry
from astra_node.permissions.manager import PermissionManager
from astra_node.permissions.types import PermissionDecision
from astra_node.providers.base import LLMProvider
from astra_node.utils.errors import (
    PermissionDeniedError,
    ProviderError,
    ToolExecutionError,
)


class QueryEngine:
    """Drives the agent loop for a single conversation session.

    One QueryEngine instance corresponds to one ongoing conversation. It
    maintains internal MessageHistory state across multiple run() calls,
    so the agent has full conversational context.

    The post_turn_hook slot accepts an async callable that fires after each
    TurnEnd event. At Step 9 this is always None (StubMemory). At Step 11,
    PersistentMemory registers its memory extraction task here as a
    fire-and-forget asyncio task.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        permission_manager: PermissionManager,
        system_prompt: str = "",
        max_turns: int = 10,
        memory: MemorySystem | None = None,
        post_turn_hook: Callable[[list[dict]], Awaitable[None]] | None = None,
    ) -> None:
        """Initialise the agent engine.

        Args:
            provider: LLM provider adapter (AnthropicProvider, OpenAIProvider, …).
            registry: Tool registry with available tools for this session.
            permission_manager: Decides whether tool calls may proceed.
            system_prompt: Base system prompt. Memory context is prepended to this.
            max_turns: Maximum number of LLM-calls before forcing a stop.
            memory: Memory backend. Defaults to StubMemory (no-op) if None.
            post_turn_hook: Async callable fired after TurnEnd (fire-and-forget).
                            Receives the current history messages list.
        """
        from astra_node.core.memory_stub import (
            StubMemory,
        )  # local import avoids circular

        self._provider = provider
        self._registry = registry
        self._permission_manager = permission_manager
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._memory: MemorySystem = memory if memory is not None else StubMemory()
        self._post_turn_hook = post_turn_hook
        self._history = MessageHistory()

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """Run one user turn through the agent loop.

        Appends the user message to history, then drives the LLM through
        tool calls until it produces a final text response or hits max_turns.

        Args:
            user_message: The user's input text.

        Yields:
            AgentEvent instances (TextDelta, ToolStart, ToolResult,
            AgentError, UsageUpdate, TurnEnd).

        Raises:
            ProviderError: If the LLM provider fails fatally (auth, network).
                           This propagates up uncaught — the CLI handles it.
        """
        from astra_node.core.prompt_guard import check_injection, wrap_user_message

        check_injection(user_message)
        safe_message = wrap_user_message(user_message)
        self._history.add_user(safe_message)
        turns_used = 0

        while turns_used < self._max_turns:
            turns_used += 1

            # Enrich system prompt with memory context
            enriched_system = self._memory.inject_into_system_prompt(
                self._system_prompt
            )

            # Build tool schemas for the provider
            provider_name = self._detect_provider()
            tool_schemas = self._registry.to_api_format(provider_name)
            messages = self._history.to_api_format(provider_name)

            # --- LLM call ---
            # Drain the streaming generator and collect TextDelta events,
            # then read stop_reason and tool_calls from last_response.
            async for event in self._provider.complete(
                messages=messages,
                tools=tool_schemas,
                system=enriched_system,
            ):
                if isinstance(event, (TextDelta, UsageUpdate)):
                    yield event

            # Retrieve the full response (populated by complete() on AnthropicProvider
            # and OpenAIProvider after their stream is exhausted)
            response = getattr(self._provider, "last_response", None)
            if response is None:
                # Safety net — should never happen with correct provider impl
                yield TurnEnd(stop_reason="end_turn")
                return

            # Build the assistant's content blocks for history
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    }
                )
            if assistant_content:
                self._history.add_assistant(assistant_content)

            # --- Tool execution ---
            if response.stop_reason == "tool_use":
                if not response.tool_calls:
                    yield TurnEnd(stop_reason="end_turn")
                    self._fire_post_turn_hook()
                    return
                for tc in response.tool_calls:
                    yield ToolStart(
                        tool_name=tc.name,
                        tool_input=tc.input,
                        tool_use_id=tc.id,
                    )

                    # Permission check
                    try:
                        tool = self._registry.get(tc.name)
                    except KeyError:
                        error_msg = f"Tool '{tc.name}' is not registered."
                        self._history.add_tool_result(tc.id, error_msg, is_error=True)
                        yield ToolResult(
                            tool_use_id=tc.id,
                            tool_name=tc.name,
                            output=error_msg,
                            is_error=True,
                        )
                        yield AgentError(
                            error=error_msg,
                            tool_name=tc.name,
                            tool_use_id=tc.id,
                        )
                        continue

                    decision = self._permission_manager.check_level(
                        tc.name, tool.permission_level, tc.input
                    )
                    if decision == PermissionDecision.DENY:
                        exc = PermissionDeniedError(tc.name, tc.input)
                        error_msg = str(exc)
                        self._history.add_tool_result(tc.id, error_msg, is_error=True)
                        yield ToolResult(
                            tool_use_id=tc.id,
                            tool_name=tc.name,
                            output=error_msg,
                            is_error=True,
                        )
                        yield AgentError(
                            error=error_msg,
                            tool_name=tc.name,
                            tool_use_id=tc.id,
                        )
                        continue

                    # Input validation via Pydantic
                    try:
                        validated_input = tool.input_schema(**tc.input)
                    except ValidationError as exc:
                        error_msg = f"Invalid input for tool '{tc.name}': {exc}"
                        self._history.add_tool_result(tc.id, error_msg, is_error=True)
                        yield ToolResult(
                            tool_use_id=tc.id,
                            tool_name=tc.name,
                            output=error_msg,
                            is_error=True,
                        )
                        yield AgentError(
                            error=error_msg,
                            tool_name=tc.name,
                            tool_use_id=tc.id,
                        )
                        continue

                    # Execute the tool
                    from astra_node.core.tool import ToolContext

                    ctx = ToolContext()
                    try:
                        result = tool.execute(validated_input, ctx)
                    except ToolExecutionError as exc:
                        error_msg = str(exc)
                        self._history.add_tool_result(tc.id, error_msg, is_error=True)
                        yield ToolResult(
                            tool_use_id=tc.id,
                            tool_name=tc.name,
                            output=error_msg,
                            is_error=True,
                        )
                        yield AgentError(
                            error=error_msg,
                            tool_name=tc.name,
                            tool_use_id=tc.id,
                            recoverable=True,
                        )
                        continue
                    except Exception as exc:
                        # Catch unexpected exceptions to keep the loop alive
                        error_msg = f"Unexpected error in tool '{tc.name}': {exc}"
                        self._history.add_tool_result(tc.id, error_msg, is_error=True)
                        yield ToolResult(
                            tool_use_id=tc.id,
                            tool_name=tc.name,
                            output=error_msg,
                            is_error=True,
                        )
                        yield AgentError(
                            error=error_msg,
                            tool_name=tc.name,
                            tool_use_id=tc.id,
                            recoverable=True,
                        )
                        continue

                    # Successful tool result — scan for injection before
                    # adding to history so malicious file/web content cannot
                    # silently hijack the next LLM turn.
                    from astra_node.core.prompt_guard import (
                        scan_tool_result,
                        wrap_tool_result,
                    )

                    tool_output = result.output
                    if not result.is_error and tool_output:
                        injection_warning = scan_tool_result(tool_output, tc.name)
                        if injection_warning:
                            import logging as _logging
                            _logging.getLogger(__name__).warning(
                                "Possible prompt injection detected in output of tool '%s'",
                                tc.name,
                            )
                            tool_output = injection_warning + tool_output
                        tool_output = wrap_tool_result(tool_output, tc.name)

                    self._history.add_tool_result(
                        tc.id, tool_output, is_error=result.is_error
                    )
                    yield ToolResult(
                        tool_use_id=tc.id,
                        tool_name=tc.name,
                        output=result.output,  # yield original to renderer, not wrapped
                        is_error=result.is_error,
                    )
                    if result.is_error:
                        yield AgentError(
                            error=result.output,
                            tool_name=tc.name,
                            tool_use_id=tc.id,
                            recoverable=True,
                        )

                # Loop back for next LLM call with tool results in history
                continue

            # --- End turn ---
            yield TurnEnd(stop_reason=response.stop_reason or "end_turn")
            self._fire_post_turn_hook()
            return

        # max_turns exhausted
        yield TurnEnd(stop_reason="max_turns")
        self._fire_post_turn_hook()

    def _detect_provider(self) -> str:
        """Return the wire format name ('anthropic' or 'openai') for tool/history serialization.

        Logical provider names like 'openrouter' and 'ollama' both use the
        OpenAI wire format, so they map to 'openai' here.
        """
        class_name = type(self._provider).__name__.lower()
        if "anthropic" in class_name:
            return "anthropic"
        return "openai"

    def _fire_post_turn_hook(self) -> None:
        """Fire the post_turn_hook as a fire-and-forget asyncio task."""
        if self._post_turn_hook is None:
            return
        try:
            task = asyncio.create_task(self._post_turn_hook(self._history.messages))
            # Log exceptions from the fire-and-forget task instead of silencing them.
            task.add_done_callback(self._on_post_turn_hook_done)
        except RuntimeError:
            # No running event loop — skip silently (e.g. called from sync context).
            pass

    @staticmethod
    def _on_post_turn_hook_done(task: "asyncio.Task") -> None:  # type: ignore[name-defined]
        import logging
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).error(
                "post_turn_hook raised an exception: %s", task.exception(), exc_info=task.exception()
            )
