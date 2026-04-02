"""Custom exception hierarchy for the agent framework.

Design principle: only ProviderError is fatal (stops the agent loop).
ToolExecutionError and PermissionDeniedError are non-fatal — they get sent
back to the LLM as is_error=True tool results, and the model decides whether
to retry, try a different approach, or give up.
"""


class AgentFrameworkError(Exception):
    """Base exception for all framework errors.

    Every custom exception in this framework inherits from this, so callers
    can catch AgentFrameworkError to handle any framework-specific error.
    """


class ProviderError(AgentFrameworkError):
    """LLM provider failed — auth error, network down, rate limited, etc.

    This is FATAL. When this is raised, the agent loop cannot continue
    because it cannot communicate with the LLM. The caller (CLI/REPL)
    should catch this and display an actionable message.
    """

    def __init__(self, message: str, provider: str, cause: Exception | None = None) -> None:
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}")


class ToolExecutionError(AgentFrameworkError):
    """A tool raised an exception during execution.

    This is NON-FATAL. The query engine catches this, wraps the error
    message into a tool_result with is_error=True, and sends it back to
    the LLM. The model can then retry, try a different tool, or inform
    the user.
    """

    def __init__(self, message: str, tool_name: str, cause: Exception | None = None) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"[{tool_name}] {message}")


class PermissionDeniedError(AgentFrameworkError):
    """User denied permission for a tool invocation.

    This is NON-FATAL. Same recovery path as ToolExecutionError — the
    denial is sent back to the LLM as an error tool_result so the model
    knows the tool call was rejected and can adjust its plan.
    """

    def __init__(self, tool_name: str, tool_input: dict | None = None) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input or {}
        super().__init__(f"Permission denied for tool '{tool_name}'")
