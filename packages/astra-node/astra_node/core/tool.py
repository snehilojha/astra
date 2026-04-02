"""BaseTool ABC, ToolResult, and ToolContext.

Every built-in and user-defined tool inherits from BaseTool. The tool
registry, permission manager, and query engine all work against this
interface — never against concrete tool classes.

Design: input validation is delegated to Pydantic v2. Each tool declares
an input_schema class (a BaseModel subclass). The query engine parses the
LLM's raw JSON input through that schema before calling execute(), so
tools always receive typed, validated input.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel


class PermissionLevel(Enum):
    """Risk level of a tool invocation.

    The permission manager uses this to decide whether to execute
    immediately, prompt the user, or refuse entirely.

    ALWAYS_ALLOW: Read-only, side-effect-free operations (file reads,
                  searches). Never prompt the user.
    ASK_USER:     Mutating or potentially dangerous operations (bash,
                  file writes). Prompt before executing unless the user
                  has granted a session-level override.
    DENY:         Blocked entirely. Even session overrides cannot enable
                  these (reserved for future policy enforcement).
    """

    ALWAYS_ALLOW = "always_allow"
    ASK_USER = "ask_user"
    DENY = "deny"


@dataclass
class ToolResult:
    """The output of a single tool execution.

    If is_error is True, output contains the error message that will be
    sent back to the LLM as a tool_result with is_error=True. The LLM
    can then decide whether to retry, take a different approach, or
    report the failure to the user.
    """

    output: str
    is_error: bool = False
    error: str | None = None

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        """Convenience constructor for a successful result."""
        return cls(output=output, is_error=False)

    @classmethod
    def err(cls, error: str) -> "ToolResult":
        """Convenience constructor for an error result."""
        return cls(output=error, is_error=True, error=error)


@dataclass
class ToolContext:
    """Execution context passed to every tool at runtime.

    Provides the tool with the current working directory, environment
    variables, and an extensible metadata dict. The metadata dict is
    used by the swarm coordinator to pass per-worker scratchpad state
    without coupling tools to swarm concepts.
    """

    cwd: Path = field(default_factory=Path.cwd)
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Concrete tools must set name, description, input_schema, and
    permission_level as class attributes, and implement execute().

    The name must be unique within a ToolRegistry. The description is
    what the LLM sees when deciding which tool to call — keep it
    precise and action-oriented.
    """

    name: str
    description: str
    input_schema: Type[BaseModel]
    permission_level: PermissionLevel

    @abstractmethod
    def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult:
        """Execute the tool with validated input.

        Args:
            input: Validated Pydantic model instance. The concrete type
                   matches this tool's input_schema class.
            ctx: Runtime context (cwd, env, metadata).

        Returns:
            ToolResult with output text. Set is_error=True for failures
            that should be reported to the LLM as tool errors.

        Raises:
            ToolExecutionError: For unrecoverable execution failures that
                the tool cannot express as a ToolResult. The query engine
                will catch this and convert it to an error ToolResult.
        """
        ...
