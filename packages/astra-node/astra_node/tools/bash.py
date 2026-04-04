"""BashTool — executes shell commands in a subprocess.

Permission: ASK_USER (mutating, potentially dangerous).
"""

import subprocess
from pydantic import BaseModel, Field, field_validator

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


class BashInput(BaseModel):
    """Input schema for BashTool."""

    command: str = Field(..., description="The shell command to execute.")
    timeout: int = Field(
        default=30,
        ge=1,
        description="Maximum seconds to wait for the command. Default 30.",
    )

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError("timeout must be at least 1 second")
        return v


class BashTool(BaseTool):
    """Execute a shell command and return its stdout/stderr output.

    Runs the command in a subprocess with a configurable timeout.
    Returns stderr merged with stdout on failure.
    """

    name = "bash"
    description = (
        "Run a shell command and return the output. "
        "Use for running scripts, reading command output, or interacting with the OS."
    )
    input_schema = BashInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: BashInput, ctx: ToolContext) -> ToolResult:
        """Run the command in a subprocess.

        Args:
            input: Validated BashInput with command and timeout.
            ctx: Tool context (cwd is used as working directory).

        Returns:
            ToolResult with stdout on success, stderr on failure.
        """
        try:
            result = subprocess.run(
                input.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=input.timeout,
                cwd=str(ctx.cwd),
                env=ctx.env if ctx.env else None,
            )
            if result.returncode != 0:
                output = (
                    result.stderr
                    or result.stdout
                    or f"Command exited with code {result.returncode}"
                )
                return ToolResult.err(output)
            return ToolResult.ok(result.stdout)
        except subprocess.TimeoutExpired:
            return ToolResult.err(f"Command timed out after {input.timeout} seconds.")
        except Exception as exc:
            return ToolResult.err(f"Failed to run command: {exc}")
