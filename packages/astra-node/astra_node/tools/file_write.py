"""FileWriteTool — writes content to a file.

Permission: ASK_USER (creates/overwrites files — mutating).
"""

from pathlib import Path
from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


class FileWriteInput(BaseModel):
    """Input schema for FileWriteTool."""

    path: str = Field(..., description="Absolute or relative path to the file to write.")
    content: str = Field(..., description="Content to write to the file.")


class FileWriteTool(BaseTool):
    """Write content to a file, creating it (and parent directories) if needed.

    Overwrites the file if it already exists.
    """

    name = "file_write"
    description = (
        "Write content to a file. Creates the file and any parent directories "
        "if they do not exist. Overwrites existing content."
    )
    input_schema = FileWriteInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: FileWriteInput, ctx: ToolContext) -> ToolResult:
        """Write content to the file at the given path.

        Args:
            input: Validated FileWriteInput with path and content.
            ctx: Tool context (cwd used to resolve relative paths).

        Returns:
            ToolResult confirming write, or an error on failure.
        """
        path = Path(input.path)
        if not path.is_absolute():
            path = ctx.cwd / path

        # Resolve ".." sequences before the file exists; use the parent dir for the
        # boundary check because the file itself may not exist yet.
        try:
            resolved_parent = path.parent.resolve()
        except OSError as exc:
            return ToolResult.err(f"Invalid path: {exc}")
        cwd_resolved = ctx.cwd.resolve()
        if not resolved_parent.is_relative_to(cwd_resolved):
            return ToolResult.err(
                f"Access denied: path is outside the working directory ({cwd_resolved})"
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input.content, encoding="utf-8")
            return ToolResult.ok(f"Wrote {len(input.content)} characters to {path}")
        except Exception as exc:
            return ToolResult.err(f"Failed to write file: {exc}")
