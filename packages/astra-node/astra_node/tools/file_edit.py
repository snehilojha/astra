"""FileEditTool — applies a targeted string replacement to a file.

Permission: ASK_USER (mutating — modifies existing file content).
"""

from pathlib import Path
from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


class FileEditInput(BaseModel):
    """Input schema for FileEditTool."""

    path: str = Field(..., description="Absolute or relative path to the file to edit.")
    old_string: str = Field(..., description="The exact string to find and replace.")
    new_string: str = Field(..., description="The replacement string.")


class FileEditTool(BaseTool):
    """Apply a targeted string replacement to an existing file.

    Replaces the first occurrence of old_string with new_string.
    Returns an error if old_string is not found in the file.
    """

    name = "file_edit"
    description = (
        "Edit a file by replacing an exact string with a new string. "
        "The old_string must match the file contents exactly (including whitespace). "
        "Returns an error if old_string is not found."
    )
    input_schema = FileEditInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: FileEditInput, ctx: ToolContext) -> ToolResult:
        """Apply the string replacement to the file.

        Args:
            input: Validated FileEditInput with path, old_string, new_string.
            ctx: Tool context (cwd used to resolve relative paths).

        Returns:
            ToolResult confirming the edit, or an error if old_string not found.
        """
        path = Path(input.path)
        if not path.is_absolute():
            path = ctx.cwd / path

        # Resolve symlinks / ".." to prevent path traversal outside cwd.
        path = path.resolve()
        cwd_resolved = ctx.cwd.resolve()
        if not path.is_relative_to(cwd_resolved):
            return ToolResult.err(
                f"Access denied: path is outside the working directory ({cwd_resolved})"
            )

        if not path.exists():
            return ToolResult.err(f"File not found: {path}")
        if not path.is_file():
            return ToolResult.err(f"Path is not a file: {path}")

        try:
            original = path.read_text(encoding="utf-8", errors="replace")
            if input.old_string not in original:
                return ToolResult.err(
                    f"old_string not found in {path}. "
                    "Ensure the string matches exactly including whitespace and indentation."
                )
            updated = original.replace(input.old_string, input.new_string, 1)
            path.write_text(updated, encoding="utf-8")
            return ToolResult.ok(f"Applied edit to {path}")
        except Exception as exc:
            return ToolResult.err(f"Failed to edit file: {exc}")
