"""FileReadTool — reads the contents of a file.

Permission: ALWAYS_ALLOW (read-only, no side effects).
"""

import fnmatch
from pathlib import Path
from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult

# Credential file patterns that must never be read by the agent.
_BLOCKED_PATTERNS = (
    "*.pem", "*.key", "*.p12", "*.pfx", "*.crt",
    "id_rsa", "id_ecdsa", "id_ed25519", "id_dsa",
    ".env", ".env.*",
    "config.json",  # catches ~/.astra/config.json when outside cwd
)
# Directories whose contents should never be read.
_BLOCKED_DIRS = (
    Path.home() / ".ssh",
    Path.home() / ".astra",
    Path.home() / ".aws",
    Path.home() / ".config" / "gcloud",
)


class FileReadInput(BaseModel):
    """Input schema for FileReadTool."""

    path: str = Field(..., description="Absolute or relative path to the file to read.")
    offset: int = Field(
        default=0,
        description="Line number (1-indexed) to start reading from. 0 means read from the beginning.",
    )
    limit: int = Field(
        default=0,
        description="Maximum number of lines to return. 0 means read the entire file.",
    )


class FileReadTool(BaseTool):
    """Read the contents of a file and return them as text.

    Supports optional line offset and limit for reading large files in chunks.
    """

    name = "file_read"
    description = (
        "Read the contents of a file. "
        "Optionally specify offset (starting line) and limit (max lines) "
        "for reading large files in chunks."
    )
    input_schema = FileReadInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: FileReadInput, ctx: ToolContext) -> ToolResult:
        """Read the file at the given path.

        Args:
            input: Validated FileReadInput with path, optional offset and limit.
            ctx: Tool context (cwd used to resolve relative paths).

        Returns:
            ToolResult with file contents, or an error if the file is missing.
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

        # Block reads of known credential files regardless of location.
        filename = path.name
        if any(fnmatch.fnmatch(filename, pat) for pat in _BLOCKED_PATTERNS):
            return ToolResult.err(
                f"Access denied: reading credential files is not permitted ({filename})"
            )
        for blocked_dir in _BLOCKED_DIRS:
            try:
                if path.is_relative_to(blocked_dir.resolve()):
                    return ToolResult.err(
                        f"Access denied: reading from {blocked_dir} is not permitted"
                    )
            except ValueError:
                pass

        if not path.exists():
            return ToolResult.err(f"File not found: {path}")
        if not path.is_file():
            return ToolResult.err(f"Path is not a file: {path}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if input.offset > 0 or input.limit > 0:
                lines = text.splitlines(keepends=True)
                start = max(0, input.offset - 1) if input.offset > 0 else 0
                end = start + input.limit if input.limit > 0 else len(lines)
                text = "".join(lines[start:end])
            return ToolResult.ok(text)
        except Exception as exc:
            return ToolResult.err(f"Failed to read file: {exc}")
