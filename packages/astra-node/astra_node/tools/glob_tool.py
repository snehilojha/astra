"""GlobTool — finds files matching a glob pattern.

Permission: ALWAYS_ALLOW (read-only directory listing, no side effects).
"""

from pathlib import Path
from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


class GlobInput(BaseModel):
    """Input schema for GlobTool."""

    pattern: str = Field(
        ...,
        description="Glob pattern to match (e.g. '**/*.py', 'src/**/*.ts').",
    )
    path: str = Field(
        default=".",
        description="Root directory to search in. Defaults to current directory.",
    )


class GlobTool(BaseTool):
    """Find files matching a glob pattern and return their paths.

    Returns up to 200 matched paths, one per line, sorted.
    """

    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Returns a list of matching file paths. "
        "Use ** for recursive matching (e.g. '**/*.py')."
    )
    input_schema = GlobInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    _MAX_RESULTS = 200
    _IGNORE_DIRS = frozenset(
        {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".idea",
            ".vscode",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
        }
    )

    def execute(self, input: GlobInput, ctx: ToolContext) -> ToolResult:
        """Search for files matching the pattern.

        Args:
            input: Validated GlobInput with pattern and root path.
            ctx: Tool context (cwd used to resolve relative paths).

        Returns:
            ToolResult with newline-separated file paths.
        """
        root = Path(input.path)
        if not root.is_absolute():
            root = ctx.cwd / root

        # Prevent traversal outside cwd via ".." sequences.
        root = root.resolve()
        cwd_resolved = ctx.cwd.resolve()
        if not root.is_relative_to(cwd_resolved):
            return ToolResult.err(
                f"Access denied: path is outside the working directory ({cwd_resolved})"
            )

        if not root.exists():
            return ToolResult.err(f"Path not found: {root}")

        try:
            matches = sorted(root.glob(input.pattern))
            file_matches = [
                str(p)
                for p in matches
                if p.is_file()
                and not any(part in self._IGNORE_DIRS for part in p.parts)
            ]

            if not file_matches:
                return ToolResult.ok("No files matched.")

            truncated = file_matches[: self._MAX_RESULTS]
            output = "\n".join(truncated)
            if len(file_matches) > self._MAX_RESULTS:
                output += f"\n... (showing {self._MAX_RESULTS} of {len(file_matches)} matches)"
            return ToolResult.ok(output)
        except Exception as exc:
            return ToolResult.err(f"Glob search failed: {exc}")
