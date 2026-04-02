"""GrepTool — searches for a pattern in files using the re module.

Permission: ALWAYS_ALLOW (read-only search, no side effects).
"""

import re
from pathlib import Path
from pydantic import BaseModel, Field

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult


class GrepInput(BaseModel):
    """Input schema for GrepTool."""

    pattern: str = Field(..., description="Regular expression pattern to search for.")
    path: str = Field(
        default=".",
        description="Directory or file to search in. Defaults to current directory.",
    )
    include: str = Field(
        default="*.py",
        description="Glob pattern for files to include (e.g. '*.py', '*.ts'). Default *.py.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the search is case-sensitive. Default False.",
    )


class GrepTool(BaseTool):
    """Search for a regex pattern across files and return matching lines.

    Returns up to 100 matches in the format 'filepath:line_number:matched_line'.
    """

    name = "grep"
    description = (
        "Search for a pattern in files using regular expressions. "
        "Returns matching lines with file path and line number. "
        "Use include to filter file types."
    )
    input_schema = GrepInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    _MAX_RESULTS = 100

    def execute(self, input: GrepInput, ctx: ToolContext) -> ToolResult:
        """Search for the pattern in the specified path.

        Args:
            input: Validated GrepInput with pattern, path, include, case_sensitive.
            ctx: Tool context (cwd used to resolve relative paths).

        Returns:
            ToolResult with matching lines, or empty result if no matches.
        """
        search_path = Path(input.path)
        if not search_path.is_absolute():
            search_path = ctx.cwd / search_path

        if not search_path.exists():
            return ToolResult.err(f"Path not found: {search_path}")

        flags = 0 if input.case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(input.pattern, flags)
        except re.error as exc:
            return ToolResult.err(f"Invalid regex pattern: {exc}")

        results: list[str] = []

        if search_path.is_file():
            files = [search_path]
        else:
            files = list(search_path.rglob(input.include))

        for file_path in sorted(files):
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                for line_num, line in enumerate(text.splitlines(), start=1):
                    if compiled.search(line):
                        results.append(f"{file_path}:{line_num}:{line.rstrip()}")
                        if len(results) >= self._MAX_RESULTS:
                            results.append(f"... (truncated at {self._MAX_RESULTS} results)")
                            return ToolResult.ok("\n".join(results))
            except Exception:
                continue

        if not results:
            return ToolResult.ok("No matches found.")
        return ToolResult.ok("\n".join(results))
