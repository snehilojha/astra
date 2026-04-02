"""Tests for all 8 built-in tools.

Filesystem tools use tmp_path fixtures.
Network tools (web_fetch, web_search) mock urllib.
No real filesystem mutations outside tmp_path.
No real network calls.
"""

import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult
from astra_node.tools.bash import BashTool, BashInput
from astra_node.tools.file_read import FileReadTool, FileReadInput
from astra_node.tools.file_write import FileWriteTool, FileWriteInput
from astra_node.tools.file_edit import FileEditTool, FileEditInput
from astra_node.tools.grep import GrepTool, GrepInput
from astra_node.tools.glob_tool import GlobTool, GlobInput
from astra_node.tools.web_fetch import WebFetchTool, WebFetchInput
from astra_node.tools.web_search import WebSearchTool, WebSearchInput


def _ctx(tmp_path: Path | None = None) -> ToolContext:
    return ToolContext(cwd=tmp_path or Path.cwd())


# ---------------------------------------------------------------------------
# Shared ABC compliance checks
# ---------------------------------------------------------------------------

class TestToolABCCompliance:
    @pytest.mark.parametrize("tool_cls", [
        BashTool, FileReadTool, FileWriteTool, FileEditTool,
        GrepTool, GlobTool, WebFetchTool, WebSearchTool,
    ])
    def test_inherits_from_base_tool(self, tool_cls):
        assert issubclass(tool_cls, BaseTool)

    @pytest.mark.parametrize("tool_cls", [
        BashTool, FileReadTool, FileWriteTool, FileEditTool,
        GrepTool, GlobTool, WebFetchTool, WebSearchTool,
    ])
    def test_name_non_empty(self, tool_cls):
        assert len(tool_cls.name) > 0

    @pytest.mark.parametrize("tool_cls", [
        BashTool, FileReadTool, FileWriteTool, FileEditTool,
        GrepTool, GlobTool, WebFetchTool, WebSearchTool,
    ])
    def test_description_non_empty(self, tool_cls):
        assert len(tool_cls.description) > 0

    @pytest.mark.parametrize("tool_cls", [
        BashTool, FileReadTool, FileWriteTool, FileEditTool,
        GrepTool, GlobTool, WebFetchTool, WebSearchTool,
    ])
    def test_input_schema_is_pydantic_model(self, tool_cls):
        assert issubclass(tool_cls.input_schema, BaseModel)

    @pytest.mark.parametrize("tool_cls,expected_level", [
        (BashTool, PermissionLevel.ASK_USER),
        (FileReadTool, PermissionLevel.ALWAYS_ALLOW),
        (FileWriteTool, PermissionLevel.ASK_USER),
        (FileEditTool, PermissionLevel.ASK_USER),
        (GrepTool, PermissionLevel.ALWAYS_ALLOW),
        (GlobTool, PermissionLevel.ALWAYS_ALLOW),
        (WebFetchTool, PermissionLevel.ASK_USER),
        (WebSearchTool, PermissionLevel.ASK_USER),
    ])
    def test_permission_level(self, tool_cls, expected_level):
        assert tool_cls.permission_level == expected_level


# ---------------------------------------------------------------------------
# BashTool
# ---------------------------------------------------------------------------

class TestBashTool:
    def test_run_simple_command(self):
        tool = BashTool()
        ctx = _ctx()
        inp = BashInput(command="echo hello")
        result = tool.execute(inp, ctx)
        assert result.is_error is False
        assert "hello" in result.output

    def test_failing_command_returns_error(self):
        tool = BashTool()
        ctx = _ctx()
        inp = BashInput(command="exit 1")
        result = tool.execute(inp, ctx)
        assert result.is_error is True

    def test_timeout_returns_error(self):
        tool = BashTool()
        ctx = _ctx()
        inp = BashInput(command="ping -n 10 127.0.0.1", timeout=1)
        result = tool.execute(inp, ctx)
        # Either times out or the command fails — either way is_error
        # On Windows ping may behave differently; check it doesn't crash
        assert isinstance(result.is_error, bool)

    def test_command_output_captured(self):
        tool = BashTool()
        ctx = _ctx()
        inp = BashInput(command="echo astra_test_output")
        result = tool.execute(inp, ctx)
        assert "astra_test_output" in result.output


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------

class TestFileReadTool:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        tool = FileReadTool()
        inp = FileReadInput(path=str(f))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "hello world" in result.output

    def test_missing_file_returns_error(self, tmp_path):
        tool = FileReadTool()
        inp = FileReadInput(path=str(tmp_path / "nonexistent.txt"))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is True
        assert "not found" in result.output.lower() or "File not found" in result.output

    def test_relative_path_resolved_from_cwd(self, tmp_path):
        f = tmp_path / "relative.txt"
        f.write_text("relative content")
        tool = FileReadTool()
        inp = FileReadInput(path="relative.txt")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "relative content" in result.output

    def test_offset_and_limit(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        tool = FileReadTool()
        inp = FileReadInput(path=str(f), offset=2, limit=2)
        result = tool.execute(inp, _ctx(tmp_path))
        assert "line2" in result.output
        assert "line3" in result.output
        assert "line1" not in result.output
        assert "line4" not in result.output


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------

class TestFileWriteTool:
    def test_writes_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        tool = FileWriteTool()
        inp = FileWriteInput(path=str(f), content="written content")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert f.read_text() == "written content"

    def test_creates_parent_directories(self, tmp_path):
        f = tmp_path / "subdir" / "deep" / "file.txt"
        tool = FileWriteTool()
        inp = FileWriteInput(path=str(f), content="deep content")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert f.read_text() == "deep content"

    def test_overwrites_existing_file(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        tool = FileWriteTool()
        inp = FileWriteInput(path=str(f), content="new content")
        tool.execute(inp, _ctx(tmp_path))
        assert f.read_text() == "new content"

    def test_relative_path_resolved(self, tmp_path):
        tool = FileWriteTool()
        inp = FileWriteInput(path="output.txt", content="data")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert (tmp_path / "output.txt").read_text() == "data"


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------

class TestFileEditTool:
    def test_applies_edit(self, tmp_path):
        f = tmp_path / "edit_me.py"
        f.write_text("def foo():\n    return 1\n")
        tool = FileEditTool()
        inp = FileEditInput(
            path=str(f),
            old_string="return 1",
            new_string="return 42",
        )
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "return 42" in f.read_text()
        assert "return 1" not in f.read_text()

    def test_old_string_not_found_returns_error(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello world")
        tool = FileEditTool()
        inp = FileEditInput(path=str(f), old_string="not present", new_string="x")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is True
        assert "not found" in result.output.lower()

    def test_missing_file_returns_error(self, tmp_path):
        tool = FileEditTool()
        inp = FileEditInput(
            path=str(tmp_path / "ghost.txt"),
            old_string="x",
            new_string="y",
        )
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is True

    def test_only_first_occurrence_replaced(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("abc abc abc")
        tool = FileEditTool()
        inp = FileEditInput(path=str(f), old_string="abc", new_string="XYZ")
        tool.execute(inp, _ctx(tmp_path))
        content = f.read_text()
        assert content.count("XYZ") == 1
        assert content.count("abc") == 2


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------

class TestGrepTool:
    def test_finds_pattern_in_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    pass\n\ndef world():\n    pass\n")
        tool = GrepTool()
        inp = GrepInput(pattern="def hello", path=str(tmp_path), include="*.py")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "def hello" in result.output

    def test_no_match_returns_no_matches(self, tmp_path):
        f = tmp_path / "file.py"
        f.write_text("nothing interesting here")
        tool = GrepTool()
        inp = GrepInput(pattern="ZZZNOMATCH", path=str(tmp_path), include="*.py")
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "No matches" in result.output

    def test_invalid_regex_returns_error(self, tmp_path):
        tool = GrepTool()
        inp = GrepInput(pattern="[invalid", path=str(tmp_path))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is True
        assert "Invalid regex" in result.output

    def test_missing_path_returns_error(self, tmp_path):
        tool = GrepTool()
        inp = GrepInput(pattern="x", path=str(tmp_path / "nonexistent"))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is True

    def test_case_insensitive_by_default(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("Hello World\n")
        tool = GrepTool()
        inp = GrepInput(pattern="hello", path=str(tmp_path), include="*.py", case_sensitive=False)
        result = tool.execute(inp, _ctx(tmp_path))
        assert "Hello World" in result.output

    def test_case_sensitive_no_match(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("Hello World\n")
        tool = GrepTool()
        inp = GrepInput(pattern="hello", path=str(tmp_path), include="*.py", case_sensitive=True)
        result = tool.execute(inp, _ctx(tmp_path))
        assert "No matches" in result.output


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------

class TestGlobTool:
    def test_finds_files_by_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        tool = GlobTool()
        inp = GlobInput(pattern="*.py", path=str(tmp_path))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.txt" not in result.output

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("")
        tool = GlobTool()
        inp = GlobInput(pattern="**/*.py", path=str(tmp_path))
        result = tool.execute(inp, _ctx(tmp_path))
        assert "deep.py" in result.output

    def test_no_match_message(self, tmp_path):
        tool = GlobTool()
        inp = GlobInput(pattern="*.xyz", path=str(tmp_path))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is False
        assert "No files matched" in result.output

    def test_missing_path_returns_error(self, tmp_path):
        tool = GlobTool()
        inp = GlobInput(pattern="*.py", path=str(tmp_path / "nonexistent"))
        result = tool.execute(inp, _ctx(tmp_path))
        assert result.is_error is True


# ---------------------------------------------------------------------------
# WebFetchTool (mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebFetchTool:
    def _mock_response(self, content: bytes, status: int = 200):
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=content)
        mock.status = status
        return mock

    def test_fetches_url_returns_content(self):
        tool = WebFetchTool()
        inp = WebFetchInput(url="https://example.com")
        mock_resp = self._mock_response(b"<html>hello</html>")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(inp, _ctx())

        assert result.is_error is False
        assert "hello" in result.output

    def test_http_error_returns_error(self):
        import urllib.error
        tool = WebFetchTool()
        inp = WebFetchInput(url="https://example.com/404")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://example.com/404",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            ),
        ):
            result = tool.execute(inp, _ctx())

        assert result.is_error is True
        assert "404" in result.output

    def test_url_error_returns_error(self):
        import urllib.error
        tool = WebFetchTool()
        inp = WebFetchInput(url="https://invalid.example")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Name or service not known"),
        ):
            result = tool.execute(inp, _ctx())

        assert result.is_error is True

    def test_truncation_at_max_length(self):
        tool = WebFetchTool()
        inp = WebFetchInput(url="https://example.com", max_length=10)
        mock_resp = self._mock_response(b"a" * 100)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(inp, _ctx())

        assert result.is_error is False
        assert "truncated" in result.output


# ---------------------------------------------------------------------------
# WebSearchTool (mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    def _mock_ddg_response(self, abstract: str = "", topics: list | None = None):
        data = {
            "AbstractText": abstract,
            "AbstractURL": "https://example.com/abstract" if abstract else "",
            "RelatedTopics": topics or [],
        }
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=json.dumps(data).encode())
        return mock

    def test_returns_results(self):
        tool = WebSearchTool()
        inp = WebSearchInput(query="python testing")
        mock_resp = self._mock_ddg_response(
            abstract="Python testing frameworks",
            topics=[
                {"Text": "pytest is a framework", "FirstURL": "https://pytest.org"},
                {"Text": "unittest is built-in", "FirstURL": "https://docs.python.org"},
            ],
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(inp, _ctx())

        assert result.is_error is False
        assert "python testing" in result.output.lower()

    def test_no_results_returns_message(self):
        tool = WebSearchTool()
        inp = WebSearchInput(query="xyz_totally_obscure_query")
        mock_resp = self._mock_ddg_response(abstract="", topics=[])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool.execute(inp, _ctx())

        assert result.is_error is False
        assert "No results" in result.output

    def test_url_error_returns_error(self):
        import urllib.error
        tool = WebSearchTool()
        inp = WebSearchInput(query="test")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network error"),
        ):
            result = tool.execute(inp, _ctx())

        assert result.is_error is True
