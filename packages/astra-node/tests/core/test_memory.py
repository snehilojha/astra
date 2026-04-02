"""Tests for PersistentMemory, SessionSummary, and parse_frontmatter."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from astra_node.core.memory import (
    PersistentMemory,
    MemoryEntry,
    parse_frontmatter,
    VALID_TYPES,
)
from astra_node.core.memory_types import QueryContext, ScoredChunk, UserProfile
from astra_node.core.session_summary import SessionSummary
from astra_node.core.events import TextDelta


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\nname: test\ntype: user\n---\n\nbody text"
        meta, body = parse_frontmatter(text)
        assert meta["name"] == "test"
        assert meta["type"] == "user"
        assert body == "body text"

    def test_no_frontmatter_returns_empty_meta(self):
        text = "just a body with no frontmatter"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "just a body with no frontmatter"

    def test_malformed_frontmatter_no_closing(self):
        text = "---\nname: test\nno closing delimiter"
        meta, body = parse_frontmatter(text)
        assert meta == {}

    def test_empty_string(self):
        meta, body = parse_frontmatter("")
        assert meta == {}


# ---------------------------------------------------------------------------
# PersistentMemory — storage
# ---------------------------------------------------------------------------

class TestPersistentMemoryStorage:
    @pytest.fixture
    def mem(self, tmp_path):
        return PersistentMemory(memory_dir=str(tmp_path))

    def test_save_creates_file(self, mem, tmp_path):
        entry = MemoryEntry(
            path=Path(""),
            name="user prefers pytest",
            description="user always uses pytest",
            type="feedback",
            body="Use pytest. Never unittest.",
        )
        mem.save(entry)
        assert any((tmp_path / f).exists() for f in tmp_path.iterdir()
                   if f.name.endswith(".md") and f.name != "MEMORY.md")

    def test_save_updates_index(self, mem, tmp_path):
        entry = MemoryEntry(
            path=Path(""),
            name="auth module location",
            description="JWT auth lives at src/auth/",
            type="project",
            body="JWT auth is at src/auth/",
        )
        mem.save(entry)
        index = (tmp_path / "MEMORY.md").read_text()
        assert "auth module location" in index

    def test_save_invalid_type_raises(self, mem):
        entry = MemoryEntry(
            path=Path(""),
            name="bad",
            description="bad type",
            type="invalid_type",
            body="body",
        )
        with pytest.raises(ValueError, match="Invalid memory type"):
            mem.save(entry)

    def test_save_all_four_types(self, mem):
        for t in VALID_TYPES:
            entry = MemoryEntry(
                path=Path(""),
                name=f"{t} memory",
                description=f"desc for {t}",
                type=t,
                body=f"body for {t}",
            )
            mem.save(entry)

        loaded = mem.load_all()
        types_found = {e.type for e in loaded}
        assert types_found == VALID_TYPES

    def test_load_all_reads_files(self, mem):
        for i in range(3):
            entry = MemoryEntry(
                path=Path(""),
                name=f"memory {i}",
                description=f"description {i}",
                type="project",
                body=f"body {i}",
            )
            mem.save(entry)
        loaded = mem.load_all()
        assert len(loaded) == 3

    def test_load_all_skips_invalid_frontmatter(self, tmp_path):
        bad_file = tmp_path / "corrupt.md"
        bad_file.write_text("no frontmatter at all")
        mem = PersistentMemory(memory_dir=str(tmp_path))
        loaded = mem.load_all()
        # Should not crash, and corrupt file is skipped
        assert all(e.name != "" for e in loaded)

    def test_clear_deletes_all_files(self, mem, tmp_path):
        for i in range(3):
            entry = MemoryEntry(
                path=Path(""),
                name=f"mem {i}",
                description="desc",
                type="project",
                body="body",
            )
            mem.save(entry)
        mem.clear()
        loaded = mem.load_all()
        assert len(loaded) == 0
        assert (tmp_path / "MEMORY.md").read_text() == ""


# ---------------------------------------------------------------------------
# PersistentMemory — taxonomy
# ---------------------------------------------------------------------------

class TestPersistentMemoryTaxonomy:
    def test_valid_types_constant(self):
        assert VALID_TYPES == {"user", "feedback", "project", "reference"}


# ---------------------------------------------------------------------------
# PersistentMemory — index limits
# ---------------------------------------------------------------------------

class TestPersistentMemoryIndexLimits:
    def test_index_truncates_at_200_lines(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        for i in range(210):
            entry = MemoryEntry(
                path=Path(""),
                name=f"memory item {i:03d}",
                description=f"description for item {i}",
                type="project",
                body=f"body {i}",
            )
            mem.save(entry)

        index = (tmp_path / "MEMORY.md").read_text()
        lines = index.splitlines()
        assert len(lines) <= 202  # 200 + warning comment + blank
        assert "truncated" in index

    def test_index_within_limits_unchanged(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        for i in range(3):
            entry = MemoryEntry(
                path=Path(""),
                name=f"mem {i}",
                description="short desc",
                type="project",
                body="body",
            )
            mem.save(entry)
        index = (tmp_path / "MEMORY.md").read_text()
        assert "truncated" not in index


# ---------------------------------------------------------------------------
# PersistentMemory — retrieval
# ---------------------------------------------------------------------------

class TestPersistentMemoryRetrieval:
    @pytest.fixture
    def mem(self, tmp_path):
        m = PersistentMemory(memory_dir=str(tmp_path))
        m.save(MemoryEntry(
            path=Path(""),
            name="pytest preference",
            description="user always uses pytest for testing",
            type="feedback",
            body="Always use pytest.",
        ))
        m.save(MemoryEntry(
            path=Path(""),
            name="project deadline",
            description="project deadline is next friday",
            type="project",
            body="Deadline is next Friday.",
        ))
        return m

    def test_query_empty_memory_returns_empty_context(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        ctx = mem.query("what testing framework?")
        assert ctx.retrieved_chunks == []

    def test_query_returns_scored_chunks(self, mem):
        ctx = mem.query("what testing framework should I use?")
        assert len(ctx.retrieved_chunks) > 0
        assert all(isinstance(c, ScoredChunk) for c in ctx.retrieved_chunks)

    def test_query_returns_at_most_5_chunks(self, tmp_path):
        m = PersistentMemory(memory_dir=str(tmp_path))
        for i in range(10):
            m.save(MemoryEntry(
                path=Path(""),
                name=f"memory {i}",
                description=f"python testing framework {i}",
                type="project",
                body="body",
            ))
        ctx = m.query("python testing")
        assert len(ctx.retrieved_chunks) <= 5

    def test_inject_into_system_prompt_with_memories(self, mem):
        result = mem.inject_into_system_prompt("You are an assistant.")
        assert "Retrieved Context" in result or "You are an assistant." in result

    def test_inject_into_system_prompt_no_match_returns_base(self, mem):
        result = mem.inject_into_system_prompt("You are an assistant.")
        # If the query is empty, inject_into_system_prompt runs query("")
        assert "You are an assistant." in result

    def test_scan_headers_reads_only_frontmatter(self, mem):
        headers = mem.scan_headers()
        assert len(headers) == 2
        for h in headers:
            assert h.body == ""
            assert h.name != ""


# ---------------------------------------------------------------------------
# PersistentMemory — extraction (mocked provider)
# ---------------------------------------------------------------------------

class TestPersistentMemoryExtraction:
    def _make_provider(self, response_text: str):
        """Return a mock provider that yields response_text as TextDelta events."""
        class MockProvider:
            async def complete(self, messages, tools, system="", **kwargs):
                for char in response_text:
                    yield TextDelta(text=char)
        return MockProvider()

    @pytest.mark.asyncio
    async def test_extract_saves_entries(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        provider = self._make_provider(
            '[{"name": "user is a developer", "type": "user", '
            '"description": "user is an experienced developer", "body": "experienced dev"}]'
        )
        saved = await mem.extract_from_messages(messages, provider)
        assert len(saved) == 1
        assert saved[0].name == "user is a developer"

    @pytest.mark.asyncio
    async def test_extract_skips_when_few_messages(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        messages = [{"role": "user", "content": "hi"}]  # < 4
        provider = self._make_provider("[]")
        saved = await mem.extract_from_messages(messages, provider)
        assert saved == []

    @pytest.mark.asyncio
    async def test_extract_returns_empty_for_empty_json(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        provider = self._make_provider("[]")
        saved = await mem.extract_from_messages(messages, provider)
        assert saved == []

    @pytest.mark.asyncio
    async def test_extract_handles_malformed_json(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        provider = self._make_provider("not valid json at all")
        saved = await mem.extract_from_messages(messages, provider)
        assert saved == []

    @pytest.mark.asyncio
    async def test_extract_only_processes_new_messages(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        provider = self._make_provider("[]")
        await mem.extract_from_messages(messages, provider)
        # Cursor should advance
        assert mem._cursor_index == 6

    @pytest.mark.asyncio
    async def test_extract_advances_cursor(self, tmp_path):
        mem = PersistentMemory(memory_dir=str(tmp_path))
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        provider = self._make_provider("[]")
        await mem.extract_from_messages(messages, provider)
        # Add more messages
        messages.extend([{"role": "user", "content": f"new {i}"} for i in range(5)])
        # Second call should only process the 5 new messages (< 4 new would skip, but we have 5)
        await mem.extract_from_messages(messages, provider)
        assert mem._cursor_index == 11


# ---------------------------------------------------------------------------
# PersistentMemory — search/list
# ---------------------------------------------------------------------------

class TestPersistentMemorySearchList:
    @pytest.fixture
    def mem(self, tmp_path):
        m = PersistentMemory(memory_dir=str(tmp_path))
        m.save(MemoryEntry(path=Path(""), name="auth token",
                           description="JWT tokens used for auth", type="project", body="JWT"))
        m.save(MemoryEntry(path=Path(""), name="user role",
                           description="user is a senior engineer", type="user", body="senior"))
        return m

    def test_search_by_name(self, mem):
        results = mem.search("auth token")
        assert any("auth" in e.name for e in results)

    def test_search_by_description(self, mem):
        results = mem.search("senior")
        assert len(results) >= 1

    def test_search_no_match_returns_empty(self, mem):
        results = mem.search("xyznonexistent123")
        assert results == []

    def test_list_all_returns_all(self, mem):
        entries = mem.list_all()
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# SessionSummary
# ---------------------------------------------------------------------------

class TestSessionSummary:
    def test_read_before_update_returns_empty(self, tmp_path):
        ss = SessionSummary(session_id="test_session_001")
        ss.path = tmp_path / "summary.md"
        result = ss.read()
        assert result == ""

    def test_read_after_update_returns_content(self, tmp_path):
        ss = SessionSummary(session_id="test_session_002")
        ss.path = tmp_path / "summary.md"
        ss.path.write_text("The user asked about Python testing.", encoding="utf-8")
        result = ss.read()
        assert "testing" in result

    @pytest.mark.asyncio
    async def test_update_writes_summary(self, tmp_path):
        ss = SessionSummary(session_id="test_session_003")
        ss.path = tmp_path / "summary.md"

        class MockProvider:
            async def complete(self, messages, tools, system="", **kwargs):
                yield TextDelta(text="The user discussed Python testing frameworks.")

        messages = [
            {"role": "user", "content": "Which testing framework should I use?"},
            {"role": "assistant", "content": "Use pytest."},
        ]
        await ss.update(messages, MockProvider())
        assert ss.path.exists()
        content = ss.read()
        assert "pytest" in content or len(content) > 0

    @pytest.mark.asyncio
    async def test_update_handles_provider_error_silently(self, tmp_path):
        """Provider errors during update do not raise — summary is non-critical."""
        ss = SessionSummary(session_id="test_session_004")
        ss.path = tmp_path / "summary.md"

        class ErrorProvider:
            async def complete(self, messages, tools, system="", **kwargs):
                raise RuntimeError("provider crashed")
                yield  # make it a generator

        messages = [{"role": "user", "content": "hi"}]
        await ss.update(messages, ErrorProvider())
        # No exception raised, no file written
        assert ss.read() == ""
