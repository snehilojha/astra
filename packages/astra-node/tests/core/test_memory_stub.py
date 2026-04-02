"""Tests for StubMemory — the no-op MemorySystem implementation."""

import pytest

from astra_node.core.memory_stub import StubMemory
from astra_node.core.memory_types import MemorySystem, QueryContext, UserProfile
from astra_node.core.query_engine import QueryEngine
from astra_node.core.registry import ToolRegistry
from astra_node.permissions.manager import PermissionManager


# ---------------------------------------------------------------------------
# StubMemory unit tests
# ---------------------------------------------------------------------------

class TestStubMemory:
    def test_is_memory_system(self):
        mem = StubMemory()
        assert isinstance(mem, MemorySystem)

    def test_query_returns_empty_query_context(self):
        mem = StubMemory()
        ctx = mem.query("what is the user's name?")
        assert isinstance(ctx, QueryContext)
        assert ctx.retrieved_chunks == []
        assert ctx.user_profile is None

    def test_query_empty_string(self):
        mem = StubMemory()
        ctx = mem.query("")
        assert ctx.retrieved_chunks == []

    def test_update_is_no_op(self):
        mem = StubMemory()
        mem.update("some query", ["chunk1", "chunk2"])  # must not raise

    def test_update_empty_chunks_no_op(self):
        mem = StubMemory()
        mem.update("query", [])  # must not raise

    def test_ingest_is_no_op(self):
        mem = StubMemory()
        mem.ingest(["doc1", "doc2"])  # must not raise

    def test_ingest_empty_list_no_op(self):
        mem = StubMemory()
        mem.ingest([])  # must not raise

    def test_get_user_context_returns_empty_profile(self):
        mem = StubMemory()
        profile = mem.get_user_context()
        assert isinstance(profile, UserProfile)
        assert profile.topics == {}

    def test_inject_into_system_prompt_returns_base_unchanged(self):
        """StubMemory has no chunks → inject_into_system_prompt returns base."""
        mem = StubMemory()
        base = "You are a helpful assistant."
        result = mem.inject_into_system_prompt(base)
        assert result == base

    def test_inject_empty_base(self):
        mem = StubMemory()
        result = mem.inject_into_system_prompt("")
        assert result == ""


# ---------------------------------------------------------------------------
# Integration: QueryEngine with StubMemory
# ---------------------------------------------------------------------------

class TestStubMemoryQueryEngineIntegration:
    """QueryEngine works with StubMemory (memory=None and explicit)."""

    def _make_provider(self, text: str):
        """Return a minimal mock provider that yields one text response."""
        from astra_node.providers.base import LLMProvider, LLMResponse, Usage
        from astra_node.core.events import TextDelta, UsageUpdate

        class SimpleProvider(LLMProvider):
            def __init__(self):
                self._last_response = LLMResponse(
                    content=text,
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=5, output_tokens=3),
                )

            async def complete(self, messages, tools, system="", **kwargs):
                yield TextDelta(text=text)
                yield UsageUpdate(input_tokens=5, output_tokens=3)

            @property
            def last_response(self):
                return self._last_response

        return SimpleProvider()

    @pytest.mark.asyncio
    async def test_query_engine_defaults_to_stub_memory(self):
        """QueryEngine with memory=None uses StubMemory internally."""
        engine = QueryEngine(
            provider=self._make_provider("hello"),
            registry=ToolRegistry(),
            permission_manager=PermissionManager(),
            memory=None,
        )
        assert isinstance(engine._memory, StubMemory)

    @pytest.mark.asyncio
    async def test_query_engine_explicit_stub_memory(self):
        """QueryEngine with explicit StubMemory works correctly."""
        from astra_node.core.events import TurnEnd

        engine = QueryEngine(
            provider=self._make_provider("ok"),
            registry=ToolRegistry(),
            permission_manager=PermissionManager(),
            memory=StubMemory(),
        )
        events = []
        async for event in engine.run("hi"):
            events.append(event)

        assert any(isinstance(e, TurnEnd) for e in events)

    @pytest.mark.asyncio
    async def test_query_engine_memory_none_works(self):
        """QueryEngine with memory=None completes a turn without error."""
        from astra_node.core.events import TurnEnd

        engine = QueryEngine(
            provider=self._make_provider("response"),
            registry=ToolRegistry(),
            permission_manager=PermissionManager(),
            memory=None,
        )
        events = []
        async for event in engine.run("test"):
            events.append(event)

        assert any(isinstance(e, TurnEnd) for e in events)
