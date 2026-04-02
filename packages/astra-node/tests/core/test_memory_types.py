"""Tests for MemorySystem ABC, ScoredChunk, UserProfile, QueryContext."""

import pytest

from astra_node.core.memory_types import (
    MemorySystem,
    QueryContext,
    ScoredChunk,
    UserProfile,
)


class TestScoredChunk:
    def test_creation(self):
        chunk = ScoredChunk(text="some memory", score=0.9)
        assert chunk.text == "some memory"
        assert chunk.score == 0.9

    def test_metadata_defaults_empty(self):
        chunk = ScoredChunk(text="x", score=0.5)
        assert chunk.metadata == {}

    def test_metadata_explicit(self):
        chunk = ScoredChunk(text="x", score=0.5, metadata={"type": "user"})
        assert chunk.metadata["type"] == "user"


class TestUserProfile:
    def test_creation_empty(self):
        profile = UserProfile()
        assert profile.topics == {}

    def test_creation_with_topics(self):
        profile = UserProfile(topics={"python": 1.0, "testing": 0.8})
        assert profile.topics["python"] == 1.0


class TestQueryContext:
    def test_render_no_chunks_returns_empty_string(self):
        ctx = QueryContext()
        assert ctx.render() == ""

    def test_render_with_chunks(self):
        ctx = QueryContext(
            retrieved_chunks=[
                ScoredChunk(text="user prefers pytest", score=0.9),
                ScoredChunk(text="project uses Python 3.11", score=0.7),
            ]
        )
        rendered = ctx.render()
        assert "## Retrieved Context" in rendered
        assert "user prefers pytest" in rendered
        assert "project uses Python 3.11" in rendered

    def test_render_single_chunk(self):
        ctx = QueryContext(
            retrieved_chunks=[ScoredChunk(text="fact", score=1.0)]
        )
        rendered = ctx.render()
        assert rendered.startswith("## Retrieved Context")
        assert "fact" in rendered

    def test_user_profile_defaults_none(self):
        ctx = QueryContext()
        assert ctx.user_profile is None

    def test_retrieved_chunks_default_empty(self):
        ctx = QueryContext()
        assert ctx.retrieved_chunks == []


class TestMemorySystemABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            MemorySystem()  # type: ignore[abstract]

    def test_concrete_without_methods_raises(self):
        class IncompleteMemory(MemorySystem):
            pass

        with pytest.raises(TypeError):
            IncompleteMemory()  # type: ignore[abstract]

    def test_concrete_with_all_methods_can_instantiate(self):
        class ConcreteMemory(MemorySystem):
            def query(self, user_message: str) -> QueryContext:
                return QueryContext()

            def update(self, query: str, used_chunks: list[str]) -> None:
                pass

            def ingest(self, documents: list[str]) -> None:
                pass

            def get_user_context(self) -> UserProfile:
                return UserProfile()

        mem = ConcreteMemory()
        assert mem is not None

    def test_inject_into_system_prompt_no_chunks_returns_base(self):
        """inject_into_system_prompt returns base unchanged when no chunks."""

        class EmptyMemory(MemorySystem):
            def query(self, user_message: str) -> QueryContext:
                return QueryContext()

            def update(self, query: str, used_chunks: list[str]) -> None:
                pass

            def ingest(self, documents: list[str]) -> None:
                pass

            def get_user_context(self) -> UserProfile:
                return UserProfile()

        mem = EmptyMemory()
        result = mem.inject_into_system_prompt("You are an assistant.")
        assert result == "You are an assistant."

    def test_inject_into_system_prompt_with_chunks_prepends(self):
        """inject_into_system_prompt prepends context when chunks exist."""

        class RichMemory(MemorySystem):
            def query(self, user_message: str) -> QueryContext:
                return QueryContext(
                    retrieved_chunks=[ScoredChunk(text="user prefers dark mode", score=1.0)]
                )

            def update(self, query: str, used_chunks: list[str]) -> None:
                pass

            def ingest(self, documents: list[str]) -> None:
                pass

            def get_user_context(self) -> UserProfile:
                return UserProfile()

        mem = RichMemory()
        result = mem.inject_into_system_prompt("Base prompt.")
        assert "Retrieved Context" in result
        assert "user prefers dark mode" in result
        assert "Base prompt." in result
        # Context comes before base
        assert result.index("Retrieved Context") < result.index("Base prompt.")
