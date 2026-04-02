"""MemorySystem ABC and associated data types.

This module defines the interface that all memory implementations must
satisfy. The query engine depends only on this ABC — never on a concrete
implementation. This lets StubMemory (no-op), PersistentMemory (file-based),
and future AdaptiveMemory (vector DB) drop in without changing any agent code.

Design mirrors Claude Code's memoryTypes.ts — 4-type taxonomy, scored
retrieval results, and a user profile for personalisation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ScoredChunk:
    """A memory fragment returned from a retrieval query.

    score indicates relevance (higher is more relevant). metadata carries
    optional provenance information (file path, type, update time, etc.).
    """

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class UserProfile:
    """Lightweight model of the user built from 'user'-type memories.

    topics maps topic strings to relative importance weights. Used by the
    adaptive memory system (v2) to bias retrieval. In v1 (PersistentMemory)
    it is populated from the 'user' memory files but weights are uniform.
    """

    topics: dict[str, float] = field(default_factory=dict)


@dataclass
class QueryContext:
    """The result of a memory query — injected into the system prompt.

    retrieved_chunks: ranked list of relevant memory fragments.
    user_profile: personalisation data for the current user.
    """

    retrieved_chunks: list[ScoredChunk] = field(default_factory=list)
    user_profile: UserProfile | None = None

    def render(self) -> str:
        """Format retrieved chunks as a markdown section for prompt injection.

        Returns an empty string if there are no retrieved chunks, so callers
        can safely append this to a system prompt without extra blank lines.
        """
        if not self.retrieved_chunks:
            return ""
        chunks = "\n".join(f"- {c.text}" for c in self.retrieved_chunks)
        return f"## Retrieved Context\n{chunks}"


class MemorySystem(ABC):
    """Abstract base for all memory backends.

    The query engine calls query() at the start of each turn to retrieve
    relevant context, and update() at the end to record which chunks were
    actually used. ingest() is called to write new memories.

    inject_into_system_prompt() is a convenience method with a default
    implementation — subclasses only need to implement the four abstract
    methods.
    """

    @abstractmethod
    def query(self, user_message: str) -> QueryContext:
        """Retrieve memory chunks relevant to the user's message.

        Args:
            user_message: The incoming user message text.

        Returns:
            QueryContext with ranked chunks and optional user profile.
        """
        ...

    @abstractmethod
    def update(self, query: str, used_chunks: list[str]) -> None:
        """Record which chunks were used in a response.

        Called at turn end. Implementations may use this to update access
        frequency weights or recency scores.

        Args:
            query: The original user message.
            used_chunks: Text of the chunks that appeared in the response.
        """
        ...

    @abstractmethod
    def ingest(self, documents: list[str]) -> None:
        """Write new documents into the memory store.

        Args:
            documents: List of text strings to store as new memories.
        """
        ...

    @abstractmethod
    def get_user_context(self) -> UserProfile:
        """Return the current user profile built from stored memories.

        Returns:
            UserProfile — may be empty if no user-type memories exist.
        """
        ...

    def inject_into_system_prompt(self, base: str) -> str:
        """Prepend relevant memory context to the system prompt.

        Queries with an empty string to get general context (index-level),
        then prepends it before the base system prompt. Returns base
        unchanged if there are no retrieved chunks.

        Args:
            base: The base system prompt text.

        Returns:
            Enriched system prompt with memory context prepended, or base
            unchanged if nothing relevant was found.
        """
        ctx = self.query("")
        rendered = ctx.render()
        if not rendered:
            return base
        return f"{rendered}\n\n{base}"
