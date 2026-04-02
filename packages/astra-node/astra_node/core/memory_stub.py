"""StubMemory — no-op MemorySystem implementation.

Ships as the default memory backend so the framework works out-of-the-box
without any extra configuration. All methods are no-ops or return empty
results. The query engine defaults to StubMemory when memory=None is passed.
"""

from astra_node.core.memory_types import MemorySystem, QueryContext, UserProfile


class StubMemory(MemorySystem):
    """No-op MemorySystem. Framework works without memory extras installed."""

    def query(self, user_message: str) -> QueryContext:
        """Return an empty QueryContext — no memory to retrieve."""
        return QueryContext(retrieved_chunks=[], user_profile=None)

    def update(self, query: str, used_chunks: list[str]) -> None:
        """No-op — stub has no state to update."""
        pass

    def ingest(self, documents: list[str]) -> None:
        """No-op — stub discards all documents."""
        pass

    def get_user_context(self) -> UserProfile:
        """Return an empty UserProfile — stub has no user data."""
        return UserProfile()
