"""PersistentMemory — file-based MemorySystem implementation.

Each memory is a markdown file with YAML-like frontmatter stored under
~/.astra/memory/. MEMORY.md is the index — always injected into the system
prompt. Individual topic files hold the full content.

The model reads/writes these files with the same tools it already has.
No extra dependencies — stdlib only (pathlib, re, datetime).

Storage layout:
    ~/.astra/memory/
        MEMORY.md                     ← index (always injected)
        user_<slug>.md                ← user-type memories
        feedback_<slug>.md            ← feedback-type memories
        project_<slug>.md             ← project-type memories
        reference_<slug>.md           ← reference-type memories
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astra_node.core.memory_types import (
    MemorySystem,
    QueryContext,
    ScoredChunk,
    UserProfile,
)

VALID_TYPES = {"user", "feedback", "project", "reference"}

# Hard limits matching Claude Code
_INDEX_MAX_LINES = 200
_INDEX_MAX_BYTES = 25 * 1024  # 25 KB


@dataclass
class MemoryEntry:
    """A single parsed memory file."""

    path: Path
    name: str
    description: str
    type: str
    body: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from a markdown file.

    Frontmatter is delimited by '---\\n' lines. Parses simple key: value
    pairs only — no nested YAML. Uses stdlib only (no PyYAML).

    Args:
        text: Raw file text.

    Returns:
        Tuple of (metadata_dict, body_text). Both are empty/empty if
        frontmatter is absent or malformed.
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    front = text[4:end]
    body = text[end + 5:]
    meta: dict[str, str] = {}
    for line in front.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body.strip()


def _slug(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:50].strip("_")


class PersistentMemory(MemorySystem):
    """File-based persistent memory. One markdown file per memory, MEMORY.md as index.

    Retrieval is keyword-based (description overlap) — no LLM call, no vector DB.
    Memory extraction from conversations is a secondary LLM call fired as a
    fire-and-forget task at turn end via the QueryEngine post_turn_hook.
    """

    def __init__(self, memory_dir: str = "~/.astra/memory/") -> None:
        """Initialise persistent memory storage.

        Args:
            memory_dir: Directory to store memory files. Created if absent.
        """
        self.memory_dir = Path(memory_dir).expanduser()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._last_extracted_message_id: str | None = None

    # ------------------------------------------------------------------
    # MemorySystem ABC
    # ------------------------------------------------------------------

    def query(self, user_message: str) -> QueryContext:
        """Scan memory file headers and return entries relevant to user_message.

        Uses word-overlap between the user message and each entry's description.
        No LLM call. Returns up to 5 most relevant chunks.

        Args:
            user_message: The incoming user message text.

        Returns:
            QueryContext with ranked ScoredChunks and optional UserProfile.
        """
        entries = self.scan_headers()
        if not entries:
            return QueryContext()

        query_words = set(user_message.lower().split())

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in entries:
            desc_words = set(entry.description.lower().split())
            if not desc_words:
                continue
            overlap = len(query_words & desc_words)
            if overlap > 0:
                score = overlap / len(desc_words | query_words)
                scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:5]

        chunks = [
            ScoredChunk(
                text=f"{entry.name}: {entry.description}",
                score=score,
                metadata={"type": entry.type, "path": str(entry.path)},
            )
            for score, entry in top
        ]

        profile = self.get_user_context()
        return QueryContext(retrieved_chunks=chunks, user_profile=profile)

    def inject_into_system_prompt(self, base: str) -> str:
        """Prepend MEMORY.md index to the system prompt.

        Overrides the ABC default (which calls query("") and returns nothing
        for an empty user message). PersistentMemory always injects the full
        MEMORY.md index so the model has its persistent context every turn.

        Args:
            base: The base system prompt text.

        Returns:
            System prompt with MEMORY.md prepended, or base if index is empty.
        """
        index_path = self.memory_dir / "MEMORY.md"
        if not index_path.exists():
            return base
        try:
            raw = index_path.read_text(encoding="utf-8").strip()
        except Exception:
            return base
        if not raw:
            return base
        return f"## Memory\n{raw}\n\n{base}"

    def update(self, query: str, used_chunks: list[str]) -> None:
        """No-op for v1. V2 will track access patterns (F-Monitor)."""
        pass

    def ingest(self, documents: list[str]) -> None:
        """Write documents as memory entries. Defaults to 'project' type.

        Args:
            documents: List of text strings to store as new memories.
        """
        for doc in documents:
            first_line = doc.splitlines()[0][:80] if doc.strip() else "ingested document"
            entry = MemoryEntry(
                path=Path(""),  # will be set by save()
                name=first_line,
                description=first_line,
                type="project",
                body=doc,
            )
            self.save(entry)

    def get_user_context(self) -> UserProfile:
        """Return UserProfile built from type='user' memory entries."""
        all_entries = self.load_all()
        user_entries = [e for e in all_entries if e.type == "user"]
        topics: dict[str, float] = {}
        for entry in user_entries:
            words = entry.description.lower().split()
            for word in words:
                if len(word) > 3:
                    topics[word] = topics.get(word, 0.0) + 1.0
        return UserProfile(topics=topics)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def save(self, entry: MemoryEntry) -> None:
        """Write a memory file and update the MEMORY.md index.

        Args:
            entry: MemoryEntry to persist. type must be in VALID_TYPES.

        Raises:
            ValueError: If entry.type is not in VALID_TYPES.
        """
        if entry.type not in VALID_TYPES:
            raise ValueError(
                f"Invalid memory type '{entry.type}'. Must be one of {sorted(VALID_TYPES)}."
            )
        slug = _slug(entry.name)
        filename = f"{entry.type}_{slug}.md"
        path = self.memory_dir / filename

        now = datetime.now(timezone.utc).isoformat()
        content = (
            f"---\n"
            f"name: {entry.name}\n"
            f"type: {entry.type}\n"
            f"description: {entry.description}\n"
            f"updated_at: {now}\n"
            f"---\n\n"
            f"{entry.body}\n"
        )
        path.write_text(content, encoding="utf-8")
        entry.path = path
        self.update_index()

    def load_all(self) -> list[MemoryEntry]:
        """Read all memory files. Skips files with invalid frontmatter.

        Returns:
            List of MemoryEntry objects sorted by updated_at descending.
        """
        entries: list[MemoryEntry] = []
        for md_file in self.memory_dir.glob("*.md"):
            if md_file.name == "MEMORY.md":
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                meta, body = parse_frontmatter(text)
                if not meta or "name" not in meta:
                    continue
                entry_type = meta.get("type", "project")
                if entry_type not in VALID_TYPES:
                    continue
                updated_str = meta.get("updated_at", "")
                try:
                    updated = datetime.fromisoformat(updated_str)
                except (ValueError, TypeError):
                    updated = datetime.now(timezone.utc)
                entries.append(
                    MemoryEntry(
                        path=md_file,
                        name=meta.get("name", ""),
                        description=meta.get("description", ""),
                        type=entry_type,
                        body=body,
                        updated_at=updated,
                    )
                )
            except Exception:
                continue

        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries

    def scan_headers(self) -> list[MemoryEntry]:
        """Read only the frontmatter of each file (fast — no body parsing).

        Returns:
            List of MemoryEntry objects with empty body.
        """
        entries: list[MemoryEntry] = []
        for md_file in self.memory_dir.glob("*.md"):
            if md_file.name == "MEMORY.md":
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                meta, _ = parse_frontmatter(text)
                if not meta or "name" not in meta:
                    continue
                entry_type = meta.get("type", "project")
                if entry_type not in VALID_TYPES:
                    continue
                entries.append(
                    MemoryEntry(
                        path=md_file,
                        name=meta.get("name", ""),
                        description=meta.get("description", ""),
                        type=entry_type,
                        body="",
                    )
                )
            except Exception:
                continue
        return entries

    def update_index(self) -> None:
        """Rebuild MEMORY.md from current memory files.

        Enforces 200-line / 25 KB hard limits — truncates with a warning
        comment if exceeded.
        """
        entries = self.load_all()
        index_path = self.memory_dir / "MEMORY.md"

        lines: list[str] = []
        for entry in entries:
            rel = entry.path.name
            lines.append(f"- [{entry.name}]({rel}) — {entry.description}")

        content = "\n".join(lines) + "\n" if lines else ""

        # Enforce line limit
        line_list = content.splitlines(keepends=True)
        if len(line_list) > _INDEX_MAX_LINES:
            line_list = line_list[:_INDEX_MAX_LINES]
            line_list.append("<!-- truncated: too many entries -->\n")
            content = "".join(line_list)

        # Enforce byte limit
        encoded = content.encode("utf-8")
        if len(encoded) > _INDEX_MAX_BYTES:
            while len(content.encode("utf-8")) > _INDEX_MAX_BYTES:
                lines_remaining = content.splitlines(keepends=True)
                if not lines_remaining:
                    break
                lines_remaining = lines_remaining[:-1]
                content = "".join(lines_remaining)
            content += "<!-- truncated: index too large -->\n"

        index_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract_from_messages(
        self,
        messages: list[dict],
        provider,
        since_message_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Secondary LLM call: extract memories from conversation messages.

        Processes only messages after since_message_id (cursor-based).
        Skips if fewer than 4 new messages since last extraction.
        Fires the LLM with the extraction prompt (no tools, max 3 turns).

        Args:
            messages: Full conversation history.
            provider: An LLMProvider instance used for the extraction call.
            since_message_id: ID of the last processed message (cursor).
                              Not used in v1 (simple index-based cursor).

        Returns:
            List of saved MemoryEntry objects (empty if nothing extracted).
        """
        # v1 cursor: use message count as a simple cursor
        cursor = getattr(self, "_cursor_index", 0)
        new_messages = messages[cursor:]

        if len(new_messages) < 4:
            return []

        extraction_prompt = (
            "You are extracting memories from a conversation.\n\n"
            "SAVE memories of type:\n"
            "- user: facts about the person (role, expertise, preferences)\n"
            "- feedback: corrections they made or approaches they confirmed\n"
            "- project: decisions, deadlines, incidents — things NOT in the code\n"
            "- reference: pointers to external systems (URLs, tool names, channels)\n\n"
            "DO NOT SAVE:\n"
            "- Code patterns or architecture — these are in the code\n"
            "- File paths or function names — these are in the code\n"
            "- Git history or recent changes — git log is authoritative\n"
            "- In-progress work or current task state — ephemeral\n\n"
            "Return JSON only:\n"
            '[\n  {\n    "name": "short memorable name",\n'
            '    "type": "user|feedback|project|reference",\n'
            '    "description": "one line, ~100 chars",\n'
            '    "body": "full memory text"\n  }\n]\n\n'
            "Return [] if nothing is worth saving."
        )

        # Build a minimal message list for the extraction call
        convo_text = "\n".join(
            f"{m.get('role', 'user')}: {_extract_text(m)}"
            for m in new_messages
        )
        extraction_messages = [
            {"role": "user", "content": f"Conversation to analyse:\n\n{convo_text}"}
        ]

        collected_text = ""
        try:
            async for event in provider.complete(
                messages=extraction_messages,
                tools=[],
                system=extraction_prompt,
            ):
                from astra_node.core.events import TextDelta
                if isinstance(event, TextDelta):
                    collected_text += event.text
        except Exception:
            return []

        # Parse the JSON response
        try:
            # Strip markdown fences if present
            text = collected_text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            raw_entries = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(raw_entries, list):
            return []

        saved: list[MemoryEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            entry_type = raw.get("type", "project")
            if entry_type not in VALID_TYPES:
                continue
            entry = MemoryEntry(
                path=Path(""),
                name=raw.get("name", "extracted memory")[:80],
                description=raw.get("description", "")[:200],
                type=entry_type,
                body=raw.get("body", ""),
            )
            try:
                self.save(entry)
                saved.append(entry)
            except ValueError:
                continue

        # Advance cursor
        self._cursor_index = len(messages)
        self._last_extracted_message_id = str(len(messages))
        return saved

    # ------------------------------------------------------------------
    # CLI utilities
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[MemoryEntry]:
        """Search by keyword across name, description, and body.

        Args:
            query: Search keyword(s).

        Returns:
            List of matching MemoryEntry objects.
        """
        lower_query = query.lower()
        results = []
        for entry in self.load_all():
            if (
                lower_query in entry.name.lower()
                or lower_query in entry.description.lower()
                or lower_query in entry.body.lower()
            ):
                results.append(entry)
        return results

    def list_all(self) -> list[MemoryEntry]:
        """List all entries sorted by updated_at descending."""
        return self.load_all()

    def clear(self) -> None:
        """Delete all memory files and reset MEMORY.md."""
        for md_file in self.memory_dir.glob("*.md"):
            md_file.unlink(missing_ok=True)
        index_path = self.memory_dir / "MEMORY.md"
        index_path.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(message: dict) -> str:
    """Extract plain text from a message dict (handles content blocks)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)
