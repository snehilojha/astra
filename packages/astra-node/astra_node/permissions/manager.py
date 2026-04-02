"""Permission manager — decides whether a tool call may proceed.

The manager combines the tool's declared PermissionLevel with any
session-level overrides set by the user (allow_always / deny_always)
and optional persistent configuration loaded from YAML.

Decision priority (highest first):
  1. Session deny override  → DENY
  2. Session allow override → ALLOW
  3. Tool's PermissionLevel.DENY → DENY
  4. Tool's PermissionLevel.ALWAYS_ALLOW → ALLOW
  5. Tool's PermissionLevel.ASK_USER → ASK
"""

from pathlib import Path
from typing import Any

import yaml

from astra_node.core.tool import PermissionLevel
from astra_node.permissions.types import PermissionDecision


class PermissionManager:
    """Evaluates whether a tool invocation should proceed.

    Session overrides are in-memory only — they are never written back to
    the YAML config file. This matches Claude Code's behaviour: user grants
    within a session don't persist by default.
    """

    def __init__(self) -> None:
        self._session_allow: set[str] = set()
        self._session_deny: set[str] = set()
        # Persistent overrides from YAML (tool_name -> "allow" | "deny")
        self._persistent: dict[str, str] = {}

    def check(self, tool_name: str, tool_input: dict[str, Any] | None = None) -> PermissionDecision:
        """Determine whether a tool call should proceed.

        Args:
            tool_name: The name attribute of the tool being invoked.
            tool_input: The validated input dict (reserved for future
                        input-level permission rules).

        Returns:
            PermissionDecision.ALLOW, .ASK, or .DENY.
        """
        # Session-level deny overrides everything
        if tool_name in self._session_deny:
            return PermissionDecision.DENY

        # Session-level allow beats everything else
        if tool_name in self._session_allow:
            return PermissionDecision.ALLOW

        # Persistent config from YAML
        if tool_name in self._persistent:
            decision = self._persistent[tool_name]
            if decision == "allow":
                return PermissionDecision.ALLOW
            if decision == "deny":
                return PermissionDecision.DENY

        # Fall through to tool's declared permission level.
        # The manager doesn't hold tool objects — callers must resolve
        # the tool's PermissionLevel and pass it via check_level() when
        # they need that path. check() is the fast path for the query engine
        # which passes the tool name + already-resolved level via check_level().
        return PermissionDecision.ASK

    def check_level(
        self,
        tool_name: str,
        level: PermissionLevel,
        tool_input: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """Determine permission given the tool's declared PermissionLevel.

        This is the primary method used by the query engine. It resolves
        session overrides first, then falls back to the tool's level.

        Args:
            tool_name: The tool's name attribute.
            level: The tool's declared PermissionLevel.
            tool_input: The validated input dict (reserved for future use).

        Returns:
            PermissionDecision.ALLOW, .ASK, or .DENY.
        """
        # Session-level deny overrides everything
        if tool_name in self._session_deny:
            return PermissionDecision.DENY

        # Session-level allow beats everything else (including tool-level DENY)
        if tool_name in self._session_allow:
            return PermissionDecision.ALLOW

        # Persistent config from YAML
        if tool_name in self._persistent:
            decision = self._persistent[tool_name]
            if decision == "allow":
                return PermissionDecision.ALLOW
            if decision == "deny":
                return PermissionDecision.DENY

        # Fall back to the tool's declared level
        if level == PermissionLevel.DENY:
            return PermissionDecision.DENY
        if level == PermissionLevel.ALWAYS_ALLOW:
            return PermissionDecision.ALLOW
        # PermissionLevel.ASK_USER
        return PermissionDecision.ASK

    def allow_always(self, tool_name: str) -> None:
        """Grant permanent-for-session ALLOW for a tool.

        Removes any session deny for the same tool so the grants are
        consistent.

        Args:
            tool_name: Tool name to unconditionally allow for this session.
        """
        self._session_allow.add(tool_name)
        self._session_deny.discard(tool_name)

    def deny_always(self, tool_name: str) -> None:
        """Grant permanent-for-session DENY for a tool.

        Args:
            tool_name: Tool name to unconditionally deny for this session.
        """
        self._session_deny.add(tool_name)
        self._session_allow.discard(tool_name)

    def load_from_yaml(self, path: Path | str) -> None:
        """Load persistent permission overrides from a YAML file.

        Expected format:
            permissions:
              bash: deny
              file_read: allow

        Args:
            path: Path to the YAML config file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is malformed or contains invalid values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Permission config file not found: {path}"
            )
        with path.open() as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid permission config in {path}: expected a YAML mapping.")

        perms = raw.get("permissions", {})
        if not isinstance(perms, dict):
            raise ValueError(
                f"Invalid 'permissions' key in {path}: expected a mapping of tool_name -> allow|deny."
            )

        valid_values = {"allow", "deny"}
        for tool_name, value in perms.items():
            if value not in valid_values:
                raise ValueError(
                    f"Invalid permission value '{value}' for tool '{tool_name}' in {path}. "
                    f"Expected 'allow' or 'deny'."
                )
            self._persistent[tool_name] = value
