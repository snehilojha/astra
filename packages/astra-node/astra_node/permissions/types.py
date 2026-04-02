"""Permission decision types.

PermissionLevel lives in core/tool.py (alongside BaseTool) because tools
declare their own permission level. This module holds the decision enum
returned by PermissionManager.check() — kept separate so consumers can
import just the decision without pulling in tool machinery.
"""

from enum import Enum


class PermissionDecision(Enum):
    """The outcome of a permission check.

    ALLOW: Proceed with the tool call immediately.
    ASK:   Pause and prompt the user before proceeding.
    DENY:  Refuse the tool call outright; send error back to LLM.
    """

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"
