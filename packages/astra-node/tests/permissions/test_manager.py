"""Tests for PermissionManager and PermissionDecision."""

import pytest
from pathlib import Path

from astra_node.core.tool import PermissionLevel
from astra_node.permissions.types import PermissionDecision
from astra_node.permissions.manager import PermissionManager


class TestPermissionDecision:
    def test_has_three_values(self):
        assert len(PermissionDecision) == 3

    def test_values(self):
        assert PermissionDecision.ALLOW.value == "allow"
        assert PermissionDecision.ASK.value == "ask"
        assert PermissionDecision.DENY.value == "deny"


class TestPermissionManagerCheckLevel:
    def test_always_allow_tool_returns_allow(self):
        mgr = PermissionManager()
        result = mgr.check_level("file_read", PermissionLevel.ALWAYS_ALLOW)
        assert result == PermissionDecision.ALLOW

    def test_ask_user_tool_returns_ask(self):
        mgr = PermissionManager()
        result = mgr.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.ASK

    def test_deny_tool_returns_deny(self):
        mgr = PermissionManager()
        result = mgr.check_level("blocked_tool", PermissionLevel.DENY)
        assert result == PermissionDecision.DENY


class TestPermissionManagerSessionOverrides:
    def test_allow_always_overrides_ask_user(self):
        mgr = PermissionManager()
        mgr.allow_always("bash")
        result = mgr.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.ALLOW

    def test_allow_always_overrides_deny(self):
        """Session allow even overrides a DENY-level tool."""
        mgr = PermissionManager()
        mgr.allow_always("blocked_tool")
        result = mgr.check_level("blocked_tool", PermissionLevel.DENY)
        assert result == PermissionDecision.ALLOW

    def test_deny_always_overrides_always_allow(self):
        mgr = PermissionManager()
        mgr.deny_always("file_read")
        result = mgr.check_level("file_read", PermissionLevel.ALWAYS_ALLOW)
        assert result == PermissionDecision.DENY

    def test_deny_always_overrides_ask_user(self):
        mgr = PermissionManager()
        mgr.deny_always("bash")
        result = mgr.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.DENY

    def test_allow_then_deny_results_in_deny(self):
        """deny_always called after allow_always replaces the allow."""
        mgr = PermissionManager()
        mgr.allow_always("bash")
        mgr.deny_always("bash")
        result = mgr.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.DENY

    def test_deny_then_allow_results_in_allow(self):
        """allow_always called after deny_always replaces the deny."""
        mgr = PermissionManager()
        mgr.deny_always("bash")
        mgr.allow_always("bash")
        result = mgr.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.ALLOW

    def test_session_overrides_do_not_affect_other_tools(self):
        mgr = PermissionManager()
        mgr.allow_always("bash")
        result = mgr.check_level("file_write", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.ASK


class TestPermissionManagerLoadFromYAML:
    def test_load_sets_allow(self, tmp_path):
        config = tmp_path / "permissions.yaml"
        config.write_text("permissions:\n  file_read: allow\n")
        mgr = PermissionManager()
        mgr.load_from_yaml(config)
        result = mgr.check_level("file_read", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.ALLOW

    def test_load_sets_deny(self, tmp_path):
        config = tmp_path / "permissions.yaml"
        config.write_text("permissions:\n  bash: deny\n")
        mgr = PermissionManager()
        mgr.load_from_yaml(config)
        result = mgr.check_level("bash", PermissionLevel.ALWAYS_ALLOW)
        assert result == PermissionDecision.DENY

    def test_load_missing_file_raises(self, tmp_path):
        mgr = PermissionManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_from_yaml(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        config = tmp_path / "permissions.yaml"
        config.write_text("permissions:\n  bash: maybe\n")  # invalid value
        mgr = PermissionManager()
        with pytest.raises(ValueError, match="maybe"):
            mgr.load_from_yaml(config)

    def test_load_non_mapping_raises(self, tmp_path):
        config = tmp_path / "permissions.yaml"
        config.write_text("- just a list\n")
        mgr = PermissionManager()
        with pytest.raises(ValueError, match="mapping"):
            mgr.load_from_yaml(config)

    def test_session_overrides_take_precedence_over_yaml(self, tmp_path):
        """Session deny beats YAML allow."""
        config = tmp_path / "permissions.yaml"
        config.write_text("permissions:\n  bash: allow\n")
        mgr = PermissionManager()
        mgr.load_from_yaml(config)
        mgr.deny_always("bash")
        result = mgr.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.DENY

    def test_session_overrides_not_persisted_to_yaml(self, tmp_path):
        """allow_always/deny_always only live in memory."""
        config = tmp_path / "permissions.yaml"
        config.write_text("permissions:\n  bash: deny\n")
        mgr = PermissionManager()
        mgr.load_from_yaml(config)
        mgr.allow_always("bash")

        # Create a fresh manager from the same file — session override is gone
        mgr2 = PermissionManager()
        mgr2.load_from_yaml(config)
        result = mgr2.check_level("bash", PermissionLevel.ASK_USER)
        assert result == PermissionDecision.DENY
