# tests/pipeline/test_user_plugins.py
"""Tests for user plugin directory discovery."""
import pytest
import json
from pathlib import Path

from fwbg.pipeline.registry import (
    PluginRegistry,
    get_user_plugins_dir,
    get_core_plugins_dir,
)


class TestUserPluginDiscovery:
    """Tests for user plugin directory functions."""

    def test_get_user_plugins_dir(self):
        """Should return path in user home directory."""
        user_dir = get_user_plugins_dir()
        assert "fwbg" in str(user_dir).lower()
        assert "plugins" in str(user_dir).lower()
        assert user_dir.parent.name == ".fwbg"

    def test_get_core_plugins_dir(self):
        """Should return path to builtins directory."""
        core_dir = get_core_plugins_dir()
        assert core_dir.name == "builtins"
        assert core_dir.exists()


class TestAutoDiscover:
    """Tests for auto_discover functionality."""

    def test_registry_has_auto_discover(self):
        """Registry should have auto_discover method."""
        registry = PluginRegistry()
        assert hasattr(registry, "auto_discover")
        assert callable(registry.auto_discover)

    def test_auto_discover_finds_core_plugins(self):
        """auto_discover should find core indicator plugins."""
        registry = PluginRegistry()
        discovered = registry.auto_discover()

        # Should find at least some core plugins
        assert len(discovered) > 0

        # Should include known core indicators
        assert "trend" in discovered
        assert "momentum" in discovered


class TestCustomUserPlugin:
    """Tests for custom user plugins."""

    def test_user_plugin_loaded(self, tmp_path):
        """User plugins should be loaded and available."""
        # Create a mock user plugin
        pkg_dir = tmp_path / "my_custom_plugins"
        pkg_dir.mkdir()

        # Manifest
        manifest = {
            "name": "my_custom_plugins",
            "version": "1.0.0",
            "description": "Custom plugin for testing",
            "phase": "indicators",
            "author": "test",
            "dependencies": [],
        }
        (pkg_dir / "manifest.json").write_text(json.dumps(manifest))

        # Plugin code
        plugin_code = '''
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext


class MyCustomIndicator(BasePlugin):
    name = "my_custom_indicator"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True

    def execute(self, ctx, **params):
        ctx.df["custom"] = 42
        return ctx

    def validate(self):
        return True
'''
        (pkg_dir / "__init__.py").write_text(plugin_code)

        # Discover
        registry = PluginRegistry()
        discovered = registry.discover_from_directory(tmp_path)

        assert "my_custom_indicator" in discovered

        # Should be usable
        plugin_cls = registry.get("my_custom_indicator")
        assert plugin_cls.name == "my_custom_indicator"
        assert plugin_cls.version == "1.0.0"
        assert plugin_cls.phase == PluginPhase.INDICATORS


# Import PluginPhase for test
from fwbg.pipeline.base import PluginPhase
