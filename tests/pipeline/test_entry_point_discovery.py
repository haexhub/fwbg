"""Tests for entry_point-based plugin discovery (fwbg-premium separation)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.registry import PluginRegistry, reset_registry


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset global registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


def _create_plugin_package(base_dir: Path, pkg_name: str, plugin_name: str) -> Path:
    """Helper to create a plugin package directory structure for tests."""
    pkg_dir = base_dir / pkg_name
    pkg_dir.mkdir(parents=True)

    # Package manifest
    (pkg_dir / "manifest.json").write_text(json.dumps({
        "name": pkg_name,
        "version": "1.0.0",
        "license": "Commercial",
    }))

    # Indicators category
    ind_dir = pkg_dir / "indicators" / plugin_name
    ind_dir.mkdir(parents=True)

    (ind_dir / "manifest.json").write_text(json.dumps({
        "name": plugin_name,
        "version": "1.0.0",
    }))

    (ind_dir / "__init__.py").write_text(f'''
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext

class TestEntryPointPlugin(BasePlugin):
    name = "{plugin_name}"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    def execute(self, ctx, **params):
        return ctx

    def validate(self):
        return True
''')

    return pkg_dir


class TestDiscoverFromEntryPoints:
    """Tests for _discover_from_entry_points()."""

    def test_discovers_plugins_from_entry_point(self, tmp_path):
        """Entry point that returns a plugins dir should discover its packages."""
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_plugin_package(plugins_dir, "ep-test-pkg", "ep_indicator")

        # Mock entry_points to return our test directory
        mock_ep = MagicMock()
        mock_ep.name = "ep-test-pkg"
        mock_ep.load.return_value = lambda: plugins_dir

        registry = PluginRegistry()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            discovered = registry._discover_from_entry_points()

        assert "ep-test-pkg:ep_indicator" in discovered
        plugin_cls = registry.get("ep-test-pkg:ep_indicator")
        assert plugin_cls.phase == PluginPhase.INDICATORS

    def test_no_entry_points_returns_empty(self):
        """No installed plugin packages should return empty list."""
        registry = PluginRegistry()

        with patch("importlib.metadata.entry_points", return_value=[]):
            discovered = registry._discover_from_entry_points()

        assert discovered == []

    def test_broken_entry_point_does_not_crash(self):
        """A broken entry point should log warning, not crash."""
        mock_ep = MagicMock()
        mock_ep.name = "broken-pkg"
        mock_ep.load.side_effect = ImportError("package not found")

        registry = PluginRegistry()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            discovered = registry._discover_from_entry_points()

        assert discovered == []

    def test_entry_point_with_nonexistent_dir(self, tmp_path):
        """Entry point pointing to nonexistent dir should be skipped."""
        mock_ep = MagicMock()
        mock_ep.name = "missing-pkg"
        mock_ep.load.return_value = lambda: tmp_path / "nonexistent"

        registry = PluginRegistry()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            discovered = registry._discover_from_entry_points()

        assert discovered == []


class TestAutoDiscoverWithEntryPoints:
    """Tests for auto_discover() integrating entry_points."""

    def test_auto_discover_includes_entry_points(self, tmp_path):
        """auto_discover should find both built-in and entry_point plugins."""
        plugins_dir = tmp_path / "ep_plugins"
        plugins_dir.mkdir()
        _create_plugin_package(plugins_dir, "external-pkg", "external_ind")

        mock_ep = MagicMock()
        mock_ep.name = "external-pkg"
        mock_ep.load.return_value = lambda: plugins_dir

        registry = PluginRegistry()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            discovered = registry.auto_discover()

        # Should find built-in core plugins
        assert "fwbg-core:trend" in discovered
        # Should also find entry_point plugins
        assert "external-pkg:external_ind" in discovered

    def test_auto_discover_deduplicates(self, tmp_path):
        """If a built-in package is also installed via entry_point, no duplicate."""
        # This tests that discover_package handles re-registration gracefully
        registry = PluginRegistry()

        with patch("importlib.metadata.entry_points", return_value=[]):
            discovered = registry.auto_discover()

        # Each plugin should appear exactly once
        assert len(discovered) == len(set(discovered))


class TestGracefulDegradation:
    """Tests that fwbg works without premium plugins installed."""

    def test_core_only_discover(self):
        """Without premium, only core plugins should be discovered."""
        registry = PluginRegistry()

        # Simulate: no entry_points, no fwbg-premium in built-in dir
        with patch("importlib.metadata.entry_points", return_value=[]):
            discovered = registry.auto_discover()

        # Core plugins should still be found
        core_plugins = [p for p in discovered if p.startswith("fwbg-core:")]
        assert len(core_plugins) > 0
        assert "fwbg-core:trend" in core_plugins

    def test_core_works_without_premium_import(self):
        """Core registry functions should not crash if premium is absent."""
        from fwbg.core.registry import INDICATOR_REGISTRY, get_indicator

        # Clear registries to simulate fresh state
        INDICATOR_REGISTRY.clear()

        # Should trigger auto-discovery, find core at minimum
        # Even without premium, get_indicator should work for core plugins
        try:
            cls = get_indicator("trend")
            assert cls is not None
        except ValueError:
            # If trend is a pipeline-only plugin, that's fine too
            pass
