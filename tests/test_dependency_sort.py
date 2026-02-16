"""
Tests for plugin dependency validation and topological sorting.

Tests:
- Missing dependency raises ValueError
- Circular dependency raises ValueError
- Topological sort produces correct order
- Plugins without depends_on are unaffected
- Short name resolution works with FQ names in PluginConfig
"""
import pytest

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.config import PluginConfig
from fwbg.pipeline.registry import PluginRegistry
from fwbg.pipeline.runner import PipelineRunner, _topological_sort, _short_name


# === Test Plugins ===

class PluginA(BasePlugin):
    name = "plugin_a"
    phase = PluginPhase.INDICATORS
    depends_on = []

    def execute(self, ctx, **params):
        return ctx


class PluginB(BasePlugin):
    name = "plugin_b"
    phase = PluginPhase.INDICATORS
    depends_on = ["plugin_a"]

    def execute(self, ctx, **params):
        return ctx


class PluginC(BasePlugin):
    name = "plugin_c"
    phase = PluginPhase.INDICATORS
    depends_on = ["plugin_b"]

    def execute(self, ctx, **params):
        return ctx


class PluginCycleX(BasePlugin):
    name = "cycle_x"
    phase = PluginPhase.INDICATORS
    depends_on = ["cycle_y"]

    def execute(self, ctx, **params):
        return ctx


class PluginCycleY(BasePlugin):
    name = "cycle_y"
    phase = PluginPhase.INDICATORS
    depends_on = ["cycle_x"]

    def execute(self, ctx, **params):
        return ctx


class PluginNoDeps(BasePlugin):
    name = "no_deps"
    phase = PluginPhase.INDICATORS

    def execute(self, ctx, **params):
        return ctx


# === Fixtures ===

@pytest.fixture
def registry():
    reg = PluginRegistry()
    reg.register(PluginA, "test")
    reg.register(PluginB, "test")
    reg.register(PluginC, "test")
    reg.register(PluginNoDeps, "test")
    reg.register(PluginCycleX, "test")
    reg.register(PluginCycleY, "test")
    return reg


# === Tests ===

class TestShortName:
    def test_fq_name(self):
        assert _short_name("fwbg-premium:regime") == "regime"

    def test_already_short(self):
        assert _short_name("regime") == "regime"


class TestTopologicalSort:
    def test_no_dependencies(self, registry):
        configs = [
            PluginConfig(name="test:no_deps", params={}),
            PluginConfig(name="test:plugin_a", params={}),
        ]
        result = _topological_sort(configs, registry)
        assert len(result) == 2

    def test_simple_dependency_order(self, registry):
        """B depends on A → A must come first."""
        configs = [
            PluginConfig(name="test:plugin_b", params={}),
            PluginConfig(name="test:plugin_a", params={}),
        ]
        result = _topological_sort(configs, registry)
        names = [_short_name(c.name) for c in result]
        assert names.index("plugin_a") < names.index("plugin_b")

    def test_chain_dependency(self, registry):
        """C → B → A must produce A, B, C order."""
        configs = [
            PluginConfig(name="test:plugin_c", params={}),
            PluginConfig(name="test:plugin_a", params={}),
            PluginConfig(name="test:plugin_b", params={}),
        ]
        result = _topological_sort(configs, registry)
        names = [_short_name(c.name) for c in result]
        assert names.index("plugin_a") < names.index("plugin_b")
        assert names.index("plugin_b") < names.index("plugin_c")

    def test_missing_dependency_raises(self, registry):
        """B depends on A, but A is not in the pipeline."""
        configs = [
            PluginConfig(name="test:plugin_b", params={}),
        ]
        with pytest.raises(ValueError, match="depends on 'plugin_a'"):
            _topological_sort(configs, registry)

    def test_circular_dependency_raises(self, registry):
        """Cycle X ↔ Y must raise ValueError."""
        configs = [
            PluginConfig(name="test:cycle_x", params={}),
            PluginConfig(name="test:cycle_y", params={}),
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            _topological_sort(configs, registry)

    def test_empty_list(self, registry):
        result = _topological_sort([], registry)
        assert result == []

    def test_no_depends_on_unchanged(self, registry):
        """Plugins without depends_on preserve their relative order."""
        configs = [
            PluginConfig(name="test:no_deps", params={}),
            PluginConfig(name="test:plugin_a", params={}),
        ]
        result = _topological_sort(configs, registry)
        assert len(result) == 2
        # Both have no incoming edges, so original order is preserved by Kahn's
        names = [_short_name(c.name) for c in result]
        assert "no_deps" in names
        assert "plugin_a" in names


class TestPipelineRunnerDependencies:
    def test_runner_validates_on_initialize(self, registry):
        """Runner raises ValueError for missing dependency during init."""
        from fwbg.pipeline.config import PipelineConfig
        config = PipelineConfig(
            indicators=[PluginConfig(name="test:plugin_b", params={})]
        )
        runner = PipelineRunner(registry, config)
        with pytest.raises(ValueError, match="depends on 'plugin_a'"):
            runner._initialize()

    def test_runner_sorts_correctly(self, registry):
        """Runner sorts plugins with dependencies in correct order."""
        from fwbg.pipeline.config import PipelineConfig
        config = PipelineConfig(
            indicators=[
                PluginConfig(name="test:plugin_b", params={}),
                PluginConfig(name="test:plugin_a", params={}),
            ]
        )
        runner = PipelineRunner(registry, config)
        runner._initialize()
        names = [_short_name(c.name) for c, _ in runner._execution_order]
        assert names.index("plugin_a") < names.index("plugin_b")
