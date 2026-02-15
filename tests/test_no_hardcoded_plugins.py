"""
Tests that production code uses plugin registry, not hardcoded logic.

Verifies:
1. targets.py dispatches to all exit strategies via registry (no if/else branching)
2. No hardcoded exit strategy or risk management defaults in config/context
3. FixedExitStrategy supports return_durations (same interface as atr_based)
"""
import inspect
import pytest


class TestTargetsNoHardcodedBranching:
    """targets.py must not branch on exit strategy names."""

    def test_no_atr_based_string_check(self):
        """targets.py must not have 'if exit_strategy_mode == \"atr_based\"'."""
        from fwbg.optimization import targets
        source = inspect.getsource(targets)
        assert '== "atr_based"' not in source, (
            "targets.py still branches on 'atr_based' string"
        )

    def test_no_inline_numba_for_fixed(self):
        """targets.py must not call compute_targets_numba directly."""
        from fwbg.optimization import targets
        source = inspect.getsource(targets.compute_targets_cached)
        assert "compute_targets_numba(" not in source, (
            "targets.py still calls compute_targets_numba inline "
            "instead of going through FixedExitStrategy plugin"
        )

    def test_uses_get_strategy(self):
        """targets.py must use get_strategy() for all exit strategies."""
        from fwbg.optimization import targets
        source = inspect.getsource(targets.compute_targets_cached)
        assert "get_strategy(" in source


class TestFixedExitStrategyInterface:
    """FixedExitStrategy must support same interface as AtrExitStrategy."""

    def test_supports_params_kwarg(self):
        """FixedExitStrategy.compute_targets must accept params=GridParams."""
        from fwbg.plugins import import_plugin_module
        mod = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
        sig = inspect.signature(mod.FixedExitStrategy.compute_targets)
        param_names = list(sig.parameters.keys())
        assert "params" in param_names or "kwargs" in param_names

    def test_supports_return_durations(self):
        """FixedExitStrategy.compute_targets must accept return_durations."""
        from fwbg.plugins import import_plugin_module
        mod = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
        sig = inspect.signature(mod.FixedExitStrategy.compute_targets)
        param_names = list(sig.parameters.keys())
        assert "return_durations" in param_names or "kwargs" in param_names


class TestDefaultPluginsResolvable:
    """Default plugin names must resolve to actual registered plugins."""

    def test_default_exit_strategy_exists_in_registry(self):
        """StrategyConfig default exit_strategy must be a registered plugin."""
        from fwbg.core.config import StrategyConfig
        from fwbg.core import get_exit_strategy
        config = StrategyConfig()
        cls = get_exit_strategy(config.exit_strategy)
        assert cls is not None

    def test_default_risk_management_exists_in_registry(self):
        """StrategyConfig default risk_management must be a registered plugin."""
        from fwbg.core.config import StrategyConfig
        from fwbg.core import get_risk_manager
        config = StrategyConfig()
        cls = get_risk_manager(config.risk_management)
        assert cls is not None

    def test_context_exit_strategy_exists_in_registry(self):
        """SimulationContext default exit_strategy must be a registered plugin."""
        from fwbg.core.context import SimulationContext
        from fwbg.core import get_exit_strategy
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX",
            spread=0.0002, point=0.0001,
        )
        cls = get_exit_strategy(ctx.exit_strategy)
        assert cls is not None
