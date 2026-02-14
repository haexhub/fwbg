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


class TestNoHardcodedDefaults:
    """Config must not hardcode specific plugin names as defaults."""

    def test_exit_strategy_no_hardcoded_default(self):
        """StrategyConfig.exit_strategy should not default to 'atr_based'."""
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig()
        assert config.exit_strategy != "atr_based", (
            "exit_strategy still defaults to 'atr_based'"
        )

    def test_risk_management_no_hardcoded_default(self):
        """StrategyConfig.risk_management should not default to 'kelly'."""
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig()
        assert config.risk_management != "kelly", (
            "risk_management still defaults to 'kelly'"
        )

    def test_context_exit_strategy_no_hardcoded_default(self):
        """SimulationContext.exit_strategy should not default to 'atr_based'."""
        from fwbg.core.context import SimulationContext
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX",
            spread=0.0002, point=0.0001,
        )
        assert ctx.exit_strategy != "atr_based"
