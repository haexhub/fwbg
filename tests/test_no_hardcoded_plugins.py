"""
Tests that default plugin names resolve to actual registered plugins.

Catches bugs like setting risk_management="none" when no "none" plugin exists.
"""


class TestDefaultPluginsResolvable:
    """Default plugin names must resolve to actual registered plugins."""

    def test_default_exit_strategy_exists_in_registry(self):
        from fwbg.core.config import StrategyConfig
        from fwbg.core import get_exit_strategy
        config = StrategyConfig()
        cls = get_exit_strategy(config.exit_strategy)
        assert cls is not None

    def test_default_risk_management_exists_in_registry(self):
        from fwbg.core.config import StrategyConfig
        from fwbg.core import get_risk_manager
        config = StrategyConfig()
        cls = get_risk_manager(config.risk_management)
        assert cls is not None

    def test_context_exit_strategy_exists_in_registry(self):
        from fwbg.core.context import SimulationContext
        from fwbg.core import get_exit_strategy
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX",
            spread=0.0002, point=0.0001,
        )
        cls = get_exit_strategy(ctx.exit_strategy)
        assert cls is not None
