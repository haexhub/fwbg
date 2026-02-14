"""
Tests for generic regime filter.

Verifies:
1. RegimeFilterConfig uses generic conditions (no hardcoded ADX/VIX/Hurst)
2. RegimeFilterGridConfig uses condition_grids
3. compute_regime_filter evaluates conditions generically
4. No hardcoded indicator names in config classes
5. Strategy JSONs work with new format
"""
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Test: Generic RegimeCondition / RegimeFilterConfig
# =============================================================================

class TestRegimeCondition:
    """RegimeCondition represents a single column condition."""

    def test_condition_dataclass_exists(self):
        from fwbg.core.config import RegimeCondition
        cond = RegimeCondition(column="trend_adx_14", operator=">=", value=25.0)
        assert cond.column == "trend_adx_14"
        assert cond.operator == ">="
        assert cond.value == 25.0


class TestGenericRegimeFilterConfig:
    """RegimeFilterConfig uses a list of generic conditions."""

    def test_has_conditions_field(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig()
        assert hasattr(config, "conditions")

    def test_default_has_no_conditions(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig()
        assert config.conditions == []

    def test_from_dict_none_returns_empty(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig.from_dict(None)
        assert config.conditions == []

    def test_from_dict_with_conditions(self):
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        data = {
            "conditions": [
                {"column": "trend_adx_14", "operator": ">=", "value": 25},
                {"column": "macro_vix", "operator": "<=", "value": 30},
            ]
        }
        config = RegimeFilterConfig.from_dict(data)
        assert len(config.conditions) == 2
        assert config.conditions[0].column == "trend_adx_14"
        assert config.conditions[1].operator == "<="

    def test_no_adx_enabled_field(self):
        """RegimeFilterConfig must NOT have hardcoded adx_enabled field."""
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig()
        assert not hasattr(config, "adx_enabled")
        assert not hasattr(config, "vix_enabled")
        assert not hasattr(config, "hurst_enabled")


# =============================================================================
# Test: Generic RegimeFilterGridConfig
# =============================================================================

class TestGenericRegimeFilterGridConfig:
    """RegimeFilterGridConfig uses condition_grids for cartesian product."""

    def test_default_produces_one_combination(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig()
        combos = config.get_combinations()
        assert len(combos) == 1
        # The one combination should have empty conditions
        assert combos[0]["conditions"] == []

    def test_condition_grids_produce_combinations(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
            ]
        })
        combos = config.get_combinations()
        # 2 x 2 = 4 combinations
        assert len(combos) == 4

    def test_null_value_means_no_condition(self):
        """null in values list means this condition is not applied."""
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
            ]
        })
        combos = config.get_combinations()
        assert len(combos) == 2

        # First combo (None) should have no conditions
        assert combos[0]["conditions"] == []

        # Second combo (25) should have the condition
        assert len(combos[1]["conditions"]) == 1
        assert combos[1]["conditions"][0]["column"] == "trend_adx_14"
        assert combos[1]["conditions"][0]["value"] == 25

    def test_total_combinations_matches(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
                {"column": "regime_hurst_100", "operator": ">=", "values": [None, 0.45]},
            ]
        })
        assert config.total_combinations() == 8
        assert config.total_combinations() == len(config.get_combinations())

    def test_from_dict_none_returns_default(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict(None)
        assert config.total_combinations() == 1

    def test_no_adx_min_field(self):
        """RegimeFilterGridConfig must NOT have hardcoded adx_min field."""
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig()
        assert not hasattr(config, "adx_min")
        assert not hasattr(config, "vix_max")
        assert not hasattr(config, "hurst")


# =============================================================================
# Test: Generic compute_regime_filter
# =============================================================================

class TestGenericComputeRegimeFilter:
    """compute_regime_filter evaluates conditions generically."""

    @pytest.fixture
    def df_with_indicators(self):
        n = 100
        return pd.DataFrame({
            "O": np.full(n, 100.0),
            "H": np.full(n, 101.0),
            "L": np.full(n, 99.0),
            "C": np.full(n, 100.0),
            "trend_adx_14": np.linspace(10, 40, n),
            "macro_vix": np.linspace(15, 35, n),
            "regime_hurst_100": np.linspace(0.3, 0.7, n),
        })

    def test_no_conditions_allows_all(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig()
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert result.all()

    def test_none_params_allows_all(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        result = compute_regime_filter(df_with_indicators, regime_params=None)
        assert result.all()

    def test_gte_condition(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert result.dtype == bool
        assert not result.all()  # Some bars have ADX < 25
        assert result.any()  # Some bars have ADX >= 25

    def test_lte_condition(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="macro_vix", operator="<=", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert not result.all()
        assert result.any()

    def test_gt_condition(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert result.dtype == bool

    def test_lt_condition(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="macro_vix", operator="<", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert result.dtype == bool

    def test_multiple_conditions_and_combined(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0),
            RegimeCondition(column="macro_vix", operator="<=", value=25.0),
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)

        # Combined should be more restrictive than either alone
        single_adx = compute_regime_filter(df_with_indicators, RegimeFilterConfig(
            conditions=[RegimeCondition("trend_adx_14", ">=", 25.0)]
        ))
        assert result.sum() <= single_adx.sum()

    def test_missing_column_skipped(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="nonexistent_col", operator=">=", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert result.all()  # No column = no filter


# =============================================================================
# Test: No hardcoded regime logic
# =============================================================================

class TestNoHardcodedRegimeLogic:
    """Config and features must not have hardcoded ADX/VIX/Hurst logic."""

    def test_no_adx_in_compute_regime_filter(self):
        """compute_regime_filter must not reference 'adx' hardcoded."""
        import inspect
        from fwbg.pipeline.features import compute_regime_filter
        source = inspect.getsource(compute_regime_filter)
        assert "adx_min" not in source
        assert "adx_enabled" not in source

    def test_no_vix_in_compute_regime_filter(self):
        """compute_regime_filter must not reference 'vix' hardcoded."""
        import inspect
        from fwbg.pipeline.features import compute_regime_filter
        source = inspect.getsource(compute_regime_filter)
        assert "vix_max" not in source
        assert "vix_enabled" not in source
        assert "sent_vix" not in source

    def test_no_hurst_in_compute_regime_filter(self):
        """compute_regime_filter must not reference 'hurst' hardcoded."""
        import inspect
        from fwbg.pipeline.features import compute_regime_filter
        source = inspect.getsource(compute_regime_filter)
        assert "hurst_enabled" not in source
        assert "_compute_rolling_hurst" not in source

    def test_no_hardcoded_regime_logging_in_process(self):
        """process.py must not have hardcoded ADX/VIX/Hurst logging."""
        import inspect
        from fwbg.optimization import process
        source = inspect.getsource(process)
        assert "adx_enabled" not in source
        assert "vix_enabled" not in source
        assert "hurst_enabled" not in source


# =============================================================================
# Test: Strategy JSON compatibility
# =============================================================================

class TestStrategyJsonNewFormat:
    """Strategy JSONs must use new condition_grids format."""

    def test_exploration_json_loads(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_exploration_atr_loads(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration_atr.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_exploration_fast_loads(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_combinations_have_conditions_format(self):
        """Combos should use conditions list format."""
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        combos = grid.regime_filter_grid.get_combinations()
        for combo in combos:
            assert "conditions" in combo
            assert isinstance(combo["conditions"], list)
