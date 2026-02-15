"""
Tests for generic regime filter.

Verifies:
1. RegimeFilterConfig parses conditions from dicts
2. RegimeFilterGridConfig produces correct cartesian product combinations
3. compute_regime_filter evaluates conditions correctly
4. Strategy JSONs work with condition_grids format
"""
import numpy as np
import pandas as pd
import pytest


class TestGenericRegimeFilterConfig:
    """RegimeFilterConfig parses and holds generic conditions."""

    def test_default_has_no_conditions(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig()
        assert config.conditions == []

    def test_from_dict_none_returns_empty(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig.from_dict(None)
        assert config.conditions == []

    def test_from_dict_with_conditions(self):
        from fwbg.core.config import RegimeFilterConfig
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


class TestGenericRegimeFilterGridConfig:
    """RegimeFilterGridConfig produces cartesian product of condition values."""

    def test_default_produces_one_combination(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig()
        combos = config.get_combinations()
        assert len(combos) == 1
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
        assert len(combos) == 4  # 2 x 2

    def test_null_value_means_no_condition(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
            ]
        })
        combos = config.get_combinations()
        assert len(combos) == 2
        assert combos[0]["conditions"] == []
        assert len(combos[1]["conditions"]) == 1
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


class TestGenericComputeRegimeFilter:
    """compute_regime_filter evaluates conditions on actual DataFrames."""

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
        result = compute_regime_filter(df_with_indicators, regime_params=RegimeFilterConfig())
        assert result.all()

    def test_none_params_allows_all(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        result = compute_regime_filter(df_with_indicators, regime_params=None)
        assert result.all()

    def test_gte_condition_filters(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert not result.all()
        assert result.any()

    def test_lte_condition_filters(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="macro_vix", operator="<=", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert not result.all()
        assert result.any()

    def test_multiple_conditions_more_restrictive(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        combined = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0),
            RegimeCondition(column="macro_vix", operator="<=", value=25.0),
        ])
        single = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0),
        ])
        result_combined = compute_regime_filter(df_with_indicators, regime_params=combined)
        result_single = compute_regime_filter(df_with_indicators, regime_params=single)
        assert result_combined.sum() <= result_single.sum()

    def test_missing_column_skipped(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_filter
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="nonexistent_col", operator=">=", value=25.0)
        ])
        result = compute_regime_filter(df_with_indicators, regime_params=config)
        assert result.all()


class TestStrategyJsonNewFormat:
    """Strategy JSONs load correctly with condition_grids format."""

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
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        combos = grid.regime_filter_grid.get_combinations()
        for combo in combos:
            assert "conditions" in combo
            assert isinstance(combo["conditions"], list)
