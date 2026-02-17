"""
Tests for bitmask regime filter.

Verifies:
1. RegimeFilterConfig parses conditions with directions/else_directions
2. RegimeFilterGridConfig produces correct cartesian product combinations
3. compute_regime_bitmask evaluates conditions to int8 bitmask arrays
4. Strategy JSONs work with condition_grids format
5. Bitmask encoding: Long=4, Short=2, Sideways=1 (0-7)
"""
import numpy as np
import pandas as pd
import pytest


class TestRegimeConditionBitmask:
    """RegimeCondition supports directions and else_directions."""

    def test_default_directions(self):
        from fwbg.core.config import RegimeCondition
        cond = RegimeCondition(column="adx", operator=">=", value=25)
        assert cond.directions == 6       # Long+Short
        assert cond.else_directions == 0  # blocked

    def test_custom_directions(self):
        from fwbg.core.config import RegimeCondition
        cond = RegimeCondition(column="ema_diff", operator=">", value=0,
                               directions=4, else_directions=2)
        assert cond.directions == 4       # Long only
        assert cond.else_directions == 2  # Short only


class TestRegimeFilterConfig:
    """RegimeFilterConfig parses and holds generic conditions."""

    def test_default_has_no_conditions(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig()
        assert config.conditions == []

    def test_from_dict_none_returns_empty(self):
        from fwbg.core.config import RegimeFilterConfig
        config = RegimeFilterConfig.from_dict(None)
        assert config.conditions == []

    def test_from_dict_with_conditions_and_directions(self):
        from fwbg.core.config import RegimeFilterConfig
        data = {
            "conditions": [
                {"column": "trend_adx_14", "operator": ">=", "value": 25,
                 "directions": 6, "else_directions": 0},
                {"column": "ema_diff", "operator": ">", "value": 0,
                 "directions": 4, "else_directions": 2},
            ]
        }
        config = RegimeFilterConfig.from_dict(data)
        assert len(config.conditions) == 2
        assert config.conditions[0].directions == 6
        assert config.conditions[1].directions == 4
        assert config.conditions[1].else_directions == 2


class TestRegimeFilterGridConfig:
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
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25],
                 "directions": 6, "else_directions": 0},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30],
                 "directions": 6, "else_directions": 0},
            ]
        })
        combos = config.get_combinations()
        assert len(combos) == 4  # 2 x 2

    def test_null_value_means_no_condition(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25],
                 "directions": 6, "else_directions": 0},
            ]
        })
        combos = config.get_combinations()
        assert len(combos) == 2
        assert combos[0]["conditions"] == []
        assert len(combos[1]["conditions"]) == 1
        assert combos[1]["conditions"][0]["value"] == 25
        assert combos[1]["conditions"][0]["directions"] == 6

    def test_total_combinations_matches(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25],
                 "directions": 6, "else_directions": 0},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30],
                 "directions": 6, "else_directions": 0},
                {"column": "regime_hurst_100", "operator": ">=", "values": [None, 0.45],
                 "directions": 6, "else_directions": 0},
            ]
        })
        assert config.total_combinations() == 8
        assert config.total_combinations() == len(config.get_combinations())

    def test_directions_propagated_to_combinations(self):
        from fwbg.core.config import RegimeFilterGridConfig
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "ema_diff", "operator": ">", "values": [None, 0],
                 "directions": 4, "else_directions": 2},
            ]
        })
        combos = config.get_combinations()
        assert combos[1]["conditions"][0]["directions"] == 4
        assert combos[1]["conditions"][0]["else_directions"] == 2


class TestComputeRegimeBitmask:
    """compute_regime_bitmask evaluates conditions to int8 bitmask arrays."""

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

    def test_no_conditions_returns_all_allowed(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_bitmask
        from fwbg.core.config import RegimeFilterConfig
        result = compute_regime_bitmask(df_with_indicators, regime_params=RegimeFilterConfig())
        assert (result == 7).all()
        assert result.dtype == np.int8

    def test_none_params_returns_all_allowed(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_bitmask
        result = compute_regime_bitmask(df_with_indicators, regime_params=None)
        assert (result == 7).all()

    def test_gte_condition_produces_bitmask(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_bitmask
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0,
                           directions=6, else_directions=0)
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=config)
        # Where ADX >= 25: 6 (Long+Short), where ADX < 25: 0 (blocked)
        assert (result[result > 0] == 6).all()
        assert (result == 0).any()
        assert (result == 6).any()

    def test_directional_filter(self, df_with_indicators):
        """EMA diff > 0 → Long only (4), otherwise Short only (2)."""
        from fwbg.pipeline.features import compute_regime_bitmask
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        df = df_with_indicators.copy()
        df["ema_diff"] = np.linspace(-1, 1, len(df))
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="ema_diff", operator=">", value=0,
                           directions=4, else_directions=2)
        ])
        result = compute_regime_bitmask(df, regime_params=config)
        # First half: ema_diff <= 0 → 2 (Short only)
        # Second half: ema_diff > 0 → 4 (Long only)
        assert (result[:50] == 2).all()
        assert (result[51:] == 4).all()

    def test_multiple_conditions_and_combine(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_bitmask
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0,
                           directions=6, else_directions=0),
            RegimeCondition(column="macro_vix", operator="<=", value=25.0,
                           directions=6, else_directions=0),
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=config)
        # Both must pass → more restrictive than single
        single = RegimeFilterConfig(conditions=[
            RegimeCondition(column="trend_adx_14", operator=">=", value=25.0,
                           directions=6, else_directions=0),
        ])
        result_single = compute_regime_bitmask(df_with_indicators, regime_params=single)
        assert (result > 0).sum() <= (result_single > 0).sum()

    def test_missing_column_keeps_all_allowed(self, df_with_indicators):
        from fwbg.pipeline.features import compute_regime_bitmask
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="nonexistent_col", operator=">=", value=25.0)
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=config)
        assert (result == 7).all()

    def test_bitmask_and_logic(self):
        """Two conditions: one allows Long+Short (6), other allows Long only (4).
        AND result: only Long (4)."""
        from fwbg.pipeline.features import compute_regime_bitmask
        from fwbg.core.config import RegimeFilterConfig, RegimeCondition
        df = pd.DataFrame({
            "a": [30.0],
            "b": [1.0],
        })
        config = RegimeFilterConfig(conditions=[
            RegimeCondition(column="a", operator=">=", value=20, directions=6, else_directions=0),
            RegimeCondition(column="b", operator=">", value=0, directions=4, else_directions=2),
        ])
        result = compute_regime_bitmask(df, regime_params=config)
        assert result[0] == 4  # 6 & 4 = 4 (Long only)


class TestStrategyJsonNewFormat:
    """Strategy JSONs load correctly with condition_grids format."""

    def test_exploration_json_loads(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid("EURUSD", "FOREX")
        assert grid.regime_filter_grid.total_combinations() == 24

    def test_exploration_atr_loads(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration_atr.json")
        grid = config.get_grid("EURUSD", "FOREX")
        assert grid.regime_filter_grid.total_combinations() == 24

    def test_exploration_fast_loads(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        grid = config.get_grid("EURUSD", "FOREX")
        assert grid.regime_filter_grid.total_combinations() == 24

    def test_combinations_have_directions(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid("EURUSD", "FOREX")
        combos = grid.regime_filter_grid.get_combinations()
        for combo in combos:
            assert "conditions" in combo
            for cond in combo["conditions"]:
                assert "directions" in cond
                assert "else_directions" in cond
