"""
Tests für Regime-Detection Features und Bitmask-Regime-Filter.

Testet:
- Regime-Filter (compute_regime_bitmask mit Bitmask-Encoding)
- RegimeFilterGridConfig (Kombinationen, Parsing, directions/else_directions)
- RegimeFilterConfig (conditions-based)
- Regime-Plugin Feature-Berechnung
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline.features import compute_regime_bitmask
from fwbg.core.config import (
    RegimeFilterGridConfig,
    RegimeFilterConfig,
    RegimeCondition,
    GridConfig,
    StrategyConfig,
)


# === FIXTURES ===

@pytest.fixture
def sample_ohlc():
    """Sample OHLC-Daten für Regime-Filter Tests."""
    np.random.seed(42)
    n = 300
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close * (1 + np.random.randn(n) * 0.002)

    df = pd.DataFrame({
        'O': open_price,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


@pytest.fixture
def df_with_indicators(sample_ohlc):
    """OHLC mit vorberechneten Indikator-Spalten für Regime-Filter."""
    df = sample_ohlc.copy()
    n = len(df)
    df["trend_adx_14"] = np.linspace(10, 40, n)
    df["macro_vix"] = np.linspace(15, 35, n)
    df["regime_hurst_100"] = np.linspace(0.3, 0.7, n)
    return df


# === TESTS FÜR compute_regime_bitmask ===

class TestComputeRegimeBitmask:
    """Tests für compute_regime_bitmask() mit Bitmask-Encoding."""

    def test_no_params_returns_all_allowed(self, sample_ohlc):
        result = compute_regime_bitmask(sample_ohlc, regime_params=None)
        assert result.dtype == np.int8
        assert len(result) == len(sample_ohlc)
        assert (result == 7).all()

    def test_empty_conditions_returns_all_allowed(self, df_with_indicators):
        params = RegimeFilterConfig(conditions=[])
        result = compute_regime_bitmask(df_with_indicators, regime_params=params)
        assert (result == 7).all()

    def test_gte_condition_produces_bitmask(self, df_with_indicators):
        """ADX >= 25 → directions (6), else → 0."""
        params = RegimeFilterConfig(conditions=[
            RegimeCondition("trend_adx_14", ">=", 25.0, directions=6, else_directions=0)
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=params)
        assert result.dtype == np.int8
        assert (result == 0).any()
        assert (result == 6).any()

    def test_lte_condition(self, df_with_indicators):
        """VIX <= 25 → directions (6), else → 0."""
        params = RegimeFilterConfig(conditions=[
            RegimeCondition("macro_vix", "<=", 25.0, directions=6, else_directions=0)
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=params)
        assert (result == 0).any()
        assert (result == 6).any()

    def test_missing_column_keeps_all_allowed(self, df_with_indicators):
        """Condition on missing column has no effect."""
        params = RegimeFilterConfig(conditions=[
            RegimeCondition("nonexistent_col", ">=", 25.0)
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=params)
        assert (result == 7).all()

    def test_combined_conditions_and_logic(self, df_with_indicators):
        """Multiple conditions are AND-combined (intersection of bitmasks)."""
        params = RegimeFilterConfig(conditions=[
            RegimeCondition("trend_adx_14", ">=", 25.0, directions=6, else_directions=0),
            RegimeCondition("macro_vix", "<=", 25.0, directions=6, else_directions=0),
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=params)
        assert result.dtype == np.int8

        # Combined should be more restrictive
        single = compute_regime_bitmask(df_with_indicators, RegimeFilterConfig(
            conditions=[RegimeCondition("trend_adx_14", ">=", 25.0, directions=6, else_directions=0)]
        ))
        assert (result > 0).sum() <= (single > 0).sum()

    def test_three_conditions(self, df_with_indicators):
        """ADX + VIX + Hurst combined."""
        params = RegimeFilterConfig(conditions=[
            RegimeCondition("trend_adx_14", ">=", 25.0, directions=6, else_directions=0),
            RegimeCondition("macro_vix", "<=", 25.0, directions=6, else_directions=0),
            RegimeCondition("regime_hurst_100", ">=", 0.45, directions=6, else_directions=0),
        ])
        result = compute_regime_bitmask(df_with_indicators, regime_params=params)
        assert result.dtype == np.int8
        assert len(result) == len(df_with_indicators)

    def test_directional_bitmask(self, df_with_indicators):
        """directions=4 (Long), else_directions=2 (Short) based on condition."""
        df = df_with_indicators.copy()
        df["ema_diff"] = np.linspace(-1, 1, len(df))
        params = RegimeFilterConfig(conditions=[
            RegimeCondition("ema_diff", ">", 0, directions=4, else_directions=2)
        ])
        result = compute_regime_bitmask(df, regime_params=params)
        # First half ema_diff <= 0 → Short only (2)
        assert (result[:len(df) // 2] == 2).all()
        # Last bars ema_diff > 0 → Long only (4)
        assert (result[-10:] == 4).all()


# === TESTS FÜR RegimeFilterGridConfig ===

class TestRegimeFilterGridConfig:
    """Tests für RegimeFilterGridConfig Konfiguration und Kombinationen."""

    def test_default_produces_one_combination(self):
        config = RegimeFilterGridConfig()
        combos = config.get_combinations()
        assert len(combos) == 1
        assert combos[0]["conditions"] == []

    def test_total_combinations_matches_get_combinations(self):
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
                {"column": "regime_hurst_100", "operator": ">=", "values": [None, 0.45]},
            ]
        })
        assert config.total_combinations() == len(config.get_combinations())

    def test_exploration_config_produces_8_combinations(self):
        """2 ADX × 2 VIX × 2 Hurst = 8."""
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
                {"column": "regime_hurst_100", "operator": ">=", "values": [None, 0.45]},
            ]
        })
        combos = config.get_combinations()
        assert len(combos) == 8

    def test_no_filter_combo_has_empty_conditions(self):
        """Null values in all grids → no conditions."""
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
            ]
        })
        combos = config.get_combinations()
        no_filter = [c for c in combos if len(c["conditions"]) == 0]
        assert len(no_filter) == 1

    def test_all_filter_combo_has_all_conditions(self):
        """Non-null values in all grids → all conditions present with directions."""
        config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25],
                 "directions": 6, "else_directions": 0},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30],
                 "directions": 6, "else_directions": 0},
            ]
        })
        combos = config.get_combinations()
        all_conds = [c for c in combos if len(c["conditions"]) == 2]
        assert len(all_conds) == 1
        assert all_conds[0]["conditions"][0]["column"] == "trend_adx_14"
        assert all_conds[0]["conditions"][0]["value"] == 25
        assert all_conds[0]["conditions"][0]["directions"] == 6
        assert all_conds[0]["conditions"][1]["column"] == "macro_vix"
        assert all_conds[0]["conditions"][1]["value"] == 30

    def test_from_dict_none_returns_default(self):
        config = RegimeFilterGridConfig.from_dict(None)
        assert config.total_combinations() == 1

    def test_grid_config_includes_regime_filter(self):
        """GridConfig.from_dict parsed regime_filter_grid."""
        data = {
            "tp": [10, 20],
            "sl": [20, 30],
            "ct": [0.6],
            "regime_filter_grid": {
                "condition_grids": [
                    {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                    {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
                    {"column": "regime_hurst_100", "operator": ">=", "values": [None, 0.45]},
                ]
            },
        }
        grid = GridConfig.from_dict(data)
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_grid_config_without_regime_filter(self):
        """GridConfig ohne regime_filter_grid hat Default (1 Kombi)."""
        data = {"tp": [10], "sl": [20], "ct": [0.6]}
        grid = GridConfig.from_dict(data)
        assert grid.regime_filter_grid.total_combinations() == 1


# === TESTS FÜR RegimeFilterConfig ===

class TestRegimeFilterConfig:
    """Tests für RegimeFilterConfig (conditions-based)."""

    def test_default_empty_conditions(self):
        config = RegimeFilterConfig()
        assert config.conditions == []

    def test_from_dict_none_returns_empty(self):
        config = RegimeFilterConfig.from_dict(None)
        assert config.conditions == []

    def test_from_dict_with_conditions(self):
        data = {
            "conditions": [
                {"column": "trend_adx_14", "operator": ">=", "value": 25},
                {"column": "macro_vix", "operator": "<=", "value": 30},
            ]
        }
        config = RegimeFilterConfig.from_dict(data)
        assert len(config.conditions) == 2
        assert config.conditions[0].column == "trend_adx_14"
        assert config.conditions[0].value == 25

    def test_grid_combo_to_filter_config(self):
        """RegimeFilterGridConfig Combos can be used as RegimeFilterConfig."""
        grid_config = RegimeFilterGridConfig.from_dict({
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 25]},
                {"column": "macro_vix", "operator": "<=", "values": [None, 30]},
            ]
        })
        combos = grid_config.get_combinations()

        for combo_dict in combos:
            config = RegimeFilterConfig.from_dict(combo_dict)
            assert isinstance(config.conditions, list)


# === TESTS FÜR Strategy-Laden mit Regime-Filter ===

class TestStrategyRegimeConfig:
    """Tests dass Exploration-Strategien korrekte Regime-Config haben."""

    def test_exploration_json_has_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 24

    def test_exploration_atr_has_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration_atr.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 24

    def test_exploration_fast_has_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 24

    def test_all_asset_classes_have_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        for asset_class in ["FOREX", "INDEX", "COMMODITY", "CRYPTO"]:
            grid = config.get_grid_for_class(asset_class)
            assert grid.regime_filter_grid.total_combinations() == 24, (
                f"{asset_class} sollte 8 Regime-Kombinationen haben"
            )

    def test_regime_combos_include_no_filter_baseline(self):
        """Jede Strategie hat eine 'kein Filter' Baseline."""
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        combos = grid.regime_filter_grid.get_combinations()

        no_filter = [c for c in combos if len(c["conditions"]) == 0]
        assert len(no_filter) == 1


# === TESTS FÜR Regime-Plugin ===

class TestRegimePlugin:
    """Tests für das RegimeIndicators Plugin."""

    def test_plugin_importable(self):
        from fwbg_sdk import BaseIndicator
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-premium:regime")
        assert plugin_cls is not None

    def test_plugin_computes_features(self, sample_ohlc):
        """Plugin berechnet Regime-Features korrekt."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        # Brauchen mehr Daten für Hurst
        np.random.seed(42)
        n = 600
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.randn(n) * 0.005))
        low = close * (1 - np.abs(np.random.randn(n) * 0.005))
        open_price = close * (1 + np.random.randn(n) * 0.002)

        df = pd.DataFrame({
            'O': open_price, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[50], vr_windows=[100], vr_lags=[5], step=10)

        assert "regime_hurst_100" in result.columns
        assert "regime_entropy_50" in result.columns
        assert "regime_vr_100_5" in result.columns

    def test_plugin_hurst_values_valid(self, sample_ohlc):
        """Hurst-Werte sind zwischen 0 und 1."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'O': close, 'H': close * 1.005, 'L': close * 0.995, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[], vr_windows=[], vr_lags=[], step=10)

        hurst = result["regime_hurst_100"].dropna()
        assert len(hurst) > 0
        assert all(0 <= v <= 1 for v in hurst)

    def test_plugin_uses_original_close(self):
        """Plugin nutzt _original_close wenn vorhanden (Frac-Diff)."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'O': close, 'H': close + 0.5, 'L': close - 0.5,
            'C': np.random.randn(n),  # Frac-diff'd (nicht-originale Close)
            '_original_close': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[], vr_windows=[], vr_lags=[], step=10)

        hurst = result["regime_hurst_100"].dropna()
        assert len(hurst) > 0
        assert all(0 <= v <= 1 for v in hurst)

    def test_plugin_benefits_from_stationary_false(self):
        """Regime-Plugin sollte benefits_from_stationary=False haben."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        assert plugin_cls.benefits_from_stationary is False
