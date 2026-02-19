"""Tests for liquidity_levels indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_liq = import_plugin_module("fwbg-core", "indicators", "liquidity_levels")
if _liq is None:
    pytest.skip("liquidity_levels plugin not available", allow_module_level=True)


def _make_ohlc(n=500, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.001, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ranging_df(n=500):
    """Create range-bound data that should produce equal highs/lows."""
    rng = np.random.default_rng(55)
    oscillation = 5 * np.sin(np.linspace(0, 20 * np.pi, n))
    prices = 105 + oscillation + rng.normal(0, 0.2, n)
    df = pd.DataFrame({
        "O": prices + rng.normal(0, 0.1, n),
        "H": prices + np.abs(rng.normal(0, 0.3, n)) + 0.3,
        "L": prices - np.abs(rng.normal(0, 0.3, n)) - 0.3,
        "C": prices,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


def _get_indicator():
    return _liq.LiquidityLevelsIndicator()


class TestLiquidityFeatures:
    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values_in_ranging(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        late = result.iloc[100:]
        for col in ["liq_eqh_active", "liq_eql_active"]:
            non_null = late[col].dropna()
            assert non_null.sum() > 0, f"{col} never activated in ranging data"

    def test_binary_features(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ["liq_eqh_active", "liq_eql_active",
                     "liq_sweep_up", "liq_sweep_down"]:
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_distances_positive_when_active(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        for col in ["liq_eqh_dist", "liq_eql_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals > 0).all(), f"{col} should be positive"

    def test_counts_at_least_min_touches(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        for col in ["liq_eqh_count", "liq_eql_count"]:
            vals = result[col].dropna()
            active_vals = vals[vals > 0]
            if len(active_vals) > 0:
                assert (active_vals >= 2).all(), f"{col} should be >= min_touches"


class TestLiquidityShiftAndInf:
    def test_shift_applied(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted"

    def test_no_inf_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=1000))
        for col in ind.get_feature_columns():
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} has inf"

    def test_no_undeclared_features(self):
        ind = _get_indicator()
        df = _make_ohlc()
        original = set(df.columns)
        result = ind.compute(df)
        undeclared = set(result.columns) - original - set(ind.get_feature_columns())
        assert not undeclared, f"Undeclared: {undeclared}"

    def test_feature_count(self):
        ind = _get_indicator()
        assert len(ind.get_feature_columns()) == 8


class TestLiquidityParameters:
    def test_get_default_params(self):
        params = _liq.LiquidityLevelsIndicator.get_default_params()
        assert params["swing_lookback"] == 10
        assert params["min_touches"] == 2

    def test_get_param_schema(self):
        schema = _liq.LiquidityLevelsIndicator.get_param_schema()
        assert "swing_lookback" in schema
        assert "tolerance_atr_mult" in schema

    def test_custom_params(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(), swing_lookback=5, min_touches=3)
        assert "liq_eqh_dist" in result.columns


class TestLiquidityHelpers:
    def test_find_equal_levels_basic(self):
        prices = np.array([100.0, 100.1, 100.05, 105.0, 105.1])
        indices = np.array([0, 10, 20, 30, 40])
        clusters = _liq._find_equal_levels(prices, indices, tolerance=0.2, min_touches=2)
        assert len(clusters) >= 2

    def test_find_equal_levels_no_clusters(self):
        prices = np.array([100.0, 110.0, 120.0])
        indices = np.array([0, 10, 20])
        clusters = _liq._find_equal_levels(prices, indices, tolerance=0.5, min_touches=2)
        assert len(clusters) == 0

    def test_find_equal_levels_empty(self):
        clusters = _liq._find_equal_levels(
            np.array([]), np.array([]), tolerance=0.5, min_touches=2)
        assert len(clusters) == 0


class TestLiquidityDiscovery:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("liquidity_levels")
        assert cls is not None
