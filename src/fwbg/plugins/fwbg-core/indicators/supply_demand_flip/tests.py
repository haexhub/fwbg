"""Tests for supply_demand_flip indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_sdf = import_plugin_module("fwbg-core", "indicators", "supply_demand_flip")
if _sdf is None:
    pytest.skip("supply_demand_flip plugin not available", allow_module_level=True)


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


def _make_trending_df(n=600):
    """Create data with alternating trend phases that produce both bull and bear flip zones.

    Phase 1 (0-199): uptrend -- breaks resistance, creates bull flip zones
    Phase 2 (200-399): downtrend -- breaks support, creates bear flip zones
    Phase 3 (400-599): uptrend again
    """
    rng = np.random.default_rng(77)
    phase1 = np.linspace(100, 120, 200)
    phase2 = np.linspace(120, 100, 200)
    phase3 = np.linspace(100, 115, 200)
    prices = np.concatenate([phase1, phase2, phase3])
    oscillation = 2 * np.sin(np.linspace(0, 30 * np.pi, n))
    prices = prices + oscillation + rng.normal(0, 0.2, n)
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
    return _sdf.SupplyDemandFlipIndicator()


class TestSDFFeatures:
    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values_in_trending(self):
        ind = _get_indicator()
        result = ind.compute(_make_trending_df())
        late = result.iloc[100:]
        for col in ["sdf_bull_active", "sdf_bear_active"]:
            non_null = late[col].dropna()
            assert non_null.sum() > 0, f"{col} never activated in trending data"

    def test_binary_features(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ["sdf_bull_active", "sdf_bear_active"]:
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_distances_positive_when_active(self):
        ind = _get_indicator()
        result = ind.compute(_make_trending_df())
        for col in ["sdf_bull_dist", "sdf_bear_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals > 0).all(), f"{col} should be positive"

    def test_strength_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_trending_df())
        for col in ["sdf_bull_strength", "sdf_bear_strength"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all(), f"{col} should be non-negative"

    def test_touches_non_negative(self):
        ind = _get_indicator()
        result = ind.compute(_make_trending_df())
        for col in ["sdf_bull_touches", "sdf_bear_touches"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all(), f"{col} should be non-negative"


class TestSDFShiftAndInf:
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


class TestSDFParameters:
    def test_get_default_params(self):
        params = _sdf.SupplyDemandFlipIndicator.get_default_params()
        assert params["swing_lookback"] == 10
        assert params["zone_atr_width"] == 0.3
        assert params["zone_expiry"] == 200

    def test_get_param_schema(self):
        schema = _sdf.SupplyDemandFlipIndicator.get_param_schema()
        assert "swing_lookback" in schema
        assert "zone_atr_width" in schema

    def test_custom_params(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(), swing_lookback=5, zone_expiry=100)
        assert "sdf_bull_active" in result.columns


class TestSDFDiscovery:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("supply_demand_flip")
        assert cls is not None
