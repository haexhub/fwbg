"""Tests for displacement indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_disp = import_plugin_module("fwbg-core", "indicators", "displacement")
if _disp is None:
    pytest.skip("displacement plugin not available", allow_module_level=True)


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


def _make_impulse_df(n=200):
    """Create data with clear impulse moves that produce FVGs."""
    rng = np.random.default_rng(99)
    prices = 100 + np.cumsum(rng.normal(0, 0.3, n))
    for i in range(30, n - 2, 30):
        prices[i] += 5.0
        prices[i + 1] += 6.0
    df = pd.DataFrame({
        "O": prices + rng.normal(0, 0.1, n),
        "H": prices + np.abs(rng.normal(0, 0.5, n)) + 0.5,
        "L": prices - np.abs(rng.normal(0, 0.5, n)) - 0.5,
        "C": prices,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


def _get_indicator():
    return _disp.DisplacementIndicator()


class TestDisplacementFeatures:
    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=500))
        late = result.iloc[50:]
        for col in ind.get_feature_columns():
            non_null = late[col].dropna()
            assert len(non_null) > 0, f"{col} all NaN after warmup"

    def test_ratios_between_0_and_1(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=1000))
        for col in ["disp_body_ratio", "disp_upper_wick_ratio",
                     "disp_lower_wick_ratio", "disp_close_position"]:
            vals = result[col].dropna()
            assert (vals >= -0.01).all() and (vals <= 1.01).all(), \
                f"{col} out of [0, 1] range"

    def test_fvg_binary(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        vals = result["disp_fvg_formed"].dropna()
        assert set(vals.unique()).issubset({0.0, 1.0})

    def test_fvg_detected_in_impulse_data(self):
        ind = _get_indicator()
        result = ind.compute(_make_impulse_df())
        fvg_count = result["disp_fvg_formed"].dropna().sum()
        assert fvg_count > 0, "Impulse data should produce FVGs"

    def test_range_expansion_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        vals = result["disp_range_expansion"].dropna()
        assert (vals >= 0).all()

    def test_body_atr_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        vals = result["disp_body_atr"].dropna()
        assert (vals >= 0).all()


class TestDisplacementShiftAndInf:
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


class TestDisplacementParameters:
    def test_get_default_params(self):
        params = _disp.DisplacementIndicator.get_default_params()
        assert params["atr_period"] == 14
        assert params["range_avg_period"] == 20

    def test_get_param_schema(self):
        schema = _disp.DisplacementIndicator.get_param_schema()
        assert "atr_period" in schema
        assert "range_avg_period" in schema

    def test_custom_params(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(), atr_period=7, range_avg_period=10)
        assert "disp_body_atr" in result.columns


class TestDisplacementDiscovery:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("displacement")
        assert cls is not None
