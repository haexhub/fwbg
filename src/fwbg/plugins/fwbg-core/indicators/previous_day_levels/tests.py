"""Tests for previous_day_levels indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_pdl = import_plugin_module("fwbg-core", "indicators", "previous_day_levels")
if _pdl is None:
    pytest.skip("previous_day_levels plugin not available", allow_module_level=True)


def _make_ohlc_15min(n=2000, seed=42):
    """Create OHLCV DataFrame with 15-minute frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.001, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.003, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.003, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ohlc_hourly(n=2000, seed=42):
    """Create OHLCV DataFrame with hourly frequency."""
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


def _make_ohlc_daily(n=500, seed=42):
    """Create OHLCV DataFrame with daily frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.005, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _get_indicator():
    return _pdl.PreviousDayLevelsIndicator()


class TestPDLFeatures:
    """Tests for previous day level features."""

    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        late = result.iloc[200:]
        for col in ind.get_feature_columns():
            non_null = late[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN after warmup"

    def test_position_reasonable_range(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        pos = result["pdl_position"].dropna()
        within = ((pos >= -1) & (pos <= 2)).mean()
        assert within > 0.8, "Most position values should be in [-1, 2]"

    def test_binary_features(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        for col in ["pdl_above_high", "pdl_below_low",
                     "pdl_high_break", "pdl_low_break",
                     "pdl_day_range_expanding"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert set(vals.unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_distances_atr_normalized(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        for col in ["pdl_high_dist", "pdl_low_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.abs().median() < 50, f"{col} too large"

    def test_range_vs_atr_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        vals = result["pdl_range_vs_atr"].dropna()
        assert (vals >= 0).all(), "Range vs ATR should be non-negative"


class TestPDLShiftAndInf:
    """Lookahead prevention and inf checks."""

    def test_shift_applied(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min())
        for col in ind.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted"

    def test_no_inf_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        for col in ind.get_feature_columns():
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} has inf values"

    def test_no_undeclared_features(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        original_cols = set(df.columns)
        result = ind.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(ind.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared: {undeclared}"

    def test_feature_count(self):
        ind = _get_indicator()
        assert len(ind.get_feature_columns()) == 10


class TestPDLDailySkip:
    """Daily data should not produce PDL features."""

    def test_daily_returns_unchanged(self):
        ind = _get_indicator()
        df = _make_ohlc_daily()
        result = ind.compute(df)
        pdl_cols = [c for c in result.columns if c.startswith("pdl_")]
        assert len(pdl_cols) == 0

    def test_hourly_data_works(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_hourly(n=500))
        assert "pdl_position" in result.columns


class TestPDLParameters:
    """Test parameter methods."""

    def test_get_default_params(self):
        params = _pdl.PreviousDayLevelsIndicator.get_default_params()
        assert params["atr_period"] == 14
        assert params["ma_period"] == 20

    def test_get_param_schema(self):
        schema = _pdl.PreviousDayLevelsIndicator.get_param_schema()
        assert "atr_period" in schema
        assert "ma_period" in schema
        assert schema["atr_period"]["type"] == "int"

    def test_custom_atr_period(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(), atr_period=7)
        assert "pdl_high_dist" in result.columns


class TestPDLDiscovery:
    """Plugin discovery tests."""

    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("previous_day_levels")
        assert cls is not None
