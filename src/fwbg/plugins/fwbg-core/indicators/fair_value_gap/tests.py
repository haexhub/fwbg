"""Tests for the fair_value_gap indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_fvg = import_plugin_module("fwbg-core", "indicators", "fair_value_gap")
if _fvg is None:
    pytest.skip("fair_value_gap plugin not available", allow_module_level=True)


class TestFVGDetection:
    """Tests for _detect_fvgs helper."""

    def test_bullish_fvg_detected(self):
        """Bullish FVG: H[i-2] < L[i] (gap up through candle 2)."""
        highs = np.array([100, 105, 108], dtype=np.float64)
        lows = np.array([98, 99, 101], dtype=np.float64)
        # H[0]=100 < L[2]=101 → bullish FVG zone [100, 101]
        fvgs = _fvg._detect_fvgs(highs, lows)
        bull = [f for f in fvgs if f["type"] == "bullish"]
        assert len(bull) == 1
        assert bull[0]["top"] == 101.0
        assert bull[0]["bottom"] == 100.0
        assert bull[0]["bar"] == 2

    def test_bearish_fvg_detected(self):
        """Bearish FVG: L[i-2] > H[i] (gap down through candle 2)."""
        highs = np.array([108, 105, 100], dtype=np.float64)
        lows = np.array([106, 101, 98], dtype=np.float64)
        # L[0]=106 > H[2]=100 → bearish FVG zone [100, 106]
        fvgs = _fvg._detect_fvgs(highs, lows)
        bear = [f for f in fvgs if f["type"] == "bearish"]
        assert len(bear) == 1
        assert bear[0]["top"] == 106.0
        assert bear[0]["bottom"] == 100.0

    def test_no_fvg_in_flat_data(self):
        """Flat data should produce no FVGs."""
        n = 50
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        fvgs = _fvg._detect_fvgs(highs, lows)
        assert len(fvgs) == 0

    def test_no_fvg_when_overlapping(self):
        """Candles that overlap produce no gap."""
        highs = np.array([102, 105, 103], dtype=np.float64)
        lows = np.array([98, 99, 101], dtype=np.float64)
        # H[0]=102 > L[2]=101 → no bullish FVG (candle 1 high overlaps candle 3 low)
        # L[0]=98 < H[2]=103 → no bearish FVG
        fvgs = _fvg._detect_fvgs(highs, lows)
        assert len(fvgs) == 0

    def test_multiple_fvgs(self):
        """Multiple FVGs in sequence."""
        # Two separate impulse moves with gaps, flat sections between
        highs = np.array([100, 106, 108, 108, 108, 108, 115, 120], dtype=np.float64)
        lows = np.array([98, 102, 101, 107, 107, 107, 110, 116], dtype=np.float64)
        # Bar 2: H[0]=100 < L[2]=101 → bullish FVG
        # Bar 7: H[5]=108 < L[7]=116 → bullish FVG
        fvgs = _fvg._detect_fvgs(highs, lows)
        bull = [f for f in fvgs if f["type"] == "bullish"]
        assert len(bull) >= 2


def _make_fvg_df(n=500):
    """Create OHLC data with clear trends that produce FVGs."""
    np.random.seed(42)
    trend = np.linspace(100, 130, n) + np.random.randn(n) * 0.3
    oscillation = 2 * np.sin(np.linspace(0, 10 * np.pi, n))
    prices = trend + oscillation

    # Add occasional impulse moves to create FVGs
    for i in range(20, n - 2, 50):
        prices[i] += 3.0  # impulse candle
        prices[i + 1] += 4.0  # follow-through

    df = pd.DataFrame(
        {
            "O": prices + np.random.randn(n) * 0.1,
            "H": prices + np.abs(np.random.randn(n) * 0.5) + 0.5,
            "L": prices - np.abs(np.random.randn(n) * 0.5) - 0.5,
            "C": prices,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


class TestComputeFeatures:
    """Tests for full compute() output."""

    def test_all_features_present(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df()
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing feature: {col}"

    def test_binary_features(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df()
        result = indicator.compute(df)

        for col in ["fvg_bull_active", "fvg_bear_active", "fvg_in_gap"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"

    def test_distances_atr_normalized(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df()
        result = indicator.compute(df)

        for col in ["fvg_bull_dist", "fvg_bear_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.median() < 50, f"{col} too large, probably not ATR-normalized"

    def test_count_non_negative(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df()
        result = indicator.compute(df)

        vals = result["fvg_count"].dropna()
        assert (vals >= 0).all()

    def test_shift_applied(self):
        """Features must be shifted by 1 bar (lookahead prevention)."""
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df(n=300)
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted"


class TestPluginIntegration:
    """Integration tests."""

    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator

        discover_plugins()
        cls = get_indicator("fair_value_gap")
        assert cls is not None

    def test_all_declared_features_in_output(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df(n=500)
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Declared {col} missing from output"

    def test_no_undeclared_features(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df(n=300)
        original_cols = set(df.columns)
        result = indicator.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(indicator.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared features: {undeclared}"

    def test_feature_count(self):
        indicator = _fvg.FairValueGapIndicator()
        assert len(indicator.get_feature_columns()) == 8

    def test_no_inf_values(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df(n=500)
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            inf_count = np.isinf(result[col].dropna()).sum()
            assert inf_count == 0, f"{col} has {inf_count} inf values"

    def test_not_all_nan(self):
        indicator = _fvg.FairValueGapIndicator()
        df = _make_fvg_df(n=500)
        result = indicator.compute(df)

        late = result.iloc[100:]
        for col in indicator.get_feature_columns():
            vals = late[col].dropna()
            assert len(vals) > 0, f"{col} is all NaN after warmup"
