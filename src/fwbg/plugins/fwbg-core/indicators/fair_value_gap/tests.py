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

        for col in ["fvg_bull_active", "fvg_bear_active", "fvg_in_gap",
                    "fvg_bull_confirmed", "fvg_bear_confirmed"]:
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
        assert len(indicator.get_feature_columns()) == 10

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


class TestFVGConfirmation:
    """Tests for fvg_bull_confirmed and fvg_bear_confirmed features.

    Rule (from SMC): after price tests the gap, an engulfing candle must form
    at the gap to confirm it will hold.
    - Bullish confirmation: bullish engulfing candle while price is in the bull FVG zone.
    - Bearish confirmation: bearish engulfing candle while price is in the bear FVG zone.
    """

    def test_confirmed_columns_present(self):
        indicator = _fvg.FairValueGapIndicator()
        result = indicator.compute(_make_fvg_df())
        assert "fvg_bull_confirmed" in result.columns
        assert "fvg_bear_confirmed" in result.columns

    def test_confirmed_binary(self):
        indicator = _fvg.FairValueGapIndicator()
        result = indicator.compute(_make_fvg_df())
        for col in ["fvg_bull_confirmed", "fvg_bear_confirmed"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"

    def test_no_confirmation_without_engulfing(self):
        """Price tests gap with a non-engulfing candle → confirmed stays 0."""
        # Bars 0-2: bullish FVG [100, 103]  (H[0]=100 < L[2]=103)
        # Bar 3: red candle tests gap (L=101 in zone), NOT a bullish engulfing
        data = {
            "O": [97,  99, 103, 106],
            "H": [100, 101, 110, 106],
            "L": [96,  99, 103, 101],
            "C": [99, 101, 109, 102],
            "V": [100] * 4,
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=4, freq="h"))
        result = _fvg.FairValueGapIndicator().compute(df)
        assert result["fvg_bull_confirmed"].fillna(0).sum() == 0.0

    def test_bull_confirmed_fires_on_engulfing(self):
        """Bullish engulfing candle at gap sets fvg_bull_confirmed=1.

        Setup:
          Bar 0-2: create bullish FVG [100, 103]  (H[0]=100 < L[2]=103)
          Bar 3:   red candle, price in gap (L=101), NOT engulfing
          Bar 4:   bullish engulfing at gap:
                     C[4]=107 > O[4]=101  (green)
                     C[4]=107 > O[3]=106  (close above prev open)
                     O[4]=101 < C[3]=102  (open below prev close)
                     L[4]=101 <= fvg_top=103  (at gap)
                     L[4]=101 > fvg_bottom=100  (not filled)
          Bars 5-7: price rallies, FVG stays active and confirmed
        """
        data = {
            "O": [97,  99, 103, 106, 101, 108, 109, 110],
            "H": [100, 101, 110, 106, 108, 110, 111, 112],
            "L": [96,  99, 103, 101, 101, 107, 108, 109],
            "C": [99, 101, 109, 102, 107, 109, 110, 111],
            "V": [100] * 8,
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="h"))
        result = _fvg.FairValueGapIndicator().compute(df)

        # After shift: confirmation at bar 4 → result.iloc[5]; persists at iloc[6], [7]
        assert result["fvg_bull_confirmed"].iloc[5] == 1.0, (
            f"Expected confirmed=1 at iloc[5], got {result['fvg_bull_confirmed'].iloc[5]}"
        )
        assert result["fvg_bull_confirmed"].iloc[6] == 1.0, "Confirmation should persist"
        # Bear confirmed should stay 0 throughout
        assert result["fvg_bear_confirmed"].fillna(0).sum() == 0.0

    def test_bear_confirmed_fires_on_engulfing(self):
        """Bearish engulfing candle at gap sets fvg_bear_confirmed=1.

        Setup:
          Bar 0-2: create bearish FVG [103, 108]  (L[0]=108 > H[2]=103)
          Bar 3:   green candle tests gap (H=106), NOT bearish engulfing
          Bar 4:   bearish engulfing at gap:
                     C[4]=100 < O[4]=107  (red)
                     O[4]=107 > C[3]=106  (open above prev close)
                     C[4]=100 < O[3]=101  (close below prev open)
                     H[4]=107 >= fvg_bottom=103  (at gap)
                     H[4]=107 < fvg_top=108  (not filled)
          Bars 5-7: price falls, FVG stays active and confirmed
        """
        data = {
            "O": [109, 108, 103, 101, 107, 102, 101, 100],
            "H": [110, 108, 103, 106, 107, 103, 102, 101],
            "L": [108, 104, 101, 101,  99,  98,  97,  96],
            "C": [108, 105, 102, 106, 100, 100,  99,  98],
            "V": [100] * 8,
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=8, freq="h"))
        result = _fvg.FairValueGapIndicator().compute(df)

        assert result["fvg_bear_confirmed"].iloc[5] == 1.0, (
            f"Expected confirmed=1 at iloc[5], got {result['fvg_bear_confirmed'].iloc[5]}"
        )
        assert result["fvg_bear_confirmed"].iloc[6] == 1.0, "Confirmation should persist"
        assert result["fvg_bull_confirmed"].fillna(0).sum() == 0.0

    def test_confirmation_resets_when_fvg_filled(self):
        """Once the FVG is filled, confirmed feature drops back to 0."""
        # Bullish FVG [100, 103], confirmed at bar 4, filled at bar 5 (L <= 100)
        data = {
            "O": [97,  99, 103, 106, 101, 100, 100],
            "H": [100, 101, 110, 106, 108, 104, 104],
            "L": [96,  99, 103, 101, 101, 98,  98],  # bar 5: L=98 <= bottom=100 → fills FVG
            "C": [99, 101, 109, 102, 107, 100, 100],
            "V": [100] * 7,
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=7, freq="h"))
        result = _fvg.FairValueGapIndicator().compute(df)

        # Confirmation fires at bar 4 → result.iloc[5] = 1
        assert result["fvg_bull_confirmed"].iloc[5] == 1.0
        # FVG filled at bar 5 (L=98 <= bottom=100) → result.iloc[6] = 0
        assert result["fvg_bull_confirmed"].iloc[6] == 0.0
