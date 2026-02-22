"""Tests for MultiTimeframeIndicators plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.registry import get_indicator


def make_h1(n=2000, seed=42):
    np.random.seed(seed)
    ret = np.random.randn(n) * 0.001
    c = 10000 * np.exp(np.cumsum(ret))
    o = np.roll(c, 1)
    o[0] = c[0]
    sp = c * 0.0001
    h = np.maximum(o, c) + np.abs(np.random.randn(n)) * sp
    low = np.minimum(o, c) - np.abs(np.random.randn(n)) * sp
    v = np.random.randint(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"O": o, "H": h, "L": low, "C": c, "V": v},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="h"),
    )


# Expected feature columns from get_feature_columns()
EXPECTED_COLS = [
    # H4
    "mtf_h4_trend", "mtf_h4_range_pos",
    "mtf_h4_ema20_dist", "mtf_h4_ema50_dist",
    "mtf_h4_adx", "mtf_h4_rsi", "mtf_h4_atr_pct", "mtf_h4_bb_pband",
    # D1
    "mtf_d1_range_pos",
    "mtf_d1_ema20_dist", "mtf_d1_ema50_dist",
    "mtf_d1_trend_strength",
    # W1
    "mtf_w1_range_pos",
    "mtf_w1_ema20_dist", "mtf_w1_ema50_dist",
    "mtf_w1_trend_strength",
    # Y1
    "mtf_y1_ema200d_dist",
    "mtf_y1_52w_range_pos", "mtf_y1_52w_high_dist", "mtf_y1_52w_low_dist",
    # Alignment
    "mtf_trend_alignment_h1h4", "mtf_trend_alignment_h4d1",
    "mtf_trend_alignment_d1w1",
    "mtf_consensus", "mtf_trend_strength",
    # Volatility & Divergence
    "mtf_vol_ratio_h1h4", "mtf_rsi_divergence",
    # Support/Resistance
    "mtf_d1_above_prev_high", "mtf_d1_below_prev_low",
    "mtf_d1_dist_to_high", "mtf_d1_dist_to_low",
]

# Warmup: w1_bars=120 + shift=1 + some EMA warmup. Use 200 bars as safe warmup.
WARMUP = 200


@pytest.fixture(scope="module")
def result():
    cls = get_indicator("multi_timeframe")
    indicator = cls()
    df = make_h1(n=2000)
    return indicator.compute(df)


class TestMTFFeaturePresence:
    """Features exist after warmup, first rows NaN, no inf, only declared columns added."""

    def test_plugin_registered(self):
        cls = get_indicator("multi_timeframe")
        assert cls is not None

    def test_declared_columns_present(self, result):
        for col in EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_only_declared_columns_added(self, result):
        orig_cols = {"O", "H", "L", "C", "V"}
        new_cols = set(result.columns) - orig_cols
        declared = set(EXPECTED_COLS)
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared columns added: {undeclared}"

    def test_first_row_is_nan(self, result):
        # shift_features shifts by 1, so row 0 must be NaN for all feature cols
        for col in EXPECTED_COLS:
            assert pd.isna(result[col].iloc[0]), f"{col}: first row should be NaN"

    def test_features_valid_after_warmup(self, result):
        # After warmup period most features should have real values
        tail = result.iloc[WARMUP:]
        for col in ["mtf_h4_trend", "mtf_h4_range_pos", "mtf_d1_range_pos", "mtf_w1_range_pos"]:
            non_nan = tail[col].dropna()
            assert len(non_nan) > 0, f"{col}: all NaN after warmup"

    def test_no_inf_values(self, result):
        for col in EXPECTED_COLS:
            series = result[col]
            assert not np.isinf(series.dropna()).any(), f"{col} contains inf values"

    def test_get_feature_columns_matches(self):
        cls = get_indicator("multi_timeframe")
        indicator = cls()
        assert set(indicator.get_feature_columns()) == set(EXPECTED_COLS)


class TestMTFH4Features:
    """H4-level feature value-range tests."""

    def test_h4_trend_bounded(self, result):
        # mtf_h4_trend = safe_divide(close - open, range): bounded to [-1, 1] for valid bars
        tail = result["mtf_h4_trend"].dropna()
        assert (tail >= -1.0).all(), "mtf_h4_trend contains values < -1"
        assert (tail <= 1.0).all(), "mtf_h4_trend contains values > 1"

    def test_h4_range_pos_bounded(self, result):
        # mtf_h4_range_pos = safe_divide(close - low, range): [0, 1]
        tail = result["mtf_h4_range_pos"].dropna()
        assert (tail >= 0.0).all(), "mtf_h4_range_pos contains values < 0"
        assert (tail <= 1.0).all(), "mtf_h4_range_pos contains values > 1"

    def test_h4_adx_non_negative(self, result):
        tail = result["mtf_h4_adx"].dropna()
        assert (tail >= 0.0).all(), "mtf_h4_adx should be non-negative"

    def test_h4_rsi_bounded(self, result):
        tail = result["mtf_h4_rsi"].dropna()
        assert (tail >= 0.0).all(), "mtf_h4_rsi < 0"
        assert (tail <= 100.0).all(), "mtf_h4_rsi > 100"

    def test_h4_atr_pct_non_negative(self, result):
        tail = result["mtf_h4_atr_pct"].dropna()
        assert (tail >= 0.0).all(), "mtf_h4_atr_pct should be non-negative"

    def test_h4_features_populated_after_warmup(self, result):
        after = result.iloc[WARMUP:]
        for col in ["mtf_h4_trend", "mtf_h4_range_pos", "mtf_h4_adx", "mtf_h4_rsi"]:
            assert after[col].notna().any(), f"{col}: no values after warmup"


class TestMTFD1Features:
    """D1-level feature value-range tests."""

    def test_d1_range_pos_bounded(self, result):
        # Need at least d1_bars=24 bars of data before d1 range is valid
        after = result.iloc[25:]
        tail = after["mtf_d1_range_pos"].dropna()
        assert len(tail) > 0, "mtf_d1_range_pos: no values after 24 bars"
        assert (tail >= 0.0).all(), "mtf_d1_range_pos contains values < 0"
        assert (tail <= 1.0).all(), "mtf_d1_range_pos contains values > 1"

    def test_d1_ema_dists_exist(self, result):
        for col in ["mtf_d1_ema20_dist", "mtf_d1_ema50_dist"]:
            assert result[col].notna().any(), f"{col}: entirely NaN"

    def test_d1_trend_strength_finite(self, result):
        vals = result["mtf_d1_trend_strength"].dropna()
        assert len(vals) > 0
        assert np.isfinite(vals).all()

    def test_d1_support_resistance_binary(self, result):
        for col in ["mtf_d1_above_prev_high", "mtf_d1_below_prev_low"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1, 0.0, 1.0}), f"{col} is not binary: {vals}"


class TestMTFTrendAlignment:
    """Alignment columns exist and are bounded in {0, 1}."""

    ALIGN_COLS = [
        "mtf_trend_alignment_h1h4",
        "mtf_trend_alignment_h4d1",
        "mtf_trend_alignment_d1w1",
        "mtf_consensus",
    ]

    def test_alignment_columns_present(self, result):
        for col in self.ALIGN_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_alignment_binary(self, result):
        for col in self.ALIGN_COLS:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0, 1, 0.0, 1.0}), (
                f"{col} should be binary, got: {vals}"
            )

    def test_trend_strength_bounded(self, result):
        # Sum of three binary alignment flags: range [0, 3]
        vals = result["mtf_trend_strength"].dropna()
        assert (vals >= 0).all()
        assert (vals <= 3).all()

    def test_consensus_implies_full_strength(self, result):
        # When mtf_consensus=1, mtf_trend_strength must equal 3
        both = result[["mtf_consensus", "mtf_trend_strength"]].dropna()
        aligned_rows = both[both["mtf_consensus"] == 1]
        if len(aligned_rows) > 0:
            assert (aligned_rows["mtf_trend_strength"] == 3).all(), (
                "consensus=1 should imply trend_strength=3"
            )


class TestMTFVolatilityRatios:
    """Volatility ratio columns are non-negative."""

    def test_vol_ratio_h1h4_non_negative(self, result):
        vals = result["mtf_vol_ratio_h1h4"].dropna()
        assert len(vals) > 0, "mtf_vol_ratio_h1h4: entirely NaN"
        assert (vals >= 0).all(), "mtf_vol_ratio_h1h4 contains negative values"

    def test_rsi_divergence_bounded(self, result):
        vals = result["mtf_rsi_divergence"].dropna()
        assert len(vals) > 0
        # RSI divergence = h1_rsi - h4_rsi; both in [0,100], so result in [-100,100]
        assert (vals >= -100).all()
        assert (vals <= 100).all()

    def test_w1_range_pos_non_empty(self, result):
        # w1_range_pos uses rolling(120) which populates well within 2000 bars
        vals = result["mtf_w1_range_pos"].dropna()
        assert len(vals) > 0, "mtf_w1_range_pos: entirely NaN with 2000 bars"

    def test_w1_ema_columns_declared(self, result):
        # W1 EMA columns (window=20*120=2400, 50*120=6000) need >2400 bars to populate.
        # With 2000 bars they remain NaN, but the columns must still be present.
        for col in ["mtf_w1_ema20_dist", "mtf_w1_ema50_dist", "mtf_w1_trend_strength"]:
            assert col in result.columns, f"Column not present: {col}"
