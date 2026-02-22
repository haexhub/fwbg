"""Tests for the market_structure indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_ms = import_plugin_module("fwbg-core", "indicators", "market_structure")
if _ms is None:
    pytest.skip("market_structure plugin not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_df(highs, lows, closes, opens=None):
    """Build a minimal OHLC DataFrame from arrays."""
    n = len(closes)
    if opens is None:
        opens = closes
    return pd.DataFrame(
        {
            "O": opens,
            "H": highs,
            "L": lows,
            "C": closes,
            "V": [100.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def _make_bull_bos_df(swing_lookback=5, n_post=4):
    """DataFrame with a clear bullish BOS.

    Bars 0..swing_lookback-1: stable at 100 (swing high = 100)
    Bar swing_lookback:       C=106 → breaks above swing high (100) → bull BOS
    Bars after:               C=107, price stays above
    """
    _n = swing_lookback + 1 + n_post
    highs  = [100.0] * swing_lookback + [106.0] + [107.0] * n_post
    lows   = [98.0]  * swing_lookback + [104.0] + [105.0] * n_post
    closes = [99.0]  * swing_lookback + [106.0] + [107.0] * n_post
    return _make_df(highs, lows, closes)


def _make_bear_bos_df(swing_lookback=5, n_post=4):
    """DataFrame with a clear bearish BOS.

    Bars 0..swing_lookback-1: stable at 100 (swing low = 100)
    Bar swing_lookback:       C=94 → breaks below swing low (100) → bear BOS
    Bars after:               C=93, price stays below
    """
    _n = swing_lookback + 1 + n_post
    highs  = [102.0] * swing_lookback + [99.0] + [98.0] * n_post
    lows   = [100.0] * swing_lookback + [93.0] + [92.0] * n_post
    closes = [101.0] * swing_lookback + [94.0] + [93.0] * n_post
    return _make_df(highs, lows, closes)


def _make_choch_df(swing_lookback=5, n_post=4):
    """DataFrame with bearish trend then bullish CHOCH.

    Phase 1 (bars 0..swing_lookback): declining — establishes bear BOS
    Phase 2 (bars swing_lookback+1): strong rally — bull BOS is the CHOCH
    """
    sw = swing_lookback
    # Phase 1: descending — swing low at bar sw, close breaks below → bear BOS
    _n = sw + 1 + 1 + n_post
    lows   = [100.0 - i for i in range(sw)] + [90.0] + [92.0] + [94.0] * n_post
    highs  = [102.0 - i for i in range(sw)] + [94.0] + [105.0] + [103.0] * n_post
    closes = [101.0 - i for i in range(sw)] + [91.0] + [104.0] + [103.0] * n_post
    return _make_df(highs, lows, closes)


def _make_random_df(n=500, seed=42):
    """General OHLC data with enough structure events for property tests."""
    np.random.seed(seed)
    prices = np.cumsum(np.random.randn(n) * 0.5) + 100
    df = pd.DataFrame(
        {
            "O": prices + np.random.randn(n) * 0.1,
            "H": prices + np.abs(np.random.randn(n) * 0.3) + 0.3,
            "L": prices - np.abs(np.random.randn(n) * 0.3) - 0.3,
            "C": prices,
            "V": np.ones(n) * 100,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


# ---------------------------------------------------------------------------
# Bullish BOS detection
# ---------------------------------------------------------------------------

class TestBullishBOS:
    """Tests for bullish Break-of-Structure detection."""

    def test_bos_bull_fires_when_close_breaks_above_swing_high(self):
        """C[i] > rolling_max_high[i-1] → ms_bos_bull=1."""
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        # BOS at bar sw; after shift appears at iloc[sw+1]
        assert result["ms_bos_bull"].iloc[sw + 1] == 1.0, (
            f"Expected ms_bos_bull=1 at iloc[{sw+1}], "
            f"got {result['ms_bos_bull'].iloc[sw+1]}"
        )

    def test_no_bos_bull_when_close_stays_at_swing_high(self):
        """C[i] == rolling_max_high → no bull BOS at that bar (strict >)."""
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw)
        # Close equals the swing high exactly → not a break at bar sw
        df.iloc[sw, df.columns.get_loc("C")] = 100.0
        df.iloc[sw, df.columns.get_loc("H")] = 100.0
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        # Only the specific bar where C==swing_high should not fire
        assert result["ms_bos_bull"].iloc[sw + 1] == 0.0, (
            "C == swing_high should not trigger BOS (strict >)"
        )

    def test_bos_bull_persists_trend_as_bullish(self):
        """After a bull BOS, ms_trend stays +1 in subsequent bars."""
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw, n_post=4)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        for offset in range(1, 4):
            assert result["ms_trend"].iloc[sw + 1 + offset] == 1.0, (
                f"Expected ms_trend=+1 at iloc[{sw+1+offset}]"
            )


# ---------------------------------------------------------------------------
# Bearish BOS detection
# ---------------------------------------------------------------------------

class TestBearishBOS:
    """Tests for bearish Break-of-Structure detection."""

    def test_bos_bear_fires_when_close_breaks_below_swing_low(self):
        """C[i] < rolling_min_low[i-1] → ms_bos_bear=1."""
        sw = 5
        df = _make_bear_bos_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        assert result["ms_bos_bear"].iloc[sw + 1] == 1.0, (
            f"Expected ms_bos_bear=1 at iloc[{sw+1}], "
            f"got {result['ms_bos_bear'].iloc[sw+1]}"
        )

    def test_no_bos_bear_when_close_stays_at_swing_low(self):
        """C[i] == rolling_min_low → no bear BOS at that bar (strict <)."""
        sw = 5
        df = _make_bear_bos_df(swing_lookback=sw)
        df.iloc[sw, df.columns.get_loc("C")] = 100.0
        df.iloc[sw, df.columns.get_loc("L")] = 100.0
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        # Only the specific bar where C==swing_low should not fire
        assert result["ms_bos_bear"].iloc[sw + 1] == 0.0, (
            "C == swing_low should not trigger BOS (strict <)"
        )

    def test_bos_bear_persists_trend_as_bearish(self):
        """After a bear BOS, ms_trend stays -1 in subsequent bars."""
        sw = 5
        df = _make_bear_bos_df(swing_lookback=sw, n_post=4)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        for offset in range(1, 4):
            assert result["ms_trend"].iloc[sw + 1 + offset] == -1.0, (
                f"Expected ms_trend=-1 at iloc[{sw+1+offset}]"
            )


# ---------------------------------------------------------------------------
# Trend state
# ---------------------------------------------------------------------------

class TestTrendState:
    """Tests for ms_trend state machine."""

    def test_trend_starts_neutral(self):
        """Before any BOS, ms_trend should be 0."""
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        # All bars before the BOS (after shift: iloc[1..sw]) should be neutral
        pre_bos = result["ms_trend"].iloc[1: sw + 1]
        assert (pre_bos == 0.0).all(), (
            f"Expected ms_trend=0 before BOS, got {pre_bos.values}"
        )

    def test_trend_flips_bull_after_bos_bull(self):
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        assert result["ms_trend"].iloc[sw + 1] == 1.0

    def test_trend_flips_bear_after_bos_bear(self):
        sw = 5
        df = _make_bear_bos_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        assert result["ms_trend"].iloc[sw + 1] == -1.0


# ---------------------------------------------------------------------------
# CHOCH detection
# ---------------------------------------------------------------------------

class TestCHOCH:
    """Tests for Change of Character detection."""

    def test_choch_bull_fires_on_first_bull_bos_after_bear_trend(self):
        """Bull BOS after bearish trend = bullish CHOCH."""
        sw = 5
        df = _make_choch_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        # bear BOS at bar sw → trend=-1; bull BOS at bar sw+1 → CHOCH
        choch_bar = sw + 1
        assert result["ms_choch_bull"].iloc[choch_bar + 1] == 1.0, (
            f"Expected ms_choch_bull=1 at iloc[{choch_bar+1}], "
            f"got {result['ms_choch_bull'].iloc[choch_bar+1]}"
        )

    def test_choch_bull_only_fires_once(self):
        """CHOCH fires on transition, not on every subsequent BOS."""
        sw = 5
        df = _make_choch_df(swing_lookback=sw, n_post=4)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        # Total CHOCH_bull count should be 1 (only on the transition bar)
        total = result["ms_choch_bull"].fillna(0).sum()
        assert total == 1.0, f"Expected exactly 1 CHOCH_bull, got {total}"

    def test_no_choch_on_first_bos_from_neutral(self):
        """First BOS from neutral trend is BOS but NOT CHOCH."""
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)
        # BOS fires, but CHOCH should not (trend was neutral, not bearish)
        assert result["ms_choch_bull"].fillna(0).sum() == 0.0, (
            "No CHOCH_bull expected when transitioning from neutral"
        )

    def test_choch_bear_fires_on_first_bear_bos_after_bull_trend(self):
        """Bear BOS after bullish trend = bearish CHOCH."""
        sw = 5
        # Build a bull BOS first, then a bear BOS
        _n = sw + 3 + 4
        # Bars 0..sw-1: low at 100 (establishing swing low), high at 110
        # Bar sw:       C=106 > swing_high(=102) → bull BOS, trend=+1
        # Bar sw+1:     C=108
        # Bar sw+2:     C=94 < swing_low_prev → bear BOS → CHOCH_bear
        highs  = [102.0] * sw + [106.0, 108.0, 96.0] + [95.0] * 4
        lows   = [100.0] * sw + [104.0, 106.0, 93.0] + [92.0] * 4
        closes = [101.0] * sw + [106.0, 107.0, 94.0] + [93.0] * 4
        df = _make_df(highs, lows, closes)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        choch_bar = sw + 2
        assert result["ms_choch_bear"].iloc[choch_bar + 1] == 1.0, (
            f"Expected ms_choch_bear=1 at iloc[{choch_bar+1}]"
        )


# ---------------------------------------------------------------------------
# Distance features
# ---------------------------------------------------------------------------

class TestDistanceFeatures:
    """Tests for BOS-level distance and swing-distance features."""

    def test_bull_bos_dist_positive_after_bull_bos(self):
        """After a bull BOS, close is above the broken level → dist > 0."""
        sw = 5
        df = _make_bull_bos_df(swing_lookback=sw, n_post=3)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        # After BOS at bar sw, dist at iloc[sw+1] should be positive
        dist = result["ms_bull_bos_dist"].iloc[sw + 1]
        assert not np.isnan(dist), "ms_bull_bos_dist should not be NaN after bull BOS"
        assert dist >= 0, f"Expected positive dist after bull BOS, got {dist}"

    def test_bear_bos_dist_positive_after_bear_bos(self):
        """After a bear BOS, close is below the broken level → dist > 0."""
        sw = 5
        df = _make_bear_bos_df(swing_lookback=sw, n_post=3)
        result = _ms.MarketStructureIndicator().compute(df, swing_lookback=sw)

        dist = result["ms_bear_bos_dist"].iloc[sw + 1]
        assert not np.isnan(dist), "ms_bear_bos_dist should not be NaN after bear BOS"
        assert dist >= 0, f"Expected positive dist after bear BOS, got {dist}"

    def test_swing_high_dist_non_negative(self):
        """Distance to swing high above should be >= 0 (close is below or at swing high)."""
        df = _make_random_df()
        result = _ms.MarketStructureIndicator().compute(df)
        vals = result["ms_swing_high_dist"].dropna()
        assert (vals >= 0).all(), "ms_swing_high_dist contains negative values"

    def test_swing_low_dist_non_negative(self):
        """Distance to swing low below should be >= 0 (close is above or at swing low)."""
        df = _make_random_df()
        result = _ms.MarketStructureIndicator().compute(df)
        vals = result["ms_swing_low_dist"].dropna()
        assert (vals >= 0).all(), "ms_swing_low_dist contains negative values"


# ---------------------------------------------------------------------------
# Feature properties
# ---------------------------------------------------------------------------

class TestFeatureProperties:
    """Tests for feature shapes, types and values."""

    def test_all_features_present(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df())
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self):
        indicator = _ms.MarketStructureIndicator()
        assert len(indicator.get_feature_columns()) == 10

    def test_binary_features(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df())
        for col in ["ms_bos_bull", "ms_bos_bear", "ms_choch_bull", "ms_choch_bear"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"

    def test_trend_values_in_valid_set(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df())
        vals = result["ms_trend"].dropna().unique()
        assert set(vals).issubset({-1.0, 0.0, 1.0}), f"ms_trend has unexpected values: {vals}"

    def test_recency_in_unit_interval(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df())
        vals = result["ms_choch_recency"].dropna()
        if len(vals) > 0:
            assert vals.min() >= 0.0, "ms_choch_recency < 0"
            assert vals.max() <= 1.0, "ms_choch_recency > 1"

    def test_shift_applied(self):
        """All features must be shifted by 1 bar (lookahead prevention)."""
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df(n=100))
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted (iloc[0] not NaN)"

    def test_no_inf_values(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df())
        for col in indicator.get_feature_columns():
            inf_count = np.isinf(result[col].dropna()).sum()
            assert inf_count == 0, f"{col} has {inf_count} inf values"

    def test_not_all_nan(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df(n=500))
        late = result.iloc[50:]
        for col in indicator.get_feature_columns():
            vals = late[col].dropna()
            assert len(vals) > 0, f"{col} is all NaN after warmup"

    def test_no_undeclared_features(self):
        indicator = _ms.MarketStructureIndicator()
        df = _make_random_df(n=200)
        original_cols = set(df.columns)
        result = indicator.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(indicator.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared features: {undeclared}"


# ---------------------------------------------------------------------------
# Plugin integration
# ---------------------------------------------------------------------------

class TestPluginIntegration:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("market_structure")
        assert cls is not None

    def test_all_declared_features_in_output(self):
        indicator = _ms.MarketStructureIndicator()
        result = indicator.compute(_make_random_df(n=200))
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Declared {col} missing from output"
