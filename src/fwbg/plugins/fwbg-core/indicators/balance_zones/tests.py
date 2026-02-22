"""Tests for the balance_zones indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_bz = import_plugin_module("fwbg-core", "indicators", "balance_zones")
if _bz is None:
    pytest.skip("balance_zones plugin not available", allow_module_level=True)


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


def _make_balanced_df(lookback=10, n_post=5):
    """DataFrame with a tight balance zone then a bullish breakout.

    Bars 0..lookback-1: O=99, C=101 → body [99, 101].
    Bar lookback:       O=102, C=105 → bullish breakout (close > 101).
    Bars after:         O=102, C=103 → price stays above (continuation).
    """
    _n = lookback + 1 + n_post
    opens  = [99.0]  * lookback + [102.0] + [102.0] * n_post
    closes = [101.0] * lookback + [105.0] + [103.0] * n_post
    highs  = [c + 0.5 for c in closes]
    lows   = [o - 0.5 for o in opens]
    return _make_df(highs, lows, closes, opens)


def _make_bear_breakout_df(lookback=10, n_post=5):
    """DataFrame with a tight balance zone then a bearish breakout.

    Bars 0..lookback-1: O=101, C=99 → body [99, 101].
    Bar lookback:       O=98, C=95 → bearish breakout (close < 99).
    Bars after:         stay below.
    """
    _n = lookback + 1 + n_post
    opens  = [101.0] * lookback + [98.0]  + [98.0]  * n_post
    closes = [99.0]  * lookback + [95.0]  + [97.0]  * n_post
    highs  = [c + 0.5 for c in closes]
    lows   = [o - 0.5 for o in opens]
    return _make_df(highs, lows, closes, opens)


def _make_fake_bear_df(lookback=10):
    """DataFrame demonstrating a fake bull breakout (bz_fake_bear signal).

    Bars 0..lookback-1: stable bodies [99, 101].
        → zone_top[lookback-1] = 101, zone_bottom[lookback-1] = 99.
    Bar lookback:       O=102, C=104 → breakout above 101.
    Bar lookback+1:     O=100, C=100 → close back below old zone_top (101).
        → bz_fake_bear fires (after shift appears at iloc[lookback+2]).
    """
    lb = lookback
    _n = lb + 2 + 3
    opens  = [99.0]  * lb + [102.0, 100.0] + [100.0] * 3
    closes = [101.0] * lb + [104.0, 100.0] + [100.0] * 3
    highs  = [101.5] * lb + [104.5, 100.5] + [100.5] * 3
    lows   = [98.5]  * lb + [101.5,  99.5] + [ 99.5] * 3
    return _make_df(highs, lows, closes, opens)


def _make_fake_bull_df(lookback=10):
    """DataFrame demonstrating a fake bear breakout (bz_fake_bull signal).

    Bars 0..lookback-1: stable bodies [99, 101].
        → zone_bottom[lookback-1] = 99.
    Bar lookback:       O=98, C=96 → breakout below 99.
    Bar lookback+1:     O=100, C=100 → close back above old zone_bottom (99).
        → bz_fake_bull fires (after shift appears at iloc[lookback+2]).
    """
    lb = lookback
    _n = lb + 2 + 3
    opens  = [101.0] * lb + [ 98.0, 100.0] + [100.0] * 3
    closes = [ 99.0] * lb + [ 96.0, 100.0] + [100.0] * 3
    highs  = [ 99.5] * lb + [ 98.5, 100.5] + [100.5] * 3
    lows   = [ 98.5] * lb + [ 95.5,  99.5] + [ 99.5] * 3
    return _make_df(highs, lows, closes, opens)


def _make_random_df(n=500, seed=42):
    """General OHLC data for property tests."""
    np.random.seed(seed)
    prices = np.cumsum(np.random.randn(n) * 0.5) + 100
    opens_arr = prices + np.random.randn(n) * 0.1
    closes_arr = prices
    df = pd.DataFrame(
        {
            "O": opens_arr,
            "H": prices + np.abs(np.random.randn(n) * 0.3) + 0.3,
            "L": prices - np.abs(np.random.randn(n) * 0.3) - 0.3,
            "C": closes_arr,
            "V": np.ones(n) * 100,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

class TestBreakout:
    """Tests for bullish and bearish breakout detection."""

    def test_breakout_bull_fires_when_close_above_zone_top(self):
        """C > rolling_max(body_top, lookback)[i-1] → bz_breakout_bull=1."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Breakout at bar lb; after shift appears at iloc[lb+1]
        assert result["bz_breakout_bull"].iloc[lb + 1] == 1.0, (
            f"Expected bz_breakout_bull=1 at iloc[{lb+1}], "
            f"got {result['bz_breakout_bull'].iloc[lb+1]}"
        )

    def test_no_breakout_bull_when_close_equals_zone_top(self):
        """C == zone_top → no bull breakout (strict >)."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        # Force close at bar lb to exactly match zone_top = 101
        df.iloc[lb, df.columns.get_loc("C")] = 101.0
        df.iloc[lb, df.columns.get_loc("H")] = 101.5
        df.iloc[lb, df.columns.get_loc("O")] = 100.0
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)
        assert result["bz_breakout_bull"].iloc[lb + 1] == 0.0, (
            "C == zone_top should not trigger breakout_bull (strict >)"
        )

    def test_breakout_bear_fires_when_close_below_zone_bottom(self):
        """C < rolling_min(body_bottom, lookback)[i-1] → bz_breakout_bear=1."""
        lb = 10
        df = _make_bear_breakout_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        assert result["bz_breakout_bear"].iloc[lb + 1] == 1.0, (
            f"Expected bz_breakout_bear=1 at iloc[{lb+1}], "
            f"got {result['bz_breakout_bear'].iloc[lb+1]}"
        )

    def test_no_breakout_bear_when_close_equals_zone_bottom(self):
        """C == zone_bottom → no bear breakout (strict <)."""
        lb = 10
        df = _make_bear_breakout_df(lookback=lb)
        # Force close at bar lb to exactly match zone_bottom = 99
        df.iloc[lb, df.columns.get_loc("C")] = 99.0
        df.iloc[lb, df.columns.get_loc("L")] = 98.5
        df.iloc[lb, df.columns.get_loc("O")] = 100.0
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)
        assert result["bz_breakout_bear"].iloc[lb + 1] == 0.0, (
            "C == zone_bottom should not trigger breakout_bear (strict <)"
        )

    def test_no_breakout_when_inside_zone(self):
        """While price stays inside the zone, neither breakout fires."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Bars 1..lb are in-zone (after shift: iloc[2..lb+1])
        for i in range(2, lb + 1):
            assert result["bz_breakout_bull"].iloc[i] == 0.0, (
                f"bz_breakout_bull should be 0 inside zone at iloc[{i}]"
            )
            assert result["bz_breakout_bear"].iloc[i] == 0.0, (
                f"bz_breakout_bear should be 0 inside zone at iloc[{i}]"
            )


# ---------------------------------------------------------------------------
# In-zone detection
# ---------------------------------------------------------------------------

class TestInZone:
    """Tests for bz_in_zone detection."""

    def test_in_zone_true_when_close_inside_zone(self):
        """Close inside [zone_bottom, zone_top] → bz_in_zone=1."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Bars 1..lb-1: close=101 inside zone [99, 101] → after shift iloc[2..lb]
        for i in range(2, lb + 1):
            assert result["bz_in_zone"].iloc[i] == 1.0, (
                f"Expected bz_in_zone=1 at iloc[{i}], got {result['bz_in_zone'].iloc[i]}"
            )

    def test_in_zone_false_when_close_above_zone_top(self):
        """Close above zone_top → bz_in_zone=0."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Bar lb: close=105 > zone_top=101 → after shift iloc[lb+1]
        assert result["bz_in_zone"].iloc[lb + 1] == 0.0, (
            f"Expected bz_in_zone=0 when close above zone, got {result['bz_in_zone'].iloc[lb+1]}"
        )

    def test_in_zone_false_when_close_below_zone_bottom(self):
        """Close below zone_bottom → bz_in_zone=0."""
        lb = 10
        df = _make_bear_breakout_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Bar lb: close=95 < zone_bottom=99 → after shift iloc[lb+1]
        assert result["bz_in_zone"].iloc[lb + 1] == 0.0, (
            f"Expected bz_in_zone=0 when close below zone, got {result['bz_in_zone'].iloc[lb+1]}"
        )


# ---------------------------------------------------------------------------
# In-balance detection
# ---------------------------------------------------------------------------

class TestInBalance:
    """Tests for bz_in_balance (tight zone relative to ATR)."""

    def test_in_balance_true_when_zone_width_small(self):
        """Zone width / ATR <= balance_atr_threshold → bz_in_balance=1."""
        lb = 10
        # Tight zone: bodies [99, 101] → width=2, ATR ≈ 2 (H-L per bar ≈ 2)
        df = _make_balanced_df(lookback=lb)
        # With default balance_atr_threshold=2.0 and zone_width/ATR ≈ 1.0, should be in balance
        result = _bz.BalanceZonesIndicator().compute(
            df, lookback=lb, balance_atr_threshold=2.0
        )
        # Bars in the stable zone (iloc[2..lb]) should be in balance
        for i in range(2, lb + 1):
            val = result["bz_in_balance"].iloc[i]
            assert val == 1.0, (
                f"Expected bz_in_balance=1 for tight zone at iloc[{i}], got {val}"
            )

    def test_in_balance_false_when_zone_width_large(self):
        """Zone width / ATR > balance_atr_threshold → bz_in_balance=0."""
        lb = 10
        # Wide zone: bodies ranging from 90 to 110, ATR small
        _n = lb + 5
        opens  = list(range(90, 90 + lb)) + [100.0] * 5
        closes = list(range(91, 91 + lb)) + [100.0] * 5
        highs  = [c + 0.2 for c in closes]
        lows   = [o - 0.2 for o in opens]
        df = _make_df(highs, lows, closes, opens)
        # Zone width after lb bars ≈ 20 (body_top max=100+lb-1 - body_bottom min=90)
        # ATR ≈ 1.2 (tight candles), so width/ATR >> 2.0 → not in balance
        result = _bz.BalanceZonesIndicator().compute(
            df, lookback=lb, balance_atr_threshold=2.0
        )
        # The later bars (after zone has expanded) should not be in balance
        val = result["bz_in_balance"].iloc[lb]
        assert val == 0.0, (
            f"Expected bz_in_balance=0 for wide zone, got {val}"
        )


# ---------------------------------------------------------------------------
# Fake breakout detection
# ---------------------------------------------------------------------------

class TestFakeBreakout:
    """Tests for bz_fake_bear and bz_fake_bull signals."""

    def test_fake_bear_fires_after_bull_breakout_reversal(self):
        """Fake bull breakout: close was above old zone_top, now back below it."""
        lb = 10
        df = _make_fake_bear_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Breakout at bar lb (close=104 > zone_top=101)
        # Re-entry at bar lb+1 (close=100 < old_zone_top=101)
        # After shift → appears at iloc[lb+2]
        fake_bar = lb + 2
        assert result["bz_fake_bear"].iloc[fake_bar] == 1.0, (
            f"Expected bz_fake_bear=1 at iloc[{fake_bar}], "
            f"got {result['bz_fake_bear'].iloc[fake_bar]}"
        )

    def test_fake_bull_fires_after_bear_breakout_reversal(self):
        """Fake bear breakout: close was below old zone_bottom, now back above it."""
        lb = 10
        df = _make_fake_bull_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Breakout at bar lb (close=96 < zone_bottom=99)
        # Re-entry at bar lb+1 (close=100 > old_zone_bottom=99)
        # After shift → appears at iloc[lb+2]
        fake_bar = lb + 2
        assert result["bz_fake_bull"].iloc[fake_bar] == 1.0, (
            f"Expected bz_fake_bull=1 at iloc[{fake_bar}], "
            f"got {result['bz_fake_bull'].iloc[fake_bar]}"
        )

    def test_no_fake_bear_without_prior_breakout(self):
        """No fake bear signal while price just stays inside the zone."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # During the stable phase (before breakout), no fake signals
        for i in range(2, lb + 1):
            assert result["bz_fake_bear"].iloc[i] == 0.0, (
                f"bz_fake_bear should be 0 without prior breakout at iloc[{i}]"
            )

    def test_no_fake_bull_without_prior_breakout(self):
        """No fake bull signal while price just stays inside the zone."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        for i in range(2, lb + 1):
            assert result["bz_fake_bull"].iloc[i] == 0.0, (
                f"bz_fake_bull should be 0 without prior breakout at iloc[{i}]"
            )


# ---------------------------------------------------------------------------
# Zone distance features
# ---------------------------------------------------------------------------

class TestDistances:
    """Tests for bz_zone_top_dist and bz_zone_bottom_dist."""

    def test_zone_top_dist_zero_when_above_zone_top(self):
        """Close > zone_top → bz_zone_top_dist=0 (clamped, price is above)."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # After breakout, close=105 > zone_top=101 → dist = 0
        assert result["bz_zone_top_dist"].iloc[lb + 1] == 0.0, (
            f"Expected bz_zone_top_dist=0 when above zone top, "
            f"got {result['bz_zone_top_dist'].iloc[lb + 1]}"
        )

    def test_zone_top_dist_positive_when_below_zone_top(self):
        """Close < zone_top while in zone → bz_zone_top_dist > 0."""
        lb = 10
        df = _make_balanced_df(lookback=lb)
        _result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # In-zone bars: close=101 == zone_top=101 → dist = 0; use earlier bars
        # close=101, zone_top after 5 bars = 101; but at bar 2 (close=101, zone so far)
        # After warmup with close < zone_top:
        # Let's check that the feature is non-negative (covered by property test)
        # Verify positivity with a bar clearly inside
        n_small = lb + 5
        opens2  = [99.0] * n_small
        closes2 = [100.0] * n_small  # always in middle of zone [99, 101]
        highs2  = [101.5] * n_small
        lows2   = [98.5]  * n_small
        df2 = _make_df(highs2, lows2, closes2, opens2)
        _result2 = _bz.BalanceZonesIndicator().compute(df2, lookback=lb)

        # zone_top = 101 (body_top = max(99, 100) = 100... wait body_top = max(O,C) = 100)
        # zone_top = 100, close = 100 → dist = 0. Let me use close < body_top...
        opens3  = [99.0]  * n_small
        closes3 = [99.5]  * n_small  # close=99.5, zone_top=rolling max body_top = max(99,99.5)=99.5
        highs3  = [100.5] * n_small
        lows3   = [98.5]  * n_small
        _df3 = _make_df(highs3, lows3, closes3, opens3)
        # After zone is established (zone_top=99.5), close=99.5 == zone_top → dist = 0
        # To test dist > 0, we need close < zone_top
        opens4  = [99.0]  * lb + [98.0] * 5
        closes4 = [101.0] * lb + [99.0] * 5  # after lb bars zone_top=101, close=99 < 101 → dist>0
        highs4  = [101.5] * lb + [99.5] * 5
        lows4   = [98.5]  * lb + [97.5] * 5
        df4 = _make_df(highs4, lows4, closes4, opens4)
        result4 = _bz.BalanceZonesIndicator().compute(df4, lookback=lb)

        # At bar lb+1: zone_top[lb] = 101 (from the stable phase), close=99 → dist > 0
        val = result4["bz_zone_top_dist"].iloc[lb + 2]
        assert val > 0.0, (
            f"Expected bz_zone_top_dist > 0 when close below zone_top, got {val}"
        )

    def test_zone_bottom_dist_zero_when_below_zone_bottom(self):
        """Close < zone_bottom (after bear breakout) → bz_zone_bottom_dist=0."""
        lb = 10
        df = _make_bear_breakout_df(lookback=lb)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # After breakout, close=95 < zone_bottom=99 → dist = 0
        assert result["bz_zone_bottom_dist"].iloc[lb + 1] == 0.0, (
            f"Expected bz_zone_bottom_dist=0 when below zone bottom, "
            f"got {result['bz_zone_bottom_dist'].iloc[lb + 1]}"
        )

    def test_zone_bottom_dist_positive_when_above_zone_bottom(self):
        """Close inside zone (above zone_bottom) → bz_zone_bottom_dist > 0."""
        lb = 10
        # Build a scenario where zone_bottom=99 and close > 99
        opens_  = [99.0]  * lb + [100.0] * 5
        closes_ = [101.0] * lb + [100.0] * 5  # zone_bottom=99, close=100>99
        highs_  = [101.5] * lb + [100.5] * 5
        lows_   = [ 98.5] * lb + [ 99.5] * 5
        df = _make_df(highs_, lows_, closes_, opens_)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # At bar lb+1: zone_bottom[lb]=99, close=100 → dist = (100-99)/ATR > 0
        val = result["bz_zone_bottom_dist"].iloc[lb + 2]
        assert val > 0.0, (
            f"Expected bz_zone_bottom_dist > 0 when close inside zone, got {val}"
        )


# ---------------------------------------------------------------------------
# Balance bars
# ---------------------------------------------------------------------------

class TestBalanceBars:
    """Tests for bz_balance_bars accumulation and reset."""

    def test_balance_bars_increases_while_in_zone(self):
        """bz_balance_bars should increase as price stays inside zone."""
        lb = 10
        df = _make_balanced_df(lookback=lb, n_post=0)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # While in zone (bars 1..lb-1), balance_bars should grow monotonically
        # After shift: visible at iloc[2..lb]
        prev_val = -1.0
        for i in range(2, lb + 1):
            val = result["bz_balance_bars"].iloc[i]
            if not np.isnan(val):
                assert val >= prev_val, (
                    f"bz_balance_bars should be non-decreasing while in zone, "
                    f"dropped at iloc[{i}]: {prev_val} → {val}"
                )
                prev_val = val

    def test_balance_bars_drops_after_breakout(self):
        """After a breakout, bz_balance_bars should drop (reset or decrease)."""
        lb = 10
        df = _make_balanced_df(lookback=lb, n_post=3)
        result = _bz.BalanceZonesIndicator().compute(df, lookback=lb)

        # Just before breakout (iloc[lb]): accumulated in-balance bars
        val_before = result["bz_balance_bars"].iloc[lb]
        # After breakout (iloc[lb+1]): close is outside zone → should be lower/reset
        val_after = result["bz_balance_bars"].iloc[lb + 1]
        assert val_after < val_before, (
            f"bz_balance_bars should drop after breakout: "
            f"{val_before} → {val_after}"
        )


# ---------------------------------------------------------------------------
# Feature properties
# ---------------------------------------------------------------------------

class TestFeatureProperties:
    """Tests for feature shapes, types, and value ranges."""

    def test_all_features_present(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df())
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self):
        indicator = _bz.BalanceZonesIndicator()
        assert len(indicator.get_feature_columns()) == 10

    def test_binary_features(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df())
        for col in [
            "bz_in_balance", "bz_in_zone",
            "bz_breakout_bull", "bz_breakout_bear",
            "bz_fake_bear", "bz_fake_bull",
        ]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"

    def test_zone_width_non_negative(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df())
        vals = result["bz_zone_width"].dropna()
        assert (vals >= 0).all(), "bz_zone_width contains negative values"

    def test_distance_features_non_negative(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df())
        for col in ["bz_zone_top_dist", "bz_zone_bottom_dist"]:
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"{col} contains negative values"

    def test_balance_bars_in_unit_interval(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df(n=500))
        vals = result["bz_balance_bars"].dropna()
        if len(vals) > 0:
            assert vals.min() >= 0.0, "bz_balance_bars < 0"
            assert vals.max() <= 1.0, "bz_balance_bars > 1"

    def test_shift_applied(self):
        """All features must be shifted by 1 bar (lookahead prevention)."""
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df(n=100))
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), (
                f"{col} not shifted (iloc[0] not NaN)"
            )

    def test_no_inf_values(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df())
        for col in indicator.get_feature_columns():
            inf_count = np.isinf(result[col].dropna()).sum()
            assert inf_count == 0, f"{col} has {inf_count} inf values"

    def test_not_all_nan(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df(n=500))
        late = result.iloc[50:]
        for col in indicator.get_feature_columns():
            vals = late[col].dropna()
            assert len(vals) > 0, f"{col} is all NaN after warmup"

    def test_no_undeclared_features(self):
        indicator = _bz.BalanceZonesIndicator()
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
        cls = get_indicator("balance_zones")
        assert cls is not None

    def test_all_declared_features_in_output(self):
        indicator = _bz.BalanceZonesIndicator()
        result = indicator.compute(_make_random_df(n=200))
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Declared {col} missing from output"
