"""End-to-end tests: PDH/PDL retest strategy correctness.

Verifies that the bot trades ONLY when the strategy rules are met:
1. Breakout above PDH → retracement to midpoint (PDH+PDL)/2 → Long entry
2. Breakout below PDL → retracement to midpoint → Short entry
3. No breakout → no trade
4. Breakout but no retracement → no trade
5. Max 1 trade per direction per day (event signal, not state)

Uses synthetic OHLC data with known PDH/PDL levels, runs the full flow:
  indicator → SignalModel → trade simulation → verify entries.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import _simulate_trades_core
from fwbg.plugins import import_plugin_module

_pdl_mod = import_plugin_module("fwbg-core", "indicators", "previous_day_levels")
_signal_mod = import_plugin_module("fwbg-core", "models", "signal")
if _pdl_mod is None or _signal_mod is None:
    pytest.skip("Required plugins not available", allow_module_level=True)

from fwbg_sdk.models import TrainingContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_ctx(**overrides) -> SimulationContext:
    """Minimal SimulationContext for testing with fixed exit strategy."""
    defaults = dict(
        symbol="TEST",
        asset_class="forex",
        spread=0.5,
        point=0.01,
        min_trades=1,
        long_enabled=True,
        short_enabled=True,
        exit_strategy="fixed",
        model_type="signal",
        model_hyperparameters={
            "signal_column_long": "pdl_retest_bull",
            "signal_column_short": "pdl_retest_bear",
        },
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def _run_strategy(df: pd.DataFrame, ctx: SimulationContext, tp: float = 6, sl: float = 4):
    """Full pipeline: indicator → SignalModel → trade simulation.

    tp/sl are multipliers of ctx.spread.  With spread=0.5:
      tp_dist = 0.5 * 6 = 3.0 points, sl_dist = 0.5 * 4 = 2.0 points.

    Returns list of detailed trade dicts.
    """
    # 1) Compute indicator
    ind = _pdl_mod.PreviousDayLevelsIndicator()
    df_feat = ind.compute(df.copy(), retest_atr_width=0.5)

    # 2) Train SignalModel for long and short
    model_long = _signal_mod.SignalModel()
    model_short = _signal_mod.SignalModel()

    feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]
    features = df_feat[feature_cols].fillna(0)

    # Dummy targets (SignalModel ignores them)
    dummy_targets = np.zeros(len(features))

    hp = ctx.model_hyperparameters
    model_long.train(features, dummy_targets, TrainingContext(direction="long"), **hp)
    model_short.train(features, dummy_targets, TrainingContext(direction="short"), **hp)

    # 3) Get predictions
    probs_long = model_long.predict_probability(features)
    probs_short = model_short.predict_probability(features)

    long_win_idx = np.where(model_long.trained_classes == 1)[0][0]
    short_win_idx = np.where(model_short.trained_classes == 1)[0][0]

    # 4) Run trade simulation
    result = _simulate_trades_core(
        df=df_feat,
        probs_long=probs_long,
        probs_short=probs_short,
        long_win_idx=long_win_idx,
        short_win_idx=short_win_idx,
        ct_long=0.5,
        ct_short=0.5,
        tp=tp,
        sl=sl,
        ctx=ctx,
        return_detailed=True,
    )

    return result.get("trades_detailed", [])


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_pdhl_bull_retest():
    """3-day hourly data: Day 1 has breakout above PDH + retracement to midpoint.

    Day 0 (Jan 1): range 90..110 → PDH=110, PDL=90, midpoint=100
    Day 1 (Jan 2):
      - 00:00-08:00: price at 105 (between PDL and PDH, no breakout)
      - 09:00: breakout above PDH=110 → close=115
      - 10:00-11:00: stays above → close=112
      - 12:00: retracement to midpoint=100 → retest_bull should fire
      - 13:00-23:00: stays around 102, no second signal
    Day 2 (Jan 3): flat at 100, no signals (PDH/PDL from day 1)
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    # Day 0: set range 90-110
    for i in range(24):
        high[i] = 110.0
        low[i] = 90.0

    # Day 1
    for i in range(24, 48):
        h = idx[i].hour
        if h < 9:
            close[i] = 105.0; high[i] = 106.0; low[i] = 104.0; opn[i] = 105.0
        elif h == 9:
            # Breakout above PDH=110
            close[i] = 115.0; high[i] = 116.0; low[i] = 108.0; opn[i] = 108.0
        elif h <= 11:
            close[i] = 112.0; high[i] = 113.0; low[i] = 111.0; opn[i] = 112.0
        elif h == 12:
            # Retracement to midpoint=100
            close[i] = 100.0; high[i] = 105.0; low[i] = 99.0; opn[i] = 105.0
        elif h == 13:
            # Entry bar (signal at 13:00, entry at Open of 14:00)
            close[i] = 101.0; high[i] = 102.0; low[i] = 100.0; opn[i] = 100.0
        else:
            # Price rallies after entry → should hit TP
            close[i] = 108.0; high[i] = 110.0; low[i] = 106.0; opn[i] = 106.0

    # Day 2: flat at 108 (no new signals)
    for i in range(48, 72):
        close[i] = 108.0; high[i] = 109.0; low[i] = 107.0; opn[i] = 108.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


def _make_pdhl_bear_retest():
    """3-day hourly data: Day 1 has breakout below PDL + retracement to midpoint.

    Day 0: range 90..110 → PDH=110, PDL=90, midpoint=100
    Day 1:
      - 00:00-08:00: price at 95 (inside range)
      - 09:00: breakout below PDL=90 → close=85
      - 10:00-11:00: stays below → close=87
      - 12:00: retracement UP to midpoint=100 → retest_bear should fire
      - 13:00+: stays around 98
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    for i in range(24):
        high[i] = 110.0; low[i] = 90.0

    for i in range(24, 48):
        h = idx[i].hour
        if h < 9:
            close[i] = 95.0; high[i] = 96.0; low[i] = 94.0; opn[i] = 95.0
        elif h == 9:
            close[i] = 85.0; high[i] = 92.0; low[i] = 84.0; opn[i] = 92.0
        elif h <= 11:
            close[i] = 87.0; high[i] = 88.0; low[i] = 86.0; opn[i] = 87.0
        elif h == 12:
            close[i] = 100.0; high[i] = 101.0; low[i] = 95.0; opn[i] = 95.0
        elif h == 13:
            # Entry bar (signal at 13:00, entry at Open of 14:00)
            close[i] = 99.0; high[i] = 100.0; low[i] = 98.0; opn[i] = 100.0
        else:
            # Price drops after entry → should hit TP for short
            close[i] = 92.0; high[i] = 94.0; low[i] = 90.0; opn[i] = 94.0

    for i in range(48, 72):
        close[i] = 92.0; high[i] = 93.0; low[i] = 91.0; opn[i] = 92.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


def _make_no_breakout():
    """3-day hourly data: price never breaks PDH/PDL → no trades.

    Day 0: range 90..110
    Day 1-2: price stays at 100 (inside range)
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    for i in range(24):
        high[i] = 110.0; low[i] = 90.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


def _make_breakout_no_retracement():
    """3-day hourly data: breakout above PDH but price never returns to midpoint.

    Day 0: range 90..110 → PDH=110, PDL=90, midpoint=100
    Day 1: breaks above 110, stays high (112-120), never touches midpoint=100
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    for i in range(24):
        high[i] = 110.0; low[i] = 90.0

    for i in range(24, 48):
        h = idx[i].hour
        if h < 9:
            close[i] = 108.0; high[i] = 109.0; low[i] = 107.0; opn[i] = 108.0
        elif h == 9:
            close[i] = 115.0; high[i] = 116.0; low[i] = 112.0; opn[i] = 112.0
        else:
            # Stays high, never near midpoint=100
            close[i] = 118.0; high[i] = 120.0; low[i] = 116.0; opn[i] = 118.0

    for i in range(48, 72):
        close[i] = 118.0; high[i] = 119.0; low[i] = 117.0; opn[i] = 118.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPDHLBullRetest:
    """Breakout above PDH + retracement to midpoint → exactly 1 Long trade."""

    def test_exactly_one_long_trade(self):
        df = _make_pdhl_bull_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        assert len(long_trades) == 1, (
            f"Expected exactly 1 LONG trade, got {len(long_trades)}. "
            f"All trades: {[(t['direction'], t.get('signal_time')) for t in trades]}"
        )

    def test_long_entry_on_day1_after_retracement(self):
        df = _make_pdhl_bull_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        assert len(long_trades) >= 1

        trade = long_trades[0]
        signal_time = pd.Timestamp(trade["signal_time"])
        # Indicator computes retest at 12:00, shift_features shifts to 13:00
        assert signal_time.date() == pd.Timestamp("2024-01-02").date(), (
            f"Trade signal should be on 2024-01-02, got {signal_time.date()}"
        )
        assert signal_time.hour == 13, (
            f"Signal should be at 13:00 (shift from 12:00 retest), got {signal_time.hour}:00"
        )

    def test_no_trades_before_breakout(self):
        """No trades on day 0 or before the breakout on day 1."""
        df = _make_pdhl_bull_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        for trade in trades:
            signal_time = pd.Timestamp(trade["signal_time"])
            # No trades on day 0 (no previous day data)
            assert signal_time.date() != pd.Timestamp("2024-01-01").date(), (
                f"Should not trade on day 0: {signal_time}"
            )

    def test_no_short_trades(self):
        """Bull breakout scenario should not produce short trades."""
        df = _make_pdhl_bull_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        short_trades = [t for t in trades if t["direction"] == "SHORT"]
        assert len(short_trades) == 0, (
            f"Expected no SHORT trades in bull scenario, got {len(short_trades)}"
        )


class TestPDHLBearRetest:
    """Breakout below PDL + retracement to midpoint → exactly 1 Short trade."""

    def test_exactly_one_short_trade(self):
        df = _make_pdhl_bear_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        short_trades = [t for t in trades if t["direction"] == "SHORT"]
        assert len(short_trades) == 1, (
            f"Expected exactly 1 SHORT trade, got {len(short_trades)}. "
            f"All trades: {[(t['direction'], t.get('signal_time')) for t in trades]}"
        )

    def test_short_entry_on_day1_after_retracement(self):
        df = _make_pdhl_bear_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        short_trades = [t for t in trades if t["direction"] == "SHORT"]
        assert len(short_trades) >= 1

        trade = short_trades[0]
        signal_time = pd.Timestamp(trade["signal_time"])
        assert signal_time.date() == pd.Timestamp("2024-01-02").date()
        assert signal_time.hour == 13, (
            f"Signal should be at 13:00 (shift from 12:00 retest), got {signal_time.hour}:00"
        )

    def test_no_long_trades(self):
        df = _make_pdhl_bear_retest()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        assert len(long_trades) == 0, (
            f"Expected no LONG trades in bear scenario, got {len(long_trades)}"
        )


class TestNoSignalCases:
    """Cases where no trades should occur."""

    def test_no_breakout_no_trades(self):
        """Price stays inside PDH/PDL range → zero trades."""
        df = _make_no_breakout()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)
        assert len(trades) == 0, (
            f"Expected 0 trades without breakout, got {len(trades)}: "
            f"{[(t['direction'], t.get('signal_time')) for t in trades]}"
        )

    def test_breakout_without_retracement_no_trades(self):
        """Breakout above PDH but price never retraces to midpoint → zero trades."""
        df = _make_breakout_no_retracement()
        ctx = _minimal_ctx()
        trades = _run_strategy(df, ctx)
        assert len(trades) == 0, (
            f"Expected 0 trades without retracement, got {len(trades)}: "
            f"{[(t['direction'], t.get('signal_time')) for t in trades]}"
        )


class TestSignalProperties:
    """Verify signal column properties directly (not through trades)."""

    def test_retest_bull_fires_once_per_day(self):
        """pdl_retest_bull fires at most once per calendar day."""
        df = _make_pdhl_bull_retest()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(df.copy(), retest_atr_width=0.5)

        for day_str in ["2024-01-01", "2024-01-02", "2024-01-03"]:
            try:
                day = result.loc[day_str]
            except KeyError:
                continue
            count = day["pdl_retest_bull"].dropna().sum()
            assert count <= 1.0, (
                f"pdl_retest_bull fired {count} times on {day_str}"
            )

    def test_retest_bear_fires_once_per_day(self):
        df = _make_pdhl_bear_retest()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(df.copy(), retest_atr_width=0.5)

        for day_str in ["2024-01-01", "2024-01-02", "2024-01-03"]:
            try:
                day = result.loc[day_str]
            except KeyError:
                continue
            count = day["pdl_retest_bear"].dropna().sum()
            assert count <= 1.0, (
                f"pdl_retest_bear fired {count} times on {day_str}"
            )

    def test_no_signal_on_first_day(self):
        """First day has no previous day data → all signals NaN."""
        df = _make_pdhl_bull_retest()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(df.copy(), retest_atr_width=0.5)

        day0 = result.loc["2024-01-01"]
        for col in ["pdl_retest_bull", "pdl_retest_bear"]:
            non_nan = day0[col].dropna()
            assert (non_nan == 0).all() or len(non_nan) == 0, (
                f"{col} fired on day 0 (no previous day data)"
            )

    def test_signal_requires_breakout_first(self):
        """Before the breakout bar, retest signal must be 0."""
        df = _make_pdhl_bull_retest()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(df.copy(), retest_atr_width=0.5)

        day1 = result.loc["2024-01-02"]
        # Breakout at 09:00, shifted to 10:00. Before that, retest_bull must be 0.
        before_breakout = day1.loc[day1.index.hour < 10, "pdl_retest_bull"].dropna()
        assert (before_breakout == 0).all(), (
            f"pdl_retest_bull fired before breakout: {before_breakout[before_breakout > 0]}"
        )
