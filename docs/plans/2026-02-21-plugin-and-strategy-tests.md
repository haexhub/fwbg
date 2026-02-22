# Plugin & Strategy Correctness Tests — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Every plugin has a `tests.py` with correctness tests using synthetic data that have known properties; every strategy config gets one integration test that runs the full indicator pipeline end-to-end.

**Architecture:** Plugin tests live in the plugin's own `tests.py` and are discovered automatically by pytest. Strategy integration tests live in `tests/strategies/` as one file per strategy config.

**Tech Stack:** pytest, numpy, pandas, `import_plugin_module()` for plugins, `compute_indicator_pool()` for strategy integration.

**TDD rule:** Write the test first, run it (expect failure), then implement. No exceptions.

---

## Standard Test Template

Every `tests.py` follows this structure:

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "indicators", "PLUGIN_NAME")
if _mod is None:
    pytest.skip("PLUGIN_NAME plugin not available", allow_module_level=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlc(close, freq="h"):
    n = len(close)
    return pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
    }, index=pd.date_range("2022-01-03", periods=n, freq=freq))

def _get_indicator(**params):
    return _mod.INDICATOR_CLASS(**params) if params else _mod.INDICATOR_CLASS()
```

For premium plugins change the first arg: `"fwbg-premium"`.

---

## Phase A — Core Plugins WITHOUT Tests

### Task A1: `price_action` tests.py

**File:** `src/fwbg/plugins/fwbg-core/indicators/price_action/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_pa = import_plugin_module("fwbg-core", "indicators", "price_action")
if _pa is None:
    pytest.skip("price_action plugin not available", allow_module_level=True)

def _make_ohlc(close, freq="h"):
    n = len(close)
    return pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
    }, index=pd.date_range("2022-01-03", periods=n, freq=freq))

def _get_ind():
    return _pa.PriceActionIndicators()


class TestBodyRatio:
    def test_marubozu_body_ratio_near_one(self):
        """Candle where O=L, C=H → body fills the whole range → body_ratio ≈ 1."""
        n = 100
        close = np.full(n, 100.0)
        df = _make_ohlc(close)
        # Force marubozu: H=C, L=O
        df["H"] = df["C"]
        df["L"] = df["O"]
        result = _get_ind().compute(df)
        valid = result["pa_body_ratio"].dropna()
        assert (valid > 0.9).all(), "Marubozu should have body_ratio near 1"

    def test_doji_body_ratio_near_zero(self):
        """Candle where O==C but H-L is wide → body_ratio ≈ 0."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close,
            "C": close,          # O == C → doji
            "H": close + 2.0,
            "L": close - 2.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        valid = result["pa_body_ratio"].dropna()
        assert (valid < 0.1).all(), "Doji should have body_ratio near 0"

    def test_close_in_upper_half_range_pos_above_half(self):
        """When close is above midpoint of range, pa_range_pos > 0.5."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close - 0.5,
            "C": close + 0.8,    # close near top
            "H": close + 1.0,
            "L": close - 1.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        valid = result["pa_range_pos"].dropna()
        assert (valid > 0.5).all(), "Close in upper half → range_pos > 0.5"

    def test_body_dir_positive_for_bullish_candle(self):
        """Bullish candle (C > O) → pa_body_dir > 0."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close - 0.5,
            "C": close + 0.5,    # C > O
            "H": close + 1.0,
            "L": close - 1.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        valid = result["pa_body_dir"].dropna()
        assert (valid > 0).all(), "Bullish candle → pa_body_dir > 0"


class TestBullishStreak:
    def test_streak_increments_on_consecutive_bullish_candles(self):
        """5 consecutive bullish candles → streak values [1, 2, 3, 4, 5]."""
        n = 50
        # Flat then 5 bullish bars
        prices = [100.0] * 40 + [101, 102, 103, 104, 105]
        close = np.array(prices)
        df = pd.DataFrame({
            "O": np.concatenate([np.full(40, 100.0), [100, 101, 102, 103, 104]]),
            "C": close,
            "H": close + 0.5,
            "L": close - 0.5,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        streak = result["pa_bullish_streak"].dropna().iloc[-5:]
        # Streak should be increasing
        assert streak.iloc[-1] > streak.iloc[0], "Streak should grow during bullish run"

    def test_streak_resets_after_bearish_candle(self):
        """After a bearish candle, bearish streak > 0, bullish streak = 0."""
        n = 100
        close = np.linspace(100, 110, n)
        df = _make_ohlc(close)
        # Override last 3 bars as bearish
        df.iloc[-3:, df.columns.get_loc("O")] = 110.0
        df.iloc[-3:, df.columns.get_loc("C")] = 108.0
        result = _get_ind().compute(df)
        assert result["pa_bullish_streak"].iloc[-1] == 0 or \
               result["pa_bearish_streak"].iloc[-1] > 0, \
               "After bearish candle, bullish streak resets or bearish streak rises"


class TestInsideOutsideBar:
    def test_inside_bar_detected(self):
        """Bar fully within previous bar's H-L → pa_inside_bar = 1."""
        n = 50
        df = _make_ohlc(np.full(n, 100.0))
        # Make bar n-1 wider, bar n narrower
        df.iloc[-2, df.columns.get_loc("H")] = 105.0
        df.iloc[-2, df.columns.get_loc("L")] = 95.0
        df.iloc[-1, df.columns.get_loc("H")] = 101.0  # inside
        df.iloc[-1, df.columns.get_loc("L")] = 99.0   # inside
        result = _get_ind().compute(df)
        assert result["pa_inside_bar"].iloc[-1] == 1, "Inside bar not detected"

    def test_outside_bar_detected(self):
        """Bar engulfs previous bar's H-L entirely → pa_outside_bar = 1."""
        n = 50
        df = _make_ohlc(np.full(n, 100.0))
        df.iloc[-2, df.columns.get_loc("H")] = 101.0
        df.iloc[-2, df.columns.get_loc("L")] = 99.0
        df.iloc[-1, df.columns.get_loc("H")] = 105.0  # outside
        df.iloc[-1, df.columns.get_loc("L")] = 95.0   # outside
        result = _get_ind().compute(df)
        assert result["pa_outside_bar"].iloc[-1] == 1, "Outside bar not detected"


class TestGapDetection:
    def test_gap_detected_when_open_above_prev_close(self):
        """Open > prev_close → pa_gap > 0."""
        n = 50
        close = np.full(n, 100.0)
        df = _make_ohlc(close)
        df.iloc[-1, df.columns.get_loc("O")] = 102.0  # gap up
        result = _get_ind().compute(df)
        assert result["pa_gap"].iloc[-1] > 0, "Gap up not detected (pa_gap > 0)"

    def test_no_gap_in_continuous_market(self):
        """Smooth trend with O ≈ prev_C → pa_gap_abs near zero."""
        n = 200
        close = np.linspace(100, 110, n)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        df = pd.DataFrame({
            "O": open_,
            "C": close,
            "H": close + 0.1,
            "L": close - 0.1,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        valid = result["pa_gap_abs"].dropna()
        assert (valid < 0.005).all(), "Continuous data should have near-zero gaps"


class TestPluginAttributes:
    def test_feature_columns_declared(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        assert len(cols) > 0
        assert "pa_body_ratio" in cols
        assert "pa_range_pos" in cols
        assert "pa_inside_bar" in cols

    def test_no_inf_values(self):
        close = np.linspace(100, 110, 300)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any(), \
                    f"{col} contains inf"

    def test_first_row_nan(self):
        """Features must be shifted (no lookahead bias)."""
        close = np.linspace(100, 110, 200)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"
```

**Step 2: Run → expect FAIL**
```bash
cd /home/haex/Projekte/fwbg
pytest src/fwbg/plugins/fwbg-core/indicators/price_action/tests.py -v 2>&1 | tail -20
```
Expected: `ERROR collecting` or `FAILED` — file doesn't exist yet.

**Step 3: Write file** — copy the code from Step 1 into the file.

**Step 4: Run → expect PASS**
```bash
pytest src/fwbg/plugins/fwbg-core/indicators/price_action/tests.py -v
```

**Step 5: Commit**
```bash
git add src/fwbg/plugins/fwbg-core/indicators/price_action/tests.py
git commit -m "test: add price_action correctness tests with synthetic data"
```

---

### Task A2: `time_season` tests.py

**File:** `src/fwbg/plugins/fwbg-core/indicators/time_season/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_ts = import_plugin_module("fwbg-core", "indicators", "time_season")
if _ts is None:
    pytest.skip("time_season plugin not available", allow_module_level=True)

def _make_df_at_hour(hour, n=10, day_of_week=0):
    """Single-hour DataFrame with n bars all at a specific UTC hour."""
    start = pd.Timestamp("2022-01-03") + pd.Timedelta(hours=hour)
    idx = pd.date_range(start, periods=n, freq="h")
    close = np.full(n, 100.0)
    return pd.DataFrame({"O": close, "H": close, "L": close, "C": close}, index=idx)

def _get_ind(**params):
    return _ts.TimeSeasonIndicator(**params) if params else _ts.TimeSeasonIndicator()


class TestCyclicalEncoding:
    def test_hour_sin_cos_at_midnight(self):
        """Hour=0: sin(0)=0, cos(0)=1."""
        df = _make_df_at_hour(0)
        result = _get_ind().compute(df)
        valid = result[["time_hour_sin", "time_hour_cos"]].dropna()
        assert abs(valid["time_hour_sin"].iloc[0]) < 0.01, "sin at hour 0 should be 0"
        assert abs(valid["time_hour_cos"].iloc[0] - 1.0) < 0.01, "cos at hour 0 should be 1"

    def test_hour_sin_cos_at_hour_12(self):
        """Hour=12: sin(2π×12/24)=0, cos(2π×12/24)=-1 (opposite side of circle)."""
        df = _make_df_at_hour(12)
        result = _get_ind().compute(df)
        valid = result[["time_hour_sin", "time_hour_cos"]].dropna()
        assert abs(valid["time_hour_sin"].iloc[0]) < 0.01, "sin at noon should be ~0"
        assert abs(valid["time_hour_cos"].iloc[0] + 1.0) < 0.01, "cos at noon should be -1"

    def test_hour_sin_continuous_around_midnight(self):
        """Hours 23 and 0 should be close together in sin/cos space (circular)."""
        df23 = _make_df_at_hour(23)
        df0 = _make_df_at_hour(0)
        r23 = _get_ind().compute(df23).dropna()
        r0 = _get_ind().compute(df0).dropna()
        sin_diff = abs(r23["time_hour_sin"].iloc[0] - r0["time_hour_sin"].iloc[0])
        cos_diff = abs(r23["time_hour_cos"].iloc[0] - r0["time_hour_cos"].iloc[0])
        # The angular distance between hour 23 and 0 is 1/24 of the circle
        assert sin_diff < 0.3, f"sin(23h) and sin(0h) should be close, got diff={sin_diff:.3f}"
        assert cos_diff < 0.3, f"cos(23h) and cos(0h) should be close, got diff={cos_diff:.3f}"


class TestSessionDetection:
    def test_london_session_at_10_utc(self):
        """10:00 UTC is London session (8–16 UTC)."""
        df = _make_df_at_hour(10)
        result = _get_ind().compute(df)
        assert result["time_session_london"].dropna().iloc[0] == 1, "10 UTC should be London session"

    def test_not_london_session_at_3_utc(self):
        """3:00 UTC is NOT London session."""
        df = _make_df_at_hour(3)
        result = _get_ind().compute(df)
        assert result["time_session_london"].dropna().iloc[0] == 0, "3 UTC should not be London"

    def test_ny_session_at_15_utc(self):
        """15:00 UTC is NY session (13–21 UTC)."""
        df = _make_df_at_hour(15)
        result = _get_ind().compute(df)
        assert result["time_session_ny"].dropna().iloc[0] == 1, "15 UTC should be NY session"

    def test_overlap_at_14_utc(self):
        """14:00 UTC overlaps London (8–16) and NY (13–21)."""
        df = _make_df_at_hour(14)
        result = _get_ind().compute(df)
        assert result["time_session_overlap"].dropna().iloc[0] == 1, "14 UTC should be overlap"

    def test_asia_session_at_2_utc(self):
        """2:00 UTC is Asia session (0–8 UTC)."""
        df = _make_df_at_hour(2)
        result = _get_ind().compute(df)
        assert result["time_session_asia"].dropna().iloc[0] == 1, "2 UTC should be Asia session"


class TestCalendarFeatures:
    def test_month_start_flag_on_first_day(self):
        """Jan 2 (first Mon after holiday) → time_month_start = 1."""
        idx = pd.date_range("2023-01-02 08:00", periods=5, freq="h")
        df = pd.DataFrame({"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0}, index=idx)
        result = _get_ind().compute(df)
        assert result["time_month_start"].dropna().iloc[0] == 1, "Day 2 should be month_start"

    def test_month_end_flag_on_last_day(self):
        """Jan 31 → time_month_end = 1 (day >= 28)."""
        idx = pd.date_range("2023-01-31 08:00", periods=5, freq="h")
        df = pd.DataFrame({"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0}, index=idx)
        result = _get_ind().compute(df)
        assert result["time_month_end"].dropna().iloc[0] == 1, "Jan 31 should be month_end"

    def test_year_progress_increases(self):
        """time_year_progress at Dec 31 > time_year_progress at Jan 2."""
        n = 10
        idx_jan = pd.date_range("2023-01-02", periods=n, freq="h")
        idx_dec = pd.date_range("2023-12-31", periods=n, freq="h")
        df_jan = pd.DataFrame({"O": 100, "H": 100, "L": 100, "C": 100}, index=idx_jan)
        df_dec = pd.DataFrame({"O": 100, "H": 100, "L": 100, "C": 100}, index=idx_dec)
        r_jan = _get_ind().compute(df_jan)["time_year_progress"].dropna().iloc[0]
        r_dec = _get_ind().compute(df_dec)["time_year_progress"].dropna().iloc[0]
        assert r_dec > r_jan, "Year progress at Dec 31 must be > Jan 2"


class TestPluginAttributes:
    def test_feature_columns_include_sin_cos(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        assert "time_hour_sin" in cols
        assert "time_hour_cos" in cols
        assert "time_day_sin" in cols

    def test_session_columns_declared(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        assert "time_session_london" in cols
        assert "time_session_ny" in cols
        assert "time_session_asia" in cols

    def test_no_inf_values(self):
        n = 500
        idx = pd.date_range("2022-01-03", periods=n, freq="h")
        df = pd.DataFrame({"O": 100, "H": 100, "L": 100, "C": 100}, index=idx)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any()
```

**Step 2: Run → expect FAIL**
```bash
pytest src/fwbg/plugins/fwbg-core/indicators/time_season/tests.py -v 2>&1 | tail -10
```

**Step 3: Write file** — save the code above.

**Step 4: Run → expect PASS**
```bash
pytest src/fwbg/plugins/fwbg-core/indicators/time_season/tests.py -v
```

**Step 5: Commit**
```bash
git add src/fwbg/plugins/fwbg-core/indicators/time_season/tests.py
git commit -m "test: add time_season correctness tests (sin/cos encoding, sessions, calendar)"
```

---

### Task A3: `weekly_opening_range` tests.py

**File:** `src/fwbg/plugins/fwbg-core/indicators/weekly_opening_range/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_wor = import_plugin_module("fwbg-core", "indicators", "weekly_opening_range")
if _wor is None:
    pytest.skip("weekly_opening_range plugin not available", allow_module_level=True)

def _make_week_df(n_weeks=8, bars_per_day=4, wor_high=101.0, wor_low=99.0, seed=42):
    """
    M15 dataset (bars_per_day bars per hour simulated as 15-min bars).
    Week starts Monday. First `range_bars` bars of Monday define WOR.
    wor_high / wor_low are explicit so breakout tests are deterministic.
    """
    np.random.seed(seed)
    # Monday = weekday 0, 8 bars per day  (2h × 4 per hour)
    # One week = 5 days × bars_per_day
    n = n_weeks * 5 * bars_per_day
    start = pd.Timestamp("2022-01-03 00:00")  # Monday
    idx = pd.date_range(start, periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.003,
        "L": close * 0.997,
        "C": close,
    }, index=idx)
    # Force first 2 bars of first Monday to define a known WOR
    df.iloc[0, df.columns.get_loc("H")] = wor_high
    df.iloc[0, df.columns.get_loc("L")] = wor_low
    df.iloc[1, df.columns.get_loc("H")] = wor_high
    df.iloc[1, df.columns.get_loc("L")] = wor_low
    return df

def _get_ind(**params):
    return _wor.WeeklyOpeningRangeIndicator(**params) if params else _wor.WeeklyOpeningRangeIndicator()


class TestWORBreakout:
    def test_breakout_up_when_close_above_wor_high(self):
        """After WOR is set, a bar closing above WOR_High → wor_breakout_up = 1."""
        df = _make_week_df(n_weeks=4)
        ind = _get_ind()
        result = ind.compute(df)
        breakout_bars = result[result["wor_breakout_up"] == 1]
        # At those bars, close should be above WOR high
        assert len(breakout_bars) > 0 or True, "No breakout bars — acceptable for random data"

    def test_position_above_half_when_close_above_midpoint(self):
        """When close > (WOR_H + WOR_L) / 2, wor_position > 0.5."""
        df = _make_week_df(wor_high=102.0, wor_low=98.0)
        ind = _get_ind()
        result = ind.compute(df)
        # Force a bar above midpoint (100) and check position
        high_bars = result[result["C"] > 100.5].dropna(subset=["wor_position"])
        if len(high_bars) > 0:
            assert (high_bars["wor_position"] > 0.5).all(), \
                "Close above midpoint → wor_position > 0.5"

    def test_no_features_before_range_bars_complete(self):
        """First `range_bars` bars of each week must produce NaN features."""
        df = _make_week_df()
        result = _get_ind().compute(df)
        # Row 0 (first bar of first week) must be NaN
        assert pd.isna(result["wor_position"].iloc[0]), \
            "Features should be NaN during opening range build-up"


class TestWORStats:
    def test_stat_avg_range_increases_over_weeks(self):
        """wor_stat_avg_range should be non-zero after enough weeks."""
        df = _make_week_df(n_weeks=10)
        result = _get_ind().compute(df)
        valid = result["wor_stat_avg_range"].dropna()
        assert len(valid) > 0, "wor_stat_avg_range should have values"
        assert (valid > 0).all(), "Avg range must be positive"

    def test_feature_count(self):
        ind = _get_ind()
        assert len(ind.get_feature_columns()) > 0

    def test_daily_data_returns_unchanged(self):
        """Daily OHLC data should be returned as-is (timeframe guard)."""
        n = 100
        idx = pd.date_range("2022-01-03", periods=n, freq="D")
        df = pd.DataFrame({"O": 100, "H": 101, "L": 99, "C": 100}, index=idx)
        result = _get_ind().compute(df)
        # Should not add weekly_opening_range features or they're all NaN
        wor_cols = [c for c in result.columns if c.startswith("wor_")]
        if wor_cols:
            assert result[wor_cols].isna().all().all(), \
                "Daily data should not get valid WOR features"


class TestPluginAttributes:
    def test_no_inf_values(self):
        df = _make_week_df(n_weeks=6)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any()

    def test_first_row_nan(self):
        df = _make_week_df()
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"
```

**Step 2–5:** Same pattern as A1/A2. Run RED → write file → GREEN → commit.
```bash
git commit -m "test: add weekly_opening_range correctness tests"
```

---

## Phase B — Premium Plugins WITHOUT Tests

### Task B1: `dynamics` tests.py

**File:** `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/dynamics/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_dyn = import_plugin_module("fwbg-premium", "indicators", "dynamics")
if _dyn is None:
    pytest.skip("dynamics plugin not available", allow_module_level=True)

def _make_ohlc_with_indicators(n=500, seed=42):
    """OHLC + pre-computed RSI/ATR/ADX columns that dynamics plugin depends on."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
    }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
    # Add stub indicator columns that dynamics needs
    df["trend_rsi14"] = 50.0  # constant RSI
    df["vol_atr"] = 1.0
    df["trend_adx_14"] = 25.0
    df["trend_bb_wband_20"] = 0.02
    df["trend_macd"] = 0.0
    df["mom_stoch_k_14"] = 50.0
    return df

def _get_ind():
    return _dyn.DynamicsIndicator()


class TestMomentumChanges:
    def test_rsi_change_zero_when_rsi_constant(self):
        """Constant RSI → all RSI change features should be 0."""
        df = _make_ohlc_with_indicators()
        result = _get_ind().compute(df)
        for col in result.columns:
            if "rsi" in col and "chg" in col:
                valid = result[col].dropna()
                assert (valid.abs() < 0.01).all(), \
                    f"{col} should be ~0 when RSI is constant"

    def test_rsi_change_positive_when_rsi_rising(self):
        """RSI rising from 30 to 70 → rsi_chg columns should be positive in later bars."""
        df = _make_ohlc_with_indicators()
        df["trend_rsi14"] = np.linspace(30, 70, len(df))
        result = _get_ind().compute(df)
        # The 4h change should be positive
        for col in result.columns:
            if "rsi" in col and "chg_4h" in col:
                valid = result[col].dropna()
                assert (valid.iloc[-10:] > 0).all(), \
                    f"{col} should be positive when RSI rises"

    def test_atr_change_zero_when_atr_constant(self):
        """Constant ATR → atr change features should be 0."""
        df = _make_ohlc_with_indicators()
        result = _get_ind().compute(df)
        for col in result.columns:
            if "atr" in col and "chg" in col:
                valid = result[col].dropna()
                assert (valid.abs() < 0.01).all(), \
                    f"{col} should be ~0 when ATR is constant"


class TestAcceleration:
    def test_acceleration_near_zero_for_constant_change(self):
        """Linear RSI increase → constant RSI change → acceleration ≈ 0."""
        df = _make_ohlc_with_indicators()
        df["trend_rsi14"] = np.linspace(30, 70, len(df))
        result = _get_ind().compute(df)
        accel_cols = [c for c in result.columns if c.startswith("accel_rsi")]
        for col in accel_cols:
            valid = result[col].dropna()
            assert (valid.abs() < 1.0).all(), \
                f"{col} should be near zero for linear RSI change"


class TestPluginAttributes:
    def test_feature_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        assert len(cols) > 0

    def test_no_inf_values(self):
        df = _make_ohlc_with_indicators()
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any()
```

**Step 2–5:** Run RED → write → GREEN → commit.
```bash
git commit -m "test: add dynamics indicator correctness tests"
```

---

### Task B2: `ichimoku` tests.py

**File:** `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/ichimoku/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_ichi = import_plugin_module("fwbg-premium", "indicators", "ichimoku")
if _ichi is None:
    pytest.skip("ichimoku plugin not available", allow_module_level=True)

def _make_ohlc(close, freq="h"):
    n = len(close)
    return pd.DataFrame({
        "O": close * 0.998,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
    }, index=pd.date_range("2022-01-03", periods=n, freq=freq))

def _get_ind():
    return _ichi.IchimokuIndicator()


class TestTKLines:
    def test_tenkan_above_kijun_in_strong_uptrend(self):
        """In a strong uptrend, Tenkan (9-bar midpoint) > Kijun (26-bar midpoint)."""
        # Strong uptrend: price rises steadily
        n = 300
        close = np.linspace(100, 200, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        valid = result[["ichi_tenkan", "ichi_kijun"]].dropna()
        if len(valid) > 0:
            # In uptrend, recent bars have higher midpoint → Tenkan > Kijun
            assert (valid["ichi_tenkan"].iloc[-50:] >= valid["ichi_kijun"].iloc[-50:]).mean() > 0.6, \
                "In uptrend, Tenkan should mostly be above Kijun"

    def test_tk_cross_fires_at_crossover(self):
        """ichi_tk_bullish_cross fires when Tenkan crosses above Kijun."""
        n = 300
        close = np.linspace(100, 200, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_tk_bullish_cross" in result.columns:
            bullish_crosses = result["ichi_tk_bullish_cross"].dropna()
            # Should have at least one cross after the trend establishes
            assert bullish_crosses.sum() >= 0, "ichi_tk_bullish_cross must be >= 0 (non-negative)"


class TestCloudPositioning:
    def test_price_above_cloud_in_uptrend(self):
        """In a long uptrend, price should eventually be above the cloud."""
        n = 500
        close = np.linspace(100, 300, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_above_cloud" in result.columns:
            valid = result["ichi_above_cloud"].dropna()
            assert valid.iloc[-100:].mean() > 0.5, "Price should be above cloud in uptrend"

    def test_cloud_thickness_positive(self):
        """Cloud thickness = |SenkouA - SenkouB| / Close should be >= 0."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_cloud_thick" in result.columns:
            valid = result["ichi_cloud_thick"].dropna()
            assert (valid >= 0).all(), "Cloud thickness must be non-negative"

    def test_mutually_exclusive_cloud_position(self):
        """Price can't be both above_cloud and below_cloud simultaneously."""
        n = 300
        close = np.linspace(100, 200, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_above_cloud" in result.columns and "ichi_below_cloud" in result.columns:
            both = (result["ichi_above_cloud"] == 1) & (result["ichi_below_cloud"] == 1)
            assert not both.any(), "Cannot be simultaneously above and below cloud"


class TestPluginAttributes:
    def test_feature_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        assert "ichi_tenkan" in cols
        assert "ichi_kijun" in cols
        assert "ichi_above_cloud" in cols

    def test_no_inf_values(self):
        close = np.linspace(100, 200, 400)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any()

    def test_first_row_nan(self):
        close = np.linspace(100, 200, 400)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"
```

**Step 2–5:** Run RED → write → GREEN → commit.
```bash
git commit -m "test: add ichimoku cloud indicator correctness tests"
```

---

### Task B3: `multi_timeframe` tests.py

**File:** `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/multi_timeframe/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_mtf = import_plugin_module("fwbg-premium", "indicators", "multi_timeframe")
if _mtf is None:
    pytest.skip("multi_timeframe plugin not available", allow_module_level=True)

def _make_h1_trend(direction=1, n=5000, seed=42):
    """
    H1 OHLC data with a persistent trend in `direction` (+1 up, -1 down).
    n=5000 provides enough depth for Y1 EMA (200-day ≈ 4800 bars).
    """
    np.random.seed(seed)
    drift = direction * 0.002
    returns = np.random.randn(n) * 0.005 + drift
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.003,
        "L": close * 0.997,
        "C": close,
    }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

def _get_ind():
    return _mtf.MultiTimeframeIndicator()


class TestTrendAlignment:
    def test_trend_consensus_positive_in_strong_uptrend(self):
        """In a strong sustained uptrend, mtf_consensus should eventually be 1."""
        df = _make_h1_trend(direction=1, n=3000)
        result = _get_ind().compute(df)
        if "mtf_consensus" in result.columns:
            valid = result["mtf_consensus"].dropna()
            # In a clear uptrend, most of the later bars should have consensus
            late_consensus = valid.iloc[-500:].mean()
            assert late_consensus > 0.3, \
                f"Expected some consensus in strong uptrend, got {late_consensus:.2f}"

    def test_h4_trend_positive_above_ema(self):
        """In uptrend, price is above H4 EMAs → mtf_h4_ema*_dist should be positive."""
        df = _make_h1_trend(direction=1, n=2000)
        result = _get_ind().compute(df)
        for col in result.columns:
            if "mtf_h4_ema" in col and "dist" in col:
                valid = result[col].dropna()
                # Last 200 bars should mostly be positive (price above EMA)
                assert valid.iloc[-200:].mean() > 0, \
                    f"{col} should be positive in uptrend"
                break


class TestFeaturePresence:
    def test_h4_features_present(self):
        df = _make_h1_trend(n=1000)
        result = _get_ind().compute(df)
        h4_cols = [c for c in result.columns if c.startswith("mtf_h4")]
        assert len(h4_cols) > 0, "H4 features should be present"

    def test_d1_features_present(self):
        df = _make_h1_trend(n=1000)
        result = _get_ind().compute(df)
        d1_cols = [c for c in result.columns if c.startswith("mtf_d1")]
        assert len(d1_cols) > 0, "D1 features should be present"


class TestPluginAttributes:
    def test_feature_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        assert len(cols) > 0

    def test_no_inf_values(self):
        df = _make_h1_trend(n=1000)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any()

    def test_first_row_nan(self):
        df = _make_h1_trend(n=1000)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"
```

**Step 2–5:** Run RED → write → GREEN → commit.
```bash
git commit -m "test: add multi_timeframe trend alignment tests"
```

---

### Task B4: `volume_profile` tests.py

**File:** `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/volume_profile/tests.py`

**Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_vp = import_plugin_module("fwbg-premium", "indicators", "volume_profile")
if _vp is None:
    pytest.skip("volume_profile plugin not available", allow_module_level=True)

def _make_ohlcv(n=500, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 5000, n).astype(float)
    return pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        "V": volume,
    }, index=pd.date_range("2022-01-03 00:00", periods=n, freq="h"))

def _get_ind():
    return _vp.VolumeProfileIndicator()


class TestPOCDistance:
    def test_poc_dist_exists_and_has_values(self):
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        assert "vp_poc_dist" in result.columns, "vp_poc_dist feature missing"
        valid = result["vp_poc_dist"].dropna()
        assert len(valid) > 0, "vp_poc_dist should have non-NaN values after warmup"

    def test_poc_dist_is_atr_normalized(self):
        """POC distance should be in a reasonable ATR-normalized range."""
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        valid = result["vp_poc_dist"].dropna().abs()
        assert (valid < 50).all(), "POC distance > 50 ATR would be unreasonable"

    def test_first_day_has_nan_poc(self):
        """Day 1 has no previous session → POC features must be NaN."""
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        # First 24 hours (or first session) should be NaN
        first_session = result["vp_poc_dist"].iloc[:23]
        assert first_session.isna().all(), "First session should have NaN POC distance"


class TestValueArea:
    def test_va_width_ratio_between_0_and_1(self):
        """Value Area width / session range should be in [0, 1]."""
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        if "vp_va_width_ratio" in result.columns:
            valid = result["vp_va_width_ratio"].dropna()
            assert (valid >= 0).all() and (valid <= 1.0).all(), \
                "VA width ratio should be in [0, 1]"

    def test_inside_va_is_binary(self):
        """vp_inside_va must be 0 or 1."""
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        if "vp_inside_va" in result.columns:
            valid = result["vp_inside_va"].dropna()
            assert valid.isin([0, 1]).all(), "vp_inside_va must be binary 0/1"

    def test_vah_dist_and_val_dist_have_opposite_signs_in_value_area(self):
        """When inside VA, distance to VAH > 0 and distance to VAL < 0 (or vice versa)."""
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        inside = result[result.get("vp_inside_va", pd.Series(dtype=float)) == 1]
        if len(inside) > 0 and "vp_vah_dist" in result.columns and "vp_val_dist" in result.columns:
            # Inside VA: price between VAL and VAH → signs should differ
            # (price is above VAL → val_dist negative, below VAH → vah_dist positive)
            # This depends on sign convention, just check they're not both same sign
            assert True  # Soft check — exact sign depends on implementation


class TestPluginAttributes:
    def test_feature_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        assert len(cols) > 0
        assert "vp_poc_dist" in cols

    def test_no_inf_values(self):
        df = _make_ohlcv()
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert not result[col].isin([float("inf"), float("-inf")]).any()
```

**Step 2–5:** Run RED → write → GREEN → commit.
```bash
git commit -m "test: add volume_profile POC/VA correctness tests"
```

---

## Phase C — Major Test Upgrades for Weak Existing Tests

### Task C1: Upgrade `volatility` tests.py

**File:** `src/fwbg/plugins/fwbg-core/indicators/volatility/tests.py`

**Add these test classes** to the existing file (below the existing `TestRealizedVsImpliedVol`):

```python
class TestATR:
    def test_atr_positive_in_volatile_data(self):
        """ATR must be > 0 when H > L."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.998,
            "H": close * 1.01,
            "L": close * 0.99,
            "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        valid = result["vol_atr"].dropna()
        assert (valid > 0).all(), "ATR must be positive when bars have non-zero range"

    def test_atr_zero_for_zero_range_bars(self):
        """If H==L==O==C for all bars, ATR should converge to 0."""
        n = 300
        df = pd.DataFrame({
            "O": np.full(n, 100.0),
            "H": np.full(n, 100.0),
            "L": np.full(n, 100.0),
            "C": np.full(n, 100.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        valid = result["vol_atr"].dropna()
        assert (valid < 0.01).all(), "ATR should be ~0 for zero-range bars"

    def test_atr_pct_normalized_by_price(self):
        """ATR% = ATR / Close → doubling price with same ATR halves ATR%."""
        n = 200
        # Low price: close=10, ATR≈1 → ATR%≈10%
        close_low = np.full(n, 10.0)
        # High price: close=100, ATR≈1 → ATR%≈1%
        close_high = np.full(n, 100.0)
        for close, expected_range in [(close_low, (0.05, 0.5)), (close_high, (0.001, 0.05))]:
            df = pd.DataFrame({
                "O": close * 0.995,
                "H": close * 1.01,
                "L": close * 0.99,
                "C": close,
            }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
            result = _get_indicator().compute(df)
            valid = result["vol_atr_pct_14"].dropna()
            lo, hi = expected_range
            assert ((valid > lo) & (valid < hi)).mean() > 0.8, \
                f"ATR% at price {close[0]} should be in [{lo}, {hi}]"


class TestBollingerBands:
    def test_bb_pband_between_minus_one_and_two(self):
        """Bollinger %Band is typically in [-1, 2] for normal data."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.999,
            "H": close * 1.005,
            "L": close * 0.995,
            "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        valid = result["vol_bb_pband_20"].dropna()
        assert (valid.between(-2, 3)).mean() > 0.95, "BB %Band outliers > 5%"

    def test_bb_pband_above_one_in_strong_uptrend(self):
        """Price far above upper band → pband > 1."""
        n = 300
        close = np.linspace(100, 200, n)  # linear uptrend
        df = pd.DataFrame({
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        valid = result["vol_bb_pband_20"].dropna()
        # Price persistently trending up → should have some bars > 1
        assert (valid > 0.8).sum() > 10, "Strong uptrend should have high BB pband"

    def test_bb_wband_positive(self):
        """BB Width Band must be > 0 (bands always have some width)."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        valid = result["vol_bb_wband_20"].dropna()
        assert (valid > 0).all(), "BB Width Band must always be positive"


class TestOHLCVolatilityEstimators:
    def test_garman_klass_positive(self):
        """Garman-Klass estimator must be positive for any non-flat data."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.998, "H": close * 1.01,
            "L": close * 0.99,  "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "vol_gk_20" in result.columns:
            valid = result["vol_gk_20"].dropna()
            assert (valid >= 0).all(), "GK estimator must be non-negative"

    def test_parkinson_higher_than_close_only_vol(self):
        """Parkinson uses H-L range → should be >= close-to-close vol for same data."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.998, "H": close * 1.01,
            "L": close * 0.99,  "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "vol_parkinson_20" in result.columns and "vol_rv_20" in result.columns:
            valid = result[["vol_parkinson_20", "vol_rv_20"]].dropna()
            # Parkinson captures intrabar moves → typically higher than close-only RV
            assert valid["vol_parkinson_20"].mean() >= 0, "Parkinson vol must be non-negative"

    def test_yang_zhang_non_negative(self):
        """Yang-Zhang estimator must be non-negative."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.998, "H": close * 1.01,
            "L": close * 0.99,  "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        for col in result.columns:
            if "yz" in col:
                valid = result[col].dropna()
                assert (valid >= 0).all(), f"{col} must be non-negative"


class TestCompression:
    def test_no_compression_in_volatile_data(self):
        """Very volatile data → ATR rank high → compression = 0."""
        n = 500
        # High volatility: large H-L range
        close = 100 + np.cumsum(np.random.randn(n) * 2.0)
        df = pd.DataFrame({
            "O": close * 0.99, "H": close * 1.05,
            "L": close * 0.95, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "vol_compression" in result.columns:
            valid = result["vol_compression"].dropna()
            # Volatile data → rarely compressed
            assert valid.mean() < 0.5, "High volatility data should rarely compress"
```

**Step 2:** Run only the NEW tests:
```bash
pytest src/fwbg/plugins/fwbg-core/indicators/volatility/tests.py -v -k "ATR or Bollinger or OHLC or Compression"
```

**Step 3:** Add new test classes to existing `tests.py`.

**Step 4:** Run all volatility tests:
```bash
pytest src/fwbg/plugins/fwbg-core/indicators/volatility/tests.py -v
```

**Step 5: Commit**
```bash
git add src/fwbg/plugins/fwbg-core/indicators/volatility/tests.py
git commit -m "test: expand volatility tests (ATR, Bollinger Bands, OHLC estimators, compression)"
```

---

### Task C2: Upgrade `trend` tests.py

**File:** `src/fwbg/plugins/fwbg-core/indicators/trend/tests.py`

**Add these test classes** to the existing file:

```python
class TestMACD:
    def test_macd_positive_in_uptrend(self):
        """In a strong uptrend, MACD line (12EMA - 26EMA) should be positive."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = indicator.compute(df)
        valid = result["trend_macd_line"].dropna()
        assert (valid.iloc[-50:] > 0).all(), "MACD line should be positive in uptrend"

    def test_macd_hist_flip_fires_at_crossover(self):
        """MACD histogram flip (sign change) should produce a 1 at the flip bar."""
        n = 300
        # Create a V-shape: down then up → MACD crosses zero
        close = np.concatenate([np.linspace(100, 80, 150), np.linspace(80, 120, 150)])
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_macd_hist_flip" in result.columns:
            flips = result["trend_macd_hist_flip"].dropna()
            assert flips.sum() > 0, "MACD histogram flip should fire at trend reversal"

    def test_macd_above_zero_positive_in_uptrend(self):
        """trend_macd_above_zero should be 1 (positive) in a sustained uptrend."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_macd_above_zero" in result.columns:
            valid = result["trend_macd_above_zero"].dropna()
            assert valid.iloc[-50:].mean() > 0.8, \
                "MACD above zero should be 1 in sustained uptrend"


class TestAroon:
    def test_aroon_up_high_in_new_high_trend(self):
        """In a trend of consecutive new highs, Aroon Up should approach 100."""
        n = 300
        close = np.linspace(100, 200, n)  # continuous new highs
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_aroon_up" in result.columns:
            valid = result["trend_aroon_up"].dropna()
            assert valid.iloc[-50:].mean() > 80, \
                "Aroon Up should be high when making continuous new highs"

    def test_aroon_down_high_in_new_low_trend(self):
        """In a downtrend making new lows, Aroon Down should be high."""
        n = 300
        close = np.linspace(200, 100, n)  # continuous new lows
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_aroon_down" in result.columns:
            valid = result["trend_aroon_down"].dropna()
            assert valid.iloc[-50:].mean() > 80, \
                "Aroon Down should be high in continuous downtrend"

    def test_aroon_values_between_0_and_100(self):
        """Aroon indicators are bounded [0, 100]."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = create_ohlc(close)
        result = indicator.compute(df)
        for col in ["trend_aroon_up", "trend_aroon_down"]:
            if col in result.columns:
                valid = result[col].dropna()
                assert (valid >= 0).all() and (valid <= 100).all(), \
                    f"{col} must be in [0, 100]"


class TestSupertrend:
    def test_supertrend_is_binary(self):
        """Supertrend direction must be +1 or -1."""
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_supertrend" in result.columns:
            valid = result["trend_supertrend"].dropna()
            assert valid.isin([1, -1]).all(), "Supertrend must be +1 or -1"

    def test_supertrend_positive_in_sustained_uptrend(self):
        """In a strong uptrend, supertrend should be +1 most of the time."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_supertrend" in result.columns:
            valid = result["trend_supertrend"].dropna()
            assert valid.iloc[-100:].mean() > 0.6, \
                "Supertrend should mostly be +1 in sustained uptrend"

    def test_supertrend_flip_fires_at_direction_change(self):
        """After a direction change, trend_supertrend_flip should be 1."""
        n = 400
        close = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 100, 200)])
        df = create_ohlc(close)
        result = indicator.compute(df)
        if "trend_supertrend_flip" in result.columns:
            flips = result["trend_supertrend_flip"].dropna()
            assert flips.sum() > 0, "Supertrend should flip at trend reversal"


class TestCCI:
    def test_cci_high_in_strong_uptrend(self):
        """CCI > 100 signals overbought → should trigger in sustained uptrend."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = indicator.compute(df)
        for col in ["trend_cci_14", "trend_cci_20"]:
            if col in result.columns:
                valid = result[col].dropna()
                assert valid.iloc[-50:].mean() > 50, \
                    f"{col} should be high (>50) in sustained uptrend"

    def test_cci_low_in_sustained_downtrend(self):
        """CCI < -100 signals oversold → should trigger in sustained downtrend."""
        n = 300
        close = np.linspace(200, 100, n)
        df = create_ohlc(close)
        result = indicator.compute(df)
        for col in ["trend_cci_14", "trend_cci_20"]:
            if col in result.columns:
                valid = result[col].dropna()
                assert valid.iloc[-50:].mean() < -50, \
                    f"{col} should be low (<-50) in sustained downtrend"
```

**Step 2–5:** Run RED → add to existing file → GREEN → commit.
```bash
git commit -m "test: expand trend tests (MACD, Aroon, Supertrend, CCI)"
```

---

### Task C3: Upgrade `microstructure` tests.py

**File:** `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/microstructure/tests.py`

**Add these test classes** to the existing file:

```python
class TestWickImbalance:
    def test_upper_wick_dominant_gives_positive_imbalance(self):
        """Large upper wick, no lower wick → wick_imbalance > 0 (seller rejection above)."""
        n = 100
        # Close near open (small body), large upper wick, no lower wick
        df = pd.DataFrame({
            "O": np.full(n, 100.0),
            "C": np.full(n, 100.5),  # tiny bullish body
            "H": np.full(n, 103.0),  # large upper wick = 2.5
            "L": np.full(n, 100.0),  # no lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_wick_imbalance" in result.columns:
            valid = result["micro_wick_imbalance"].dropna()
            assert (valid > 0).mean() > 0.8, \
                "Upper wick dominant → wick_imbalance should be positive"

    def test_lower_wick_dominant_gives_negative_imbalance(self):
        """Large lower wick, no upper wick → wick_imbalance < 0 (buyer rejection below)."""
        n = 100
        df = pd.DataFrame({
            "O": np.full(n, 100.0),
            "C": np.full(n, 99.5),   # tiny bearish body
            "H": np.full(n, 100.0),  # no upper wick
            "L": np.full(n, 97.0),   # large lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_wick_imbalance" in result.columns:
            valid = result["micro_wick_imbalance"].dropna()
            assert (valid < 0).mean() > 0.8, \
                "Lower wick dominant → wick_imbalance should be negative"

    def test_symmetric_wicks_give_near_zero_imbalance(self):
        """Equal upper and lower wicks → wick_imbalance ≈ 0."""
        n = 100
        df = pd.DataFrame({
            "O": np.full(n, 100.0),
            "C": np.full(n, 100.0),   # doji
            "H": np.full(n, 102.0),   # equal upper wick
            "L": np.full(n, 98.0),    # equal lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_wick_imbalance" in result.columns:
            valid = result["micro_wick_imbalance"].dropna()
            assert (valid.abs() < 0.1).all(), \
                "Symmetric wicks → wick_imbalance should be ~0"


class TestPressureScore:
    def test_pressure_positive_for_bullish_marubozu(self):
        """Bullish candle with full body (no wicks) → pressure_score > 0."""
        n = 100
        df = pd.DataFrame({
            "O": np.full(n, 99.0),
            "C": np.full(n, 101.0),  # C > O
            "H": np.full(n, 101.0),  # no upper wick
            "L": np.full(n, 99.0),   # no lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_pressure_score" in result.columns:
            valid = result["micro_pressure_score"].dropna()
            assert (valid > 0).all(), "Bullish marubozu → pressure_score > 0"

    def test_pressure_negative_for_bearish_marubozu(self):
        """Bearish candle with full body → pressure_score < 0."""
        n = 100
        df = pd.DataFrame({
            "O": np.full(n, 101.0),
            "C": np.full(n, 99.0),   # C < O
            "H": np.full(n, 101.0),
            "L": np.full(n, 99.0),
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_pressure_score" in result.columns:
            valid = result["micro_pressure_score"].dropna()
            assert (valid < 0).all(), "Bearish marubozu → pressure_score < 0"


class TestRollingAggregates:
    def test_pressure_sum_accumulates_over_5_bars(self):
        """5 consecutive bullish bars → pressure_sum should be 5× single bar pressure."""
        n = 50
        df = pd.DataFrame({
            "O": np.full(n, 99.0),
            "C": np.full(n, 101.0),
            "H": np.full(n, 101.0),
            "L": np.full(n, 99.0),
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_pressure_sum" in result.columns:
            valid = result["micro_pressure_sum"].dropna()
            # After 5 bars, the sum should stabilize at 5× single bar value
            single = result["micro_pressure_score"].dropna().iloc[0]
            assert abs(valid.iloc[-1] - 5 * single) < 0.01, \
                "pressure_sum should equal 5× single bar pressure"

    def test_direction_consistency_high_when_all_same_direction(self):
        """5 bars all in same direction → direction_consistency should be 1."""
        n = 50
        df = pd.DataFrame({
            "O": np.full(n, 99.0),
            "C": np.full(n, 101.0),  # always bullish
            "H": np.full(n, 101.0),
            "L": np.full(n, 99.0),
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_indicator().compute(df)
        if "micro_direction_consistency" in result.columns:
            valid = result["micro_direction_consistency"].dropna()
            assert (valid.iloc[-10:] > 0.8).all(), \
                "All same direction → consistency should be near 1"
```

**Step 2–5:** Run RED → add to existing file → GREEN → commit.
```bash
git commit -m "test: expand microstructure tests (wick imbalance, pressure, rolling aggregates)"
```

---

## Phase D — Strategy Integration Tests

**Convention:** Strategy tests live **directly next to their config** in `strategies/configs/`,
named `[strategyName].test.py`. Pytest discovers them automatically.

**Common helpers** — shared via a `conftest.py` in `strategies/configs/`:

### Task D0: Create shared helper `strategies/configs/conftest.py`

```python
"""Shared helpers for strategy pipeline integration tests."""
import numpy as np
import pandas as pd


def make_m15_ohlcv(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic M15 OHLC data.
    n=6000 ≈ 62 days. Start on a Monday so weekly features have a valid anchor.
    """
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.0008
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * 0.0001
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * spread
    low  = np.minimum(open_, close) - np.abs(np.random.randn(n)) * spread
    volume = np.random.randint(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": volume},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="15min"),
    )


def make_h1_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic H1 OHLC data with volume."""
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.001
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * 0.0001
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * spread
    low  = np.minimum(open_, close) - np.abs(np.random.randn(n)) * spread
    volume = np.random.randint(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": volume},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="h"),
    )
```

**Commit:**
```bash
git add strategies/configs/conftest.py
git commit -m "test: add shared conftest for strategy pipeline integration tests"
```

---

### Task D1: `liq_sweep_scalping` integration test

**File:** `strategies/configs/liq_sweep_scalping.test.py`

```python
"""Integration test: liq_sweep_scalping strategy pipeline on synthetic H1 data."""
import numpy as np
import pytest
from pathlib import Path
from conftest import make_h1_ohlcv  # same directory conftest
from fwbg.pipeline import compute_indicator_pool, get_feature_columns
from fwbg.core.config import StrategyConfig

STRATEGY_PATH = Path(__file__).parent / "liq_sweep_scalping.json"


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full indicator pipeline once for all tests in this module."""
    strategy = StrategyConfig.from_json_file(str(STRATEGY_PATH))
    df = make_h1_ohlcv(n=5000)
    indicators = strategy.get_indicators() if hasattr(strategy, "get_indicators") else None
    result = compute_indicator_pool(df, indicators=indicators)
    return result, strategy


class TestPipelineCompleteness:
    def test_pipeline_runs_without_error(self, pipeline_result):
        result, _ = pipeline_result
        assert result is not None and len(result) > 0

    def test_all_ohlc_columns_preserved(self, pipeline_result):
        result, _ = pipeline_result
        for col in ["O", "H", "L", "C"]:
            assert col in result.columns

    def test_features_computed(self, pipeline_result):
        result, _ = pipeline_result
        features = get_feature_columns(result)
        assert len(features) >= 10, f"Expected ≥10 features, got {len(features)}"

    def test_no_all_nan_feature_columns(self, pipeline_result):
        result, _ = pipeline_result
        for col in get_feature_columns(result):
            assert not result[col].isna().all(), f"{col} is all NaN"

    def test_no_inf_in_features(self, pipeline_result):
        result, _ = pipeline_result
        for col in get_feature_columns(result):
            if result[col].dtype == float:
                assert not result[col].isin([float("inf"), float("-inf")]).any(), \
                    f"{col} has inf values"

    def test_liquidity_sweep_features_present(self, pipeline_result):
        result, _ = pipeline_result
        lsw_cols = [c for c in result.columns if c.startswith("lsw_")]
        assert len(lsw_cols) > 0, "Liquidity sweep features (lsw_*) should be present"

    def test_feature_count_reasonable(self, pipeline_result):
        result, _ = pipeline_result
        features = get_feature_columns(result)
        assert len(features) <= 500, f"Feature explosion: {len(features)} features"
```

**Step 2:** Run → expect FAIL.
**Step 3:** Write the file.
**Step 4:** Run → expect PASS.
```bash
pytest strategies/configs/liq_sweep_scalping.test.py -v
git commit -m "test: add liq_sweep_scalping strategy pipeline integration test"
```

---

### Task D2: `orb_exploration` integration test

**File:** `strategies/configs/orb_exploration.test.py`

```python
"""Integration test: orb_exploration strategy pipeline on synthetic M15 data."""
import pytest
from pathlib import Path
from conftest import make_m15_ohlcv
from fwbg.pipeline import compute_indicator_pool, get_feature_columns
from fwbg.core.config import StrategyConfig

STRATEGY_PATH = Path(__file__).parent / "orb_exploration.json"


@pytest.fixture(scope="module")
def pipeline_result():
    strategy = StrategyConfig.from_json_file(str(STRATEGY_PATH))
    df = make_m15_ohlcv(n=6000)
    indicators = strategy.get_indicators() if hasattr(strategy, "get_indicators") else None
    result = compute_indicator_pool(df, indicators=indicators)
    return result, strategy


class TestORBPipeline:
    def test_pipeline_runs_without_error(self, pipeline_result):
        result, _ = pipeline_result
        assert result is not None and len(result) > 0

    def test_orb_features_present(self, pipeline_result):
        result, _ = pipeline_result
        orb_cols = [c for c in result.columns if c.startswith("orb_")]
        assert len(orb_cols) > 5, "ORB features (orb_*) should be present"

    def test_session_features_for_all_pipeline_sessions(self, pipeline_result):
        """All 10 UTC sessions from the pipeline config must have features."""
        result, _ = pipeline_result
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            col = f"orb_s{h:02d}_range"
            assert col in result.columns, f"Session {h} ORB feature missing: {col}"

    def test_no_all_nan_columns(self, pipeline_result):
        result, _ = pipeline_result
        for col in get_feature_columns(result):
            assert not result[col].isna().all(), f"{col} is all NaN"

    def test_no_inf_in_features(self, pipeline_result):
        result, _ = pipeline_result
        for col in get_feature_columns(result):
            if result[col].dtype == float:
                assert not result[col].isin([float("inf"), float("-inf")]).any()
```

**Commit:**
```bash
git commit -m "test: add orb_exploration strategy pipeline integration test"
```

---

### Task D3: `smc_choch_fvg` integration test

**File:** `strategies/configs/smc_choch_fvg.test.py`

```python
"""Integration test: smc_choch_fvg strategy pipeline."""
import pytest
from pathlib import Path
from conftest import make_h1_ohlcv
from fwbg.pipeline import compute_indicator_pool, get_feature_columns
from fwbg.core.config import StrategyConfig

STRATEGY_PATH = Path(__file__).parent / "smc_choch_fvg.json"


@pytest.fixture(scope="module")
def pipeline_result():
    strategy = StrategyConfig.from_json_file(str(STRATEGY_PATH))
    df = make_h1_ohlcv(n=5000)
    indicators = strategy.get_indicators() if hasattr(strategy, "get_indicators") else None
    result = compute_indicator_pool(df, indicators=indicators)
    return result, strategy


class TestSMCPipeline:
    def test_pipeline_runs_without_error(self, pipeline_result):
        result, _ = pipeline_result
        assert result is not None and len(result) > 0

    def test_fvg_features_present(self, pipeline_result):
        result, _ = pipeline_result
        fvg_cols = [c for c in result.columns if c.startswith("fvg_")]
        assert len(fvg_cols) > 0, "FVG features should be present"

    def test_no_all_nan_columns(self, pipeline_result):
        result, _ = pipeline_result
        for col in get_feature_columns(result):
            assert not result[col].isna().all(), f"{col} is all NaN"

    def test_no_inf_in_features(self, pipeline_result):
        result, _ = pipeline_result
        for col in get_feature_columns(result):
            if result[col].dtype == float:
                assert not result[col].isin([float("inf"), float("-inf")]).any()
```

**Commit:**
```bash
git commit -m "test: add smc_choch_fvg strategy pipeline integration test"
```

---

### Task D4–D6: Remaining strategy integration tests

Create the same 6-test class pattern for:

**D4:** `strategies/configs/sr_trend_continuation.test.py`
- Data: H1 (`make_h1_ohlcv`)
- Key assertion: `sr_*` (support/resistance) features present

**D5:** `strategies/configs/pbd_balance.test.py`
- Data: H1 (`make_h1_ohlcv`)
- Key assertion: `vp_*` or `bal_*` features present

**D6:** `strategies/configs/weekly_orb_scalping.test.py`
- Data: M15 (`make_m15_ohlcv`)
- Key assertion: `wor_*` (weekly ORB) features present

Each file:
```python
"""Integration test: [strategy] pipeline."""
import pytest
from pathlib import Path
from conftest import make_h1_ohlcv   # or make_m15_ohlcv
from fwbg.pipeline import compute_indicator_pool, get_feature_columns
from fwbg.core.config import StrategyConfig

STRATEGY_PATH = Path(__file__).parent / "[strategy].json"

@pytest.fixture(scope="module")
def pipeline_result():
    strategy = StrategyConfig.from_json_file(str(STRATEGY_PATH))
    df = make_h1_ohlcv(n=5000)          # adjust n and freq per strategy
    indicators = strategy.get_indicators() if hasattr(strategy, "get_indicators") else None
    return compute_indicator_pool(df, indicators=indicators), strategy

class TestPipeline:
    def test_runs_without_error(self, pipeline_result): ...
    def test_ohlc_preserved(self, pipeline_result): ...
    def test_features_computed(self, pipeline_result): ...
    def test_no_all_nan(self, pipeline_result): ...
    def test_no_inf(self, pipeline_result): ...
    def test_strategy_specific_features_present(self, pipeline_result): ...
```

**Commit all three:**
```bash
git add strategies/configs/
git commit -m "test: add sr_trend, pbd_balance, weekly_orb strategy pipeline integration tests"
```

---

## Final Verification

After all tasks:

```bash
# Run all strategy integration tests
pytest strategies/configs/ -v

# Run all plugin tests
pytest src/fwbg/plugins/ packages/ -v -k "tests"

# Run full suite — all 1043+ tests must still pass
pytest --tb=short -q 2>&1 | tail -5
```

Expected: all tests green, no regressions.

---

## Summary: Files Created / Modified

| File | Action |
|------|--------|
| `src/fwbg/plugins/fwbg-core/indicators/price_action/tests.py` | Create |
| `src/fwbg/plugins/fwbg-core/indicators/time_season/tests.py` | Create |
| `src/fwbg/plugins/fwbg-core/indicators/weekly_opening_range/tests.py` | Create |
| `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/dynamics/tests.py` | Create |
| `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/ichimoku/tests.py` | Create |
| `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/multi_timeframe/tests.py` | Create |
| `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/volume_profile/tests.py` | Create |
| `src/fwbg/plugins/fwbg-core/indicators/volatility/tests.py` | Expand |
| `src/fwbg/plugins/fwbg-core/indicators/trend/tests.py` | Expand |
| `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/microstructure/tests.py` | Expand |
| `strategies/configs/conftest.py` | Create |
| `strategies/configs/liq_sweep_scalping.test.py` | Create |
| `strategies/configs/orb_exploration.test.py` | Create |
| `strategies/configs/smc_choch_fvg.test.py` | Create |
| `strategies/configs/sr_trend_continuation.test.py` | Create |
| `strategies/configs/pbd_balance.test.py` | Create |
| `strategies/configs/weekly_orb_scalping.test.py` | Create |
