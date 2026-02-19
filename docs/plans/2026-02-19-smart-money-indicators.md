# Smart Money Indicators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4 new indicator plugins (previous_day_levels, displacement, supply_demand_flip, liquidity_levels) and a strategy config combining them with ORB for intraday scalping.

**Architecture:** Each indicator is a self-contained plugin under `src/fwbg/plugins/fwbg-core/indicators/<name>/` with `__init__.py`, `tests.py`, and `manifest.json`. All indicators follow the `BaseIndicator` pattern: compute features → shift_features → concat. The strategy config `strategies/orb_pdhl_scalping.json` combines these with existing ORB/trend/momentum indicators.

**Tech Stack:** Python, pandas, numpy, fwbg_sdk (BaseIndicator, register_indicator, shift_features, safe_divide, EPSILON)

---

## Task 1: `previous_day_levels` Indicator

Computes features relative to the previous trading day's high and low. These are key support/resistance levels used in intraday trading. On daily data, returns df unchanged (same pattern as `opening_range`).

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/indicators/previous_day_levels/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/indicators/previous_day_levels/manifest.json`
- Create: `src/fwbg/plugins/fwbg-core/indicators/previous_day_levels/tests.py`

### Features (10 total)

| Feature | Type | Description |
|---------|------|-------------|
| `pdl_high_dist` | continuous | Distance from close to PDH, normalized by ATR. Positive = below PDH |
| `pdl_low_dist` | continuous | Distance from close to PDL, normalized by ATR. Positive = above PDL |
| `pdl_position` | continuous | Position within PDH/PDL range: 0=at PDL, 1=at PDH |
| `pdl_range_vs_atr` | continuous | Previous day range / ATR — expansion vs contraction |
| `pdl_above_high` | signal | 1 if close > PDH, else 0 |
| `pdl_below_low` | signal | 1 if close < PDL, else 0 |
| `pdl_high_break` | signal | 1 on bar where close first crosses above PDH for this day |
| `pdl_low_break` | signal | 1 on bar where close first crosses below PDL for this day |
| `pdl_range_position_ma` | continuous | Rolling mean of pdl_position over `ma_period` bars — trend within daily range |
| `pdl_day_range_expanding` | signal | 1 if current day range > prev day range (expanding volatility) |

### Step 1: Write manifest.json

```json
{
  "name": "previous_day_levels",
  "version": "1.0.0",
  "description": "Previous Day High/Low features: distance, position, break detection for intraday S/R",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

### Step 2: Write the indicator implementation

```python
"""
Previous Day Levels Indicator Plugin.

Computes features relative to previous day's high and low:
- Distance to PDH/PDL in ATR units
- Position within daily range
- Break detection (first cross above PDH or below PDL)
- Range expansion/contraction

Timeframe: Intraday only (M1-H4). On daily bars returns df unchanged.
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide, EPSILON


def _compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Compute ATR from OHLC data."""
    highs = df["H"].values
    lows = df["L"].values
    close = df["C"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


def _compute_pdl_features(
    df: pd.DataFrame,
    atr: np.ndarray,
    ma_period: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all previous day level features."""
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    n = len(df)

    # Group by trading day
    day_group = df.index.normalize()

    # Per-day high/low
    day_high = df["H"].groupby(day_group).transform("max")
    day_low = df["L"].groupby(day_group).transform("min")

    # Shift by 1 day to get PREVIOUS day high/low
    # Create mapping: date -> (high, low)
    unique_days = day_group.unique()
    day_hl = pd.DataFrame({
        "high": df["H"].groupby(day_group).max(),
        "low": df["L"].groupby(day_group).min(),
    })
    prev_day_hl = day_hl.shift(1)

    # Map previous day H/L back to each bar
    pdh = prev_day_hl["high"].reindex(day_group).values
    pdl_low = prev_day_hl["low"].reindex(day_group).values
    pd_range = pdh - pdl_low

    close = df["C"].values
    safe_atr = np.where(atr > EPSILON, atr, 1.0)
    safe_range = np.where(np.abs(pd_range) > EPSILON, pd_range, np.nan)

    # Distance features (ATR-normalized)
    features["pdl_high_dist"] = (pdh - close) / safe_atr
    features["pdl_low_dist"] = (close - pdl_low) / safe_atr

    # Position within range: 0=at PDL, 1=at PDH
    features["pdl_position"] = (close - pdl_low) / safe_range

    # Range vs ATR
    features["pdl_range_vs_atr"] = pd_range / safe_atr

    # Binary: above PDH / below PDL
    features["pdl_above_high"] = (close > pdh).astype(float)
    features["pdl_below_low"] = (close < pdl_low).astype(float)

    # Break detection: first bar of the day where close crosses PDH/PDL
    above = close > pdh
    below = close < pdl_low
    day_ids = pd.Series(day_group).factorize()[0]

    high_break = np.zeros(n)
    low_break = np.zeros(n)
    prev_day_id = -1
    already_broke_high = False
    already_broke_low = False
    for i in range(n):
        if day_ids[i] != prev_day_id:
            prev_day_id = day_ids[i]
            already_broke_high = False
            already_broke_low = False
        if above[i] and not already_broke_high:
            high_break[i] = 1.0
            already_broke_high = True
        if below[i] and not already_broke_low:
            low_break[i] = 1.0
            already_broke_low = True

    features["pdl_high_break"] = high_break
    features["pdl_low_break"] = low_break

    # Rolling MA of position
    pos_series = pd.Series(features["pdl_position"])
    features["pdl_range_position_ma"] = pos_series.rolling(
        ma_period, min_periods=1
    ).mean().values

    # Range expanding: current day range > previous day range
    current_day_range = (day_high - day_low).values
    features["pdl_day_range_expanding"] = (current_day_range > pd_range).astype(float)

    # NaN out first day (no previous day data)
    first_valid_day = unique_days[1] if len(unique_days) > 1 else unique_days[0]
    mask_first_day = day_group < first_valid_day
    for key in features:
        arr = features[key]
        if isinstance(arr, np.ndarray):
            arr = arr.astype(float)
            arr[mask_first_day] = np.nan
            features[key] = arr
        else:
            features[key] = np.where(mask_first_day, np.nan, arr)

    return features


@register_indicator("previous_day_levels")
class PreviousDayLevelsIndicator(BaseIndicator):
    """Previous Day High/Low features for intraday trading."""

    name = "previous_day_levels"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    _FEATURES = [
        "pdl_high_dist",
        "pdl_low_dist",
        "pdl_position",
        "pdl_range_vs_atr",
        "pdl_above_high",
        "pdl_below_low",
        "pdl_high_break",
        "pdl_low_break",
        "pdl_range_position_ma",
        "pdl_day_range_expanding",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        ma_period: int = 20,
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        # Skip daily data — PDL is intraday only
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                return df

        atr = _compute_atr(df, atr_period)
        features = _compute_pdl_features(df, atr, ma_period)

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return [
            "pdl_above_high", "pdl_below_low",
            "pdl_high_break", "pdl_low_break",
            "pdl_day_range_expanding",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"atr_period": 14, "ma_period": 20}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing distances. PDH/PDL distances are expressed in ATR units.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "ma_period": {
                "type": "int",
                "default": 20,
                "description": "Rolling window for position moving average. Smooths the intraday position within the previous day's range.",
                "min": 5,
                "max": 100,
                "step": 5,
            },
        }


__all__ = ["PreviousDayLevelsIndicator"]
```

### Step 3: Write tests

```python
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
```

### Step 4: Run tests

```bash
python -m pytest src/fwbg/plugins/fwbg-core/indicators/previous_day_levels/tests.py -v
```

### Step 5: Commit

```bash
git add src/fwbg/plugins/fwbg-core/indicators/previous_day_levels/
git commit -m "feat: add previous_day_levels indicator plugin"
```

---

## Task 2: `displacement` Indicator

Measures breakout quality — how violently price breaks through a level. Strong displacement (large candle bodies, FVG formation) indicates high-probability breakouts. Weak displacement (wicks, consolidation) indicates likely failure.

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/indicators/displacement/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/indicators/displacement/manifest.json`
- Create: `src/fwbg/plugins/fwbg-core/indicators/displacement/tests.py`

### Features (8 total)

| Feature | Type | Description |
|---------|------|-------------|
| `disp_body_ratio` | continuous | Candle body / full range (0=doji, 1=marubozu). High = strong conviction |
| `disp_body_atr` | continuous | Candle body size / ATR. Measures impulse magnitude |
| `disp_upper_wick_ratio` | continuous | Upper wick / full range. High = rejection, sellers stepping in |
| `disp_lower_wick_ratio` | continuous | Lower wick / full range. High = rejection, buyers stepping in |
| `disp_fvg_formed` | signal | 1 if a Fair Value Gap formed at this bar (3-candle imbalance) |
| `disp_consecutive_dir` | continuous | Count of consecutive same-direction candles (signed: +bull, -bear) |
| `disp_range_expansion` | continuous | Current range / rolling avg range — detects unusual moves |
| `disp_close_position` | continuous | (Close - Low) / (High - Low). 1=closed at high (bullish), 0=closed at low (bearish) |

### Step 1: Write manifest.json

```json
{
  "name": "displacement",
  "version": "1.0.0",
  "description": "Displacement features: breakout quality, candle conviction, impulse detection",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

### Step 2: Write the indicator implementation

```python
"""
Displacement Indicator Plugin.

Measures breakout quality and candle conviction:
- Body ratio and size relative to ATR (impulse strength)
- Wick analysis (rejection detection)
- FVG formation at current bar (imbalance confirmation)
- Consecutive directional candles (momentum persistence)
- Range expansion (unusual move detection)

These features help ML models distinguish between high-probability breakouts
(strong displacement) and likely failures (weak displacement/wicks).
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide, EPSILON


def _compute_displacement_features(
    df: pd.DataFrame,
    atr_period: int,
    range_avg_period: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all displacement features."""
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    n = len(df)

    o = df["O"].values
    h = df["H"].values
    l = df["L"].values  # noqa: E741
    c = df["C"].values

    candle_range = h - l
    body = np.abs(c - o)
    safe_range = np.where(candle_range > EPSILON, candle_range, np.nan)

    # ATR
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(candle_range, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
    safe_atr = np.where(atr > EPSILON, atr, 1.0)

    # Body ratio: body / range (0=doji, 1=marubozu)
    features["disp_body_ratio"] = body / safe_range

    # Body / ATR: impulse magnitude
    features["disp_body_atr"] = body / safe_atr

    # Wick ratios
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    features["disp_upper_wick_ratio"] = upper_wick / safe_range
    features["disp_lower_wick_ratio"] = lower_wick / safe_range

    # FVG detection at current bar: H[i-2] < L[i] (bull) or L[i-2] > H[i] (bear)
    fvg_formed = np.zeros(n)
    for i in range(2, n):
        if h[i - 2] < l[i] or l[i - 2] > h[i]:
            fvg_formed[i] = 1.0
    features["disp_fvg_formed"] = fvg_formed

    # Consecutive same-direction candles (signed)
    direction = np.sign(c - o)  # +1 bull, -1 bear, 0 doji
    consecutive = np.zeros(n, dtype=float)
    for i in range(1, n):
        if direction[i] == 0:
            consecutive[i] = 0.0
        elif direction[i] == direction[i - 1]:
            consecutive[i] = consecutive[i - 1] + direction[i]
        else:
            consecutive[i] = direction[i]
    features["disp_consecutive_dir"] = consecutive

    # Range expansion: current range / rolling average range
    avg_range = pd.Series(candle_range).rolling(
        range_avg_period, min_periods=1
    ).mean().values
    safe_avg_range = np.where(avg_range > EPSILON, avg_range, 1.0)
    features["disp_range_expansion"] = candle_range / safe_avg_range

    # Close position: (C - L) / (H - L). 1=top, 0=bottom
    features["disp_close_position"] = (c - l) / safe_range

    return features


@register_indicator("displacement")
class DisplacementIndicator(BaseIndicator):
    """Displacement/breakout quality features for ML trading."""

    name = "displacement"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "price_action"

    _FEATURES = [
        "disp_body_ratio",
        "disp_body_atr",
        "disp_upper_wick_ratio",
        "disp_lower_wick_ratio",
        "disp_fvg_formed",
        "disp_consecutive_dir",
        "disp_range_expansion",
        "disp_close_position",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        range_avg_period: int = 20,
        **params,
    ) -> pd.DataFrame:
        features = _compute_displacement_features(df, atr_period, range_avg_period)

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["disp_fvg_formed"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"atr_period": 14, "range_avg_period": 20}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing body size. Body/ATR measures impulse magnitude in volatility-adjusted terms.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "range_avg_period": {
                "type": "int",
                "default": 20,
                "description": "Rolling window for average range computation. Range expansion compares current candle range to this rolling average.",
                "min": 5,
                "max": 100,
                "step": 5,
            },
        }


__all__ = ["DisplacementIndicator"]
```

### Step 3: Write tests

```python
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
    # Add impulse moves every 30 bars
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
```

### Step 4: Run tests

```bash
python -m pytest src/fwbg/plugins/fwbg-core/indicators/displacement/tests.py -v
```

### Step 5: Commit

```bash
git add src/fwbg/plugins/fwbg-core/indicators/displacement/
git commit -m "feat: add displacement indicator plugin"
```

---

## Task 3: `supply_demand_flip` Indicator

Detects zones where support flips to resistance (or vice versa) after a break of structure. Tracks active flip zones and measures the current price's relationship to them.

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/indicators/supply_demand_flip/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/indicators/supply_demand_flip/manifest.json`
- Create: `src/fwbg/plugins/fwbg-core/indicators/supply_demand_flip/tests.py`

### Features (8 total)

| Feature | Type | Description |
|---------|------|-------------|
| `sdf_bull_active` | signal | 1 if a bullish flip zone (demand→supply→demand) is active nearby |
| `sdf_bear_active` | signal | 1 if a bearish flip zone (supply→demand→supply) is active nearby |
| `sdf_bull_dist` | continuous | Distance to nearest bullish flip zone in ATR units |
| `sdf_bear_dist` | continuous | Distance to nearest bearish flip zone in ATR units |
| `sdf_bull_strength` | continuous | Strength of nearest bullish flip zone (breakout displacement / ATR) |
| `sdf_bear_strength` | continuous | Strength of nearest bearish flip zone |
| `sdf_bull_touches` | continuous | Number of times price has touched the nearest bullish flip zone |
| `sdf_bear_touches` | continuous | Number of times price has touched the nearest bearish flip zone |

### Step 1: Write manifest.json

```json
{
  "name": "supply_demand_flip",
  "version": "1.0.0",
  "description": "Supply/Demand Flip Zones: detects S/R polarity changes with distance and strength features",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

### Step 2: Write the indicator implementation

```python
"""
Supply/Demand Flip Zone Indicator Plugin.

Detects zones where supply turns into demand (or vice versa):
- Identifies swing highs/lows as potential S/R zones
- When a zone is broken, it flips polarity (support → resistance, resistance → support)
- Tracks active flip zones with distance, strength, and touch count

A bullish flip zone: price broke above resistance, now that level acts as support.
A bearish flip zone: price broke below support, now that level acts as resistance.
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON


def _find_swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int):
    """Find swing highs and swing lows using a simple N-bar lookback."""
    n = len(highs)
    swing_highs = []  # (index, price)
    swing_lows = []

    for i in range(lookback, n - lookback):
        # Swing high: highest high in window
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        # Swing low: lowest low in window
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def _compute_sdf_features(
    df: pd.DataFrame,
    swing_lookback: int,
    zone_atr_width: float,
    atr_period: int,
    max_active_zones: int,
    zone_expiry: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute supply/demand flip zone features."""
    n = len(df)
    h = df["H"].values
    l = df["L"].values  # noqa: E741
    c = df["C"].values

    # ATR
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    # Find swing points
    swing_highs, swing_lows = _find_swing_points(h, l, swing_lookback)

    # Track zones: each zone = {level, type, bar_created, strength, touches, active}
    # "resistance" zones that get broken become bullish flip zones (support)
    # "support" zones that get broken become bearish flip zones (resistance)

    # Build per-bar features
    bull_active = np.zeros(n)
    bear_active = np.zeros(n)
    bull_dist = np.full(n, np.nan)
    bear_dist = np.full(n, np.nan)
    bull_strength = np.full(n, np.nan)
    bear_strength = np.full(n, np.nan)
    bull_touches = np.zeros(n)
    bear_touches = np.zeros(n)

    # Active resistance levels (from swing highs) and support levels (from swing lows)
    resistance_zones = []  # [(level, bar, broken)]
    support_zones = []
    flip_zones = []  # [(level, type='bull'|'bear', bar_created, strength, touches)]

    # Index swing points by bar for efficient lookup
    sh_by_bar = {bar: price for bar, price in swing_highs}
    sl_by_bar = {bar: price for bar, price in swing_lows}

    for i in range(n):
        current_atr = atr[i] if atr[i] > EPSILON else 1.0
        zone_width = current_atr * zone_atr_width

        # Register new swing points as zones
        if i in sh_by_bar:
            resistance_zones.append({"level": sh_by_bar[i], "bar": i})
        if i in sl_by_bar:
            support_zones.append({"level": sl_by_bar[i], "bar": i})

        # Check if price breaks through resistance → create bullish flip zone
        surviving_resistance = []
        for zone in resistance_zones:
            if i - zone["bar"] > zone_expiry:
                continue
            if c[i] > zone["level"] + zone_width:
                # Broken! Create bullish flip zone
                strength = (c[i] - zone["level"]) / current_atr
                flip_zones.append({
                    "level": zone["level"],
                    "type": "bull",
                    "bar": i,
                    "strength": strength,
                    "touches": 0,
                })
            else:
                surviving_resistance.append(zone)
        resistance_zones = surviving_resistance

        # Check if price breaks through support → create bearish flip zone
        surviving_support = []
        for zone in support_zones:
            if i - zone["bar"] > zone_expiry:
                continue
            if c[i] < zone["level"] - zone_width:
                strength = (zone["level"] - c[i]) / current_atr
                flip_zones.append({
                    "level": zone["level"],
                    "type": "bear",
                    "bar": i,
                    "strength": strength,
                    "touches": 0,
                })
            else:
                surviving_support.append(zone)
        support_zones = surviving_support

        # Update flip zones: check touches, expiry, invalidation
        surviving_flips = []
        for fz in flip_zones:
            if i - fz["bar"] > zone_expiry:
                continue
            # Bullish flip zone invalidated if price closes below it
            if fz["type"] == "bull" and c[i] < fz["level"] - zone_width:
                continue
            # Bearish flip zone invalidated if price closes above it
            if fz["type"] == "bear" and c[i] > fz["level"] + zone_width:
                continue
            # Check touch (price within zone_width of level)
            if abs(c[i] - fz["level"]) <= zone_width:
                fz["touches"] += 1
            surviving_flips.append(fz)
        flip_zones = surviving_flips[-max_active_zones:]

        # Find nearest bull and bear flip zones
        nearest_bull_d = np.inf
        nearest_bull_s = 0.0
        nearest_bull_t = 0
        nearest_bear_d = np.inf
        nearest_bear_s = 0.0
        nearest_bear_t = 0

        for fz in flip_zones:
            if fz["type"] == "bull":
                d = (c[i] - fz["level"]) / current_atr
                if d > 0 and d < nearest_bull_d:
                    nearest_bull_d = d
                    nearest_bull_s = fz["strength"]
                    nearest_bull_t = fz["touches"]
            else:
                d = (fz["level"] - c[i]) / current_atr
                if d > 0 and d < nearest_bear_d:
                    nearest_bear_d = d
                    nearest_bear_s = fz["strength"]
                    nearest_bear_t = fz["touches"]

        if nearest_bull_d < np.inf:
            bull_active[i] = 1.0
            bull_dist[i] = nearest_bull_d
            bull_strength[i] = nearest_bull_s
            bull_touches[i] = nearest_bull_t

        if nearest_bear_d < np.inf:
            bear_active[i] = 1.0
            bear_dist[i] = nearest_bear_d
            bear_strength[i] = nearest_bear_s
            bear_touches[i] = nearest_bear_t

    return {
        "sdf_bull_active": bull_active,
        "sdf_bear_active": bear_active,
        "sdf_bull_dist": bull_dist,
        "sdf_bear_dist": bear_dist,
        "sdf_bull_strength": bull_strength,
        "sdf_bear_strength": bear_strength,
        "sdf_bull_touches": bull_touches,
        "sdf_bear_touches": bear_touches,
    }


@register_indicator("supply_demand_flip")
class SupplyDemandFlipIndicator(BaseIndicator):
    """Supply/Demand Flip Zone features for ML trading."""

    name = "supply_demand_flip"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "structure"

    _FEATURES = [
        "sdf_bull_active",
        "sdf_bear_active",
        "sdf_bull_dist",
        "sdf_bear_dist",
        "sdf_bull_strength",
        "sdf_bear_strength",
        "sdf_bull_touches",
        "sdf_bear_touches",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 10,
        zone_atr_width: float = 0.3,
        atr_period: int = 14,
        max_active_zones: int = 20,
        zone_expiry: int = 200,
        **params,
    ) -> pd.DataFrame:
        features = _compute_sdf_features(
            df, swing_lookback, zone_atr_width, atr_period,
            max_active_zones, zone_expiry,
        )

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["sdf_bull_active", "sdf_bear_active"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_lookback": 10,
            "zone_atr_width": 0.3,
            "atr_period": 14,
            "max_active_zones": 20,
            "zone_expiry": 200,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int",
                "default": 10,
                "description": "N-bar lookback for swing high/low detection. Larger values find more significant pivots but miss minor ones.",
                "min": 3,
                "max": 50,
                "step": 1,
            },
            "zone_atr_width": {
                "type": "float",
                "default": 0.3,
                "description": "Zone width as fraction of ATR. Price must break beyond level ± this width to trigger a flip. Also used for touch detection.",
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing distances, strength, and zone width.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "max_active_zones": {
                "type": "int",
                "default": 20,
                "description": "Maximum number of active flip zones to track. Keeps only the most recent ones.",
                "min": 5,
                "max": 50,
                "step": 5,
            },
            "zone_expiry": {
                "type": "int",
                "default": 200,
                "description": "Number of bars after which a zone expires and is no longer tracked.",
                "min": 50,
                "max": 1000,
                "step": 50,
            },
        }


__all__ = ["SupplyDemandFlipIndicator"]
```

### Step 3: Write tests

```python
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


def _make_trending_df(n=500):
    """Create data with clear uptrend that should produce flip zones."""
    rng = np.random.default_rng(77)
    # Uptrend with pullbacks
    trend = np.linspace(100, 140, n)
    oscillation = 3 * np.sin(np.linspace(0, 15 * np.pi, n))
    prices = trend + oscillation + rng.normal(0, 0.3, n)
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
        # At least some flip zones should be active in trending data
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
```

### Step 4: Run tests

```bash
python -m pytest src/fwbg/plugins/fwbg-core/indicators/supply_demand_flip/tests.py -v
```

### Step 5: Commit

```bash
git add src/fwbg/plugins/fwbg-core/indicators/supply_demand_flip/
git commit -m "feat: add supply_demand_flip indicator plugin"
```

---

## Task 4: `liquidity_levels` Indicator

Detects equal highs/lows where stop-loss orders cluster (liquidity pools). Tracks distance to these pools and detects liquidity sweeps (price briefly exceeds the level then reverses).

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/indicators/liquidity_levels/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/indicators/liquidity_levels/manifest.json`
- Create: `src/fwbg/plugins/fwbg-core/indicators/liquidity_levels/tests.py`

### Features (8 total)

| Feature | Type | Description |
|---------|------|-------------|
| `liq_eqh_dist` | continuous | Distance to nearest equal highs (ATR-normalized). Negative = below |
| `liq_eql_dist` | continuous | Distance to nearest equal lows (ATR-normalized). Positive = above |
| `liq_eqh_count` | continuous | Number of touches forming the equal highs level |
| `liq_eql_count` | continuous | Number of touches forming the equal lows level |
| `liq_eqh_active` | signal | 1 if equal highs liquidity pool exists above current price |
| `liq_eql_active` | signal | 1 if equal lows liquidity pool exists below current price |
| `liq_sweep_up` | signal | 1 if price swept above equal highs then closed below (stop hunt) |
| `liq_sweep_down` | signal | 1 if price swept below equal lows then closed above (stop hunt) |

### Step 1: Write manifest.json

```json
{
  "name": "liquidity_levels",
  "version": "1.0.0",
  "description": "Liquidity level detection: equal highs/lows, sweep detection, stop-loss cluster proximity",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

### Step 2: Write the indicator implementation

```python
"""
Liquidity Levels Indicator Plugin.

Detects liquidity pools from equal highs/lows and sweep events:
- Equal highs: multiple swing highs at similar price → stops cluster above
- Equal lows: multiple swing lows at similar price → stops cluster below
- Sweeps: price briefly exceeds level then reverses (stop hunt)

Smart money targets these liquidity pools before reversing.
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON


def _find_equal_levels(
    prices: np.ndarray,
    indices: np.ndarray,
    tolerance: float,
    min_touches: int,
):
    """Group price levels within tolerance and return clusters with >= min_touches.

    Returns list of (level, touch_count, last_bar_index).
    """
    if len(prices) == 0:
        return []

    # Sort by price
    order = np.argsort(prices)
    sorted_prices = prices[order]
    sorted_indices = indices[order]

    clusters = []
    cluster_start = 0

    for i in range(1, len(sorted_prices) + 1):
        if i == len(sorted_prices) or sorted_prices[i] - sorted_prices[cluster_start] > tolerance:
            count = i - cluster_start
            if count >= min_touches:
                level = np.mean(sorted_prices[cluster_start:i])
                last_bar = int(np.max(sorted_indices[cluster_start:i]))
                clusters.append((level, count, last_bar))
            cluster_start = i

    return clusters


def _compute_liquidity_features(
    df: pd.DataFrame,
    swing_lookback: int,
    tolerance_atr_mult: float,
    atr_period: int,
    min_touches: int,
    lookback_window: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all liquidity level features."""
    n = len(df)
    h = df["H"].values
    l = df["L"].values  # noqa: E741
    c = df["C"].values

    # ATR
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    # Find all swing highs and lows
    swing_high_bars = []
    swing_high_prices = []
    swing_low_bars = []
    swing_low_prices = []

    for i in range(swing_lookback, n - swing_lookback):
        if h[i] == np.max(h[i - swing_lookback:i + swing_lookback + 1]):
            swing_high_bars.append(i)
            swing_high_prices.append(h[i])
        if l[i] == np.min(l[i - swing_lookback:i + swing_lookback + 1]):
            swing_low_bars.append(i)
            swing_low_prices.append(l[i])

    swing_high_bars = np.array(swing_high_bars)
    swing_high_prices = np.array(swing_high_prices)
    swing_low_bars = np.array(swing_low_bars)
    swing_low_prices = np.array(swing_low_prices)

    # Per-bar features
    eqh_dist = np.full(n, np.nan)
    eql_dist = np.full(n, np.nan)
    eqh_count = np.zeros(n)
    eql_count = np.zeros(n)
    eqh_active = np.zeros(n)
    eql_active = np.zeros(n)
    sweep_up = np.zeros(n)
    sweep_down = np.zeros(n)

    for i in range(swing_lookback * 2, n):
        current_atr = atr[i] if atr[i] > EPSILON else 1.0
        tolerance = current_atr * tolerance_atr_mult

        # Filter swing points within lookback window
        sh_mask = (swing_high_bars < i) & (swing_high_bars >= i - lookback_window)
        sl_mask = (swing_low_bars < i) & (swing_low_bars >= i - lookback_window)

        # Find equal high clusters
        if sh_mask.any():
            eq_highs = _find_equal_levels(
                swing_high_prices[sh_mask], swing_high_bars[sh_mask],
                tolerance, min_touches,
            )
            # Nearest equal highs above price
            above = [(lvl, cnt) for lvl, cnt, _ in eq_highs if lvl > c[i]]
            if above:
                nearest = min(above, key=lambda x: x[0] - c[i])
                eqh_dist[i] = (nearest[0] - c[i]) / current_atr
                eqh_count[i] = nearest[1]
                eqh_active[i] = 1.0

                # Sweep detection: high exceeded level but close is below
                if h[i] > nearest[0] and c[i] < nearest[0]:
                    sweep_up[i] = 1.0

        # Find equal low clusters
        if sl_mask.any():
            eq_lows = _find_equal_levels(
                swing_low_prices[sl_mask], swing_low_bars[sl_mask],
                tolerance, min_touches,
            )
            # Nearest equal lows below price
            below = [(lvl, cnt) for lvl, cnt, _ in eq_lows if lvl < c[i]]
            if below:
                nearest = max(below, key=lambda x: x[0])
                eql_dist[i] = (c[i] - nearest[0]) / current_atr
                eql_count[i] = nearest[1]
                eql_active[i] = 1.0

                # Sweep detection: low went below level but close is above
                if l[i] < nearest[0] and c[i] > nearest[0]:
                    sweep_down[i] = 1.0

    return {
        "liq_eqh_dist": eqh_dist,
        "liq_eql_dist": eql_dist,
        "liq_eqh_count": eqh_count,
        "liq_eql_count": eql_count,
        "liq_eqh_active": eqh_active,
        "liq_eql_active": eql_active,
        "liq_sweep_up": sweep_up,
        "liq_sweep_down": sweep_down,
    }


@register_indicator("liquidity_levels")
class LiquidityLevelsIndicator(BaseIndicator):
    """Liquidity level detection features for ML trading."""

    name = "liquidity_levels"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "structure"

    _FEATURES = [
        "liq_eqh_dist",
        "liq_eql_dist",
        "liq_eqh_count",
        "liq_eql_count",
        "liq_eqh_active",
        "liq_eql_active",
        "liq_sweep_up",
        "liq_sweep_down",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 10,
        tolerance_atr_mult: float = 0.2,
        atr_period: int = 14,
        min_touches: int = 2,
        lookback_window: int = 200,
        **params,
    ) -> pd.DataFrame:
        features = _compute_liquidity_features(
            df, swing_lookback, tolerance_atr_mult, atr_period,
            min_touches, lookback_window,
        )

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["liq_eqh_active", "liq_eql_active", "liq_sweep_up", "liq_sweep_down"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_lookback": 10,
            "tolerance_atr_mult": 0.2,
            "atr_period": 14,
            "min_touches": 2,
            "lookback_window": 200,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int",
                "default": 10,
                "description": "N-bar lookback for swing high/low detection.",
                "min": 3,
                "max": 50,
                "step": 1,
            },
            "tolerance_atr_mult": {
                "type": "float",
                "default": 0.2,
                "description": "How close swing highs/lows must be (in ATR units) to count as 'equal'. Smaller = stricter matching.",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing distances and tolerance.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "min_touches": {
                "type": "int",
                "default": 2,
                "description": "Minimum number of swing points at similar price to form a liquidity level.",
                "min": 2,
                "max": 5,
                "step": 1,
            },
            "lookback_window": {
                "type": "int",
                "default": 200,
                "description": "Number of bars to look back for swing points when building liquidity levels.",
                "min": 50,
                "max": 1000,
                "step": 50,
            },
        }


__all__ = ["LiquidityLevelsIndicator"]
```

### Step 3: Write tests

```python
"""Tests for liquidity_levels indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_liq = import_plugin_module("fwbg-core", "indicators", "liquidity_levels")
if _liq is None:
    pytest.skip("liquidity_levels plugin not available", allow_module_level=True)


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


def _make_ranging_df(n=500):
    """Create range-bound data that should produce equal highs/lows."""
    rng = np.random.default_rng(55)
    # Range between 100 and 110 with oscillation
    oscillation = 5 * np.sin(np.linspace(0, 20 * np.pi, n))
    prices = 105 + oscillation + rng.normal(0, 0.2, n)
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
    return _liq.LiquidityLevelsIndicator()


class TestLiquidityFeatures:
    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values_in_ranging(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        late = result.iloc[100:]
        for col in ["liq_eqh_active", "liq_eql_active"]:
            non_null = late[col].dropna()
            assert non_null.sum() > 0, f"{col} never activated in ranging data"

    def test_binary_features(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ["liq_eqh_active", "liq_eql_active",
                     "liq_sweep_up", "liq_sweep_down"]:
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_distances_positive_when_active(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        for col in ["liq_eqh_dist", "liq_eql_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals > 0).all(), f"{col} should be positive"

    def test_counts_at_least_min_touches(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        for col in ["liq_eqh_count", "liq_eql_count"]:
            vals = result[col].dropna()
            active_vals = vals[vals > 0]
            if len(active_vals) > 0:
                assert (active_vals >= 2).all(), f"{col} should be >= min_touches"


class TestLiquidityShiftAndInf:
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


class TestLiquidityParameters:
    def test_get_default_params(self):
        params = _liq.LiquidityLevelsIndicator.get_default_params()
        assert params["swing_lookback"] == 10
        assert params["min_touches"] == 2

    def test_get_param_schema(self):
        schema = _liq.LiquidityLevelsIndicator.get_param_schema()
        assert "swing_lookback" in schema
        assert "tolerance_atr_mult" in schema

    def test_custom_params(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(), swing_lookback=5, min_touches=3)
        assert "liq_eqh_dist" in result.columns


class TestLiquidityHelpers:
    def test_find_equal_levels_basic(self):
        prices = np.array([100.0, 100.1, 100.05, 105.0, 105.1])
        indices = np.array([0, 10, 20, 30, 40])
        clusters = _liq._find_equal_levels(prices, indices, tolerance=0.2, min_touches=2)
        # Should find cluster at ~100 (3 touches) and ~105 (2 touches)
        assert len(clusters) >= 2

    def test_find_equal_levels_no_clusters(self):
        prices = np.array([100.0, 110.0, 120.0])
        indices = np.array([0, 10, 20])
        clusters = _liq._find_equal_levels(prices, indices, tolerance=0.5, min_touches=2)
        assert len(clusters) == 0

    def test_find_equal_levels_empty(self):
        clusters = _liq._find_equal_levels(
            np.array([]), np.array([]), tolerance=0.5, min_touches=2)
        assert len(clusters) == 0


class TestLiquidityDiscovery:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("liquidity_levels")
        assert cls is not None
```

### Step 4: Run tests

```bash
python -m pytest src/fwbg/plugins/fwbg-core/indicators/liquidity_levels/tests.py -v
```

### Step 5: Commit

```bash
git add src/fwbg/plugins/fwbg-core/indicators/liquidity_levels/
git commit -m "feat: add liquidity_levels indicator plugin"
```

---

## Task 5: Strategy Config `orb_pdhl_scalping.json`

Combines Opening Range Breakout + Previous Day Levels + Displacement + supply/demand flip zones for intraday scalping. Uses M15 timeframe, focused on the intraday strategy from the video transcript.

**Files:**
- Create: `strategies/orb_pdhl_scalping.json`

### Step 1: Write the strategy config

```json
{
  "name": "ORB + PDH/PDL Scalping",
  "description": "Intraday scalping strategy combining Opening Range Breakout with Previous Day High/Low levels, displacement quality, S/D flip zones, and liquidity detection on M15 data.",
  "tags": ["orb", "pdhl", "scalping", "intraday", "smart_money"],
  "timeframe": "MINUTE_15",
  "hypothesis": "ORB breakouts with displacement towards PDH/PDL produce high-probability intraday setups. Supply/demand flip zones and liquidity levels add confluence. ML learns which combinations of ORB break direction, displacement quality, and proximity to daily levels are profitable.",
  "expected_outcome": "Strong edge on indices and FX majors during NY/London sessions. PDL features and displacement features selected as important by feature selection. Higher win rate when displacement is present.",
  "pipeline": {
    "preprocessing": [
      {
        "name": "fractional_diff",
        "params": {
          "auto_d": false,
          "default_d": 0.4,
          "columns": ["O", "H", "L", "C"]
        }
      }
    ],
    "indicators": [
      {
        "name": "opening_range",
        "params": {
          "range_bars": 1,
          "atr_period": 14,
          "sessions": [0, 8, 13, 14],
          "stat_window": 20,
          "enable_rolling": true,
          "enable_session": true,
          "enable_stats": true
        }
      },
      {
        "name": "previous_day_levels",
        "params": {
          "atr_period": 14,
          "ma_period": 20
        }
      },
      {
        "name": "displacement",
        "params": {
          "atr_period": 14,
          "range_avg_period": 20
        }
      },
      {
        "name": "supply_demand_flip",
        "params": {
          "swing_lookback": 10,
          "zone_atr_width": 0.3,
          "atr_period": 14,
          "max_active_zones": 20,
          "zone_expiry": 200
        }
      },
      {
        "name": "liquidity_levels",
        "params": {
          "swing_lookback": 10,
          "tolerance_atr_mult": 0.2,
          "atr_period": 14,
          "min_touches": 2,
          "lookback_window": 200
        }
      },
      {
        "name": "trend",
        "params": {
          "adx_periods": [7, 14],
          "ema_periods": [8, 21, 50],
          "sma_periods": [20, 50],
          "cci_periods": [14],
          "aroon_period": 25,
          "er_periods": [10, 20],
          "supertrend_period": 14,
          "supertrend_multiplier": 3.0
        }
      },
      {
        "name": "momentum",
        "params": {
          "rsi_periods": [7, 14],
          "stoch_periods": [14],
          "williams_periods": [14],
          "roc_periods": [5, 10]
        }
      },
      {
        "name": "volatility",
        "params": {
          "atr_periods": [7, 14],
          "vol_est_windows": [20]
        }
      },
      {
        "name": "price_action",
        "params": {
          "hh_ll_period": 14,
          "compute_volume": false
        }
      },
      {
        "name": "time_season",
        "params": {
          "include_raw": true,
          "include_encoded": true
        }
      },
      {
        "name": "fair_value_gap",
        "params": {
          "atr_period": 14,
          "lookback": 100
        }
      }
    ],
    "feature_selection": [
      {
        "name": "stability",
        "params": {
          "inner_selector": "boruta",
          "inner_params": {
            "n_iter": 8,
            "n_estimators": 50,
            "max_depth": 5,
            "min_z_score": 0.5
          },
          "n_bootstrap": 7,
          "threshold": 0.6,
          "bootstrap_ratio": 0.8
        }
      },
      {
        "name": "correlation_filter",
        "params": {
          "max_correlation": 0.7,
          "max_features": 50
        }
      }
    ],
    "data_loading": [
      {
        "name": "macro_data",
        "source": "forexsb"
      }
    ]
  },
  "exit_strategy": "atr_based",
  "exit_params": {
    "atr_period": 14,
    "min_tp_pips": 8,
    "min_sl_pips": 12,
    "adaptive_timeout": false
  },
  "model": {
    "type": "xgboost",
    "architecture": "long_short_separate",
    "trade_directions": ["long", "short"],
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "random_state": 42
    }
  },
  "grids": {
    "FOREX": {
      "tp": [1.5, 2.0, 2.5, 3.0],
      "sl": [3.0, 4.0, 5.0],
      "ct": [0.5, 0.55, 0.6, 0.65],
      "timeout_bars": [null, 48, 96],
      "regime_filter_grid": {
        "condition_grids": [
          {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [null, 20, 25],
            "directions": 6,
            "else_directions": 0
          }
        ]
      }
    },
    "INDEX": {
      "tp": [1.5, 2.0, 2.5, 3.0, 3.5],
      "sl": [3.0, 4.0, 5.0, 6.0],
      "ct": [0.5, 0.55, 0.6, 0.65],
      "timeout_bars": [null, 48, 96],
      "regime_filter_grid": {
        "condition_grids": [
          {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [null, 20, 25],
            "directions": 6,
            "else_directions": 0
          },
          {
            "column": "macro_vix",
            "operator": "<=",
            "values": [null, 25, 30],
            "directions": 6,
            "else_directions": 0
          }
        ]
      }
    },
    "COMMODITY": {
      "tp": [2.0, 2.5, 3.0, 3.5],
      "sl": [3.0, 4.0, 5.0, 6.0],
      "ct": [0.5, 0.55, 0.6, 0.65],
      "timeout_bars": [null, 48, 96],
      "regime_filter_grid": {
        "condition_grids": [
          {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [null, 25],
            "directions": 6,
            "else_directions": 0
          }
        ]
      }
    },
    "CRYPTO": {
      "tp": [2.0, 3.0, 4.0, 5.0],
      "sl": [3.0, 4.0, 5.0, 7.0],
      "ct": [0.5, 0.55, 0.6, 0.65],
      "timeout_bars": [null, 48, 96],
      "regime_filter_grid": {
        "condition_grids": [
          {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [null, 25],
            "directions": 6,
            "else_directions": 0
          }
        ]
      }
    }
  },
  "validation": {
    "method": "walk_forward",
    "folds": 8,
    "oos_size": 8000,
    "min_trades": 50,
    "n_inner_folds": 5,
    "embargo_bars": 400,
    "sample_weights": true,
    "early_pruning": {
      "enabled": true,
      "keep_ratio": 0.5,
      "min_survivors": 10
    },
    "probability_calibration": false,
    "calibration_method": "isotonic"
  },
  "filters": {
    "min_rrr": 0,
    "min_trades": 30,
    "min_annual_return": 0,
    "min_sharpe": 0,
    "max_drawdown": 1.0
  },
  "resources": {
    "ram_per_worker_gb": 4.0,
    "min_free_ram_percent": 0.15,
    "max_cpu_percent": 0.8,
    "max_concurrent_assets": 1,
    "xgboost_n_jobs": 0
  }
}
```

### Step 2: Commit

```bash
git add strategies/orb_pdhl_scalping.json
git commit -m "feat: add ORB + PDH/PDL scalping strategy config"
```

---

## Task 6: Run Full Test Suite

Verify all new indicators pass tests and don't break existing tests.

### Step 1: Run new indicator tests

```bash
python -m pytest src/fwbg/plugins/fwbg-core/indicators/previous_day_levels/tests.py src/fwbg/plugins/fwbg-core/indicators/displacement/tests.py src/fwbg/plugins/fwbg-core/indicators/supply_demand_flip/tests.py src/fwbg/plugins/fwbg-core/indicators/liquidity_levels/tests.py -v
```

### Step 2: Run existing indicator tests (regression)

```bash
python -m pytest tests/ -x -q
```

Expected: All existing tests pass, 5 skipped (pre-existing). Any new failures must be investigated and fixed before marking complete.
