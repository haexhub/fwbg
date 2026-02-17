# Support & Resistance Indicator — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `support_resistance` indicator plugin that identifies S/R zones on H1 and D1 timeframes, classifies trend strength (Rayner-style), and produces ~28 features for ML-based trading decisions.

**Architecture:** Premium indicator plugin (`BaseIndicator`) with three tiers: (1) Swing-based S/R zone detection with clustering, (2) Trend classification via MA alignment, (3) Interaction features combining S/R proximity with trend context. Multi-timeframe via rolling-window aggregation (H1 + D1).

**Tech Stack:** numpy, pandas, fwbg_sdk (BaseIndicator, shift_features, safe_divide, register_indicator)

**Design Doc:** `docs/plans/2026-02-17-support-resistance-indicator-design.md`

---

## Task 1: Scaffold — manifest.json + empty plugin class

**Files:**
- Create: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/__init__.py`
- Create: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/manifest.json`

**Step 1: Create manifest.json**

```json
{
  "name": "support_resistance",
  "version": "1.0.0",
  "description": "Support/Resistance zones with trend context (Rayner-style)",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

`benefits_from_stationary = false` because S/R detection needs raw OHLC prices, not fractionally-differenced data.

**Step 2: Create empty plugin class**

```python
"""
Support & Resistance Indicator Plugin.

Identifies S/R zones on H1 and D1 timeframes, classifies trend strength
(Rayner Teo style), and produces interaction features for ML trading decisions.

Trading logic:
- Uptrend → Long an Support (Pullback-Entry)
- Downtrend → Short an Resistance (Rally-Entry)
- Sideways → Long an Support + Short an Resistance (Range-Trading)
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, safe_divide, EPSILON, register_indicator


@register_indicator("support_resistance")
class SupportResistanceIndicator(BaseIndicator):
    """S/R zones + trend context features."""

    name = "support_resistance"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        features = {}
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return []

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_periods": [5, 10, 20],
            "lookback": 200,
            "cluster_threshold": 1.5,
            "atr_period": 14,
            "ma_periods": [20, 50, 200],
            "zone_proximity_atr_mult": 0.5,
            "d1_bars": 24,
        }


__all__ = ["SupportResistanceIndicator"]
```

**Step 3: Verify plugin discovery**

Run: `python -c "from fwbg.core import discover_plugins; discover_plugins(); from fwbg.core import get_indicator; print(get_indicator('support_resistance'))"`

Expected: `<class '...SupportResistanceIndicator'>`

**Step 4: Commit**

```
feat: scaffold support_resistance indicator plugin
```

---

## Task 2: Swing Detection — tests + implementation

**Files:**
- Create: `tests/test_support_resistance.py`
- Modify: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/__init__.py`

**Step 1: Write tests for `_detect_swings`**

```python
"""Tests for the support_resistance indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_sr = import_plugin_module("fwbg-premium", "indicators", "support_resistance")
if _sr is None:
    pytest.skip("support_resistance plugin not available", allow_module_level=True)


class TestSwingDetection:
    """Tests for _detect_swings helper function."""

    def test_detects_obvious_swing_high(self):
        """A clear peak must be detected as swing high."""
        #                          v peak at index 5
        highs = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float64)
        lows = highs - 0.5
        period = 5

        swing_highs, _ = _sr._detect_swings(highs, lows, period)

        # Peak is at index 5 (value 6), confirmed at index 5 + period = 10
        assert swing_highs[10] == 6.0

    def test_detects_obvious_swing_low(self):
        """A clear trough must be detected as swing low."""
        #                           v trough at index 5
        lows = np.array([6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6], dtype=np.float64)
        highs = lows + 0.5
        period = 5

        _, swing_lows = _sr._detect_swings(highs, lows, period)

        assert swing_lows[10] == 1.0

    def test_no_swing_in_flat_data(self):
        """Flat data should not produce swings."""
        n = 50
        highs = np.full(n, 100.0)
        lows = np.full(n, 99.0)

        swing_highs, swing_lows = _sr._detect_swings(highs, lows, period=5)

        # All values should be NaN (no clear swing)
        assert np.all(np.isnan(swing_highs)) or swing_highs[~np.isnan(swing_highs)].size <= 2

    def test_no_lookahead_bias(self):
        """Swing at index i must NOT be confirmed before index i + period."""
        n = 30
        highs = np.array([float(i) for i in range(15)] + [float(14 - i) for i in range(15)])
        lows = highs - 0.5
        period = 5

        swing_highs, _ = _sr._detect_swings(highs, lows, period)

        # Peak is at index 14 (value 14.0)
        # Must NOT appear before index 14 + period = 19
        for i in range(19):
            assert np.isnan(swing_highs[i]) or swing_highs[i] != 14.0, \
                f"Swing high leaked at index {i} (before confirmation at 19)"

    def test_multiple_swings(self):
        """Multiple peaks should all be detected."""
        # Two peaks: index 5 and index 15
        vals = list(range(6)) + list(range(4, -1, -1)) + list(range(1, 7)) + list(range(5, -1, -1))
        highs = np.array(vals, dtype=np.float64)
        lows = highs - 0.5
        period = 5

        swing_highs, _ = _sr._detect_swings(highs, lows, period)

        detected = swing_highs[~np.isnan(swing_highs)]
        assert len(detected) >= 2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_support_resistance.py::TestSwingDetection -v`
Expected: FAIL — `_detect_swings` not found

**Step 3: Implement `_detect_swings`**

Add to `__init__.py` before the class:

```python
def _detect_swings(highs: np.ndarray, lows: np.ndarray, period: int):
    """
    Detect swing highs and lows. Lookahead-safe: confirmation at i + period.

    A swing high at index j is confirmed at index j + period, meaning
    j's high was the max across [j - period, j + period].
    We check this at bar i = j + period by looking at window [i - 2*period, i].

    Returns:
        (swing_highs, swing_lows): Arrays with price level at confirmation bar, NaN elsewhere.
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(period * 2, n):
        # Candidate is at i - period (middle of window)
        start = i - 2 * period
        end = i + 1  # exclusive
        window_h = highs[start:end]
        window_l = lows[start:end]
        mid = period  # index within window

        if window_h[mid] == np.max(window_h):
            swing_highs[i] = highs[i - period]

        if window_l[mid] == np.min(window_l):
            swing_lows[i] = lows[i - period]

    return swing_highs, swing_lows
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_support_resistance.py::TestSwingDetection -v`
Expected: All PASS

**Step 5: Commit**

```
feat: add swing detection for support_resistance plugin
```

---

## Task 3: Zone Clustering — tests + implementation

**Files:**
- Modify: `tests/test_support_resistance.py`
- Modify: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/__init__.py`

**Step 1: Write tests for `_cluster_levels` and `_find_zones`**

Append to test file:

```python
class TestZoneClustering:
    """Tests for _cluster_levels and _find_zones."""

    def test_identical_levels_cluster(self):
        """Levels at the same price must cluster into one zone."""
        levels = np.array([100.0, 100.1, 100.05])
        atr = 1.0
        zones = _sr._cluster_levels(levels, atr, threshold=1.5)
        assert len(zones) == 1
        assert zones[0]["touches"] == 3

    def test_distant_levels_separate(self):
        """Levels far apart must be separate zones."""
        levels = np.array([100.0, 110.0, 120.0])
        atr = 1.0
        zones = _sr._cluster_levels(levels, atr, threshold=1.5)
        assert len(zones) == 3
        assert all(z["touches"] == 1 for z in zones)

    def test_mixed_clustering(self):
        """Close levels cluster, distant ones separate."""
        levels = np.array([100.0, 100.5, 100.3, 110.0, 110.2])
        atr = 1.0
        zones = _sr._cluster_levels(levels, atr, threshold=1.5)
        assert len(zones) == 2
        assert zones[0]["touches"] == 3  # 100.0, 100.3, 100.5
        assert zones[1]["touches"] == 2  # 110.0, 110.2

    def test_empty_levels(self):
        """All-NaN levels must return empty."""
        levels = np.array([np.nan, np.nan, np.nan])
        zones = _sr._cluster_levels(levels, atr=1.0, threshold=1.5)
        assert len(zones) == 0

    def test_zone_has_center(self):
        """Zone center must be the mean of clustered levels."""
        levels = np.array([100.0, 101.0])
        zones = _sr._cluster_levels(levels, atr=1.0, threshold=1.5)
        assert len(zones) == 1
        assert zones[0]["center"] == pytest.approx(100.5)


class TestFindZones:
    """Tests for _find_zones (swing detection + clustering combined)."""

    def _make_v_shape(self, n=100, peak_idx=50, peak_val=110.0, base_val=100.0):
        """Create V-shape price data with a clear peak."""
        prices = np.full(n, base_val)
        # Ramp up to peak
        for i in range(peak_idx):
            prices[i] = base_val + (peak_val - base_val) * i / peak_idx
        # Ramp down from peak
        for i in range(peak_idx, n):
            prices[i] = peak_val - (peak_val - base_val) * (i - peak_idx) / (n - peak_idx)
        return prices

    def test_finds_resistance_from_peak(self):
        """A peak must produce a resistance zone."""
        prices = self._make_v_shape()
        highs = prices + 0.5
        lows = prices - 0.5
        atr = np.full(len(prices), 1.0)

        zones = _sr._find_zones(highs, lows, atr, swing_periods=[5], lookback=200, cluster_threshold=1.5)

        resistance_zones = [z for z in zones if z["type"] in ("resistance", "both")]
        assert len(resistance_zones) >= 1

    def test_finds_support_from_trough(self):
        """A trough must produce a support zone."""
        # Inverted V-shape
        n = 100
        prices = np.full(n, 110.0)
        for i in range(50):
            prices[i] = 110.0 - 10.0 * i / 50
        for i in range(50, n):
            prices[i] = 100.0 + 10.0 * (i - 50) / 50
        highs = prices + 0.5
        lows = prices - 0.5
        atr = np.full(n, 1.0)

        zones = _sr._find_zones(highs, lows, atr, swing_periods=[5], lookback=200, cluster_threshold=1.5)

        support_zones = [z for z in zones if z["type"] in ("support", "both")]
        assert len(support_zones) >= 1
```

**Step 2: Run tests, verify fail**

Run: `python -m pytest tests/test_support_resistance.py::TestZoneClustering -v`

**Step 3: Implement `_cluster_levels` and `_find_zones`**

Add to `__init__.py`:

```python
def _cluster_levels(levels: np.ndarray, atr: float, threshold: float = 1.5):
    """
    Group nearby price levels into zones.

    Args:
        levels: Array of price levels (may contain NaN).
        atr: Current ATR value for distance threshold.
        threshold: ATR multiplier — levels within threshold * atr are clustered.

    Returns:
        List of dicts: [{"center": float, "touches": int}, ...]
    """
    valid = levels[~np.isnan(levels)]
    if len(valid) == 0:
        return []

    sorted_levels = np.sort(valid)
    zones = []
    current = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        if level - np.mean(current) < threshold * atr:
            current.append(level)
        else:
            zones.append({"center": float(np.mean(current)), "touches": len(current)})
            current = [level]

    zones.append({"center": float(np.mean(current)), "touches": len(current)})
    return zones


def _find_zones(
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    swing_periods: list,
    lookback: int,
    cluster_threshold: float,
):
    """
    Detect S/R zones by finding swings across multiple periods, then clustering.

    Returns:
        List of zone dicts: [{"center", "touches", "type"}, ...]
        type: "support" | "resistance" | "both"
    """
    n = len(highs)
    all_swing_highs = []
    all_swing_lows = []

    for period in swing_periods:
        sh, sl = _detect_swings(highs, lows, period)
        # Only use swings within lookback window from the end
        start = max(0, n - lookback)
        for i in range(start, n):
            if not np.isnan(sh[i]):
                all_swing_highs.append(sh[i])
            if not np.isnan(sl[i]):
                all_swing_lows.append(sl[i])

    # Current ATR (use median of recent values for stability)
    recent_atr = np.nanmedian(atr[max(0, n - 50):n])
    if recent_atr < EPSILON:
        recent_atr = 1.0

    # Cluster highs → resistance zones, lows → support zones
    r_zones = _cluster_levels(np.array(all_swing_highs) if all_swing_highs else np.array([np.nan]),
                              recent_atr, cluster_threshold)
    s_zones = _cluster_levels(np.array(all_swing_lows) if all_swing_lows else np.array([np.nan]),
                              recent_atr, cluster_threshold)

    # Tag types and detect flip zones (S zone close to R zone)
    for z in r_zones:
        z["type"] = "resistance"
    for z in s_zones:
        z["type"] = "support"

    # Merge and detect flip zones
    all_zones = r_zones + s_zones
    merged = []
    used = set()
    for i, z1 in enumerate(all_zones):
        if i in used:
            continue
        for j, z2 in enumerate(all_zones):
            if j <= i or j in used:
                continue
            if abs(z1["center"] - z2["center"]) < cluster_threshold * recent_atr:
                merged.append({
                    "center": (z1["center"] + z2["center"]) / 2,
                    "touches": z1["touches"] + z2["touches"],
                    "type": "both",
                })
                used.add(i)
                used.add(j)
                break
        if i not in used:
            merged.append(z1)

    return merged
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_support_resistance.py -v -k "Clustering or FindZones"`
Expected: All PASS

**Step 5: Commit**

```
feat: add zone clustering for support_resistance plugin
```

---

## Task 4: Trend Classification — tests + implementation

**Files:**
- Modify: `tests/test_support_resistance.py`
- Modify: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/__init__.py`

**Step 1: Write tests**

Append to test file:

```python
class TestTrendClassification:
    """Tests for _classify_trend (Rayner-style -3..+3)."""

    def test_strong_uptrend(self):
        """Price above all MAs, MAs bullish aligned → +3."""
        assert _sr._classify_trend(close=110, ma20=108, ma50=105, ma200=100) == 3

    def test_healthy_uptrend(self):
        """Price between MA20 and MA50, MAs bullish aligned → +2."""
        assert _sr._classify_trend(close=106, ma20=108, ma50=105, ma200=100) == 2

    def test_weak_uptrend(self):
        """Price below MA50 but MAs still bullish aligned → +1."""
        assert _sr._classify_trend(close=103, ma20=108, ma50=105, ma200=100) == 1

    def test_strong_downtrend(self):
        """Price below all MAs, MAs bearish aligned → -3."""
        assert _sr._classify_trend(close=90, ma20=92, ma50=95, ma200=100) == -3

    def test_healthy_downtrend(self):
        """Price between MA20 and MA50, MAs bearish aligned → -2."""
        assert _sr._classify_trend(close=94, ma20=92, ma50=95, ma200=100) == -2

    def test_weak_downtrend(self):
        """Price above MA50 but MAs still bearish aligned → -1."""
        assert _sr._classify_trend(close=97, ma20=92, ma50=95, ma200=100) == -1

    def test_sideways(self):
        """MAs not aligned → 0 (sideways/range)."""
        # MA20 > MA200 > MA50 → not bull or bear aligned
        assert _sr._classify_trend(close=100, ma20=102, ma50=98, ma200=100) == 0
```

**Step 2: Implement `_classify_trend`**

Add to `__init__.py`:

```python
def _classify_trend(close: float, ma20: float, ma50: float, ma200: float) -> int:
    """
    Rayner Teo style trend classification.

    Returns:
        -3 (strong down) to +3 (strong up), 0 = sideways.
    """
    if ma20 > ma50 > ma200:  # Bullish MA alignment
        if close > ma20:
            return 3
        if close > ma50:
            return 2
        return 1
    if ma20 < ma50 < ma200:  # Bearish MA alignment
        if close < ma20:
            return -3
        if close < ma50:
            return -2
        return -1
    return 0  # Sideways — MAs not aligned
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_support_resistance.py::TestTrendClassification -v`
Expected: All PASS

**Step 4: Commit**

```
feat: add Rayner-style trend classification
```

---

## Task 5: Full compute() — Tier 1 (H1 S/R zone features)

**Files:**
- Modify: `tests/test_support_resistance.py`
- Modify: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/__init__.py`

**Step 1: Write integration test for H1 zone features**

Append to test file:

```python
def _make_trending_df(n=500):
    """Create OHLC DataFrame with clear trends and S/R bounces."""
    np.random.seed(42)
    # Uptrend with pullbacks
    trend = np.linspace(100, 120, n) + np.random.randn(n) * 0.5
    # Add oscillation for S/R formation
    oscillation = 3 * np.sin(np.linspace(0, 8 * np.pi, n))
    prices = trend + oscillation

    df = pd.DataFrame({
        "O": prices + np.random.randn(n) * 0.1,
        "H": prices + np.abs(np.random.randn(n) * 0.5),
        "L": prices - np.abs(np.random.randn(n) * 0.5),
        "C": prices,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


class TestComputeTier1:
    """Tests for H1 S/R zone features."""

    def test_h1_features_present(self):
        """All H1 S/R features must be present in output."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        expected_cols = [
            "sr_dist_nearest_support", "sr_dist_nearest_resistance",
            "sr_support_strength", "sr_resistance_strength",
            "sr_in_support_zone", "sr_in_resistance_zone",
            "sr_nearest_is_flip_zone",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_distances_are_atr_normalized(self):
        """S/R distances must be in ATR units (typically 0-20 range)."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        # Skip warmup NaNs
        dist = result["sr_dist_nearest_support"].dropna()
        assert len(dist) > 0
        # ATR-normalized distances should be reasonable (not raw price)
        assert dist.median() < 50, f"Distances too large ({dist.median():.1f}), probably not ATR-normalized"

    def test_in_zone_is_binary(self):
        """in_support_zone and in_resistance_zone must be 0 or 1."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_in_support_zone", "sr_in_resistance_zone"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} has non-binary values: {vals}"

    def test_strength_is_positive_int(self):
        """Zone strength (touches) must be >= 1."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_support_strength", "sr_resistance_strength"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all(), f"{col} has negative values"

    def test_shift_applied(self):
        """Features must be shifted by 1 bar (lookahead prevention)."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=300)
        result = indicator.compute(df)

        # First row must be NaN (shifted)
        for col in ["sr_dist_nearest_support", "sr_dist_nearest_resistance"]:
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} not shifted — first row is not NaN"
```

**Step 2: Implement H1 zone features in compute()**

Replace the `compute` method body with the full implementation. This is the core logic — extract zone features per bar using a rolling window approach:

```python
def compute(
    self,
    df: pd.DataFrame,
    swing_periods: list = None,
    lookback: int = 200,
    cluster_threshold: float = 1.5,
    atr_period: int = 14,
    ma_periods: list = None,
    zone_proximity_atr_mult: float = 0.5,
    d1_bars: int = 24,
    **params,
) -> pd.DataFrame:
    if swing_periods is None:
        swing_periods = [5, 10, 20]
    if ma_periods is None:
        ma_periods = [20, 50, 200]

    features = {}
    n = len(df)
    close = df["C"].values
    highs = df["H"].values
    lows = df["L"].values

    # ATR for normalization
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - np.roll(close, 1)),
                               np.abs(lows - np.roll(close, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    # --- Tier 1: H1 S/R Zones ---
    self._compute_sr_features(features, highs, lows, close, atr, n,
                              swing_periods, lookback, cluster_threshold,
                              zone_proximity_atr_mult, prefix="sr")

    # --- Tier 1: D1 S/R Zones ---
    d1_highs = pd.Series(highs).rolling(d1_bars, min_periods=1).max().values
    d1_lows = pd.Series(lows).rolling(d1_bars, min_periods=1).min().values
    d1_atr = pd.Series(tr).rolling(atr_period * d1_bars, min_periods=1).mean().values
    d1_swing_periods = [p * d1_bars for p in swing_periods]
    d1_lookback = lookback * d1_bars

    self._compute_sr_features(features, d1_highs, d1_lows, close, d1_atr, n,
                              d1_swing_periods, d1_lookback, cluster_threshold,
                              zone_proximity_atr_mult, prefix="sr_d1")

    # --- Tier 2: Trend Context ---
    self._compute_trend_features(features, close, atr, n, ma_periods)

    # --- Tier 3: Interaction ---
    self._compute_interaction_features(features, atr, n, zone_proximity_atr_mult)

    features_df = shift_features(features, df.index)
    return pd.concat([df, features_df], axis=1)
```

Plus the helper `_compute_sr_features`:

```python
def _compute_sr_features(self, features, highs, lows, close, atr, n,
                         swing_periods, lookback, cluster_threshold,
                         zone_proximity, prefix):
    """Compute S/R zone features for a given timeframe."""
    dist_support = np.full(n, np.nan)
    dist_resistance = np.full(n, np.nan)
    strength_support = np.zeros(n)
    strength_resistance = np.zeros(n)
    in_support = np.zeros(n)
    in_resistance = np.zeros(n)
    is_flip = np.zeros(n)

    # Pre-compute all swings
    all_swing_highs = {}
    all_swing_lows = {}
    for period in swing_periods:
        sh, sl = _detect_swings(highs, lows, period)
        for i in range(n):
            if not np.isnan(sh[i]):
                all_swing_highs.setdefault(i, []).append(sh[i])
            if not np.isnan(sl[i]):
                all_swing_lows.setdefault(i, []).append(sl[i])

    for i in range(n):
        current_atr = atr[i] if atr[i] > EPSILON else 1.0
        current_close = close[i]

        # Collect swing levels within lookback
        start = max(0, i - lookback)
        swing_h_levels = []
        swing_l_levels = []
        for j in range(start, i + 1):
            swing_h_levels.extend(all_swing_highs.get(j, []))
            swing_l_levels.extend(all_swing_lows.get(j, []))

        # Cluster into zones
        r_zones = _cluster_levels(np.array(swing_h_levels) if swing_h_levels else np.array([np.nan]),
                                  current_atr, cluster_threshold)
        s_zones = _cluster_levels(np.array(swing_l_levels) if swing_l_levels else np.array([np.nan]),
                                  current_atr, cluster_threshold)

        # Find nearest support (below price) and resistance (above price)
        nearest_s_dist = np.inf
        nearest_s_strength = 0
        nearest_s_is_flip = False
        for z in s_zones:
            d = (current_close - z["center"]) / current_atr
            if 0 < d < nearest_s_dist:
                nearest_s_dist = d
                nearest_s_strength = z["touches"]
                # Check if this support level overlaps with any resistance
                nearest_s_is_flip = any(
                    abs(z["center"] - rz["center"]) < cluster_threshold * current_atr
                    for rz in r_zones
                )

        nearest_r_dist = np.inf
        nearest_r_strength = 0
        nearest_r_is_flip = False
        for z in r_zones:
            d = (z["center"] - current_close) / current_atr
            if 0 < d < nearest_r_dist:
                nearest_r_dist = d
                nearest_r_strength = z["touches"]
                nearest_r_is_flip = any(
                    abs(z["center"] - sz["center"]) < cluster_threshold * current_atr
                    for sz in s_zones
                )

        dist_support[i] = nearest_s_dist if nearest_s_dist < np.inf else np.nan
        dist_resistance[i] = nearest_r_dist if nearest_r_dist < np.inf else np.nan
        strength_support[i] = nearest_s_strength
        strength_resistance[i] = nearest_r_strength
        in_support[i] = 1.0 if nearest_s_dist < zone_proximity else 0.0
        in_resistance[i] = 1.0 if nearest_r_dist < zone_proximity else 0.0
        is_flip[i] = 1.0 if (nearest_s_is_flip or nearest_r_is_flip) else 0.0

    features[f"{prefix}_dist_nearest_support"] = dist_support
    features[f"{prefix}_dist_nearest_resistance"] = dist_resistance
    features[f"{prefix}_support_strength"] = strength_support
    features[f"{prefix}_resistance_strength"] = strength_resistance
    features[f"{prefix}_in_support_zone"] = in_support
    features[f"{prefix}_in_resistance_zone"] = in_resistance
    features[f"{prefix}_nearest_is_flip_zone"] = is_flip
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_support_resistance.py::TestComputeTier1 -v`
Expected: All PASS

**Step 4: Commit**

```
feat: implement H1 + D1 S/R zone features
```

---

## Task 6: Full compute() — Tier 2 + 3 (trend + interaction features)

**Files:**
- Modify: `tests/test_support_resistance.py`
- Modify: `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/__init__.py`

**Step 1: Write tests for Tier 2 + 3**

Append to test file:

```python
class TestComputeTier2:
    """Tests for trend context features."""

    def test_trend_features_present(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_trend_class", "sr_pullback_depth", "sr_ma_alignment",
                     "sr_price_vs_ma20", "sr_price_vs_ma50", "sr_price_vs_ma200"]:
            assert col in result.columns, f"Missing: {col}"

    def test_trend_class_range(self):
        """sr_trend_class must be in [-3, +3]."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)
        vals = result["sr_trend_class"].dropna()
        assert vals.min() >= -3
        assert vals.max() <= 3

    def test_ma_alignment_range(self):
        """sr_ma_alignment must be in [-1, +1]."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)
        vals = result["sr_ma_alignment"].dropna()
        assert vals.min() >= -1.0 - 0.01
        assert vals.max() <= 1.0 + 0.01


class TestComputeTier3:
    """Tests for interaction features."""

    def test_interaction_features_present(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_at_support_in_uptrend", "sr_at_resistance_in_downtrend",
                     "sr_at_support_in_range", "sr_at_resistance_in_range",
                     "sr_range_width", "sr_range_position",
                     "sr_breakout_up", "sr_breakout_down"]:
            assert col in result.columns, f"Missing: {col}"

    def test_range_position_bounds(self):
        """sr_range_position must be in [0, 1] (or NaN)."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)
        vals = result["sr_range_position"].dropna()
        if len(vals) > 0:
            assert vals.min() >= -0.01
            assert vals.max() <= 1.01

    def test_breakouts_are_binary(self):
        for col in ["sr_breakout_up", "sr_breakout_down"]:
            indicator = _sr.SupportResistanceIndicator()
            df = _make_trending_df()
            result = indicator.compute(df)
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"
```

**Step 2: Implement `_compute_trend_features` and `_compute_interaction_features`**

Add to the class:

```python
def _compute_trend_features(self, features, close, atr, n, ma_periods):
    """Tier 2: Rayner-style trend classification and MA features."""
    close_s = pd.Series(close)
    ma20 = close_s.rolling(ma_periods[0], min_periods=1).mean().values
    ma50 = close_s.rolling(ma_periods[1], min_periods=1).mean().values
    ma200 = close_s.rolling(ma_periods[2], min_periods=1).mean().values

    trend_class = np.zeros(n)
    pullback_depth = np.full(n, np.nan)
    ma_alignment = np.zeros(n, dtype=np.float64)

    recent_swing_high = close[0]
    recent_swing_low = close[0]

    for i in range(n):
        trend_class[i] = _classify_trend(close[i], ma20[i], ma50[i], ma200[i])
        a = atr[i] if atr[i] > EPSILON else 1.0

        # Track running swing extremes
        if close[i] > recent_swing_high:
            recent_swing_high = close[i]
        if close[i] < recent_swing_low:
            recent_swing_low = close[i]

        # Pullback depth from recent extreme
        if trend_class[i] > 0:
            pullback_depth[i] = (recent_swing_high - close[i]) / a
        elif trend_class[i] < 0:
            pullback_depth[i] = (close[i] - recent_swing_low) / a
        else:
            pullback_depth[i] = 0.0

        # Reset tracking on trend change
        if i > 0 and np.sign(trend_class[i]) != np.sign(trend_class[i - 1]):
            recent_swing_high = close[i]
            recent_swing_low = close[i]

        # MA alignment: +1 if perfectly bullish, -1 if perfectly bearish
        bull_score = float(ma20[i] > ma50[i]) + float(ma50[i] > ma200[i])
        bear_score = float(ma20[i] < ma50[i]) + float(ma50[i] < ma200[i])
        ma_alignment[i] = (bull_score - bear_score) / 2.0

    features["sr_trend_class"] = trend_class
    features["sr_pullback_depth"] = pullback_depth
    features["sr_ma_alignment"] = ma_alignment
    for j, period in enumerate(ma_periods):
        ma_vals = [ma20, ma50, ma200][j]
        features[f"sr_price_vs_ma{period}"] = np.where(
            atr > EPSILON, (close - ma_vals) / atr, 0.0
        )

def _compute_interaction_features(self, features, atr, n, zone_proximity):
    """Tier 3: Combine S/R proximity with trend context."""
    trend = features["sr_trend_class"]
    in_sup = features["sr_in_support_zone"]
    in_res = features["sr_in_resistance_zone"]
    dist_sup = features["sr_dist_nearest_support"]
    dist_res = features["sr_dist_nearest_resistance"]

    near_sup = np.where(np.isnan(dist_sup), 0.0, np.where(dist_sup < zone_proximity * 2, 1.0, 0.0))
    near_res = np.where(np.isnan(dist_res), 0.0, np.where(dist_res < zone_proximity * 2, 1.0, 0.0))

    features["sr_at_support_in_uptrend"] = np.where((near_sup == 1) & (trend > 0), 1.0, 0.0)
    features["sr_at_resistance_in_downtrend"] = np.where((near_res == 1) & (trend < 0), 1.0, 0.0)
    features["sr_at_support_in_range"] = np.where((near_sup == 1) & (trend == 0), 1.0, 0.0)
    features["sr_at_resistance_in_range"] = np.where((near_res == 1) & (trend == 0), 1.0, 0.0)

    # Range width: distance between nearest support and resistance
    range_width = np.full(n, np.nan)
    range_position = np.full(n, np.nan)
    for i in range(n):
        s = dist_sup[i] if not np.isnan(dist_sup[i]) else 0.0
        r = dist_res[i] if not np.isnan(dist_res[i]) else 0.0
        if s > 0 and r > 0:
            range_width[i] = s + r
            range_position[i] = np.clip(s / (s + r), 0.0, 1.0)
    features["sr_range_width"] = range_width
    features["sr_range_position"] = range_position

    # Breakouts: price moved beyond zone
    prev_in_res = np.roll(near_res, 1)
    prev_in_sup = np.roll(near_sup, 1)
    prev_in_res[0] = 0
    prev_in_sup[0] = 0
    features["sr_breakout_up"] = np.where(
        (prev_in_res == 1) & (near_res == 0) & (np.nan_to_num(dist_res) == 0), 1.0, 0.0
    )
    features["sr_breakout_down"] = np.where(
        (prev_in_sup == 1) & (near_sup == 0) & (np.nan_to_num(dist_sup) == 0), 1.0, 0.0
    )
```

Also update `get_feature_columns()` to return all 28 features.

**Step 3: Run all tests**

Run: `python -m pytest tests/test_support_resistance.py -v`
Expected: All PASS

**Step 4: Commit**

```
feat: implement trend classification and interaction features
```

---

## Task 7: Integration test + stationarity registration

**Files:**
- Modify: `tests/test_support_resistance.py`
- Modify: `tests/test_indicator_stationarity.py`

**Step 1: Write integration test**

Append to `tests/test_support_resistance.py`:

```python
class TestPluginIntegration:
    """Integration tests — plugin discovery, strategy JSON, all features."""

    def test_plugin_discoverable(self):
        """Plugin must be found via discover_plugins."""
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("support_resistance")
        assert cls is not None

    def test_all_feature_columns_present(self):
        """Every column in get_feature_columns() must appear in compute() output."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=600)
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Declared feature {col} missing from output"

    def test_no_extra_undeclared_features(self):
        """compute() must not add columns beyond OHLC + declared features."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=300)
        original_cols = set(df.columns)
        result = indicator.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(indicator.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared features in output: {undeclared}"

    def test_not_all_nan(self):
        """Features must not be all NaN after warmup period."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=600)
        result = indicator.compute(df)

        # Skip first 250 bars (warmup for 200-lookback + shift)
        late = result.iloc[250:]
        for col in indicator.get_feature_columns():
            vals = late[col].dropna()
            assert len(vals) > 0, f"{col} is all NaN after warmup"
```

**Step 2: Update stationarity test**

In `tests/test_indicator_stationarity.py`, add `"support_resistance"` to the `expected_false` set (since `benefits_from_stationary = false`).

**Step 3: Run full test suite**

Run: `python -m pytest tests/test_support_resistance.py tests/test_indicator_stationarity.py -v`
Expected: All PASS

**Step 4: Commit**

```
feat: complete support_resistance plugin with integration tests
```

---

## Task 8: Performance check + final cleanup

**Step 1: Benchmark compute time**

```bash
python -c "
import time
import numpy as np
import pandas as pd
from fwbg.core import discover_plugins, get_indicator
discover_plugins()
cls = get_indicator('support_resistance')
ind = cls()

np.random.seed(42)
n = 5000
prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    'O': prices, 'H': prices + 0.5, 'L': prices - 0.5, 'C': prices
}, index=pd.date_range('2024-01-01', periods=n, freq='h'))

t0 = time.time()
result = ind.compute(df)
print(f'{time.time()-t0:.2f}s for {n} bars, {len(ind.get_feature_columns())} features')
"
```

Expected: < 10s for 5000 bars. If slow (>10s), optimize the inner loop with vectorized operations or consider Numba.

**Step 2: Run full project test suite**

Run: `python -m pytest tests/ -q --deselect tests/pipeline/test_config.py::TestPipelineConfig::test_pipeline_config_parse`

Expected: All pass (+ known pre-existing failures only).

**Step 3: Final commit**

```
feat: support_resistance indicator plugin complete
```
