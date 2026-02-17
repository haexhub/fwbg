"""
Support & Resistance Indicator Plugin.

Identifies S/R zones on H1 and D1 timeframes, classifies trend strength
(Rayner Teo style), and produces interaction features for ML trading decisions.

Trading logic:
- Uptrend -> Long an Support (Pullback-Entry)
- Downtrend -> Short an Resistance (Rally-Entry)
- Sideways -> Long an Support + Short an Resistance (Range-Trading)
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, safe_divide, EPSILON, register_indicator


def _detect_swings(highs: np.ndarray, lows: np.ndarray, period: int):
    """Detect swing highs and lows. Lookahead-safe: confirmation at i + period.

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
        start = i - 2 * period
        end = i + 1  # exclusive
        window_h = highs[start:end]
        window_l = lows[start:end]
        mid = period  # index within window

        # Strict: candidate must be higher than all other bars in window
        left_h = window_h[:mid]
        right_h = window_h[mid + 1:]
        if len(left_h) > 0 and len(right_h) > 0:
            if window_h[mid] > np.max(left_h) and window_h[mid] > np.max(right_h):
                swing_highs[i] = highs[i - period]

        left_l = window_l[:mid]
        right_l = window_l[mid + 1:]
        if len(left_l) > 0 and len(right_l) > 0:
            if window_l[mid] < np.min(left_l) and window_l[mid] < np.min(right_l):
                swing_lows[i] = lows[i - period]

    return swing_highs, swing_lows


def _cluster_levels(levels: np.ndarray, atr: float, threshold: float = 1.5):
    """Group nearby price levels into zones.

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
    """Detect S/R zones by finding swings across multiple periods, then clustering.

    Returns:
        List of zone dicts: [{"center", "touches", "type"}, ...]
        type: "support" | "resistance" | "both"
    """
    n = len(highs)
    all_swing_highs = []
    all_swing_lows = []

    for period in swing_periods:
        sh, sl = _detect_swings(highs, lows, period)
        start = max(0, n - lookback)
        for i in range(start, n):
            if not np.isnan(sh[i]):
                all_swing_highs.append(sh[i])
            if not np.isnan(sl[i]):
                all_swing_lows.append(sl[i])

    recent_atr = np.nanmedian(atr[max(0, n - 50):n])
    if recent_atr < EPSILON:
        recent_atr = 1.0

    r_zones = _cluster_levels(
        np.array(all_swing_highs) if all_swing_highs else np.array([np.nan]),
        recent_atr, cluster_threshold,
    )
    s_zones = _cluster_levels(
        np.array(all_swing_lows) if all_swing_lows else np.array([np.nan]),
        recent_atr, cluster_threshold,
    )

    for z in r_zones:
        z["type"] = "resistance"
    for z in s_zones:
        z["type"] = "support"

    # Merge overlapping support/resistance into flip zones
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


def _classify_trend(close: float, ma20: float, ma50: float, ma200: float) -> int:
    """Rayner Teo style trend classification.

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


@register_indicator("support_resistance")
class SupportResistanceIndicator(BaseIndicator):
    """S/R zones + trend context features."""

    name = "support_resistance"
    version = "1.0.0"

    _H1_FEATURES = [
        "sr_dist_nearest_support", "sr_dist_nearest_resistance",
        "sr_support_strength", "sr_resistance_strength",
        "sr_in_support_zone", "sr_in_resistance_zone",
        "sr_nearest_is_flip_zone",
    ]
    _D1_FEATURES = [
        "sr_d1_dist_nearest_support", "sr_d1_dist_nearest_resistance",
        "sr_d1_support_strength", "sr_d1_resistance_strength",
        "sr_d1_in_support_zone", "sr_d1_in_resistance_zone",
        "sr_d1_nearest_is_flip_zone",
    ]
    _TREND_FEATURES = [
        "sr_trend_class", "sr_pullback_depth", "sr_ma_alignment",
        "sr_price_vs_ma20", "sr_price_vs_ma50", "sr_price_vs_ma200",
        "sr_trend_break",
    ]
    _INTERACTION_FEATURES = [
        "sr_at_support_in_uptrend", "sr_at_resistance_in_downtrend",
        "sr_at_support_in_range", "sr_at_resistance_in_range",
        "sr_range_width", "sr_range_position",
        "sr_breakout_up", "sr_breakout_down",
        "sr_at_flipped_support", "sr_at_flipped_resistance",
    ]

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
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(close, 1)),
                np.abs(lows - np.roll(close, 1)),
            ),
        )
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

        # Tier 1: H1 S/R Zones
        self._compute_sr_features(
            features, highs, lows, close, atr, n,
            swing_periods, lookback, cluster_threshold,
            zone_proximity_atr_mult, prefix="sr",
        )

        # Tier 1: D1 S/R Zones
        # Use original H1 price data with scaled periods — larger periods
        # naturally capture daily-level swings without rolling aggregation
        # (rolling max/min creates plateaus that break strict swing detection)
        d1_atr = pd.Series(tr).rolling(atr_period * d1_bars, min_periods=1).mean().values
        d1_swing_periods = [max(1, p * d1_bars) for p in swing_periods]
        d1_lookback = lookback * d1_bars

        self._compute_sr_features(
            features, highs, lows, close, d1_atr, n,
            d1_swing_periods, d1_lookback, cluster_threshold,
            zone_proximity_atr_mult, prefix="sr_d1",
        )

        # Tier 2: Trend Context
        self._compute_trend_features(features, close, atr, n, ma_periods)

        # Trend break: swing structure violated
        mid_period = swing_periods[len(swing_periods) // 2]
        sh, sl = _detect_swings(highs, lows, mid_period)
        trend_break = np.zeros(n)
        last_swing_high = np.nan
        last_swing_low = np.nan
        trend = features["sr_trend_class"]

        for i in range(n):
            if not np.isnan(sh[i]):
                last_swing_high = sh[i]
            if not np.isnan(sl[i]):
                last_swing_low = sl[i]

            if trend[i] > 0 and not np.isnan(last_swing_low):
                if close[i] < last_swing_low:
                    trend_break[i] = -1.0
            elif trend[i] < 0 and not np.isnan(last_swing_high):
                if close[i] > last_swing_high:
                    trend_break[i] = 1.0

        features["sr_trend_break"] = trend_break

        # Tier 3: Interaction
        self._compute_interaction_features(features, n, zone_proximity_atr_mult)

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def _compute_sr_features(
        self, features, highs, lows, close, atr, n,
        swing_periods, lookback, cluster_threshold,
        zone_proximity, prefix,
    ):
        """Compute S/R zone features for a given timeframe."""
        dist_support = np.full(n, np.nan)
        dist_resistance = np.full(n, np.nan)
        strength_support = np.zeros(n)
        strength_resistance = np.zeros(n)
        in_support = np.zeros(n)
        in_resistance = np.zeros(n)
        is_flip = np.zeros(n)

        # Temporal flip: only computed for H1 (prefix="sr")
        compute_flip = prefix == "sr"
        if compute_flip:
            at_flipped_support = np.zeros(n)
            at_flipped_resistance = np.zeros(n)

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

            r_zones = _cluster_levels(
                np.array(swing_h_levels) if swing_h_levels else np.array([np.nan]),
                current_atr, cluster_threshold,
            )
            s_zones = _cluster_levels(
                np.array(swing_l_levels) if swing_l_levels else np.array([np.nan]),
                current_atr, cluster_threshold,
            )

            # Find nearest support (below price) and resistance (above price)
            nearest_s_dist = np.inf
            nearest_s_strength = 0
            nearest_s_is_flip = False
            for z in s_zones:
                d = (current_close - z["center"]) / current_atr
                if 0 < d < nearest_s_dist:
                    nearest_s_dist = d
                    nearest_s_strength = z["touches"]
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

            # Temporal flip: resistance broken above → flipped support
            if compute_flip:
                for z in r_zones:
                    d = (current_close - z["center"]) / current_atr
                    if 0 < d < zone_proximity:
                        at_flipped_support[i] = 1.0
                        break
                for z in s_zones:
                    d = (z["center"] - current_close) / current_atr
                    if 0 < d < zone_proximity:
                        at_flipped_resistance[i] = 1.0
                        break

        features[f"{prefix}_dist_nearest_support"] = dist_support
        features[f"{prefix}_dist_nearest_resistance"] = dist_resistance
        features[f"{prefix}_support_strength"] = strength_support
        features[f"{prefix}_resistance_strength"] = strength_resistance
        features[f"{prefix}_in_support_zone"] = in_support
        features[f"{prefix}_in_resistance_zone"] = in_resistance
        features[f"{prefix}_nearest_is_flip_zone"] = is_flip

        if compute_flip:
            features["sr_at_flipped_support"] = at_flipped_support
            features["sr_at_flipped_resistance"] = at_flipped_resistance

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

            if close[i] > recent_swing_high:
                recent_swing_high = close[i]
            if close[i] < recent_swing_low:
                recent_swing_low = close[i]

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

            bull_score = float(ma20[i] > ma50[i]) + float(ma50[i] > ma200[i])
            bear_score = float(ma20[i] < ma50[i]) + float(ma50[i] < ma200[i])
            ma_alignment[i] = (bull_score - bear_score) / 2.0

        features["sr_trend_class"] = trend_class
        features["sr_pullback_depth"] = pullback_depth
        features["sr_ma_alignment"] = ma_alignment
        ma_arrays = [ma20, ma50, ma200]
        for j, period in enumerate(ma_periods):
            features[f"sr_price_vs_ma{period}"] = np.where(
                atr > EPSILON, (close - ma_arrays[j]) / atr, 0.0,
            )

    def _compute_interaction_features(self, features, n, zone_proximity):
        """Tier 3: Combine S/R proximity with trend context."""
        trend = features["sr_trend_class"]
        dist_sup = features["sr_dist_nearest_support"]
        dist_res = features["sr_dist_nearest_resistance"]

        near_sup = np.where(
            np.isnan(dist_sup), 0.0, np.where(dist_sup < zone_proximity * 2, 1.0, 0.0),
        )
        near_res = np.where(
            np.isnan(dist_res), 0.0, np.where(dist_res < zone_proximity * 2, 1.0, 0.0),
        )

        features["sr_at_support_in_uptrend"] = np.where(
            (near_sup == 1) & (trend > 0), 1.0, 0.0,
        )
        features["sr_at_resistance_in_downtrend"] = np.where(
            (near_res == 1) & (trend < 0), 1.0, 0.0,
        )
        features["sr_at_support_in_range"] = np.where(
            (near_sup == 1) & (trend == 0), 1.0, 0.0,
        )
        features["sr_at_resistance_in_range"] = np.where(
            (near_res == 1) & (trend == 0), 1.0, 0.0,
        )

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

        prev_in_res = np.roll(near_res, 1)
        prev_in_sup = np.roll(near_sup, 1)
        prev_in_res[0] = 0
        prev_in_sup[0] = 0
        features["sr_breakout_up"] = np.where(
            (prev_in_res == 1) & (near_res == 0) & (np.nan_to_num(dist_res) == 0),
            1.0, 0.0,
        )
        features["sr_breakout_down"] = np.where(
            (prev_in_sup == 1) & (near_sup == 0) & (np.nan_to_num(dist_sup) == 0),
            1.0, 0.0,
        )

    def get_feature_columns(self) -> List[str]:
        return (
            self._H1_FEATURES + self._D1_FEATURES
            + self._TREND_FEATURES + self._INTERACTION_FEATURES
        )

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
