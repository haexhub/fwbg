"""
Liquidity Sweep Indicator.

Detects "fake-out" / "stop-hunt" patterns where price wicks beyond a recent
swing high or low but the candle closes back inside the range (weakness):

  Bullish sweep: price wicks BELOW a swing low, closes back ABOVE it.
                 → support confirmed, zone = [wick_low, close]
  Bearish sweep: price wicks ABOVE a swing high, closes back BELOW it.
                 → resistance confirmed, zone = [close, wick_high]

The "zone" (rectangle in the source strategy) is the area between the close
and wick extreme of the sweep candle.  Entry: price re-enters the zone.
Stop: beyond the wick extreme.

Features track active sweep zones, distance to zone, recency, and whether
price is currently inside the zone.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, EPSILON, register_indicator


@register_indicator("liquidity_sweep")
class LiquiditySweepIndicator(BaseIndicator):
    """Liquidity sweep (fake-out / stop-hunt) zone features."""

    name = "liquidity_sweep"
    version = "1.0.0"

    _FEATURES = [
        "lsw_bull_active",    # 1 if ≥1 active bullish sweep zone exists
        "lsw_bear_active",    # 1 if ≥1 active bearish sweep zone exists
        "lsw_bull_dist",      # ATR-normalised distance from close to nearest bull zone midpoint
        "lsw_bear_dist",      # ATR-normalised distance from close to nearest bear zone midpoint
        "lsw_bull_size",      # ATR-normalised size of nearest bull sweep zone
        "lsw_bear_size",      # ATR-normalised size of nearest bear sweep zone
        "lsw_bull_in_zone",   # 1 if close is inside the nearest bull sweep zone
        "lsw_bear_in_zone",   # 1 if close is inside the nearest bear sweep zone
        "lsw_bull_recency",   # recency of most recent bull sweep (1=just happened, 0=old)
        "lsw_bear_recency",   # recency of most recent bear sweep
    ]

    def compute(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 20,
        zone_lookback: int = 50,
        atr_period: int = 14,
        **params,
    ) -> pd.DataFrame:
        n = len(df)
        highs = df["H"].values
        lows = df["L"].values
        closes = df["C"].values

        # ATR for normalisation
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1)),
            ),
        )
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

        # Rolling swing high / low (no lookahead: use values available up to bar i)
        swing_high = pd.Series(highs).rolling(swing_lookback, min_periods=1).max().values
        swing_low = pd.Series(lows).rolling(swing_lookback, min_periods=1).min().values

        # Output arrays
        bull_active = np.zeros(n)
        bear_active = np.zeros(n)
        bull_dist = np.full(n, np.nan)
        bear_dist = np.full(n, np.nan)
        bull_size = np.full(n, np.nan)
        bear_size = np.full(n, np.nan)
        bull_in_zone = np.zeros(n)
        bear_in_zone = np.zeros(n)
        bull_recency = np.zeros(n)
        bear_recency = np.zeros(n)

        # Active sweep zones:
        # {"type": "bull"|"bear", "zone_bottom": float, "zone_top": float, "bar": int}
        active_zones: list = []

        for i in range(1, n):
            current_atr = atr[i] if atr[i] > EPSILON else 1.0
            c = closes[i]

            # 1. Expire / invalidate existing zones
            surviving = []
            for zone in active_zones:
                age = i - zone["bar"]
                if age >= zone_lookback:
                    continue
                # Bull zone: invalidated if price re-sweeps below the wick low
                if zone["type"] == "bull" and lows[i] < zone["zone_bottom"]:
                    continue
                # Bear zone: invalidated if price re-sweeps above the wick high
                if zone["type"] == "bear" and highs[i] > zone["zone_top"]:
                    continue
                surviving.append(zone)
            active_zones = surviving

            # 2. Detect new sweep at bar i using bar i-1's swing level (no lookahead)
            prev_sh = swing_high[i - 1]
            prev_sl = swing_low[i - 1]

            # Bullish sweep: wick below swing low, close back above
            if lows[i] < prev_sl and closes[i] > prev_sl:
                active_zones.append({
                    "type": "bull",
                    "zone_bottom": float(lows[i]),    # wick extreme (stop level)
                    "zone_top": float(closes[i]),      # close (entry trigger level)
                    "bar": i,
                })

            # Bearish sweep: wick above swing high, close back below
            if highs[i] > prev_sh and closes[i] < prev_sh:
                active_zones.append({
                    "type": "bear",
                    "zone_bottom": float(closes[i]),   # close (entry trigger level)
                    "zone_top": float(highs[i]),        # wick extreme (stop level)
                    "bar": i,
                })

            # 3. Compute per-bar features from active zones
            nearest_bull_dist = np.inf
            nearest_bull_sz = 0.0
            nearest_bear_dist = np.inf
            nearest_bear_sz = 0.0
            last_bull_bar = -1
            last_bear_bar = -1

            for zone in active_zones:
                zone_sz = (zone["zone_top"] - zone["zone_bottom"]) / current_atr
                mid = (zone["zone_top"] + zone["zone_bottom"]) / 2.0

                if zone["type"] == "bull":
                    d = (c - mid) / current_atr
                    if d > 0 and d < nearest_bull_dist:
                        nearest_bull_dist = d
                        nearest_bull_sz = zone_sz
                    if zone["zone_bottom"] <= c <= zone["zone_top"]:
                        bull_in_zone[i] = 1.0
                    last_bull_bar = max(last_bull_bar, zone["bar"])

                elif zone["type"] == "bear":
                    d = (mid - c) / current_atr
                    if d > 0 and d < nearest_bear_dist:
                        nearest_bear_dist = d
                        nearest_bear_sz = zone_sz
                    if zone["zone_bottom"] <= c <= zone["zone_top"]:
                        bear_in_zone[i] = 1.0
                    last_bear_bar = max(last_bear_bar, zone["bar"])

            if nearest_bull_dist < np.inf:
                bull_active[i] = 1.0
                bull_dist[i] = nearest_bull_dist
                bull_size[i] = nearest_bull_sz

            if nearest_bear_dist < np.inf:
                bear_active[i] = 1.0
                bear_dist[i] = nearest_bear_dist
                bear_size[i] = nearest_bear_sz

            if last_bull_bar >= 0:
                age = min(i - last_bull_bar, zone_lookback)
                bull_recency[i] = 1.0 - age / zone_lookback

            if last_bear_bar >= 0:
                age = min(i - last_bear_bar, zone_lookback)
                bear_recency[i] = 1.0 - age / zone_lookback

        features = {
            "lsw_bull_active": bull_active,
            "lsw_bear_active": bear_active,
            "lsw_bull_dist": bull_dist,
            "lsw_bear_dist": bear_dist,
            "lsw_bull_size": bull_size,
            "lsw_bear_size": bear_size,
            "lsw_bull_in_zone": bull_in_zone,
            "lsw_bear_in_zone": bear_in_zone,
            "lsw_bull_recency": bull_recency,
            "lsw_bear_recency": bear_recency,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return [
            "lsw_bull_active", "lsw_bear_active",
            "lsw_bull_in_zone", "lsw_bear_in_zone",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_lookback": 20,
            "zone_lookback": 50,
            "atr_period": 14,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int",
                "default": 20,
                "description": (
                    "Bars to look back for identifying recent swing highs/lows. "
                    "The rolling max/min over this window defines the level that "
                    "gets swept. On M15: 20 bars ≈ 5 hours."
                ),
                "min": 3,
                "max": 200,
                "step": 1,
            },
            "zone_lookback": {
                "type": "int",
                "default": 50,
                "description": (
                    "Maximum bars a sweep zone stays active. Zone is also removed "
                    "if price sweeps through it again (re-test fails). "
                    "On M15: 50 bars ≈ 12.5 hours."
                ),
                "min": 5,
                "max": 500,
                "step": 5,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period for normalising distances and zone sizes.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
        }


__all__ = ["LiquiditySweepIndicator"]
