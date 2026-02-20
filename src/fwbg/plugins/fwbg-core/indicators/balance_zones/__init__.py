"""
Balance Zones Indicator.

Implements the PBD (Price/Balance/Direction) methodology from Trades to Traders:

  Candle body (O/C range) represents fair value. Overlapping bodies define a
  "balance box" — a zone where buyers and sellers agree on price. Three key
  setups emerge when price interacts with the balance zone:

    1. Fake breakout: price exits the zone then re-enters → counter-trade signal.
    2. Balance re-acceptance: price returns to zone after a breakout → continuation.
    3. Balance abandonment: breakout with no return → trend follow signal.

  The zone is computed as a rolling window over candle body extremes:
    zone_top    = rolling_max(max(O, C), lookback)
    zone_bottom = rolling_min(min(O, C), lookback)

Features are normalised by ATR to be scale-independent across instruments.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, EPSILON, register_indicator


@register_indicator("balance_zones")
class BalanceZonesIndicator(BaseIndicator):
    """Balance zone features derived from candle body overlap (PBD methodology)."""

    name = "balance_zones"
    version = "1.0.0"

    _FEATURES = [
        "bz_in_balance",       # 1 if zone_width / ATR <= balance_atr_threshold
        "bz_in_zone",          # 1 if close is inside the current balance zone
        "bz_zone_width",       # ATR-normalised width of the balance zone
        "bz_zone_top_dist",    # ATR-norm distance from close to zone top (0 if above)
        "bz_zone_bottom_dist", # ATR-norm distance from close to zone bottom (0 if below)
        "bz_breakout_bull",    # 1 if close breaks above zone top this bar
        "bz_breakout_bear",    # 1 if close breaks below zone bottom this bar
        "bz_fake_bear",        # 1 if prev close > old zone top AND now back below (bearish)
        "bz_fake_bull",        # 1 if prev close < old zone bottom AND now back above (bullish)
        "bz_balance_bars",     # normalised consecutive bars with close inside zone (0..1)
    ]

    def compute(
        self,
        df: pd.DataFrame,
        lookback: int = 10,
        balance_atr_threshold: float = 2.0,
        atr_period: int = 14,
        balance_bars_max: int = 20,
        **params,
    ) -> pd.DataFrame:
        n = len(df)
        opens = df["O"].values
        closes = df["C"].values
        highs = df["H"].values
        lows = df["L"].values

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

        # Candle body extremes
        body_top = np.maximum(opens, closes)
        body_bottom = np.minimum(opens, closes)

        # Rolling balance zone (no lookahead: zone[i] uses bars 0..i inclusive)
        zone_top = pd.Series(body_top).rolling(lookback, min_periods=1).max().values
        zone_bottom = pd.Series(body_bottom).rolling(lookback, min_periods=1).min().values

        # Output arrays
        in_balance = np.zeros(n)
        in_zone = np.zeros(n)
        zone_width = np.full(n, np.nan)
        zone_top_dist = np.full(n, np.nan)
        zone_bottom_dist = np.full(n, np.nan)
        breakout_bull = np.zeros(n)
        breakout_bear = np.zeros(n)
        fake_bear = np.zeros(n)
        fake_bull = np.zeros(n)
        balance_bars = np.zeros(n)

        consecutive_in_zone = 0

        for i in range(1, n):
            c = closes[i]
            atr_i = atr[i] if atr[i] > EPSILON else 1.0

            prev_zt = zone_top[i - 1]
            prev_zb = zone_bottom[i - 1]

            # Zone width and balance state
            width = (prev_zt - prev_zb) / atr_i
            zone_width[i] = width
            in_balance[i] = 1.0 if width <= balance_atr_threshold else 0.0

            # In-zone detection
            is_in_zone = prev_zb <= c <= prev_zt
            in_zone[i] = 1.0 if is_in_zone else 0.0

            # Distance features (clamped: 0 when price is on the "wrong" side)
            zone_top_dist[i] = max(0.0, (prev_zt - c) / atr_i)
            zone_bottom_dist[i] = max(0.0, (c - prev_zb) / atr_i)

            # Breakout detection (strict)
            breakout_bull[i] = 1.0 if c > prev_zt else 0.0
            breakout_bear[i] = 1.0 if c < prev_zb else 0.0

            # Fake breakout detection: needs two bars of history
            if i >= 2:
                old_zt = zone_top[i - 2]
                old_zb = zone_bottom[i - 2]
                prev_c = closes[i - 1]

                # Fake bull breakout (bz_fake_bear): price was above old zone_top,
                # now back at or below old zone_top → bearish rejection signal
                if prev_c > old_zt and c <= old_zt:
                    fake_bear[i] = 1.0

                # Fake bear breakout (bz_fake_bull): price was below old zone_bottom,
                # now back at or above old zone_bottom → bullish rejection signal
                if prev_c < old_zb and c >= old_zb:
                    fake_bull[i] = 1.0

            # Consecutive bars in zone (normalised)
            if is_in_zone:
                consecutive_in_zone += 1
            else:
                consecutive_in_zone = 0
            balance_bars[i] = min(consecutive_in_zone, balance_bars_max) / balance_bars_max

        features = {
            "bz_in_balance": in_balance,
            "bz_in_zone": in_zone,
            "bz_zone_width": zone_width,
            "bz_zone_top_dist": zone_top_dist,
            "bz_zone_bottom_dist": zone_bottom_dist,
            "bz_breakout_bull": breakout_bull,
            "bz_breakout_bear": breakout_bear,
            "bz_fake_bear": fake_bear,
            "bz_fake_bull": fake_bull,
            "bz_balance_bars": balance_bars,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return [
            "bz_breakout_bull", "bz_breakout_bear",
            "bz_fake_bear", "bz_fake_bull",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "lookback": 10,
            "balance_atr_threshold": 2.0,
            "atr_period": 14,
            "balance_bars_max": 20,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lookback": {
                "type": "int",
                "default": 10,
                "description": (
                    "Rolling window (bars) for building the body balance zone. "
                    "zone_top = rolling_max(max(O,C), lookback); "
                    "zone_bottom = rolling_min(min(O,C), lookback). "
                    "On M15: 10 bars = 2.5 hours; on M1: 10 bars = 10 minutes."
                ),
                "min": 3,
                "max": 200,
                "step": 1,
            },
            "balance_atr_threshold": {
                "type": "float",
                "default": 2.0,
                "description": (
                    "Maximum zone_width / ATR ratio considered 'in balance'. "
                    "A value of 2.0 means the balance zone spans ≤ 2× ATR. "
                    "Increase for wider balance detection."
                ),
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period for normalising width and distance features.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "balance_bars_max": {
                "type": "int",
                "default": 20,
                "description": (
                    "Maximum consecutive in-zone bars for normalising bz_balance_bars to 1.0. "
                    "bz_balance_bars = min(consecutive_in_zone, balance_bars_max) / balance_bars_max."
                ),
                "min": 5,
                "max": 200,
                "step": 5,
            },
        }


__all__ = ["BalanceZonesIndicator"]
