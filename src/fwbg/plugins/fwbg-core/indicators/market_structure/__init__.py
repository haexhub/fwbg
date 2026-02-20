"""
Market Structure Indicator.

Detects Break-of-Structure (BOS) and Change of Character (CHOCH) events,
the two core market-structure signals in ICT/SMC trading:

  BOS (bullish): close > rolling-max high over the last N bars.
                 Confirms bullish trend continuation.
  BOS (bearish): close < rolling-min low over the last N bars.
                 Confirms bearish trend continuation.

  CHOCH (bullish): first bullish BOS after a bearish trend.
                   Signals a potential bullish reversal.
  CHOCH (bearish): first bearish BOS after a bullish trend.
                   Signals a potential bearish reversal.

Trend state machine: neutral (0) → bullish (+1) on bull BOS,
                     neutral (0) → bearish (-1) on bear BOS,
                     bearish (-1) → bullish (+1) on bull CHOCH, and vice-versa.

Distance features capture how far price is from the most recent BOS level
and from the current rolling swing extremes — useful for ML as proximity signals.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, EPSILON, register_indicator


@register_indicator("market_structure")
class MarketStructureIndicator(BaseIndicator):
    """Market structure (BOS / CHOCH) features for ML trading."""

    name = "market_structure"
    version = "1.0.0"

    _FEATURES = [
        "ms_bos_bull",         # 1 if bullish BOS at this bar
        "ms_bos_bear",         # 1 if bearish BOS at this bar
        "ms_choch_bull",       # 1 if bullish CHOCH (first bull BOS after bearish trend)
        "ms_choch_bear",       # 1 if bearish CHOCH (first bear BOS after bullish trend)
        "ms_trend",            # current trend: +1 bull, -1 bear, 0 neutral
        "ms_bull_bos_dist",    # ATR-normalised distance from close to last bull BOS level
        "ms_bear_bos_dist",    # ATR-normalised distance from close to last bear BOS level
        "ms_swing_high_dist",  # ATR-normalised distance from close to rolling swing high
        "ms_swing_low_dist",   # ATR-normalised distance from close to rolling swing low
        "ms_choch_recency",    # recency of most recent CHOCH (1=just happened, 0=old)
    ]

    def compute(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 20,
        choch_lookback: int = 50,
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

        # Rolling swing high / low (no lookahead: values up to bar i)
        swing_high = pd.Series(highs).rolling(swing_lookback, min_periods=1).max().values
        swing_low = pd.Series(lows).rolling(swing_lookback, min_periods=1).min().values

        # Output arrays
        bos_bull = np.zeros(n)
        bos_bear = np.zeros(n)
        choch_bull = np.zeros(n)
        choch_bear = np.zeros(n)
        trend = np.zeros(n)
        bull_bos_dist = np.full(n, np.nan)
        bear_bos_dist = np.full(n, np.nan)
        swing_high_dist = np.full(n, np.nan)
        swing_low_dist = np.full(n, np.nan)
        choch_recency = np.zeros(n)

        # State
        current_trend = 0         # -1, 0, +1
        last_bull_bos_level = np.nan
        last_bear_bos_level = np.nan
        last_choch_bar = -1

        for i in range(1, n):
            c = closes[i]
            atr_i = atr[i] if atr[i] > EPSILON else 1.0

            prev_sh = swing_high[i - 1]
            prev_sl = swing_low[i - 1]

            # BOS detection (strict break of rolling extreme)
            is_bos_bull = c > prev_sh
            is_bos_bear = c < prev_sl

            if is_bos_bull:
                bos_bull[i] = 1.0
                if current_trend < 0:
                    # First bull BOS after bearish trend → CHOCH
                    choch_bull[i] = 1.0
                    last_choch_bar = i
                current_trend = 1
                last_bull_bos_level = prev_sh

            if is_bos_bear:
                bos_bear[i] = 1.0
                if current_trend > 0:
                    # First bear BOS after bullish trend → CHOCH
                    choch_bear[i] = 1.0
                    last_choch_bar = i
                current_trend = -1
                last_bear_bos_level = prev_sl

            trend[i] = float(current_trend)

            # BOS level distance (positive = price on the "right" side of the break)
            if not np.isnan(last_bull_bos_level):
                bull_bos_dist[i] = (c - last_bull_bos_level) / atr_i
            if not np.isnan(last_bear_bos_level):
                bear_bos_dist[i] = (last_bear_bos_level - c) / atr_i

            # Swing extreme distances (clamped to 0 on the break bar itself)
            swing_high_dist[i] = max(0.0, (prev_sh - c) / atr_i)
            swing_low_dist[i] = max(0.0, (c - prev_sl) / atr_i)

            # CHOCH recency
            if last_choch_bar >= 0:
                age = min(i - last_choch_bar, choch_lookback)
                choch_recency[i] = 1.0 - age / choch_lookback

        features = {
            "ms_bos_bull": bos_bull,
            "ms_bos_bear": bos_bear,
            "ms_choch_bull": choch_bull,
            "ms_choch_bear": choch_bear,
            "ms_trend": trend,
            "ms_bull_bos_dist": bull_bos_dist,
            "ms_bear_bos_dist": bear_bos_dist,
            "ms_swing_high_dist": swing_high_dist,
            "ms_swing_low_dist": swing_low_dist,
            "ms_choch_recency": choch_recency,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["ms_bos_bull", "ms_bos_bear", "ms_choch_bull", "ms_choch_bear"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_lookback": 20,
            "choch_lookback": 50,
            "atr_period": 14,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int",
                "default": 20,
                "description": (
                    "Rolling window for identifying swing highs and lows. "
                    "A BOS fires when close breaks the rolling extreme of this window. "
                    "On M1: 20 bars = 20 minutes; on M15: 20 bars = 5 hours."
                ),
                "min": 3,
                "max": 200,
                "step": 1,
            },
            "choch_lookback": {
                "type": "int",
                "default": 50,
                "description": (
                    "Number of bars over which CHOCH recency decays from 1 to 0. "
                    "Controls how 'fresh' a CHOCH is considered for the recency feature."
                ),
                "min": 5,
                "max": 500,
                "step": 5,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period for normalising distance features.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
        }


__all__ = ["MarketStructureIndicator"]
