"""
Fair Value Gap (FVG) Indicator Plugin.

Detects 3-candle imbalance zones (Smart Money Concepts):
- Bullish FVG: H[i-2] < L[i] — gap up, acts as support
- Bearish FVG: L[i-2] > H[i] — gap down, acts as resistance

Price tends to return to fill these gaps. Features track active (unfilled)
FVGs and the current price's relationship to them.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, EPSILON, register_indicator


def _detect_fvgs(highs: np.ndarray, lows: np.ndarray):
    """Detect all Fair Value Gaps in the data.

    Returns:
        List of dicts: [{"type": "bullish"|"bearish", "top": float,
                         "bottom": float, "bar": int}, ...]
    """
    fvgs = []
    for i in range(2, len(highs)):
        # Bullish FVG: candle 1's high < candle 3's low
        if highs[i - 2] < lows[i]:
            fvgs.append({
                "type": "bullish",
                "bottom": float(highs[i - 2]),
                "top": float(lows[i]),
                "bar": i,
            })
        # Bearish FVG: candle 1's low > candle 3's high
        if lows[i - 2] > highs[i]:
            fvgs.append({
                "type": "bearish",
                "bottom": float(highs[i]),
                "top": float(lows[i - 2]),
                "bar": i,
            })
    return fvgs


@register_indicator("fair_value_gap")
class FairValueGapIndicator(BaseIndicator):
    """Fair Value Gap features for ML trading."""

    name = "fair_value_gap"
    version = "1.0.0"

    _FEATURES = [
        "fvg_bull_active",
        "fvg_bear_active",
        "fvg_bull_dist",
        "fvg_bear_dist",
        "fvg_bull_size",
        "fvg_bear_size",
        "fvg_in_gap",
        "fvg_count",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        lookback: int = 100,
        **params,
    ) -> pd.DataFrame:
        n = len(df)
        highs = df["H"].values
        lows = df["L"].values
        close = df["C"].values

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

        # Detect all FVGs
        all_fvgs = _detect_fvgs(highs, lows)

        # Per-bar: track active FVGs and compute features
        bull_active = np.zeros(n)
        bear_active = np.zeros(n)
        bull_dist = np.full(n, np.nan)
        bear_dist = np.full(n, np.nan)
        bull_size = np.full(n, np.nan)
        bear_size = np.full(n, np.nan)
        in_gap = np.zeros(n)
        count = np.zeros(n)

        # Index FVGs by creation bar for efficient lookup
        fvg_by_bar = {}
        for fvg in all_fvgs:
            fvg_by_bar.setdefault(fvg["bar"], []).append(fvg)

        # Track active (unfilled) FVGs
        active_fvgs = []

        for i in range(n):
            current_atr = atr[i] if atr[i] > EPSILON else 1.0
            c = close[i]

            # Add newly created FVGs at this bar
            if i in fvg_by_bar:
                active_fvgs.extend(fvg_by_bar[i])

            # Remove filled FVGs and those outside lookback
            surviving = []
            for fvg in active_fvgs:
                if i - fvg["bar"] > lookback:
                    continue
                # Bullish FVG filled when low penetrates below gap bottom
                if fvg["type"] == "bullish" and lows[i] <= fvg["bottom"]:
                    continue
                # Bearish FVG filled when high penetrates above gap top
                if fvg["type"] == "bearish" and highs[i] >= fvg["top"]:
                    continue
                surviving.append(fvg)
            active_fvgs = surviving

            # Compute features from active FVGs
            count[i] = len(active_fvgs)

            nearest_bull_dist = np.inf
            nearest_bull_size = 0.0
            nearest_bear_dist = np.inf
            nearest_bear_size = 0.0

            for fvg in active_fvgs:
                gap_size = (fvg["top"] - fvg["bottom"]) / current_atr
                mid = (fvg["top"] + fvg["bottom"]) / 2

                if fvg["type"] == "bullish":
                    d = (c - mid) / current_atr
                    if d > 0 and d < nearest_bull_dist:
                        nearest_bull_dist = d
                        nearest_bull_size = gap_size
                    # Check if price is inside this gap
                    if fvg["bottom"] <= c <= fvg["top"]:
                        in_gap[i] = 1.0

                elif fvg["type"] == "bearish":
                    d = (mid - c) / current_atr
                    if d > 0 and d < nearest_bear_dist:
                        nearest_bear_dist = d
                        nearest_bear_size = gap_size
                    if fvg["bottom"] <= c <= fvg["top"]:
                        in_gap[i] = 1.0

            if nearest_bull_dist < np.inf:
                bull_active[i] = 1.0
                bull_dist[i] = nearest_bull_dist
                bull_size[i] = nearest_bull_size

            if nearest_bear_dist < np.inf:
                bear_active[i] = 1.0
                bear_dist[i] = nearest_bear_dist
                bear_size[i] = nearest_bear_size

        features = {
            "fvg_bull_active": bull_active,
            "fvg_bear_active": bear_active,
            "fvg_bull_dist": bull_dist,
            "fvg_bear_dist": bear_dist,
            "fvg_bull_size": bull_size,
            "fvg_bear_size": bear_size,
            "fvg_in_gap": in_gap,
            "fvg_count": count,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_period": 14,
            "lookback": 100,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period used to normalize FVG distances and sizes. FVG distances and gap sizes are expressed in ATR units for scale-independence across different price levels and instruments.",
                "min": 2,
                "max": 500,
                "step": 1,
            },
            "lookback": {
                "type": "int",
                "default": 100,
                "description": "Maximum number of bars an unfilled FVG remains active. After this many bars, stale gaps are discarded. Longer lookbacks track more historical gaps but may include zones that have lost their significance as support/resistance.",
                "min": 10,
                "max": 1000,
                "step": 10,
            },
        }


__all__ = ["FairValueGapIndicator"]
