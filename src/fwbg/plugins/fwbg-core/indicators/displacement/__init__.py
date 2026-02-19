"""
Displacement Indicator Plugin.

Measures breakout quality and candle conviction:
- Body ratio and size relative to ATR (impulse strength)
- Wick analysis (rejection detection)
- FVG formation at current bar (imbalance confirmation)
- Consecutive directional candles (momentum persistence)
- Range expansion (unusual move detection)
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON


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
    l = df["L"].values
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
    direction = np.sign(c - o)
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
                "description": "ATR period for normalizing body size.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "range_avg_period": {
                "type": "int",
                "default": 20,
                "description": "Rolling window for average range computation.",
                "min": 5,
                "max": 100,
                "step": 5,
            },
        }


__all__ = ["DisplacementIndicator"]
