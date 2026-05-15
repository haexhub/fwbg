"""
MACD (Moving Average Convergence Divergence) Indicator.

MACD = 12-EMA - 26-EMA. Signal = 9-EMA of MACD. Histogram = MACD - Signal.
Alle Werte werden durch Close normalisiert.
"""
from typing import List

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


@register_indicator("macd")
class MACDIndicator(BaseIndicator):
    """
    MACD-Indikator für Momentum und Trend-Following.

    Features:
    - macd_line: MACD-Linie (12-EMA - 26-EMA), normalisiert
    - macd_signal: Signal-Linie (9-EMA der MACD-Linie), normalisiert
    - macd_hist: Histogramm (MACD - Signal), normalisiert
    - macd_above_zero: Vorzeichen der MACD-Linie (+1/-1)
    - macd_dist_zero: Absoluter Abstand von Null, normalisiert
    - macd_hist_flip: 1 wenn Histogramm Vorzeichen wechselt
    """

    name = "macd"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **params,
    ) -> pd.DataFrame:
        macd_ind = ta.trend.MACD(
            df["C"],
            window_fast=fast_period,
            window_slow=slow_period,
            window_sign=signal_period,
        )
        macd_line = macd_ind.macd()
        macd_hist = macd_ind.macd_diff()

        features = {
            "macd_hist": safe_divide(macd_hist, df["C"]),
            "macd_signal": safe_divide(macd_ind.macd_signal(), df["C"]),
            "macd_line": safe_divide(macd_line, df["C"]),
            "macd_above_zero": np.sign(macd_line),
            "macd_dist_zero": safe_divide(macd_line.abs(), df["C"]),
            "macd_hist_flip": (np.sign(macd_hist) != np.sign(macd_hist.shift(1))).astype(float),
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return [
            "macd_hist", "macd_signal", "macd_line",
            "macd_above_zero", "macd_dist_zero", "macd_hist_flip",
        ]

    def get_signal_columns(self, params=None) -> List[str]:
        return ["macd_above_zero", "macd_hist_flip"]

    def get_overlay_columns(self, params=None) -> List[str]:
        return []

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "fast_period": {
                "type": "int",
                "default": 12,
                "min": 2,
                "max": 100,
                "step": 1,
                "description": "Fast EMA period for MACD line calculation.",
            },
            "slow_period": {
                "type": "int",
                "default": 26,
                "min": 2,
                "max": 200,
                "step": 1,
                "description": "Slow EMA period for MACD line calculation.",
            },
            "signal_period": {
                "type": "int",
                "default": 9,
                "min": 2,
                "max": 50,
                "step": 1,
                "description": "Signal line EMA period.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"macd": "MACD"}


__all__ = ["MACDIndicator"]
