"""
Supertrend Indicator (Standard + Multi-Timeframe).

ATR-basierter Trend-Following-Indikator. Direction: +1 (Aufwärts) oder -1 (Abwärts).

Enthält zwei registrierte Indikatoren:
- "supertrend": Standard-Supertrend auf OHLC-Daten
- "supertrend_mtf": Multi-Timeframe via rollende OHLC-Aggregation
"""
from typing import List

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend Berechnung.

    Returns (direction, line):
    - direction: +1 (Aufwärts) oder -1 (Abwärts)
    - line: Supertrend-Preisniveau
    """
    atr = ta.volatility.average_true_range(high, low, close, window=period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(close)
    supertrend_line = np.zeros(n)
    direction = np.ones(n)

    final_upper = upper_band.values.copy()
    final_lower = lower_band.values.copy()

    for i in range(1, n):
        if final_lower[i] < final_lower[i - 1] and close.iloc[i - 1] > final_lower[i - 1]:
            final_lower[i] = final_lower[i - 1]
        if final_upper[i] > final_upper[i - 1] and close.iloc[i - 1] < final_upper[i - 1]:
            final_upper[i] = final_upper[i - 1]

        if direction[i - 1] == 1:
            if close.iloc[i] < final_lower[i]:
                direction[i] = -1
                supertrend_line[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend_line[i] = final_lower[i]
        else:
            if close.iloc[i] > final_upper[i]:
                direction[i] = 1
                supertrend_line[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend_line[i] = final_upper[i]

    return (
        pd.Series(direction, index=close.index),
        pd.Series(supertrend_line, index=close.index),
    )


@register_indicator("supertrend")
class SupertrendIndicator(BaseIndicator):
    """
    Standard-Supertrend auf OHLC-Daten.

    Features:
    - st_direction: Trend-Richtung (+1/-1)
    - st_flip: 1 wenn Richtungswechsel
    - _st_line: Supertrend-Preisniveau (Overlay)
    """

    name = "supertrend"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 3.0,
        **params,
    ) -> pd.DataFrame:
        direction, st_line = _supertrend(df["H"], df["L"], df["C"], period, multiplier)

        features = {
            "st_direction": direction,
            "st_flip": (direction != direction.shift(1)).astype(float),
            "_st_line": st_line,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return ["st_direction", "st_flip"]

    def get_signal_columns(self, params=None) -> List[str]:
        return ["st_direction", "st_flip"]

    def get_overlay_columns(self, params=None) -> List[str]:
        return ["_st_line"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"period": 14, "multiplier": 3.0}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "period": {
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 500,
                "step": 1,
                "description": "ATR lookback period. Lower values = more responsive but noisier.",
            },
            "multiplier": {
                "type": "float",
                "default": 3.0,
                "min": 0.5,
                "max": 20.0,
                "step": 0.5,
                "description": "ATR multiplier for band width. Higher values = fewer trend flips.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"st": "Supertrend"}


@register_indicator("supertrend_mtf")
class SupertrendMTFIndicator(BaseIndicator):
    """
    Supertrend auf aggregierter Zeitebene als Trend-Filter.

    Berechnet rollende OHLC-Aggregation über d1_bars und wendet
    Supertrend darauf an. Ermöglicht höhere Zeitebenen ohne externe Daten.

    Bars pro Zeiteinheit (M15-Basis):
        H4 → d1_bars=16, D1 → d1_bars=96, W1 → d1_bars=480

    Features:
    - st_d1_direction: Richtung (+1/-1)
    - st_d1_dist_atr: Abstand zum ST-Level in ATR-Einheiten
    - _st_d1_line: Supertrend-Preisniveau (Overlay)
    """

    name = "supertrend_mtf"
    version = "1.1.0"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 3.0,
        d1_bars: int = 96,
        **params,
    ) -> pd.DataFrame:
        d1_high = df["H"].rolling(d1_bars, min_periods=1).max()
        d1_low = df["L"].rolling(d1_bars, min_periods=1).min()
        d1_close = df["C"]

        direction, st_line = _supertrend(d1_high, d1_low, d1_close, period, multiplier)

        m15_atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)
        st_dist_atr = safe_divide(d1_close - st_line, m15_atr)

        features = {
            "st_d1_direction": direction,
            "st_d1_dist_atr": st_dist_atr,
            "_st_d1_line": st_line,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return ["st_d1_direction", "st_d1_dist_atr"]

    def get_signal_columns(self, params=None) -> List[str]:
        return ["st_d1_direction"]

    def get_overlay_columns(self, params=None) -> List[str]:
        return ["_st_d1_line"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"period": 14, "multiplier": 3.0, "d1_bars": 96}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "period": {
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
                "description": "ATR period for Supertrend.",
            },
            "multiplier": {
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 6.0,
                "step": 0.5,
                "description": "ATR multiplier for band width.",
            },
            "d1_bars": {
                "type": "int",
                "default": 96,
                "min": 4,
                "max": 960,
                "step": 4,
                "description": "Number of base bars per aggregated candle. M15→D1: 96, M15→H4: 16, M15→W1: 480.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"st": "Supertrend MTF"}


__all__ = ["SupertrendIndicator", "SupertrendMTFIndicator"]
