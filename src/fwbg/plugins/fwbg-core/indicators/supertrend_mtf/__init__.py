"""
Supertrend Multi-Timeframe Indicator.

Berechnet den Supertrend auf einer höheren Zeitebene (z.B. Tagesbasis)
direkt aus M15-Daten durch rollende OHLC-Aggregation.

Prinzip:
    Statt externe Tagesdaten zu laden, wird ein rollendes Fenster von
    d1_bars M15-Bars als "einen Tag" behandelt:
    - D1 High  = max(H) der letzten d1_bars Bars
    - D1 Low   = min(L) der letzten d1_bars Bars
    - D1 Close = aktueller Close

    Auf diesen aggregierten OHLC-Werten wird der Supertrend berechnet.
    Ergebnis: Tages-Trend-Richtung (+1 = Aufwärtstrend, -1 = Abwärtstrend)
    als Filter-Signal für M15-Entries.

Bars pro Zeiteinheit (M15-Basis):
    H4  → d1_bars=16   (4h × 4 bars/h)
    D1  → d1_bars=96   (24h × 4 bars/h)  ← Default
    W1  → d1_bars=480  (5d × 96 bars/d)

Verwendung im Config:
    "indicators": [{"name": "supertrend_mtf", "params": {"d1_bars": 96}}]

    "signal_rules": {
        "long":  {"conditions": [..., {"type": "value_check",
                                       "column": "st_d1_direction",
                                       "op": ">", "value": 0}]},
        "short": {"conditions": [..., {"type": "value_check",
                                       "column": "st_d1_direction",
                                       "op": "<", "value": 0}]}
    }
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
    """Supertrend auf beliebigen OHLC-Daten (identisch zum trend-Indikator)."""
    atr = ta.volatility.average_true_range(high, low, close, window=period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(close)
    supertrend_line = np.zeros(n)
    direction = np.ones(n)  # 1 = Aufwärts, -1 = Abwärts

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


@register_indicator("supertrend_mtf")
class SupertrendMTFIndicator(BaseIndicator):
    """
    Supertrend auf aggregierter Tagesbasis als Trend-Filter.

    Features:
    - st_d1_direction:  Richtung (+1 = Aufwärtstrend, -1 = Abwärtstrend)
    - st_d1_dist_atr:   Abstand zum ST-Level in ATR-Einheiten (normalisiert)
    - _st_d1_line:      Supertrend-Preisniveau (Overlay, kein ML-Feature)
    """

    name = "supertrend_mtf"
    version = "1.0.0"
    group = "trend"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 3.0,
        d1_bars: int = 96,
        **params,
    ) -> pd.DataFrame:
        """
        Berechnet Supertrend auf aggregierten Tagesbars.

        Args:
            df:         DataFrame mit OHLC-Daten (O, H, L, C).
            period:     ATR-Periode für Supertrend (default: 14).
            multiplier: ATR-Multiplikator für Bandbreite (default: 3.0).
            d1_bars:    Anzahl M15-Bars pro "Tag" (default: 96 = 24h × 4).
                        H4→16, D1→96, W1→480.
        """
        # Tages-OHLC via rollendes Fenster aggregieren.
        # min_periods=1 damit das Fenster ab Bar 0 wächst (kein harter NaN-Sprung).
        # Während der Warmup-Phase entspricht d1_high/d1_low dem bisherigen Extremum –
        # das ist konservativ und verursacht kein Lookahead.
        d1_high = df["H"].rolling(d1_bars, min_periods=1).max()
        d1_low = df["L"].rolling(d1_bars, min_periods=1).min()
        d1_close = df["C"]

        direction, st_line = _supertrend(d1_high, d1_low, d1_close, period, multiplier)

        # ATR auf M15-Basis für Normalisierung.
        # d1_atr würde wegen NaN-Propagation scheitern (d1_high/d1_low starten als NaN).
        # M15-ATR ist immer valide nach der Warmup-Periode.
        m15_atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)

        # Abstand zum ST-Level: positiv = über ST (long-freundlich), negativ = darunter
        st_dist_atr = safe_divide(d1_close - st_line, m15_atr)

        features = {
            "st_d1_direction": direction,   # +1 oder -1
            "st_d1_dist_atr": st_dist_atr,  # normalisierter Abstand
        }
        # ST-Linie als Overlay (Präfix _ → kein ML-Feature, aber im Chart sichtbar)
        overlay = {"_st_d1_line": st_line}

        features_df = shift_features({**features, **overlay}, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return ["st_d1_direction", "st_d1_dist_atr"]

    def get_signal_columns(self) -> List[str]:
        return ["st_d1_direction"]

    def get_overlay_columns(self) -> List[str]:
        return ["_st_d1_line"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "period": 14,
            "multiplier": 3.0,
            "d1_bars": 96,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "period": {
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
                "description": "ATR-Periode für Supertrend. Standardwert 14 (Wilder). Kürzere Perioden reagieren schneller auf Trendwechsel, längere filtern mehr Noise.",
            },
            "multiplier": {
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 6.0,
                "step": 0.5,
                "description": "ATR-Multiplikator für die Bandbreite. Größere Werte = weniger Trendwechsel (seltener aber stabiler). 3.0 ist der Standardwert.",
            },
            "d1_bars": {
                "type": "int",
                "default": 96,
                "min": 4,
                "max": 960,
                "step": 4,
                "description": "Anzahl der Basis-Bars pro aggregiertem Candle. M15→D1: 96 (24h×4). M15→H4: 16. M15→W1: 480. Bestimmt welche Zeitebene der Supertrend repräsentiert.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {
            "st": "Supertrend MTF",
        }


__all__ = ["SupertrendMTFIndicator"]
