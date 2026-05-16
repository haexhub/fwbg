"""
EMA (Exponential Moving Average) Indicator.

Berechnet EMA-Linien auf konfigurierbaren OHLC-Quellen mit
Distanz-Features und Crossing-Features über alle Linien hinweg.

Config-Beispiel:
    {
        "name": "ema",
        "params": {
            "lines": [
                {"period": 5, "source": "H"},
                {"period": 5, "source": "L"},
                {"period": 100, "source": "H"},
                {"period": 100, "source": "L"},
                {"period": 200, "source": "C"}
            ],
            "crossings": true
        }
    }

Spalten-Naming:
    - ema_dist_200       → Distance Close ↔ EMA(200, C) — kein Suffix bei Source "C"
    - ema_dist_5_h       → Distance Close ↔ EMA(5, H)
    - ema_5_h_above_200  → EMA(5, H) > EMA(200, C)
    - _ema_200           → Raw EMA-Linie (Overlay)
"""
import pandas as pd
import ta

from fwbg_sdk import register_indicator
from fwbg.utils.indicator_bases import BaseMovingAverageIndicator

DEFAULT_LINES = [
    {"period": 8, "source": "C"},
    {"period": 21, "source": "C"},
    {"period": 50, "source": "C"},
    {"period": 100, "source": "C"},
    {"period": 200, "source": "C"},
]


@register_indicator("ema")
class EMAIndicator(BaseMovingAverageIndicator):
    """
    EMA-Indikatoren mit konfigurierbaren Quellpreisen.

    Features:
    - EMA Distanz (normalisiert als % vom Close)
    - EMA Crossings (alle Paare, auch cross-source)
    - EMA Overlay-Linien (Prefix '_', kein ML-Feature)
    """

    name = "ema"
    version = "1.0.0"
    DEFAULT_LINES = DEFAULT_LINES
    _human_label = "EMA"

    def _compute_ma(self, series: pd.Series, period: int) -> pd.Series:
        return ta.trend.ema_indicator(series, window=period)

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "lines": DEFAULT_LINES,
            "crossings": True,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lines": {
                "type": "list[object]",
                "default": DEFAULT_LINES,
                "description": "List of EMA lines to compute. Each entry has 'period' (int, >= 2) and 'source' (one of 'O', 'H', 'L', 'C'). Source defaults to 'C' if omitted.",
                "item_schema": {
                    "period": {"type": "int", "min": 2, "max": 1000},
                    "source": {"type": "str", "enum": ["O", "H", "L", "C"], "default": "C"},
                },
            },
            "crossings": {
                "type": "bool",
                "default": True,
                "description": "Compute crossing features for all EMA line pairs (including cross-source). Produces binary features indicating whether shorter-period EMA is above longer-period EMA.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {
            "ema_dist": "EMA Distance",
            "ema_crossing": "EMA Crossings",
            "_ema": "EMA Lines",
        }


__all__ = ["EMAIndicator"]
