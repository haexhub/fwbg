"""
SMA (Simple Moving Average) Indicator.

Gleiche Architektur wie EMA: per-line Source-Config, Distanz-Features,
Cross-Source Crossings.
"""
import pandas as pd
import ta

from fwbg_sdk import register_indicator
from fwbg.utils.indicator_bases import BaseMovingAverageIndicator

DEFAULT_LINES = [
    {"period": 20, "source": "C"},
    {"period": 50, "source": "C"},
    {"period": 200, "source": "C"},
]


@register_indicator("sma")
class SMAIndicator(BaseMovingAverageIndicator):
    """
    SMA-Indikatoren mit konfigurierbaren Quellpreisen.

    Features:
    - SMA Distanz (normalisiert als % vom Close)
    - SMA Crossings (alle Paare, auch cross-source)
    - SMA Overlay-Linien (Prefix '_', kein ML-Feature)
    """

    name = "sma"
    version = "1.0.0"
    DEFAULT_LINES = DEFAULT_LINES
    _human_label = "SMA"

    def _compute_ma(self, series: pd.Series, period: int) -> pd.Series:
        return ta.trend.sma_indicator(series, window=period)

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
                "description": "List of SMA lines to compute. Each entry has 'period' (int, >= 2) and 'source' (one of 'O', 'H', 'L', 'C'). Source defaults to 'C' if omitted.",
                "item_schema": {
                    "period": {"type": "int", "min": 2, "max": 1000},
                    "source": {"type": "str", "enum": ["O", "H", "L", "C"], "default": "C"},
                },
            },
            "crossings": {
                "type": "bool",
                "default": True,
                "description": "Compute crossing features for all SMA line pairs (including cross-source).",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {
            "sma_dist": "SMA Distance",
            "sma_crossing": "SMA Crossings",
            "_sma": "SMA Lines",
        }


__all__ = ["SMAIndicator"]
