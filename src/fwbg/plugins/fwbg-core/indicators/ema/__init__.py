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
from typing import List

import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide

VALID_SOURCES = {"O", "H", "L", "C"}

DEFAULT_LINES = [
    {"period": 8, "source": "C"},
    {"period": 21, "source": "C"},
    {"period": 50, "source": "C"},
    {"period": 100, "source": "C"},
    {"period": 200, "source": "C"},
]


def _line_key(period: int, source: str) -> str:
    """Erzeugt den Spalten-Suffix für eine EMA-Linie: '200' oder '5_h'."""
    if source == "C":
        return str(period)
    return f"{period}_{source.lower()}"


def _parse_lines(lines: list[dict]) -> list[tuple[int, str]]:
    """Validiert und normalisiert die lines-Config."""
    parsed = []
    for line in lines:
        period = line["period"]
        source = line.get("source", "C").upper()
        if source not in VALID_SOURCES:
            raise ValueError(f"Invalid EMA source '{source}', must be one of {VALID_SOURCES}")
        if period < 2:
            raise ValueError(f"EMA period must be >= 2, got {period}")
        parsed.append((period, source))
    return parsed


@register_indicator("ema")
class EMAIndicator(BaseIndicator):
    """
    EMA-Indikatoren mit konfigurierbaren Quellpreisen.

    Features:
    - EMA Distanz (normalisiert als % vom Close)
    - EMA Crossings (alle Paare, auch cross-source)
    - EMA Overlay-Linien (Prefix '_', kein ML-Feature)
    """

    name = "ema"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        lines: list[dict] = None,
        crossings: bool = True,
        **params,
    ) -> pd.DataFrame:
        if lines is None:
            lines = DEFAULT_LINES

        parsed = _parse_lines(lines)
        features = {}
        ema_series = {}

        for period, source in parsed:
            key = _line_key(period, source)
            ema = ta.trend.ema_indicator(df[source], window=period)
            ema_series[key] = ema

            features[f"ema_dist_{key}"] = safe_divide(df["C"] - ema, df["C"])
            features[f"_ema_{key}"] = ema

        if crossings:
            sorted_keys = sorted(ema_series.keys(), key=lambda k: (
                int(k.split("_")[0]),
                k.split("_")[1] if "_" in k else "",
            ))
            for i, short_key in enumerate(sorted_keys):
                for long_key in sorted_keys[i + 1:]:
                    features[f"ema_{short_key}_above_{long_key}"] = (
                        ema_series[short_key] > ema_series[long_key]
                    ).astype(float)

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def _resolve_lines(self, params=None) -> tuple[list[tuple[int, str]], bool]:
        p = {**self.get_default_params(), **(params or {})}
        lines = _parse_lines(p.get("lines", DEFAULT_LINES))
        crossings = p.get("crossings", True)
        return lines, crossings

    def get_feature_columns(self, params=None) -> List[str]:
        lines, crossings = self._resolve_lines(params)
        cols = []
        keys = []

        for period, source in lines:
            key = _line_key(period, source)
            keys.append(key)
            cols.append(f"ema_dist_{key}")

        if crossings:
            sorted_keys = sorted(keys, key=lambda k: (
                int(k.split("_")[0]),
                k.split("_")[1] if "_" in k else "",
            ))
            for i, short_key in enumerate(sorted_keys):
                for long_key in sorted_keys[i + 1:]:
                    cols.append(f"ema_{short_key}_above_{long_key}")

        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        lines, _ = self._resolve_lines(params)
        return [f"_ema_{_line_key(p, s)}" for p, s in lines]

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
