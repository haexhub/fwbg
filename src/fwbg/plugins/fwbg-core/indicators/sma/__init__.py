"""
SMA (Simple Moving Average) Indicator.

Gleiche Architektur wie EMA: per-line Source-Config, Distanz-Features,
Cross-Source Crossings.
"""
from typing import List

import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide

VALID_SOURCES = {"O", "H", "L", "C"}

DEFAULT_LINES = [
    {"period": 20, "source": "C"},
    {"period": 50, "source": "C"},
    {"period": 200, "source": "C"},
]


def _line_key(period: int, source: str) -> str:
    if source == "C":
        return str(period)
    return f"{period}_{source.lower()}"


def _parse_lines(lines: list[dict]) -> list[tuple[int, str]]:
    parsed = []
    for line in lines:
        period = line["period"]
        source = line.get("source", "C").upper()
        if source not in VALID_SOURCES:
            raise ValueError(f"Invalid SMA source '{source}', must be one of {VALID_SOURCES}")
        if period < 2:
            raise ValueError(f"SMA period must be >= 2, got {period}")
        parsed.append((period, source))
    return parsed


@register_indicator("sma")
class SMAIndicator(BaseIndicator):
    """
    SMA-Indikatoren mit konfigurierbaren Quellpreisen.

    Features:
    - SMA Distanz (normalisiert als % vom Close)
    - SMA Crossings (alle Paare, auch cross-source)
    - SMA Overlay-Linien (Prefix '_', kein ML-Feature)
    """

    name = "sma"
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
        sma_series = {}

        for period, source in parsed:
            key = _line_key(period, source)
            sma = ta.trend.sma_indicator(df[source], window=period)
            sma_series[key] = sma

            features[f"sma_dist_{key}"] = safe_divide(df["C"] - sma, df["C"])
            features[f"_sma_{key}"] = sma

        if crossings:
            sorted_keys = sorted(sma_series.keys(), key=lambda k: (
                int(k.split("_")[0]),
                k.split("_")[1] if "_" in k else "",
            ))
            for i, short_key in enumerate(sorted_keys):
                for long_key in sorted_keys[i + 1:]:
                    features[f"sma_{short_key}_above_{long_key}"] = (
                        sma_series[short_key] > sma_series[long_key]
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
            cols.append(f"sma_dist_{key}")

        if crossings:
            sorted_keys = sorted(keys, key=lambda k: (
                int(k.split("_")[0]),
                k.split("_")[1] if "_" in k else "",
            ))
            for i, short_key in enumerate(sorted_keys):
                for long_key in sorted_keys[i + 1:]:
                    cols.append(f"sma_{short_key}_above_{long_key}")

        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        lines, _ = self._resolve_lines(params)
        return [f"_sma_{_line_key(p, s)}" for p, s in lines]

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
