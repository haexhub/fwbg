"""Shared base classes for indicator plugins.

Plugin __init__.py files live in non-importable directories (the
`fwbg-core` directory has a hyphen). This module hosts shared scaffolding
that multiple plugins can import via the standard import system.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple

import pandas as pd

from fwbg_sdk import BaseIndicator, safe_divide, shift_features

VALID_OHLC_SOURCES = {"O", "H", "L", "C"}


class BaseMovingAverageIndicator(BaseIndicator):
    """Common scaffolding for moving-average indicators (EMA, SMA, ...).

    Subclass contract:
        - override `_compute_ma(series, period) -> pd.Series`
        - set class attributes `name`, `version`, `DEFAULT_LINES`
        - optionally set `_human_label` (e.g. "EMA", "SMA") for error messages
    """

    name: str = ""
    DEFAULT_LINES: List[dict] = []
    _human_label: str = "MA"

    # ---- Hook methods ---------------------------------------------------

    @abstractmethod
    def _compute_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Return the moving-average series for *series* with *period* bars."""

    # ---- Shared helpers --------------------------------------------------

    @staticmethod
    def _line_key(period: int, source: str) -> str:
        if source == "C":
            return str(period)
        return f"{period}_{source.lower()}"

    def _parse_lines(self, lines: list[dict]) -> list[Tuple[int, str]]:
        parsed: list[Tuple[int, str]] = []
        for line in lines:
            period = line["period"]
            source = line.get("source", "C").upper()
            if source not in VALID_OHLC_SOURCES:
                raise ValueError(
                    f"Invalid {self._human_label} source '{source}', "
                    f"must be one of {VALID_OHLC_SOURCES}"
                )
            if period < 2:
                raise ValueError(
                    f"{self._human_label} period must be >= 2, got {period}"
                )
            parsed.append((period, source))
        return parsed

    def _resolve_lines(self, params=None) -> Tuple[list[Tuple[int, str]], bool]:
        p = {**self.get_default_params(), **(params or {})}
        lines = self._parse_lines(p.get("lines", self.DEFAULT_LINES))
        crossings = p.get("crossings", True)
        return lines, crossings

    @staticmethod
    def _sorted_keys(keys: list[str]) -> list[str]:
        return sorted(keys, key=lambda k: (
            int(k.split("_")[0]),
            k.split("_")[1] if "_" in k else "",
        ))

    # ---- Pipeline entry points -------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        lines: list[dict] = None,
        crossings: bool = True,
        **params,
    ) -> pd.DataFrame:
        if lines is None:
            lines = self.DEFAULT_LINES

        parsed = self._parse_lines(lines)
        prefix = self.name
        features = {}
        ma_series = {}

        for period, source in parsed:
            key = self._line_key(period, source)
            ma = self._compute_ma(df[source], period)
            ma_series[key] = ma
            features[f"{prefix}_dist_{key}"] = safe_divide(df["C"] - ma, df["C"])
            features[f"_{prefix}_{key}"] = ma

        if crossings:
            sorted_keys = self._sorted_keys(list(ma_series.keys()))
            for i, short_key in enumerate(sorted_keys):
                for long_key in sorted_keys[i + 1:]:
                    features[f"{prefix}_{short_key}_above_{long_key}"] = (
                        ma_series[short_key] > ma_series[long_key]
                    ).astype(float)

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        lines, crossings = self._resolve_lines(params)
        prefix = self.name
        cols: List[str] = []
        keys: List[str] = []

        for period, source in lines:
            key = self._line_key(period, source)
            keys.append(key)
            cols.append(f"{prefix}_dist_{key}")

        if crossings:
            sorted_keys = self._sorted_keys(keys)
            for i, short_key in enumerate(sorted_keys):
                for long_key in sorted_keys[i + 1:]:
                    cols.append(f"{prefix}_{short_key}_above_{long_key}")

        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        lines, _ = self._resolve_lines(params)
        prefix = self.name
        return [f"_{prefix}_{self._line_key(p, s)}" for p, s in lines]
