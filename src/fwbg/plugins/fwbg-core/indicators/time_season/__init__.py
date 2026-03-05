"""
Time & Season Indicator Plugin.

Enthält:
- Intraday Zeit Features (Hour, Day of Week)
- Saisonalität Features (Month, Quarter, Week)
- Cyclische Encoding (Sin/Cos Transformation)
- Konfigurierbare Trading Sessions
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features

_DEFAULT_SESSIONS = {
    "asia": [0, 8],
    "london": [8, 16],
    "ny": [13, 21],
}


@register_indicator("time_season")
class TimeSeasonIndicators(BaseIndicator):
    """Zeit- und Saisonalitäts-Features mit konfigurierbaren Sessions."""

    name = "time_season"
    version = "3.0.0"

    def _resolve_params(self, params=None) -> dict:
        return {**self.get_default_params(), **(params or {})}

    @staticmethod
    def _parse_sessions(raw) -> dict[str, list[int]]:
        """Normalize sessions param to {name: [start, end]}."""
        if isinstance(raw, dict):
            return {k: list(v) for k, v in raw.items()}
        return dict(_DEFAULT_SESSIONS)

    @staticmethod
    def _session_cols(sessions: dict[str, list[int]]) -> list[str]:
        """Column names for configured sessions + overlap."""
        cols = [f"time_session_{name}" for name in sorted(sessions)]
        if len(sessions) >= 2:
            cols.append("time_session_overlap")
        return cols

    def compute(
        self,
        df: pd.DataFrame,
        include_raw: bool = True,
        include_encoded: bool = True,
        include_sessions: bool = True,
        sessions: dict | None = None,
        include_seasonality: bool = True,
        include_calendar: bool = True,
        include_year_progress: bool = True,
        trading_days: list[int] | None = None,
        **params
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DateTimeIndex haben")

        features = {}
        hour = df.index.hour

        # === Intraday Features ===
        if include_raw:
            features["time_hour"] = hour
            features["time_day"] = df.index.dayofweek

        if include_encoded:
            features["time_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features["time_hour_cos"] = np.cos(2 * np.pi * hour / 24)
            features["time_day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features["time_day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # === Trading Session Features ===
        if include_sessions:
            sess = self._parse_sessions(sessions)
            masks = {}
            for name in sorted(sess):
                start, end = sess[name]
                if start < end:
                    mask = (hour >= start) & (hour < end)
                else:
                    mask = (hour >= start) | (hour < end)
                features[f"time_session_{name}"] = mask.astype(int)
                masks[name] = mask

            if len(masks) >= 2:
                overlap = np.zeros(len(df), dtype=int)
                mask_list = list(masks.values())
                for i in range(len(mask_list)):
                    for j in range(i + 1, len(mask_list)):
                        overlap |= (mask_list[i] & mask_list[j]).astype(int)
                features["time_session_overlap"] = overlap

        # === Trading Days ===
        if trading_days is not None:
            day_set = set(trading_days)
            features["time_trading_day"] = df.index.dayofweek.isin(day_set).astype(int)

        # === Saisonalität Features ===
        if include_seasonality:
            if include_raw:
                features["season_month"] = df.index.month
                features["season_quarter"] = df.index.quarter
                features["season_week"] = df.index.isocalendar().week.astype(int).values
                features["season_dayofmonth"] = df.index.day

            if include_encoded:
                features["season_month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
                features["season_month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
                features["season_quarter_sin"] = np.sin(2 * np.pi * df.index.quarter / 4)
                features["season_quarter_cos"] = np.cos(2 * np.pi * df.index.quarter / 4)
                week = df.index.isocalendar().week.astype(int).values
                features["season_week_sin"] = np.sin(2 * np.pi * week / 52)
                features["season_week_cos"] = np.cos(2 * np.pi * week / 52)
                features["season_dayofmonth_sin"] = np.sin(2 * np.pi * df.index.day / 31)
                features["season_dayofmonth_cos"] = np.cos(2 * np.pi * df.index.day / 31)

        # === Special Calendar Features ===
        if include_calendar:
            features["time_month_start"] = (df.index.day <= 3).astype(int)
            features["time_month_end"] = (df.index.day >= 28).astype(int)
            features["time_quarter_end"] = (
                (df.index.month.isin([3, 6, 9, 12])) & (df.index.day >= 28)
            ).astype(int)
            features["time_week_start"] = (df.index.dayofweek == 0).astype(int)
            features["time_week_end"] = (df.index.dayofweek == 4).astype(int)

        # === Year Progress ===
        if include_year_progress:
            day_of_year = df.index.dayofyear
            days_in_year = 365 + df.index.is_leap_year.astype(int)
            features["time_year_progress"] = day_of_year / days_in_year

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        p = self._resolve_params(params)
        cols: List[str] = []

        if p["include_raw"]:
            cols += ["time_hour", "time_day"]
        if p["include_encoded"]:
            cols += ["time_hour_sin", "time_hour_cos", "time_day_sin", "time_day_cos"]

        if p["include_sessions"]:
            sess = self._parse_sessions(p.get("sessions"))
            cols += self._session_cols(sess)

        if p.get("trading_days") is not None:
            cols.append("time_trading_day")

        if p["include_seasonality"]:
            if p["include_raw"]:
                cols += ["season_month", "season_quarter", "season_week", "season_dayofmonth"]
            if p["include_encoded"]:
                cols += ["season_month_sin", "season_month_cos",
                         "season_quarter_sin", "season_quarter_cos",
                         "season_week_sin", "season_week_cos",
                         "season_dayofmonth_sin", "season_dayofmonth_cos"]

        if p["include_calendar"]:
            cols += ["time_month_start", "time_month_end", "time_quarter_end",
                     "time_week_start", "time_week_end"]

        if p["include_year_progress"]:
            cols += ["time_year_progress"]

        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        p = self._resolve_params(params)
        cols: List[str] = []

        if p["include_sessions"]:
            sess = self._parse_sessions(p.get("sessions"))
            cols += self._session_cols(sess)

        if p.get("trading_days") is not None:
            cols.append("time_trading_day")

        if p["include_calendar"]:
            cols += ["time_month_start", "time_month_end", "time_quarter_end",
                     "time_week_start", "time_week_end"]

        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "include_raw": True,
            "include_encoded": True,
            "include_sessions": True,
            "sessions": {
                "asia": [0, 8],
                "london": [8, 16],
                "ny": [13, 21],
            },
            "include_seasonality": True,
            "include_calendar": True,
            "include_year_progress": True,
            "trading_days": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "include_raw": {
                "type": "bool",
                "default": True,
                "description": "Include raw (integer) time features: hour of day (0-23), day of week (0-6), month (1-12), quarter (1-4), ISO week (1-52), day of month (1-31).",
            },
            "include_encoded": {
                "type": "bool",
                "default": True,
                "description": "Include sin/cos cyclical encoding of time features.",
            },
            "include_sessions": {
                "type": "bool",
                "default": True,
                "description": "Include trading session binary features based on configured session ranges.",
            },
            "sessions": {
                "type": "session_ranges",
                "default": {"asia": [0, 8], "london": [8, 16], "ny": [13, 21]},
                "description": "Trading sessions as {name: [start_hour, end_hour]}. Each session produces a binary feature time_session_{name}. Overlap is computed automatically. Supports wrap-around (e.g. [22, 6] for overnight).",
            },
            "trading_days": {
                "type": "list[int]",
                "default": None,
                "description": "Active trading days (0=Mon, 6=Sun). When set, produces time_trading_day binary feature. Leave empty to disable.",
                "min": 0,
                "max": 6,
                "choices": ["0", "1", "2", "3", "4", "5", "6"],
                "choice_labels": {
                    "0": "Monday",
                    "1": "Tuesday",
                    "2": "Wednesday",
                    "3": "Thursday",
                    "4": "Friday",
                    "5": "Saturday",
                    "6": "Sunday",
                },
            },
            "include_seasonality": {
                "type": "bool",
                "default": True,
                "description": "Include seasonality features: month, quarter, ISO week, day of month.",
            },
            "include_calendar": {
                "type": "bool",
                "default": True,
                "description": "Include special calendar features: month start/end, quarter end, week start/end.",
            },
            "include_year_progress": {
                "type": "bool",
                "default": True,
                "description": "Include year progress feature (0-1).",
            },
        }


__all__ = ["TimeSeasonIndicators"]
