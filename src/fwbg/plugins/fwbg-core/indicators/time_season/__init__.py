"""
Time & Season Indicator Plugin.

Enthält:
- Intraday Zeit Features (Hour, Day of Week)
- Saisonalität Features (Month, Quarter, Week)
- Cyclische Encoding (Sin/Cos Transformation)
- Konfigurierbare Trading Sessions

Timing: every column describes the bar's OWN timestamp (its index value), not the
previous bar's. The calendar and the session clock are fixed in advance, so the
value for a timestamp is known before the bar opens and cannot leak the future;
shifting it by one bar would make Monday's row describe Friday and the first bar
of a session look like the last bar before it. These columns are therefore not
shifted. Only the first row is set to NaN, as for every indicator.

Month start/end and quarter end are counted in trading days of the ``calendar``
parameter (default ``"weekdays"``: every Monday-Friday; an exchange such as
``"NYSE"`` also skips its holidays), so the last trading day of a month is always
flagged. A weekend after the month's last trading day keeps its flag.

Session hours are UTC (naive timestamps are UTC). A session given as
``[start, end, "Area/City"]`` is evaluated in that time zone's local hours, so it
follows daylight saving time.
"""

from typing import List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

_DEFAULT_SESSIONS = {
    "asia": [0, 8],
    "london": [8, 16],
    "ny": [13, 21],
}


def _turn_of_month_module():
    from fwbg.plugins import import_plugin_module

    return import_plugin_module("fwbg-core", "indicators", "turn_of_month")


def _calendar_choices() -> list:
    return list(_turn_of_month_module().CALENDARS)


def iso_weeks_in_year(years) -> np.ndarray:
    """Number of ISO weeks (52 or 53) of each ISO year in *years*."""
    years = np.asarray(years, dtype=int)
    unique = np.unique(years)
    # 28 December always lies in the last ISO week of its year
    last = {y: int(pd.Timestamp(int(y), 12, 28).isocalendar().week) for y in unique}
    return np.array([last[y] for y in years], dtype=float)


@register_indicator("time_season")
class TimeSeasonIndicators(BaseIndicator):
    """Zeit- und Saisonalitäts-Features mit konfigurierbaren Sessions."""

    name = "time_season"
    version = "3.1.0"

    def _resolve_params(self, params=None) -> dict:
        return {**self.get_default_params(), **(params or {})}

    @staticmethod
    def _parse_sessions(raw) -> dict[str, list[int]]:
        """Normalize sessions param to {name: [start, end]} or {name: [start, end, tz]}."""
        if isinstance(raw, dict):
            parsed = {k: list(v) for k, v in raw.items()}
            for name, spec in parsed.items():
                if len(spec) not in (2, 3):
                    raise ValueError(
                        f"time_season: session {name!r} must be [start, end] or "
                        f"[start, end, timezone], got {spec!r}"
                    )
            return parsed
        return {k: list(v) for k, v in _DEFAULT_SESSIONS.items()}

    @staticmethod
    def _session_hours(index: pd.DatetimeIndex, spec: list) -> np.ndarray:
        """Hour of day of each bar in the session's time zone (index hours if none)."""
        if len(spec) < 3 or spec[2] in (None, ""):
            return index.hour
        try:
            ZoneInfo(spec[2])
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(
                f"time_season: unknown session time zone {spec[2]!r}"
            ) from exc
        utc = index.tz_localize("UTC") if index.tz is None else index
        return utc.tz_convert(spec[2]).hour

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
        calendar: str = "weekdays",
        month_edge_days: int = 3,
        **params,
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
                start, end = sess[name][:2]
                session_hour = self._session_hours(df.index, sess[name])
                if start < end:
                    mask = (session_hour >= start) & (session_hour < end)
                else:
                    mask = (session_hour >= start) | (session_hour < end)
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
                features["season_quarter_sin"] = np.sin(
                    2 * np.pi * df.index.quarter / 4
                )
                features["season_quarter_cos"] = np.cos(
                    2 * np.pi * df.index.quarter / 4
                )
                iso = df.index.isocalendar()
                week = iso.week.astype(int).values
                # the cycle length is the ISO year's week count (52 or 53), so week 53
                # of a long year is not mapped onto week 1
                weeks_in_year = iso_weeks_in_year(iso.year.astype(int).values)
                features["season_week_sin"] = np.sin(2 * np.pi * week / weeks_in_year)
                features["season_week_cos"] = np.cos(2 * np.pi * week / weeks_in_year)
                features["season_dayofmonth_sin"] = np.sin(
                    2 * np.pi * df.index.day / 31
                )
                features["season_dayofmonth_cos"] = np.cos(
                    2 * np.pi * df.index.day / 31
                )

        # === Special Calendar Features ===
        if include_calendar:
            if len(df):
                dates = (
                    df.index.tz_localize(None) if df.index.tz is not None else df.index
                )
                tom_day, days_left = _turn_of_month_module().month_position(
                    calendar, dates.normalize()
                )
            else:
                tom_day = days_left = np.zeros(0)
            # first / last `month_edge_days` trading days of the month
            is_start = (tom_day >= 1) & (tom_day <= month_edge_days)
            is_end = days_left < month_edge_days
            features["time_month_start"] = is_start.astype(int)
            features["time_month_end"] = is_end.astype(int)
            features["time_quarter_end"] = (
                is_end & df.index.month.isin([3, 6, 9, 12])
            ).astype(int)
            features["time_week_start"] = (df.index.dayofweek == 0).astype(int)
            features["time_week_end"] = (df.index.dayofweek == 4).astype(int)

        # === Year Progress ===
        if include_year_progress:
            day_of_year = df.index.dayofyear
            days_in_year = 365 + df.index.is_leap_year.astype(int)
            features["time_year_progress"] = day_of_year / days_in_year

        features_df = pd.DataFrame(features, index=df.index).astype(float)
        if len(features_df):
            features_df.iloc[0] = np.nan  # indicator contract: the first row is NaN
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
                cols += [
                    "season_month",
                    "season_quarter",
                    "season_week",
                    "season_dayofmonth",
                ]
            if p["include_encoded"]:
                cols += [
                    "season_month_sin",
                    "season_month_cos",
                    "season_quarter_sin",
                    "season_quarter_cos",
                    "season_week_sin",
                    "season_week_cos",
                    "season_dayofmonth_sin",
                    "season_dayofmonth_cos",
                ]

        if p["include_calendar"]:
            cols += [
                "time_month_start",
                "time_month_end",
                "time_quarter_end",
                "time_week_start",
                "time_week_end",
            ]

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
            cols += [
                "time_month_start",
                "time_month_end",
                "time_quarter_end",
                "time_week_start",
                "time_week_end",
            ]

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
            "calendar": "weekdays",
            "month_edge_days": 3,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "include_raw": {
                "type": "bool",
                "default": True,
                "description": "Include raw (integer) time features: hour of day (0-23), day of week (0-6), month (1-12), quarter (1-4), ISO week (1-53), day of month (1-31).",
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
                "description": "Trading sessions as {name: [start_hour, end_hour]} in UTC hours, or {name: [start_hour, end_hour, 'Area/City']} to evaluate the hours in that time zone's local time (follows daylight saving time). Each session produces a binary feature time_session_{name}. Overlap is computed automatically. Supports wrap-around (e.g. [22, 6] for overnight).",
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
                "description": "Include special calendar features: month start/end, quarter end, week start/end. Month start/end and quarter end count trading days of the calendar parameter.",
            },
            "include_year_progress": {
                "type": "bool",
                "default": True,
                "description": "Include year progress feature (0-1).",
            },
            "calendar": {
                "type": "choice",
                "choices": _calendar_choices(),
                "default": "weekdays",
                "description": "Trading-day calendar for time_month_start/end and time_quarter_end: 'weekdays' (every Monday-Friday) or an exchange holiday calendar such as NYSE (holidays.financial_holidays).",
            },
            "month_edge_days": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "time_month_start covers the first and time_month_end the last this-many trading days of a month.",
            },
        }


__all__ = ["TimeSeasonIndicators", "iso_weeks_in_year"]
