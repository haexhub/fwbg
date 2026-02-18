"""
Time & Season Indicator Plugin.

Enthält:
- Intraday Zeit Features (Hour, Day of Week)
- Saisonalität Features (Month, Quarter, Week)
- Cyclische Encoding (Sin/Cos Transformation)

Zeitbasierte Features sind wichtig:
- Marktverhalten ändert sich je nach Tageszeit
- Saisonale Muster (January Effect, etc.)
- Periodische Zyklen erfordern Sin/Cos Encoding
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


@register_indicator("time_season")
class TimeSeasonIndicators(BaseIndicator):
    """
    Zeit- und Saisonalitäts-Features.

    Features:
    - Intraday: Hour, Day of Week (mit Sin/Cos Encoding)
    - Saisonalität: Month, Quarter, Week, Day of Month
    - Alle cyclischen Features mit Sin/Cos Transformation
    """

    # Required attributes
    name = "time_season"
    version = "2.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        include_raw: bool = True,
        include_encoded: bool = True,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Zeit- und Saisonalitäts-Features.

        Args:
            df: DataFrame mit DateTimeIndex
            include_raw: Inkludiere Raw-Werte (hour, day, etc.)
            include_encoded: Inkludiere Sin/Cos Encoding

        Returns:
            DataFrame mit Zeit-Features
        """
        # Sicherstellen dass wir einen DateTimeIndex haben
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DateTimeIndex haben")

        features = {}
        hour = df.index.hour

        # === Intraday Features ===
        if include_raw:
            features["time_hour"] = hour
            features["time_day"] = df.index.dayofweek

        if include_encoded:
            # Hour: 0-23 -> Sin/Cos
            features["time_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features["time_hour_cos"] = np.cos(2 * np.pi * hour / 24)

            # Day of Week: 0-6 -> Sin/Cos
            features["time_day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features["time_day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # === Trading Session Features ===
        # Major Trading Sessions (für Forex)
        # Asia: 0-8 UTC, London: 8-16 UTC, NY: 13-21 UTC
        features["time_session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
        features["time_session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
        features["time_session_ny"] = ((hour >= 13) & (hour < 21)).astype(int)

        # Session Overlap (höhere Liquidität)
        features["time_session_overlap"] = (
            ((hour >= 8) & (hour < 12)) |  # London-Asia
            ((hour >= 13) & (hour < 16))    # London-NY
        ).astype(int)

        # === Saisonalität Features ===
        if include_raw:
            features["season_month"] = df.index.month
            features["season_quarter"] = df.index.quarter
            features["season_week"] = df.index.isocalendar().week.astype(int).values
            features["season_dayofmonth"] = df.index.day

        if include_encoded:
            # Month: 1-12 -> Sin/Cos
            features["season_month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
            features["season_month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

            # Quarter: 1-4 -> Sin/Cos
            features["season_quarter_sin"] = np.sin(2 * np.pi * df.index.quarter / 4)
            features["season_quarter_cos"] = np.cos(2 * np.pi * df.index.quarter / 4)

            # Week: 1-52 -> Sin/Cos
            week = df.index.isocalendar().week.astype(int).values
            features["season_week_sin"] = np.sin(2 * np.pi * week / 52)
            features["season_week_cos"] = np.cos(2 * np.pi * week / 52)

            # Day of Month: 1-31 -> Sin/Cos
            features["season_dayofmonth_sin"] = np.sin(2 * np.pi * df.index.day / 31)
            features["season_dayofmonth_cos"] = np.cos(2 * np.pi * df.index.day / 31)

        # === Special Calendar Features ===
        # Monatsanfang/Monatsende (institutionelle Rebalancing)
        features["time_month_start"] = (df.index.day <= 3).astype(int)
        features["time_month_end"] = (df.index.day >= 28).astype(int)

        # Quartalsende (starkes Rebalancing)
        features["time_quarter_end"] = (
            (df.index.month.isin([3, 6, 9, 12])) & (df.index.day >= 28)
        ).astype(int)

        # Wochenanfang/Wochenende
        features["time_week_start"] = (df.index.dayofweek == 0).astype(int)  # Monday
        features["time_week_end"] = (df.index.dayofweek == 4).astype(int)    # Friday

        # === Year Progress ===
        # Linearer Fortschritt durch das Jahr (0-1)
        day_of_year = df.index.dayofyear
        days_in_year = 365 + df.index.is_leap_year.astype(int)
        features["time_year_progress"] = day_of_year / days_in_year

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        # Even though time features don't use OHLC, we shift for consistency
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            # Intraday Raw
            "time_hour", "time_day",
            # Intraday Encoded
            "time_hour_sin", "time_hour_cos",
            "time_day_sin", "time_day_cos",
            # Trading Sessions
            "time_session_asia", "time_session_london", "time_session_ny",
            "time_session_overlap",
            # Saisonalität Raw
            "season_month", "season_quarter", "season_week", "season_dayofmonth",
            # Saisonalität Encoded
            "season_month_sin", "season_month_cos",
            "season_quarter_sin", "season_quarter_cos",
            "season_week_sin", "season_week_cos",
            "season_dayofmonth_sin", "season_dayofmonth_cos",
            # Special Calendar
            "time_month_start", "time_month_end", "time_quarter_end",
            "time_week_start", "time_week_end",
            # Year Progress
            "time_year_progress",
        ]

    def get_signal_columns(self) -> List[str]:
        return [
            "time_session_asia",
            "time_session_london",
            "time_session_ny",
            "time_session_overlap",
            "time_month_start",
            "time_month_end",
            "time_quarter_end",
            "time_week_start",
            "time_week_end",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "include_raw": True,
            "include_encoded": True,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "include_raw": {
                "type": "bool",
                "default": True,
                "description": "Include raw (integer) time features: hour of day (0-23), day of week (0-6), month (1-12), quarter (1-4), ISO week (1-52), day of month (1-31). Useful for tree-based models that can split on ordinal values directly.",
            },
            "include_encoded": {
                "type": "bool",
                "default": True,
                "description": "Include sin/cos cyclical encoding of time features. Maps periodic values (hour, day, month, etc.) onto a unit circle so that e.g. hour 23 and hour 0 are close together. Essential for neural networks and linear models that cannot learn cyclical patterns from raw integers.",
            },
        }


__all__ = ["TimeSeasonIndicators"]
