"""
Calendar anomaly features — systematic market effects tied to known calendar events.

Provides deterministic features derived from the datetime index that capture
well-documented calendar anomalies: turn-of-month effect, quarter-end rebalancing,
options expiry, FOMC cycle, NFP week, January effect, and more.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, register_indicator


def _third_friday(year: int, month: int) -> int:
    """Return the day-of-month of the 3rd Friday for given year/month."""
    # 1st day of the month
    first = pd.Timestamp(year, month, 1)
    # dayofweek: Monday=0 .. Friday=4
    offset = (4 - first.dayofweek) % 7
    # First Friday
    first_friday = 1 + offset
    # Third Friday = first Friday + 14
    return first_friday + 14


def _days_in_month(ts: pd.Timestamp) -> int:
    """Return number of days in the month of the given timestamp."""
    return ts.days_in_month


@register_indicator("calendar_events")
class CalendarEventsIndicator(BaseIndicator):
    """Calendar-based anomaly features for ML trading."""

    name = "calendar_events"
    version = "1.0.0"

    _FEATURES = [
        "cal_turn_of_month",
        "cal_days_to_month_end",
        "cal_quarter_end",
        "cal_triple_witching",
        "cal_monthly_opex",
        "cal_fomc_proximity",
        "cal_nfp_week",
        "cal_week_of_month",
        "cal_year_boundary",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        include_proximity: bool = True,
        include_binary: bool = True,
        **params,
    ) -> pd.DataFrame:
        idx = df.index
        n = len(idx)

        day = idx.day
        month = idx.month
        days_in_month = idx.days_in_month
        day_of_year = idx.dayofyear

        features: dict = {}

        # --- Binary event features ---
        if include_binary:
            # Turn-of-Month: last 2 or first 3 calendar days
            features["cal_turn_of_month"] = np.where(
                (day <= 3) | (day >= days_in_month - 1), 1.0, 0.0
            )

            # Quarter-end: last 5 calendar days of Mar, Jun, Sep, Dec
            is_quarter_month = np.isin(month, [3, 6, 9, 12])
            features["cal_quarter_end"] = np.where(
                is_quarter_month & (day > days_in_month - 5), 1.0, 0.0
            )

            # Triple Witching: within 2 days of 3rd Friday of Mar/Jun/Sep/Dec
            triple_witching = np.zeros(n)
            for i in range(n):
                ts = idx[i]
                if ts.month in (3, 6, 9, 12):
                    tf_day = _third_friday(ts.year, ts.month)
                    if abs(ts.day - tf_day) <= 2:
                        triple_witching[i] = 1.0
            features["cal_triple_witching"] = triple_witching

            # Monthly OpEx: within 2 days of 3rd Friday of any month
            monthly_opex = np.zeros(n)
            for i in range(n):
                ts = idx[i]
                tf_day = _third_friday(ts.year, ts.month)
                if abs(ts.day - tf_day) <= 2:
                    monthly_opex[i] = 1.0
            features["cal_monthly_opex"] = monthly_opex

            # NFP week: first 5 calendar days of any month
            features["cal_nfp_week"] = np.where(day <= 5, 1.0, 0.0)

            # Year boundary: last 5 days of Dec or first 5 days of Jan
            features["cal_year_boundary"] = np.where(
                ((month == 12) & (day > 26)) | ((month == 1) & (day <= 5)),
                1.0,
                0.0,
            )

        # --- Continuous proximity features ---
        if include_proximity:
            # Days to month end, normalized 0..1
            features["cal_days_to_month_end"] = (
                (days_in_month - day) / days_in_month
            ).astype(np.float64)

            # FOMC proximity: cyclic approximation (~8 meetings/year, ~46 day cycle)
            features["cal_fomc_proximity"] = np.sin(
                2 * np.pi * day_of_year / 46
            ).astype(np.float64)

            # Week of month (1-5), normalized to 0..1
            features["cal_week_of_month"] = (
                ((day - 1) // 7) / 4.0
            ).astype(np.float64)

        features_df = shift_features(features, idx)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "include_proximity": True,
            "include_binary": True,
        }


__all__ = ["CalendarEventsIndicator"]
