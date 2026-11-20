"""
Calendar anomaly features — systematic market effects tied to known calendar events.

Provides deterministic features derived from the datetime index that capture
well-documented calendar anomalies: turn-of-month effect, quarter-end rebalancing,
options expiry, NFP week, January effect, and more.

Timing convention: every column describes the bar's OWN date (its index
timestamp), not the previous bar's. The calendar is fixed in advance, so the value
for a date is known long before the bar opens and cannot leak information from the
future; shifting it by one bar (``shift_features``) would only make Monday's row
describe Friday. Unlike price-derived features, these columns are therefore not
shifted. Only the first row is set to NaN, as for every indicator. Intraday bars
carry the values of their date on every bar of the day.

Trading-day rules (turn of month) use the exchange calendar of the ``calendar``
parameter (default ``NYSE``, holidays from ``holidays.financial_holidays``);
``"weekdays"`` treats every Monday-Friday as a trading day.
"""

from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator


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


def _turn_of_month_module():
    from fwbg.plugins import import_plugin_module

    return import_plugin_module("fwbg-core", "indicators", "turn_of_month")


def _calendar_choices() -> list:
    return list(_turn_of_month_module().CALENDARS)


def nfp_week(dates: pd.DatetimeIndex) -> np.ndarray:
    """1.0 on Monday-Friday of the week that contains the month's first Friday.

    The US employment report is usually published on the first Friday of the
    month. The BLS schedule follows the survey reference week and occasionally
    deviates (a release can fall in the second week or on another weekday, for
    example around holidays), and no exceptions are hard-coded here, so this is a
    rule-based approximation, not the release calendar. Weekends are 0.
    """
    friday = dates.normalize() - pd.to_timedelta(dates.dayofweek, unit="D")
    friday = friday + pd.Timedelta(days=4)
    return np.where((friday.day <= 7) & (dates.dayofweek <= 4), 1.0, 0.0)


@register_indicator("calendar_events")
class CalendarEventsIndicator(BaseIndicator):
    """Calendar-based anomaly features for ML trading."""

    name = "calendar_events"
    version = "1.1.0"

    _FEATURES = [
        "cal_turn_of_month",
        "cal_days_to_month_end",
        "cal_quarter_end",
        "cal_triple_witching",
        "cal_monthly_opex",
        "cal_nfp_week",
        "cal_week_of_month",
        "cal_year_boundary",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        include_proximity: bool = True,
        include_binary: bool = True,
        calendar: str = "NYSE",
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("calendar_events needs a DatetimeIndex")
        idx = df.index
        n = len(idx)

        day = idx.day
        month = idx.month
        days_in_month = idx.days_in_month

        features: dict = {}

        # --- Binary event features ---
        if include_binary:
            # Turn-of-Month: last 2 or first 3 trading days of the month. Counted in
            # trading days so the month's last trading day is always flagged (the
            # last calendar days can be a weekend or a holiday).
            if n:
                dates = idx.tz_localize(None) if idx.tz is not None else idx
                tom_day, days_left = _turn_of_month_module().month_position(
                    calendar, dates.normalize()
                )
            else:
                tom_day = days_left = np.zeros(0)
            features["cal_turn_of_month"] = np.where(
                ((tom_day >= 1) & (tom_day <= 3)) | (days_left <= 1), 1.0, 0.0
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

            # NFP week: Monday-Friday of the week with the month's first Friday
            features["cal_nfp_week"] = nfp_week(idx)

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

            # Week of month (1-5), normalized to 0..1
            features["cal_week_of_month"] = (((day - 1) // 7) / 4.0).astype(np.float64)

        features_df = pd.DataFrame(features, index=idx, dtype=np.float64)
        if n:
            features_df.iloc[0] = np.nan  # indicator contract: the first row is NaN
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return [
            "cal_turn_of_month",
            "cal_quarter_end",
            "cal_triple_witching",
            "cal_monthly_opex",
            "cal_nfp_week",
            "cal_year_boundary",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "include_proximity": True,
            "include_binary": True,
            "calendar": "NYSE",
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "include_proximity": {
                "type": "bool",
                "default": True,
                "description": "Include continuous proximity features: normalized days to month end and normalized week of month. These provide smooth, gradient-friendly signals for models to learn calendar-based positioning.",
            },
            "include_binary": {
                "type": "bool",
                "default": True,
                "description": "Include binary event flags: turn-of-month (last 2 / first 3 trading days), quarter-end, triple witching (quarterly options/futures expiry), monthly OpEx (options expiration), NFP week (Monday-Friday of the week with the month's first Friday; approximation of the employment report week), and year boundary. Each captures a known calendar anomaly in equity and derivatives markets.",
            },
            "calendar": {
                "type": "choice",
                "choices": _calendar_choices(),
                "default": "NYSE",
                "description": "Exchange holiday calendar (holidays.financial_holidays) used to count trading days for cal_turn_of_month, or 'weekdays' for every weekday.",
            },
        }


__all__ = ["CalendarEventsIndicator", "nfp_week"]
