"""Rebalancing-pressure Indicator Plugin.

Fixed-weight funds rebalance their stock/bond mix around month end: after stocks
outperformed bonds during the month they sell stocks in the last days of the
month, after underperformance they buy (Harvey, Mazzoleni & Melone 2025, "The
Unintended Consequences of Rebalancing").

The month-to-date drift is read from daily stock and bond series that a
``data_loading`` step added as columns (e.g. ``macro_spy``, ``macro_tlt``). Each
such column already holds, on every bar, the last close before the bar's date, so
on the month's first trading day it holds the previous month's last close (the
base) and later the close of the trading day before the bar. Trading days come from
an exchange calendar (``turn_of_month``'s ``trading_days``).

Columns (unshifted, for a decision at the bar's close; first row NaN):

- ``rebal_drift``: mean month-to-date return of the stock columns minus that of the
  bond columns (NaN when the month's first trading day is missing from the bars)
- ``rebal_entry_long``: 1 on the bar with ``entry_days_left`` trading days left in
  the month when ``rebal_drift < -threshold`` (stocks lagged: funds buy stocks)
- ``rebal_entry_short``: 1 on that bar when ``rebal_drift > threshold``
- ``rebal_exit``: 1 on the month's last trading day, an exit column for
  ``signal_exit`` (both directions)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

REBAL_COLUMNS = ["rebal_drift", "rebal_entry_long", "rebal_entry_short", "rebal_exit"]
log = logging.getLogger(__name__)


def _trading_days(calendar: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    from fwbg.plugins import import_plugin_module

    tom = import_plugin_module("fwbg-core", "indicators", "turn_of_month")
    return tom.trading_days(calendar, first, last)


@register_indicator("rebalancing_pressure")
class RebalancingPressureIndicator(BaseIndicator):
    """Month-to-date stock-minus-bond drift with month-end entry and exit flags."""

    name = "rebalancing_pressure"
    version = "1.0.0"
    group = "calendar"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        p = {**self.get_default_params(), **params}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("rebalancing_pressure needs a DatetimeIndex")
        out = pd.DataFrame(index=df.index, columns=REBAL_COLUMNS, dtype=float)
        stocks, bonds = list(p["stock_columns"]), list(p["bond_columns"])
        missing = [c for c in stocks + bonds if c not in df.columns]
        if missing:
            log.warning("rebalancing_pressure: missing input columns %s; all values NaN", missing)
        elif len(df):
            dates = (
                df.index.tz_localize(None) if df.index.tz is not None else df.index
            ).normalize()
            days = _trading_days(p["calendar"], dates.min(), dates.max())
            months = days.to_period("M")
            first_day = pd.Series(days, index=days).groupby(months).min()
            last_pos = pd.Series(np.arange(len(days)), index=days).groupby(months).max()
            bar_months = dates.to_period("M")
            pos = days.searchsorted(dates, side="right") - 1
            days_left = last_pos.reindex(bar_months).to_numpy() - pos
            is_first = dates == first_day.reindex(bar_months).to_numpy()
            month_key = pd.Series(bar_months, index=df.index)

            def mtd(cols: list[str]) -> pd.Series:
                values = df[cols].astype(float)
                base = values.where(pd.Series(is_first, index=df.index), np.nan)
                base = base.groupby(month_key).transform("first")
                has_base = pd.Series(is_first, index=df.index).groupby(month_key).transform("any")
                ret = (values / base - 1).where(has_base)
                return ret.mean(axis=1)

            drift = mtd(stocks) - mtd(bonds)
            entry_bar = days_left == p["entry_days_left"]
            out["rebal_drift"] = drift
            out["rebal_entry_long"] = (entry_bar & (drift < -p["threshold"])).astype(float)
            out["rebal_entry_short"] = (entry_bar & (drift > p["threshold"])).astype(float)
            out["rebal_exit"] = (days_left == 0).astype(float)
        if len(out):
            out.iloc[0] = np.nan  # indicator contract: the first row is NaN
        return pd.concat([df, out], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return list(REBAL_COLUMNS)

    def get_signal_columns(self, params=None) -> List[str]:
        return ["rebal_entry_long", "rebal_entry_short", "rebal_exit"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "stock_columns": ["macro_spy"],
            "bond_columns": ["macro_tlt", "macro_ief"],
            "threshold": 0.02,
            "entry_days_left": 3,
            "calendar": "NYSE",
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "stock_columns": {
                "type": "list[string]",
                "default": ["macro_spy"],
                "description": "Daily stock series columns (added by a data_loading step)",
            },
            "bond_columns": {
                "type": "list[string]",
                "default": ["macro_tlt", "macro_ief"],
                "description": "Daily bond series columns (added by a data_loading step)",
            },
            "threshold": {
                "type": "float",
                "default": 0.02,
                "min": 0.0,
                "max": 0.5,
                "description": "Minimum absolute month-to-date drift for an entry",
            },
            "entry_days_left": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 15,
                "description": "Entry flags mark the bar with this many trading days left",
            },
            "calendar": {
                "type": "string",
                "default": "NYSE",
                "description": "Exchange calendar for trading days (see turn_of_month)",
            },
        }


__all__ = ["RebalancingPressureIndicator", "REBAL_COLUMNS"]
