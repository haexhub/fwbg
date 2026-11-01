"""Internal Bar Strength (IBS) Indicator Plugin.

IBS = (close - low) / (high - low): 0 means the bar closed at its low, 1 at its
high. Closes near the low of a session bar tend to revert over the next sessions
in equity indices (Pagonidis 2013).

Columns (unshifted: each uses the bar's own close and is meant for a decision at
that close, e.g. ``entry_delay=0``; the first bar has no previous bar and is NaN):

- ``ibs``: close location in the bar's range (NaN for a bar without range)
- ``ibs_down``: 1 when the close is below the previous close
- ``ibs_above_prev_high``: 1 when the close is above the previous bar's high,
  a rebound exit for the ``signal_exit`` exit strategy
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

IBS_COLUMNS = ["ibs", "ibs_down", "ibs_above_prev_high"]


@register_indicator("ibs")
class IbsIndicator(BaseIndicator):
    """Close location within the bar's range plus down-close and rebound flags."""

    name = "ibs"
    version = "1.0.0"
    group = "price_action"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        bar_range = df["H"] - df["L"]
        out = pd.DataFrame(index=df.index)
        out["ibs"] = (df["C"] - df["L"]) / bar_range.where(bar_range > 0)
        out["ibs_down"] = (df["C"] < df["C"].shift(1)).astype(float)
        out["ibs_above_prev_high"] = (df["C"] > df["H"].shift(1)).astype(float)
        if len(out):
            out.iloc[0] = np.nan
        return pd.concat([df, out], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return list(IBS_COLUMNS)

    def get_signal_columns(self, params=None) -> List[str]:
        return ["ibs_down", "ibs_above_prev_high"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {}


__all__ = ["IbsIndicator", "IBS_COLUMNS"]
