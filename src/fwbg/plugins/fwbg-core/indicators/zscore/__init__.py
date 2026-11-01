"""Z-Score Indicator Plugin.

Distance of the close from its rolling mean in rolling standard deviations,
for mean-reversion rules: enter beyond +/- a threshold, exit back at the mean.

Columns per lookback ``L`` (unshifted: each uses the bar's own close and is meant
for a decision at that close, e.g. ``entry_delay=0``; NaN until ``L`` bars exist):

- ``zscore_L``: (x - mean_L(x)) / std_L(x), x = log close (or close)
- ``zscore_L_above_mean``: 1 when z >= 0, an exit for longs via ``signal_exit``
- ``zscore_L_below_mean``: 1 when z <= 0, an exit for shorts via ``signal_exit``

With ``entry_z`` set (one lookback only), four columns whose names do not depend
on the lookback, so ``optimization.indicator_grid`` can vary lookback and threshold
under one set of signal rules:

- ``zscore_entry_long`` / ``zscore_entry_short``: 1 when z < -entry_z / z > entry_z
- ``zscore_exit_long`` / ``zscore_exit_short``: 1 when z >= 0 / z <= 0
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator


def _lookbacks(value) -> List[int]:
    values = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 2:
            raise ValueError(f"zscore: lookbacks must be integers >= 2, not {value!r}")
        out.append(int(v))
    return out


ENTRY_COLUMNS = ["zscore_entry_long", "zscore_entry_short", "zscore_exit_long", "zscore_exit_short"]


def _entry_z(value, lookbacks: List[int]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(f"zscore: entry_z must be a positive number, not {value!r}")
    if len(lookbacks) != 1:
        raise ValueError(f"zscore: entry_z needs exactly one lookback, not {lookbacks!r}")
    return float(value)


def _columns(lookbacks: List[int], entry_z=None) -> List[str]:
    cols = [c for lb in lookbacks for c in (f"zscore_{lb}", f"zscore_{lb}_above_mean", f"zscore_{lb}_below_mean")]
    return cols + (ENTRY_COLUMNS if _entry_z(entry_z, lookbacks) is not None else [])


@register_indicator("zscore")
class ZscoreIndicator(BaseIndicator):
    """Rolling z-score of the close with back-to-mean exit flags."""

    name = "zscore"
    version = "1.1.0"
    group = "statistics"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, lookbacks=(20,), use_log: bool = True, entry_z=None,
                **params) -> pd.DataFrame:
        x = np.log(df["C"].where(df["C"] > 0)) if use_log else df["C"]
        lbs = _lookbacks(lookbacks)
        threshold = _entry_z(entry_z, lbs)
        out = {}
        for lb in lbs:
            std = x.rolling(lb).std()
            z = (x - x.rolling(lb).mean()) / std.where(std > 0)
            out[f"zscore_{lb}"] = z
            out[f"zscore_{lb}_above_mean"] = (z >= 0).astype(float).where(z.notna())
            out[f"zscore_{lb}_below_mean"] = (z <= 0).astype(float).where(z.notna())
            if threshold is not None:
                out["zscore_entry_long"] = (z < -threshold).astype(float).where(z.notna())
                out["zscore_entry_short"] = (z > threshold).astype(float).where(z.notna())
                out["zscore_exit_long"] = out[f"zscore_{lb}_above_mean"]
                out["zscore_exit_short"] = out[f"zscore_{lb}_below_mean"]
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    def get_feature_columns(self, params: Optional[dict] = None) -> List[str]:
        params = params or {}
        return _columns(_lookbacks(params.get("lookbacks", [20])), params.get("entry_z"))

    def get_signal_columns(self, params: Optional[dict] = None) -> List[str]:
        return [c for c in self.get_feature_columns(params) if c.endswith("_mean") or c in ENTRY_COLUMNS]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"lookbacks": [20], "use_log": True, "entry_z": None}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lookbacks": {"type": "list[int]", "default": [20], "min": 2, "max": 1000,
                          "description": "Rolling windows (bars) for mean and standard deviation; one column set each"},
            "use_log": {"type": "bool", "default": True,
                        "description": "Use the log close (relative moves) instead of the close"},
            "entry_z": {"type": "float", "default": None, "min": 0,
                        "description": "Entry threshold in standard deviations; adds zscore_entry_*/zscore_exit_* "
                                       "columns (needs exactly one lookback)"},
        }


__all__ = ["ZscoreIndicator", "ENTRY_COLUMNS"]
