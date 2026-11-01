"""Z-Score Indicator Plugin.

Distance of the close from its rolling mean in rolling standard deviations,
for mean-reversion rules: enter beyond +/- a threshold, exit back at the mean.

Columns per lookback ``L`` (unshifted: each uses the bar's own close and is meant
for a decision at that close, e.g. ``entry_delay=0``; NaN until ``L`` bars exist):

- ``zscore_L``: (x - mean_L(x)) / std_L(x), x = log close (or close)
- ``zscore_L_above_mean``: 1 when z >= 0, an exit for longs via ``signal_exit``
- ``zscore_L_below_mean``: 1 when z <= 0, an exit for shorts via ``signal_exit``
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


def _columns(lookbacks: List[int]) -> List[str]:
    return [c for lb in lookbacks for c in (f"zscore_{lb}", f"zscore_{lb}_above_mean", f"zscore_{lb}_below_mean")]


@register_indicator("zscore")
class ZscoreIndicator(BaseIndicator):
    """Rolling z-score of the close with back-to-mean exit flags."""

    name = "zscore"
    version = "1.0.0"
    group = "statistics"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, lookbacks=(20,), use_log: bool = True, **params) -> pd.DataFrame:
        x = np.log(df["C"].where(df["C"] > 0)) if use_log else df["C"]
        out = {}
        for lb in _lookbacks(lookbacks):
            std = x.rolling(lb).std()
            z = (x - x.rolling(lb).mean()) / std.where(std > 0)
            out[f"zscore_{lb}"] = z
            out[f"zscore_{lb}_above_mean"] = (z >= 0).astype(float).where(z.notna())
            out[f"zscore_{lb}_below_mean"] = (z <= 0).astype(float).where(z.notna())
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    def get_feature_columns(self, params: Optional[dict] = None) -> List[str]:
        return _columns(_lookbacks((params or {}).get("lookbacks", [20])))

    def get_signal_columns(self, params: Optional[dict] = None) -> List[str]:
        return [c for c in self.get_feature_columns(params) if c.endswith("_mean")]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"lookbacks": [20], "use_log": True}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "lookbacks": {"type": "list[int]", "default": [20], "min": 2, "max": 1000,
                          "description": "Rolling windows (bars) for mean and standard deviation; one column set each"},
            "use_log": {"type": "bool", "default": True,
                        "description": "Use the log close (relative moves) instead of the close"},
        }


__all__ = ["ZscoreIndicator"]
