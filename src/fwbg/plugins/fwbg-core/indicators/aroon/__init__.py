"""
Aroon Indicator.

Misst die Zeit seit dem letzten N-Perioden Hoch/Tief.
Range: 0-100. Hoher Aroon Up = kürzlich neues Hoch = Aufwärtstrend.
"""

from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


def _aroon(high: pd.Series, low: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Aroon Up/Down measured from the most recent extreme.

    Over the last ``period + 1`` bars (current bar included):
    ``up = 100 * (period - bars_since_highest_high) / period``, likewise for the
    lowest low. When the extreme is tied, the most recent bar holding it counts
    (the ``ta`` library takes the oldest, which understates the signal on flat or
    integer-valued prices). The first ``period`` rows are NaN, as is any row whose
    window contains a NaN.
    """
    if period < 1:
        raise ValueError(f"Aroon period must be >= 1, got {period}")

    def _since_extreme(values: np.ndarray, pick) -> np.ndarray:
        out = np.full(len(values), np.nan)
        if len(values) <= period:
            return out
        windows = np.lib.stride_tricks.sliding_window_view(values, period + 1)
        # argmax/argmin on the reversed window: first hit == most recent bar.
        bars_since = pick(windows[:, ::-1], axis=1)
        result = 100.0 * (period - bars_since) / period
        result[np.isnan(windows).any(axis=1)] = np.nan
        out[period:] = result
        return out

    up = _since_extreme(high.to_numpy(dtype=float), np.argmax)
    down = _since_extreme(low.to_numpy(dtype=float), np.argmin)
    return pd.Series(up, index=high.index), pd.Series(down, index=low.index)


@register_indicator("aroon")
class AroonIndicator(BaseIndicator):
    """
    Aroon-Indikator für Trend-Erkennung.

    Features:
    - aroon_up: Aroon Up (0-100)
    - aroon_down: Aroon Down (0-100)
    """

    name = "aroon"
    version = "1.1.0"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 25,
        **params,
    ) -> pd.DataFrame:
        aroon_up, aroon_down = _aroon(df["H"], df["L"], period)

        features = {
            "aroon_up": aroon_up,
            "aroon_down": aroon_down,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return ["aroon_up", "aroon_down"]

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        return []

    @classmethod
    def get_default_params(cls) -> dict:
        return {"period": 25}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "period": {
                "type": "int",
                "default": 25,
                "min": 5,
                "max": 200,
                "step": 1,
                "description": "Lookback period for Aroon Up/Down. Measures how many bars since the highest high (Aroon Up) or lowest low (Aroon Down) within the window.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"aroon": "Aroon"}


__all__ = ["AroonIndicator"]
