"""
Aroon Indicator.

Misst die Zeit seit dem letzten N-Perioden Hoch/Tief.
Range: 0-100. Hoher Aroon Up = kürzlich neues Hoch = Aufwärtstrend.
"""
from typing import List

import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


@register_indicator("aroon")
class AroonIndicator(BaseIndicator):
    """
    Aroon-Indikator für Trend-Erkennung.

    Features:
    - aroon_up: Aroon Up (0-100)
    - aroon_down: Aroon Down (0-100)
    """

    name = "aroon"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 25,
        **params,
    ) -> pd.DataFrame:
        aroon = ta.trend.AroonIndicator(df["H"], df["L"], window=period)

        features = {
            "aroon_up": aroon.aroon_up(),
            "aroon_down": aroon.aroon_down(),
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
