"""
ADX (Average Directional Index) Indicator.

Misst die Trendstärke auf einer Skala von 0-100, unabhängig von der Richtung.
"""
from typing import List

import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


@register_indicator("adx")
class ADXIndicator(BaseIndicator):
    """
    ADX-Indikator für Trendstärke-Messung.

    Features:
    - adx_{period}: ADX-Wert (0-100) für jede konfigurierte Periode
    """

    name = "adx"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        periods: list[int] = None,
        **params,
    ) -> pd.DataFrame:
        if periods is None:
            periods = [7, 14, 21]

        features = {}
        for period in periods:
            features[f"adx_{period}"] = ta.trend.adx(
                df["H"], df["L"], df["C"], window=period
            )

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        p = {**self.get_default_params(), **(params or {})}
        return [f"adx_{period}" for period in p.get("periods", [7, 14, 21])]

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        return []

    @classmethod
    def get_default_params(cls) -> dict:
        return {"periods": [7, 14, 21]}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "periods": {
                "type": "list[int]",
                "default": [7, 14, 21],
                "description": "Periods for ADX calculation. ADX measures trend strength on a 0-100 scale regardless of direction. Shorter periods react faster, longer periods smooth out noise.",
                "min": 2,
                "max": 500,
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"adx": "ADX (Average Directional Index)"}


__all__ = ["ADXIndicator"]
