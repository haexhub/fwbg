"""
CCI (Commodity Channel Index) Indicator.

Misst die Abweichung des Typical Price von seinem SMA.
Overbought: CCI > 100. Oversold: CCI < -100.
"""
from typing import List

import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


@register_indicator("cci")
class CCIIndicator(BaseIndicator):
    """
    CCI-Indikator für Mean-Reversion und Momentum.

    Features:
    - cci_{period}: CCI-Wert für jede konfigurierte Periode
    """

    name = "cci"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        periods: list[int] = None,
        **params,
    ) -> pd.DataFrame:
        if periods is None:
            periods = [14, 20]

        features = {}
        for period in periods:
            features[f"cci_{period}"] = ta.trend.cci(
                df["H"], df["L"], df["C"], window=period
            )

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        p = {**self.get_default_params(), **(params or {})}
        return [f"cci_{period}" for period in p.get("periods", [14, 20])]

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        return []

    @classmethod
    def get_default_params(cls) -> dict:
        return {"periods": [14, 20]}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "periods": {
                "type": "list[int]",
                "default": [14, 20],
                "description": "Periods for CCI calculation. CCI measures deviation of Typical Price from its SMA. Values > 100 indicate overbought, < -100 oversold.",
                "min": 2,
                "max": 500,
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"cci": "CCI (Commodity Channel Index)"}


__all__ = ["CCIIndicator"]
