"""
Efficiency Ratio (Kaufman) Indicator.

ER = |Price Change over N| / Sum of |Bar-to-Bar Changes| over N.
Wert nahe 1 = geradliniger Trend, nahe 0 = choppy/seitwärts.
"""
from typing import List

import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


@register_indicator("efficiency_ratio")
class EfficiencyRatioIndicator(BaseIndicator):
    """
    Kaufman's Efficiency Ratio für Trend-Qualitäts-Messung.

    Features:
    - er_{period}: Efficiency Ratio (0-1) für jede Periode
    - er_{period}_chg: Veränderung des ER über halbe Periode
    """

    name = "efficiency_ratio"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        periods: list[int] = None,
        **params,
    ) -> pd.DataFrame:
        if periods is None:
            periods = [10, 20, 50]

        features = {}
        for period in periods:
            change = abs(df["C"] - df["C"].shift(period))
            volatility = abs(df["C"].diff()).rolling(period).sum()
            er = safe_divide(change, volatility)
            features[f"er_{period}"] = er
            features[f"er_{period}_chg"] = er - er.shift(period // 2)

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        p = {**self.get_default_params(), **(params or {})}
        periods = p.get("periods", [10, 20, 50])
        cols = []
        for period in periods:
            cols.append(f"er_{period}")
            cols.append(f"er_{period}_chg")
        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        return []

    def get_overlay_columns(self, params=None) -> List[str]:
        return []

    @classmethod
    def get_default_params(cls) -> dict:
        return {"periods": [10, 20, 50]}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "periods": {
                "type": "list[int]",
                "default": [10, 20, 50],
                "description": "Periods for Efficiency Ratio. ER near 1 = clean trend, near 0 = choppy sideways. Change features (er_N_chg) measure ER momentum over half the period.",
                "min": 2,
                "max": 500,
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"er": "Efficiency Ratio (Kaufman)"}


__all__ = ["EfficiencyRatioIndicator"]
