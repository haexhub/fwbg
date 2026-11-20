"""
ADX (Average Directional Index) Indicator.

Misst die Trendstärke auf einer Skala von 0-100, unabhängig von der Richtung.
"""

from typing import List

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


def _wilder_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Wilder ADX with a NaN warm-up.

    The first value needs ``2 * period`` bars (smoothed +DI/-DI from bar
    ``period``, then the mean of ``period`` DX values), so rows
    ``0 .. 2 * period - 2`` are NaN. The ``ta`` library fills them with 0.0 and
    raises on shorter input; a NaN anywhere in its input also turns every later
    value into NaN. Bars with a NaN high, low or close are therefore left out of
    the recursion (the missing bar is bridged, not interpolated) and stay NaN in
    the result.
    """
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}")

    out = np.full(len(close), np.nan)
    valid = (high.notna() & low.notna() & close.notna()).to_numpy()
    if valid.sum() < 2 * period:
        return pd.Series(out, index=close.index)

    h, lo, c = high[valid], low[valid], close[valid]
    adx = ta.trend.adx(h, lo, c, window=period).to_numpy().copy()
    adx[: 2 * period - 1] = np.nan
    out[valid] = adx
    return pd.Series(out, index=close.index)


@register_indicator("adx")
class ADXIndicator(BaseIndicator):
    """
    ADX-Indikator für Trendstärke-Messung.

    Features:
    - adx_{period}: ADX-Wert (0-100) für jede konfigurierte Periode
    """

    name = "adx"
    version = "1.1.0"

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
            features[f"adx_{period}"] = _wilder_adx(df["H"], df["L"], df["C"], period)

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
