"""
Momentum Indicator Plugin.

Enthält: RSI, Stochastic, Williams %R, Ultimate Oscillator, ROC.
"""

from typing import List

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI.

    The average gain and loss are seeded with the simple mean of the first
    ``period`` price changes and then smoothed with alpha = 1 / period, so the
    first value is at index ``period`` (``period + 1`` bars) and everything
    before it is NaN. ``ta`` instead starts the average at an artificial zero
    change on bar 0 and emits values from ``period - 1`` on.

    With no movement at all (average gain and loss both 0, a flat series) the
    RSI is 50: neither side dominates. Only gains give 100, only losses 0.
    """
    delta = close.diff()
    averages = []
    for part in (delta.clip(lower=0.0), (-delta).clip(lower=0.0)):
        seeded = part.copy()
        seeded.iloc[: period + 1] = np.nan
        if len(part) > period:
            seeded.iloc[period] = part.iloc[1 : period + 1].mean()
        averages.append(seeded.ewm(alpha=1.0 / period, adjust=False).mean())
    avg_gain, avg_loss = averages
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi.where(avg_gain.notna() & avg_loss.notna())


@register_indicator("momentum")
class MomentumIndicators(BaseIndicator):
    """
    Momentum-Indikatoren für Trading-Strategien.

    Features:
    - RSI (7, 14, 21 Perioden)
    - Stochastic K/D (14, 21 Perioden)
    - Williams %R (14, 21 Perioden)
    - Ultimate Oscillator
    - Rate of Change (5, 10, 20 Perioden)
    """

    name = "momentum"
    version = "2.1.0"

    def compute(
        self,
        df: pd.DataFrame,
        rsi_periods: List[int] = None,
        stoch_periods: List[int] = None,
        williams_periods: List[int] = None,
        roc_periods: List[int] = None,
        **params,
    ) -> pd.DataFrame:
        """
        Berechnet alle Momentum-Indikatoren.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            rsi_periods: RSI-Perioden (default: [7, 14, 21])
            stoch_periods: Stochastic-Perioden (default: [14, 21])
            williams_periods: Williams %R Perioden (default: [14, 21])
            roc_periods: ROC-Perioden (default: [5, 10, 20])

        Returns:
            DataFrame mit Momentum-Features
        """
        if rsi_periods is None:
            rsi_periods = [7, 14, 21]
        if stoch_periods is None:
            stoch_periods = [14, 21]
        if williams_periods is None:
            williams_periods = [14, 21]
        if roc_periods is None:
            roc_periods = [5, 10, 20]

        features = {}

        # RSI
        for period in rsi_periods:
            features[f"mom_rsi_{period}"] = _wilder_rsi(df["C"], period)

        # Stochastic
        for period in stoch_periods:
            stoch = ta.momentum.StochasticOscillator(
                df["H"], df["L"], df["C"], window=period
            )
            features[f"mom_stoch_k_{period}"] = stoch.stoch()
            features[f"mom_stoch_d_{period}"] = stoch.stoch_signal()

        # Williams %R
        for period in williams_periods:
            features[f"mom_williams_{period}"] = ta.momentum.williams_r(
                df["H"], df["L"], df["C"], lbp=period
            )

        # Ultimate Oscillator
        features["mom_uo"] = ta.momentum.ultimate_oscillator(df["H"], df["L"], df["C"])

        # Rate of Change
        for period in roc_periods:
            features[f"mom_roc_{period}"] = ta.momentum.roc(df["C"], window=period)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        """Feature columns compute() produces for ``params`` (defaults when None)."""
        params = params or {}
        rsi = params.get("rsi_periods") or [7, 14, 21]
        stoch = params.get("stoch_periods") or [14, 21]
        williams = params.get("williams_periods") or [14, 21]
        roc = params.get("roc_periods") or [5, 10, 20]
        cols = [f"mom_rsi_{p}" for p in rsi]
        for p in stoch:
            cols += [f"mom_stoch_k_{p}", f"mom_stoch_d_{p}"]
        cols += [f"mom_williams_{p}" for p in williams]
        cols.append("mom_uo")
        cols += [f"mom_roc_{p}" for p in roc]
        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "rsi_periods": [7, 14, 21],
            "stoch_periods": [14, 21],
            "williams_periods": [14, 21],
            "roc_periods": [5, 10, 20],
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "rsi_periods": {
                "type": "list[int]",
                "default": [7, 14, 21],
                "description": "Periods for RSI (Relative Strength Index) calculation. RSI oscillates 0-100 measuring the speed and magnitude of price changes. 14 is the classic Wilder period; shorter periods (7) are more sensitive to recent moves, longer periods (21) smoother.",
                "min": 2,
                "max": 500,
            },
            "stoch_periods": {
                "type": "list[int]",
                "default": [14, 21],
                "description": "Lookback periods for Stochastic Oscillator (%K and %D). Measures where price closed relative to its high-low range over N bars. Shorter periods react faster to price swings, longer periods filter out noise.",
                "min": 2,
                "max": 500,
            },
            "williams_periods": {
                "type": "list[int]",
                "default": [14, 21],
                "description": "Lookback periods for Williams %R. Similar to Stochastic but inverted (0 to -100 scale). Identifies overbought (near 0) and oversold (near -100) conditions within the given lookback window.",
                "min": 2,
                "max": 500,
            },
            "roc_periods": {
                "type": "list[int]",
                "default": [5, 10, 20],
                "description": "Periods for Rate of Change (ROC) calculation. Measures the percentage change in price over N bars. Short periods (5) capture immediate momentum, longer periods (20) capture swing momentum.",
                "min": 1,
                "max": 500,
            },
        }

    def get_column_group_labels(self) -> dict:
        return {
            "rsi": "RSI (Relative Strength Index)",
            "stoch": "Stochastic Oscillator",
            "williams": "Williams %R",
            "uo": "Ultimate Oscillator",
            "roc": "Rate of Change",
        }


__all__ = ["MomentumIndicators"]
