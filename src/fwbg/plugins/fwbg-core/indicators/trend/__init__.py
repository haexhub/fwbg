"""
Trend Indicator Plugin.

Enthält: ADX, EMA, SMA, MACD, CCI, Aroon, Efficiency Ratio, Supertrend.
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide
from fwbg.core import register_indicator


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14, multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend Indikator.

    ATR-basierter Trend-Filter der weniger noisy ist als Parabolic SAR.
    Gibt +1 (Uptrend) oder -1 (Downtrend) zurück.
    """
    atr = ta.volatility.average_true_range(high, low, close, window=period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(close)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = uptrend, -1 = downtrend

    final_upper = upper_band.values.copy()
    final_lower = lower_band.values.copy()

    for i in range(1, n):
        # Adjust bands: band can only move in trend direction
        if final_lower[i] < final_lower[i - 1] and close.iloc[i - 1] > final_lower[i - 1]:
            final_lower[i] = final_lower[i - 1]
        if final_upper[i] > final_upper[i - 1] and close.iloc[i - 1] < final_upper[i - 1]:
            final_upper[i] = final_upper[i - 1]

        # Direction logic
        if direction[i - 1] == 1:  # was uptrend
            if close.iloc[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:  # was downtrend
            if close.iloc[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    return pd.Series(direction, index=close.index)


@register_indicator("trend")
class TrendIndicators(BaseIndicator):
    """
    Trend-Indikatoren für Trading-Strategien.

    Features:
    - ADX (7, 14, 21 Perioden)
    - EMA Distanz (8, 21, 50, 100, 200 Perioden)
    - SMA Distanz (20, 50, 200 Perioden)
    - MACD
    - CCI (14, 20 Perioden)
    - Aroon Up/Down
    - Kaufman's Efficiency Ratio
    - Supertrend (ATR-basierter Trend-Filter)
    """

    # Required attributes
    name = "trend"
    version = "3.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        adx_periods: List[int] = None,
        ema_periods: List[int] = None,
        sma_periods: List[int] = None,
        supertrend_period: int = 14,
        supertrend_multiplier: float = 3.0,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet alle Trend-Indikatoren.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            adx_periods: ADX-Perioden (default: [7, 14, 21])
            ema_periods: EMA-Perioden (default: [8, 21, 50, 100, 200])
            sma_periods: SMA-Perioden (default: [20, 50, 200])
            supertrend_period: ATR-Periode für Supertrend (default: 14)
            supertrend_multiplier: ATR-Multiplikator für Supertrend (default: 3.0)
        """
        if adx_periods is None:
            adx_periods = [7, 14, 21]
        if ema_periods is None:
            ema_periods = [8, 21, 50, 100, 200]
        if sma_periods is None:
            sma_periods = [20, 50, 200]

        features = {}

        # ADX
        for period in adx_periods:
            features[f"trend_adx_{period}"] = ta.trend.adx(
                df["H"], df["L"], df["C"], window=period
            )

        # EMA Distanz
        for period in ema_periods:
            ema = ta.trend.ema_indicator(df["C"], window=period)
            features[f"trend_ema_dist_{period}"] = safe_divide(df["C"] - ema, df["C"])

        # SMA Distanz
        for period in sma_periods:
            sma = ta.trend.sma_indicator(df["C"], window=period)
            features[f"trend_sma_dist_{period}"] = safe_divide(df["C"] - sma, df["C"])

        # MACD
        macd = ta.trend.MACD(df["C"])
        features["trend_macd"] = safe_divide(macd.macd_diff(), df["C"])
        features["trend_macd_signal"] = safe_divide(macd.macd_signal(), df["C"])

        # CCI
        for period in [14, 20]:
            features[f"trend_cci_{period}"] = ta.trend.cci(
                df["H"], df["L"], df["C"], window=period
            )

        # Aroon
        aroon = ta.trend.AroonIndicator(df["H"], df["L"], window=25)
        features["trend_aroon_up"] = aroon.aroon_up()
        features["trend_aroon_down"] = aroon.aroon_down()

        # Kaufman's Efficiency Ratio
        for period in [10, 20, 50]:
            change = abs(df["C"] - df["C"].shift(period))
            volatility = abs(df["C"].diff()).rolling(period).sum()
            features[f"trend_er_{period}"] = safe_divide(change, volatility)

        # ER Change
        features["trend_er_10_chg"] = features["trend_er_10"] - features["trend_er_10"].shift(5)
        features["trend_er_20_chg"] = features["trend_er_20"] - features["trend_er_20"].shift(10)

        # Supertrend
        st_direction = _supertrend(
            df["H"], df["L"], df["C"],
            period=supertrend_period, multiplier=supertrend_multiplier,
        )
        features["trend_supertrend"] = st_direction
        # Supertrend Flip: 1 wenn gerade gewechselt, sonst 0
        features["trend_supertrend_flip"] = (st_direction != st_direction.shift(1)).astype(float)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        """Gibt Liste aller Trend-Feature-Spalten zurück."""
        return [
            # ADX
            "trend_adx_7", "trend_adx_14", "trend_adx_21",
            # EMA
            "trend_ema_dist_8", "trend_ema_dist_21", "trend_ema_dist_50",
            "trend_ema_dist_100", "trend_ema_dist_200",
            # SMA
            "trend_sma_dist_20", "trend_sma_dist_50", "trend_sma_dist_200",
            # MACD
            "trend_macd", "trend_macd_signal",
            # CCI
            "trend_cci_14", "trend_cci_20",
            # Aroon
            "trend_aroon_up", "trend_aroon_down",
            # Efficiency Ratio
            "trend_er_10", "trend_er_20", "trend_er_50",
            "trend_er_10_chg", "trend_er_20_chg",
            # Supertrend
            "trend_supertrend", "trend_supertrend_flip",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "adx_periods": [7, 14, 21],
            "ema_periods": [8, 21, 50, 100, 200],
            "sma_periods": [20, 50, 200],
            "supertrend_period": 14,
            "supertrend_multiplier": 3.0,
        }


__all__ = ["TrendIndicators"]
