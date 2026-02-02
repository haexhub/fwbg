"""
Trend Indicator Plugin.

Enthält: ADX, EMA, SMA, MACD, CCI, Aroon, Efficiency Ratio.
"""
from typing import List, TYPE_CHECKING
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator

if TYPE_CHECKING:
    pass


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
    """

    group = "trend"

    def compute(
        self,
        df: pd.DataFrame,
        adx_periods: List[int] = None,
        ema_periods: List[int] = None,
        sma_periods: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet alle Trend-Indikatoren.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            adx_periods: ADX-Perioden (default: [7, 14, 21])
            ema_periods: EMA-Perioden (default: [8, 21, 50, 100, 200])
            sma_periods: SMA-Perioden (default: [20, 50, 200])

        Returns:
            DataFrame mit Trend-Features
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
            features[f"trend_ema_dist_{period}"] = (df["C"] - ema) / df["C"]

        # SMA Distanz
        for period in sma_periods:
            sma = ta.trend.sma_indicator(df["C"], window=period)
            features[f"trend_sma_dist_{period}"] = (df["C"] - sma) / df["C"]

        # MACD
        macd = ta.trend.MACD(df["C"])
        features["trend_macd"] = macd.macd_diff() / df["C"]
        features["trend_macd_signal"] = macd.macd_signal() / df["C"]

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
            features[f"trend_er_{period}"] = change / (volatility + 1e-10)

        # ER Change
        features["trend_er_10_chg"] = features["trend_er_10"] - features["trend_er_10"].shift(5)
        features["trend_er_20_chg"] = features["trend_er_20"] - features["trend_er_20"].shift(10)

        # Concat all features at once
        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

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
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "adx_periods": [7, 14, 21],
            "ema_periods": [8, 21, 50, 100, 200],
            "sma_periods": [20, 50, 200],
        }


__all__ = ["TrendIndicators"]
