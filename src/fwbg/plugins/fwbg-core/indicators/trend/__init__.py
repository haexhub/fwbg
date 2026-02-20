"""
Trend Indicator Plugin.

Enthält: ADX, EMA, SMA, MACD, CCI, Aroon, Efficiency Ratio, Supertrend.
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


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
        macd_only: bool = False,
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
            macd_only: Wenn True, werden nur MACD-Features berechnet (default: False)
        """
        if adx_periods is None:
            adx_periods = [7, 14, 21]
        if ema_periods is None:
            ema_periods = [8, 21, 50, 100, 200]
        if sma_periods is None:
            sma_periods = [20, 50, 200]

        features = {}

        if not macd_only:
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
        macd_ind = ta.trend.MACD(df["C"])
        macd_line = macd_ind.macd()
        macd_hist = macd_ind.macd_diff()

        # Existing: histogram (MACD line - Signal line) and signal line, normalized
        features["trend_macd"] = safe_divide(macd_hist, df["C"])
        features["trend_macd_signal"] = safe_divide(macd_ind.macd_signal(), df["C"])

        # MACD line itself (fast EMA - slow EMA), normalized — needed for zero-line filter
        features["trend_macd_line"] = safe_divide(macd_line, df["C"])
        # Zero-line side: +1 bullish bias, -1 bearish bias (System 1: zero-line rule)
        features["trend_macd_above_zero"] = np.sign(macd_line)
        # Absolute distance from zero, normalized (System 1: distance rule — far = stronger)
        features["trend_macd_dist_zero"] = safe_divide(macd_line.abs(), df["C"])
        # Histogram flip: 1 when histogram just crossed zero (crossover signal)
        features["trend_macd_hist_flip"] = (np.sign(macd_hist) != np.sign(macd_hist.shift(1))).astype(float)

        if not macd_only:
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
            "trend_macd_line", "trend_macd_above_zero",
            "trend_macd_dist_zero", "trend_macd_hist_flip",
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

    def get_signal_columns(self) -> List[str]:
        return ["trend_supertrend", "trend_supertrend_flip"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "adx_periods": [7, 14, 21],
            "ema_periods": [8, 21, 50, 100, 200],
            "sma_periods": [20, 50, 200],
            "supertrend_period": 14,
            "supertrend_multiplier": 3.0,
            "macd_only": False,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "adx_periods": {
                "type": "list[int]",
                "default": [7, 14, 21],
                "description": "Periods for ADX (Average Directional Index) calculation. ADX measures trend strength on a 0-100 scale regardless of direction. Shorter periods react faster to trend changes, longer periods smooth out noise.",
                "min": 2,
                "max": 500,
            },
            "ema_periods": {
                "type": "list[int]",
                "default": [8, 21, 50, 100, 200],
                "description": "Periods for EMA (Exponential Moving Average) distance features. Measures how far the current price deviates from each EMA as a percentage. Short EMAs (8, 21) capture immediate momentum, long EMAs (100, 200) capture macro trend positioning.",
                "min": 2,
                "max": 1000,
            },
            "sma_periods": {
                "type": "list[int]",
                "default": [20, 50, 200],
                "description": "Periods for SMA (Simple Moving Average) distance features. Similar to EMA distances but with equal weighting of all bars in the window. Classic levels like 50 and 200 are widely watched by institutional traders.",
                "min": 2,
                "max": 1000,
            },
            "supertrend_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period for the Supertrend indicator. Controls sensitivity of the ATR-based trend-following bands. Lower values make Supertrend more responsive but noisier.",
                "min": 2,
                "max": 500,
                "step": 1,
            },
            "supertrend_multiplier": {
                "type": "float",
                "default": 3.0,
                "description": "ATR multiplier for Supertrend band width. Higher values create wider bands requiring larger moves to trigger trend flips, reducing whipsaws but increasing lag.",
                "min": 0.5,
                "max": 20.0,
                "step": 0.5,
            },
            "macd_only": {
                "type": "bool",
                "default": False,
                "description": "When True, only MACD features are computed (trend_macd_line, trend_macd_above_zero, trend_macd_dist_zero, trend_macd_hist_flip, trend_macd, trend_macd_signal). All other features (ADX, EMA, SMA, CCI, Aroon, ER, Supertrend) are skipped. Use for isolated MACD strategy falsification.",
            },
        }


__all__ = ["TrendIndicators"]
