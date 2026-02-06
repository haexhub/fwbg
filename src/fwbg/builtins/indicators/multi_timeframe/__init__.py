"""
Multi-Timeframe Indicator Plugin.

Enthält:
- H4 (4-Stunden) Timeframe Features
- D1 (Daily) Timeframe Features
- Trend Alignment zwischen Timeframes
- Volatility Ratio zwischen Timeframes

Multi-Timeframe-Analyse ist entscheidend:
- Höhere Timeframes zeigen den "Big Picture" Trend
- Alignment = stärkere Setups
- Divergenz = potenzielle Reversals
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


@register_indicator("multi_timeframe")
class MultiTimeframeIndicators(BaseIndicator):
    """
    Multi-Timeframe Features.

    Berechnet Features für höhere Timeframes (H4, D1) aus H1-Daten.

    Features:
    - H4 Trend (EMA Distance)
    - H4 Range Position
    - H4 ADX, RSI, ATR
    - D1 Range Position, EMA Distance
    - Trend Alignment Scores
    - Volatility Ratio
    """

    group = "multi_timeframe"

    def compute(
        self,
        df: pd.DataFrame,
        h4_bars: int = 4,
        d1_bars: int = 24,
        ema_periods: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Multi-Timeframe Features.

        Args:
            df: DataFrame mit H1 OHLC-Daten
            h4_bars: Bars pro H4 Candle (default: 4)
            d1_bars: Bars pro D1 Candle (default: 24)
            ema_periods: EMA Perioden für MTF (default: [20, 50])

        Returns:
            DataFrame mit MTF-Features
        """
        if ema_periods is None:
            ema_periods = [20, 50]

        features = {}

        # === H4 Timeframe ===
        h4_high = df["H"].rolling(h4_bars).max()
        h4_low = df["L"].rolling(h4_bars).min()
        h4_close = df["C"]
        h4_open = df["O"].shift(h4_bars - 1)

        # H4 Trend
        features["mtf_h4_trend"] = (h4_close - h4_open) / (h4_high - h4_low + 1e-10)
        features["mtf_h4_range_pos"] = (df["C"] - h4_low) / (h4_high - h4_low + 1e-10)

        # H4 EMA Distances
        for period in ema_periods:
            h4_ema = ta.trend.ema_indicator(df["C"], window=period * h4_bars)
            features[f"mtf_h4_ema{period}_dist"] = (df["C"] - h4_ema) / df["C"]

        # H4 Technical Indicators
        features["mtf_h4_adx"] = ta.trend.adx(h4_high, h4_low, df["C"], window=14)
        h4_rsi = ta.momentum.rsi(df["C"], window=14 * h4_bars)
        features["mtf_h4_rsi"] = h4_rsi

        h4_atr = ta.volatility.average_true_range(h4_high, h4_low, df["C"], window=14)
        h4_atr_pct = h4_atr / df["C"]
        features["mtf_h4_atr_pct"] = h4_atr_pct

        h4_bb = ta.volatility.BollingerBands(df["C"], window=20 * h4_bars)
        features["mtf_h4_bb_pband"] = h4_bb.bollinger_pband()

        # === D1 Timeframe ===
        d1_high = df["H"].rolling(d1_bars).max()
        d1_low = df["L"].rolling(d1_bars).min()

        features["mtf_d1_range_pos"] = (df["C"] - d1_low) / (d1_high - d1_low + 1e-10)

        for period in ema_periods:
            d1_ema = ta.trend.ema_indicator(df["C"], window=period * d1_bars)
            features[f"mtf_d1_ema{period}_dist"] = (df["C"] - d1_ema) / df["C"]

        d1_ema_slow = ta.trend.ema_indicator(df["C"], window=20 * d1_bars)
        features["mtf_d1_trend_strength"] = d1_ema_slow.pct_change(d1_bars) * 100

        # === Trend Alignment ===
        h1_ema_21 = ta.trend.ema_indicator(df["C"], window=21)
        h1_trend = (df["C"] - h1_ema_21) / df["C"]
        h4_trend = features["mtf_h4_ema20_dist"]
        d1_trend = features["mtf_d1_ema20_dist"]

        features["mtf_trend_alignment_h1h4"] = (np.sign(h1_trend) == np.sign(h4_trend)).astype(int)
        features["mtf_trend_alignment_h4d1"] = (np.sign(h4_trend) == np.sign(d1_trend)).astype(int)
        features["mtf_consensus"] = (
            (np.sign(h1_trend) == np.sign(h4_trend)) &
            (np.sign(h4_trend) == np.sign(d1_trend))
        ).astype(int)
        features["mtf_trend_strength"] = features["mtf_trend_alignment_h1h4"] + features["mtf_trend_alignment_h4d1"]

        # === Volatility Ratio ===
        h1_atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
        h1_atr_pct = h1_atr / df["C"]
        features["mtf_vol_ratio_h1h4"] = h1_atr_pct / (h4_atr_pct + 1e-10)

        # === Momentum Divergence ===
        h1_rsi = ta.momentum.rsi(df["C"], window=14)
        features["mtf_rsi_divergence"] = h1_rsi - h4_rsi

        # === Higher Timeframe Support/Resistance ===
        d1_prev_high = d1_high.shift(d1_bars)
        d1_prev_low = d1_low.shift(d1_bars)

        features["mtf_d1_above_prev_high"] = (df["C"] > d1_prev_high).astype(int)
        features["mtf_d1_below_prev_low"] = (df["C"] < d1_prev_low).astype(int)
        features["mtf_d1_dist_to_high"] = (d1_prev_high - df["C"]) / df["C"]
        features["mtf_d1_dist_to_low"] = (df["C"] - d1_prev_low) / df["C"]

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        # At bar i, the model should use features from bar i-1, not bar i
        features_df = pd.DataFrame(features, index=df.index)
        for col in features_df.columns:
            features_df[col] = features_df[col].shift(1)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            # H4 Features
            "mtf_h4_trend", "mtf_h4_range_pos",
            "mtf_h4_ema20_dist", "mtf_h4_ema50_dist",
            "mtf_h4_adx", "mtf_h4_rsi", "mtf_h4_atr_pct", "mtf_h4_bb_pband",
            # D1 Features
            "mtf_d1_range_pos",
            "mtf_d1_ema20_dist", "mtf_d1_ema50_dist",
            "mtf_d1_trend_strength",
            # Alignment Features
            "mtf_trend_alignment_h1h4", "mtf_trend_alignment_h4d1",
            "mtf_consensus", "mtf_trend_strength",
            # Volatility & Divergence
            "mtf_vol_ratio_h1h4", "mtf_rsi_divergence",
            # Support/Resistance
            "mtf_d1_above_prev_high", "mtf_d1_below_prev_low",
            "mtf_d1_dist_to_high", "mtf_d1_dist_to_low",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "h4_bars": 4,
            "d1_bars": 24,
            "ema_periods": [20, 50],
        }


__all__ = ["MultiTimeframeIndicators"]
