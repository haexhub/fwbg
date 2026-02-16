"""
Multi-Timeframe Indicator Plugin.

Enthält:
- H4 (4-Stunden) Timeframe Features
- D1 (Daily) Timeframe Features
- W1 (Weekly) Timeframe Features
- Y1 (Yearly) Timeframe Features (200d EMA, 52-week range)
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
from fwbg.plugins.indicator import shift_features, safe_divide
from fwbg.core import register_indicator


@register_indicator("multi_timeframe")
class MultiTimeframeIndicators(BaseIndicator):
    """
    Multi-Timeframe Features.

    Berechnet Features für höhere Timeframes (H4, D1, W1, Y1) aus H1-Daten.

    Features:
    - H4 Trend (EMA Distance), Range Position, ADX, RSI, ATR
    - D1 Range Position, EMA Distance, Trend Strength
    - W1 Range Position, EMA Distance, Trend Strength
    - Y1 200-day EMA Distance, 52-week Range Position
    - Trend Alignment Scores (H1→H4→D1→W1)
    - Volatility Ratios
    """

    name = "multi_timeframe"
    version = "3.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        h4_bars: int = 4,
        d1_bars: int = 24,
        w1_bars: int = 120,
        ema_periods: List[int] = None,
        include_yearly: bool = True,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Multi-Timeframe Features.

        Args:
            df: DataFrame mit H1 OHLC-Daten
            h4_bars: Bars pro H4 Candle (default: 4)
            d1_bars: Bars pro D1 Candle (default: 24)
            w1_bars: Bars pro W1 Candle (default: 120 = 5 * 24)
            ema_periods: EMA Perioden für MTF (default: [20, 50])
            include_yearly: Ob Jahres-Features berechnet werden (default: True)

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

        h4_range = h4_high - h4_low
        features["mtf_h4_trend"] = safe_divide(h4_close - h4_open, h4_range)
        features["mtf_h4_range_pos"] = safe_divide(df["C"] - h4_low, h4_range)

        for period in ema_periods:
            h4_ema = ta.trend.ema_indicator(df["C"], window=period * h4_bars)
            features[f"mtf_h4_ema{period}_dist"] = safe_divide(df["C"] - h4_ema, df["C"])

        features["mtf_h4_adx"] = ta.trend.adx(h4_high, h4_low, df["C"], window=14)
        h4_rsi = ta.momentum.rsi(df["C"], window=14 * h4_bars)
        features["mtf_h4_rsi"] = h4_rsi

        h4_atr = ta.volatility.average_true_range(h4_high, h4_low, df["C"], window=14)
        h4_atr_pct = safe_divide(h4_atr, df["C"])
        features["mtf_h4_atr_pct"] = h4_atr_pct

        h4_bb = ta.volatility.BollingerBands(df["C"], window=20 * h4_bars)
        features["mtf_h4_bb_pband"] = h4_bb.bollinger_pband()

        # === D1 Timeframe ===
        d1_high = df["H"].rolling(d1_bars).max()
        d1_low = df["L"].rolling(d1_bars).min()

        d1_range = d1_high - d1_low
        features["mtf_d1_range_pos"] = safe_divide(df["C"] - d1_low, d1_range)

        for period in ema_periods:
            d1_ema = ta.trend.ema_indicator(df["C"], window=period * d1_bars)
            features[f"mtf_d1_ema{period}_dist"] = safe_divide(df["C"] - d1_ema, df["C"])

        d1_ema_slow = ta.trend.ema_indicator(df["C"], window=20 * d1_bars)
        features["mtf_d1_trend_strength"] = d1_ema_slow.pct_change(d1_bars) * 100

        # === W1 (Weekly) Timeframe ===
        w1_high = df["H"].rolling(w1_bars).max()
        w1_low = df["L"].rolling(w1_bars).min()

        w1_range = w1_high - w1_low
        features["mtf_w1_range_pos"] = safe_divide(df["C"] - w1_low, w1_range)

        for period in ema_periods:
            w1_ema = ta.trend.ema_indicator(df["C"], window=period * w1_bars)
            features[f"mtf_w1_ema{period}_dist"] = safe_divide(df["C"] - w1_ema, df["C"])

        w1_ema_slow = ta.trend.ema_indicator(df["C"], window=20 * w1_bars)
        features["mtf_w1_trend_strength"] = w1_ema_slow.pct_change(w1_bars) * 100

        # === Y1 (Yearly) Features ===
        if include_yearly:
            # 200-day EMA distance (~4800 hourly bars)
            y1_ema_200d = ta.trend.ema_indicator(df["C"], window=200 * d1_bars)
            features["mtf_y1_ema200d_dist"] = safe_divide(df["C"] - y1_ema_200d, df["C"])

            # 52-week range (52 * 5 * 24 = 6240 hourly bars)
            y1_window = 52 * w1_bars
            y1_high = df["H"].rolling(y1_window, min_periods=w1_bars).max()
            y1_low = df["L"].rolling(y1_window, min_periods=w1_bars).min()
            y1_range = y1_high - y1_low
            features["mtf_y1_52w_range_pos"] = safe_divide(df["C"] - y1_low, y1_range)
            features["mtf_y1_52w_high_dist"] = safe_divide(y1_high - df["C"], df["C"])
            features["mtf_y1_52w_low_dist"] = safe_divide(df["C"] - y1_low, df["C"])

        # === Trend Alignment ===
        h1_ema_21 = ta.trend.ema_indicator(df["C"], window=21)
        h1_trend = safe_divide(df["C"] - h1_ema_21, df["C"])
        h4_trend = features["mtf_h4_ema20_dist"]
        d1_trend = features["mtf_d1_ema20_dist"]
        w1_trend = features["mtf_w1_ema20_dist"]

        features["mtf_trend_alignment_h1h4"] = (np.sign(h1_trend) == np.sign(h4_trend)).astype(int)
        features["mtf_trend_alignment_h4d1"] = (np.sign(h4_trend) == np.sign(d1_trend)).astype(int)
        features["mtf_trend_alignment_d1w1"] = (np.sign(d1_trend) == np.sign(w1_trend)).astype(int)
        features["mtf_consensus"] = (
            (np.sign(h1_trend) == np.sign(h4_trend)) &
            (np.sign(h4_trend) == np.sign(d1_trend)) &
            (np.sign(d1_trend) == np.sign(w1_trend))
        ).astype(int)
        features["mtf_trend_strength"] = (
            features["mtf_trend_alignment_h1h4"]
            + features["mtf_trend_alignment_h4d1"]
            + features["mtf_trend_alignment_d1w1"]
        )

        # === Volatility Ratio ===
        h1_atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
        h1_atr_pct = safe_divide(h1_atr, df["C"])
        features["mtf_vol_ratio_h1h4"] = safe_divide(h1_atr_pct, h4_atr_pct)

        # === Momentum Divergence ===
        h1_rsi = ta.momentum.rsi(df["C"], window=14)
        features["mtf_rsi_divergence"] = h1_rsi - h4_rsi

        # === Higher Timeframe Support/Resistance ===
        d1_prev_high = d1_high.shift(d1_bars)
        d1_prev_low = d1_low.shift(d1_bars)

        features["mtf_d1_above_prev_high"] = (df["C"] > d1_prev_high).astype(int)
        features["mtf_d1_below_prev_low"] = (df["C"] < d1_prev_low).astype(int)
        features["mtf_d1_dist_to_high"] = safe_divide(d1_prev_high - df["C"], df["C"])
        features["mtf_d1_dist_to_low"] = safe_divide(df["C"] - d1_prev_low, df["C"])

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

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
            # W1 Features
            "mtf_w1_range_pos",
            "mtf_w1_ema20_dist", "mtf_w1_ema50_dist",
            "mtf_w1_trend_strength",
            # Y1 Features
            "mtf_y1_ema200d_dist",
            "mtf_y1_52w_range_pos", "mtf_y1_52w_high_dist", "mtf_y1_52w_low_dist",
            # Alignment Features
            "mtf_trend_alignment_h1h4", "mtf_trend_alignment_h4d1",
            "mtf_trend_alignment_d1w1",
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
            "w1_bars": 120,
            "ema_periods": [20, 50],
            "include_yearly": True,
        }


__all__ = ["MultiTimeframeIndicators"]
