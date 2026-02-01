"""
Cross-Indicator Features Plugin.

Enthält:
- Kombinationen aus verschiedenen Indikatoren
- Conditional Features (wenn A dann B)
- Interaktions-Features (A * B)

Cross-Features sind wichtig weil:
- RSI > 70 allein ist weniger aussagekräftig als RSI > 70 + steigend
- Volatilität * Trend-Stärke zeigt "explodierende" Moves
- Divergenzen zwischen Indikatoren zeigen potenzielle Reversals
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


@register_indicator("cross_features")
class CrossFeatureIndicators(BaseIndicator):
    """
    Cross-Indicator Features.

    Features:
    - RSI High + Rising / RSI Low + Falling
    - Volatility * Trend Interactions
    - Overbought/Oversold Conditions
    - Indicator Divergences
    - Confluence Scores
    """

    group = "cross"

    def compute(
        self,
        df: pd.DataFrame,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Cross-Indicator Features.

        Args:
            df: DataFrame mit OHLC-Daten (und optional bereits berechneten Indikatoren)
            rsi_overbought: RSI Overbought Level (default: 70)
            rsi_oversold: RSI Oversold Level (default: 30)

        Returns:
            DataFrame mit Cross-Features
        """
        # Stelle sicher dass Basis-Indikatoren vorhanden sind
        self._ensure_base_indicators(df)

        # === RSI Conditional Features ===
        rsi = df["mom_rsi_14"]
        rsi_change = rsi - rsi.shift(4)

        # RSI hoch UND steigend (Continuation Signal)
        df["cross_rsi_high_rising"] = (
            (rsi > rsi_overbought) & (rsi_change > 0)
        ).astype(int)

        # RSI niedrig UND fallend (Continuation Signal)
        df["cross_rsi_low_falling"] = (
            (rsi < rsi_oversold) & (rsi_change < 0)
        ).astype(int)

        # RSI hoch ABER fallend (Reversal Warning)
        df["cross_rsi_high_falling"] = (
            (rsi > rsi_overbought) & (rsi_change < 0)
        ).astype(int)

        # RSI niedrig ABER steigend (Reversal Warning)
        df["cross_rsi_low_rising"] = (
            (rsi < rsi_oversold) & (rsi_change > 0)
        ).astype(int)

        # === Volatility-Trend Interactions ===
        atr_change = df["vol_atr_pct_14"].pct_change(4) * 100
        adx = df["trend_adx_14"]

        # Volatility * Trend Strength (hoher ADX + steigende Vol = explosive Move)
        df["cross_vol_trend"] = atr_change * adx / 100

        # Expanding Volatility in Strong Trend
        df["cross_expanding_trend"] = (
            (atr_change > 0) & (adx > 25)
        ).astype(int)

        # Contracting Volatility (Consolidation)
        df["cross_contracting"] = (
            (atr_change < -5) & (adx < 20)
        ).astype(int)

        # === Bollinger Band Squeeze ===
        bb_width = df["vol_bb_wband_20"]
        bb_width_percentile = bb_width.rolling(100).apply(
            lambda x: (x.iloc[-1] <= np.percentile(x, 20)) if len(x) > 0 else 0
        )
        df["cross_bb_squeeze"] = (bb_width_percentile == 1).astype(int)

        # === Trend Confirmation ===
        ema_short = ta.trend.ema_indicator(df["C"], window=8)
        ema_long = ta.trend.ema_indicator(df["C"], window=21)

        # EMA aligned with ADX
        bullish_ema = ema_short > ema_long
        strong_trend = adx > 25

        df["cross_bullish_strong"] = (bullish_ema & strong_trend).astype(int)
        df["cross_bearish_strong"] = (~bullish_ema & strong_trend).astype(int)

        # === MACD-RSI Confluence ===
        macd = df["trend_macd"]

        # Bullish Confluence: MACD > 0 und RSI > 50 und RSI < 70
        df["cross_bullish_confluence"] = (
            (macd > 0) & (rsi > 50) & (rsi < rsi_overbought)
        ).astype(int)

        # Bearish Confluence: MACD < 0 und RSI < 50 und RSI > 30
        df["cross_bearish_confluence"] = (
            (macd < 0) & (rsi < 50) & (rsi > rsi_oversold)
        ).astype(int)

        # === Divergence Detection ===
        # Price macht Higher High, aber RSI nicht
        price_hh = (df["H"] > df["H"].rolling(20).max().shift(1))
        rsi_hh = (rsi > rsi.rolling(20).max().shift(1))
        df["cross_bearish_divergence"] = (price_hh & ~rsi_hh).astype(int)

        # Price macht Lower Low, aber RSI nicht
        price_ll = (df["L"] < df["L"].rolling(20).min().shift(1))
        rsi_ll = (rsi < rsi.rolling(20).min().shift(1))
        df["cross_bullish_divergence"] = (price_ll & ~rsi_ll).astype(int)

        # === Momentum-Volatility Score ===
        # Kombiniert multiple Faktoren
        rsi_score = (rsi - 50) / 50  # -1 to 1
        adx_score = adx / 50  # 0 to ~1
        vol_score = df["vol_atr_pct_14"] / df["vol_atr_pct_14"].rolling(50).mean()

        df["cross_momentum_vol_score"] = rsi_score * adx_score * vol_score

        # === Overbought/Oversold with Trend ===
        # Overbought in Uptrend (könnte weiter steigen)
        df["cross_overbought_uptrend"] = (
            (rsi > rsi_overbought) & (ema_short > ema_long) & (adx > 20)
        ).astype(int)

        # Oversold in Downtrend (könnte weiter fallen)
        df["cross_oversold_downtrend"] = (
            (rsi < rsi_oversold) & (ema_short < ema_long) & (adx > 20)
        ).astype(int)

        # === Stochastic-RSI Confluence ===
        stoch = df["mom_stoch_k_14"]

        df["cross_stoch_rsi_overbought"] = (
            (stoch > 80) & (rsi > rsi_overbought)
        ).astype(int)

        df["cross_stoch_rsi_oversold"] = (
            (stoch < 20) & (rsi < rsi_oversold)
        ).astype(int)

        # === Confluence Score ===
        # Bullish signals count
        bullish_signals = (
            df["cross_bullish_confluence"].astype(int) +
            df["cross_bullish_divergence"].astype(int) +
            (df["cross_momentum_vol_score"] > 0).astype(int)
        )

        # Bearish signals count
        bearish_signals = (
            df["cross_bearish_confluence"].astype(int) +
            df["cross_bearish_divergence"].astype(int) +
            (df["cross_momentum_vol_score"] < 0).astype(int)
        )

        df["cross_bullish_count"] = bullish_signals
        df["cross_bearish_count"] = bearish_signals
        df["cross_signal_bias"] = bullish_signals - bearish_signals

        return df

    def _ensure_base_indicators(self, df: pd.DataFrame) -> None:
        """Berechnet fehlende Basis-Indikatoren."""
        if "mom_rsi_14" not in df.columns:
            df["mom_rsi_14"] = ta.momentum.rsi(df["C"], window=14)

        if "mom_stoch_k_14" not in df.columns:
            stoch = ta.momentum.StochasticOscillator(df["H"], df["L"], df["C"], window=14)
            df["mom_stoch_k_14"] = stoch.stoch()

        if "trend_adx_14" not in df.columns:
            df["trend_adx_14"] = ta.trend.adx(df["H"], df["L"], df["C"], window=14)

        if "trend_macd" not in df.columns:
            macd = ta.trend.MACD(df["C"])
            df["trend_macd"] = macd.macd_diff() / df["C"]

        if "vol_atr_pct_14" not in df.columns:
            atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
            df["vol_atr_pct_14"] = atr / df["C"]

        if "vol_bb_wband_20" not in df.columns:
            bb = ta.volatility.BollingerBands(df["C"], window=20)
            df["vol_bb_wband_20"] = bb.bollinger_wband()

    def get_feature_columns(self) -> List[str]:
        return [
            # RSI Conditional
            "cross_rsi_high_rising", "cross_rsi_low_falling",
            "cross_rsi_high_falling", "cross_rsi_low_rising",
            # Volatility-Trend
            "cross_vol_trend", "cross_expanding_trend", "cross_contracting",
            "cross_bb_squeeze",
            # Trend Confirmation
            "cross_bullish_strong", "cross_bearish_strong",
            # Confluence
            "cross_bullish_confluence", "cross_bearish_confluence",
            # Divergence
            "cross_bearish_divergence", "cross_bullish_divergence",
            # Composite Scores
            "cross_momentum_vol_score",
            "cross_overbought_uptrend", "cross_oversold_downtrend",
            "cross_stoch_rsi_overbought", "cross_stoch_rsi_oversold",
            # Signal Counts
            "cross_bullish_count", "cross_bearish_count", "cross_signal_bias",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
        }


__all__ = ["CrossFeatureIndicators"]
