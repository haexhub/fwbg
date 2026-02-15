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

WICHTIG: Dieses Modul berechnet IMMER alle Basis-Indikatoren neu (nicht geshiptet),
um Double-Shift Probleme zu vermeiden wenn es nach anderen Modulen läuft.
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide
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

    WICHTIG: Berechnet alle Basis-Indikatoren IMMER neu um Double-Shift
    zu vermeiden. Bereits berechnete Features im DataFrame werden ignoriert.
    """

    name = "cross_features"
    version = "2.0.0"

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
        # KRITISCH: Berechne Basis-Indikatoren IMMER neu (nicht geshiptet)
        # um Double-Shift zu vermeiden wenn dieses Modul nach anderen läuft
        base = self._compute_base_indicators(df)

        features = {}

        # === RSI Conditional Features ===
        rsi = base["rsi"]
        rsi_change = rsi - rsi.shift(4)

        features["cross_rsi_high_rising"] = ((rsi > rsi_overbought) & (rsi_change > 0)).astype(int)
        features["cross_rsi_low_falling"] = ((rsi < rsi_oversold) & (rsi_change < 0)).astype(int)
        features["cross_rsi_high_falling"] = ((rsi > rsi_overbought) & (rsi_change < 0)).astype(int)
        features["cross_rsi_low_rising"] = ((rsi < rsi_oversold) & (rsi_change > 0)).astype(int)

        # === Volatility-Trend Interactions ===
        atr_pct = base["atr_pct"]
        atr_change = atr_pct.pct_change(4) * 100
        adx = base["adx"]

        features["cross_vol_trend"] = atr_change * adx / 100
        features["cross_expanding_trend"] = ((atr_change > 0) & (adx > 25)).astype(int)
        features["cross_contracting"] = ((atr_change < -5) & (adx < 20)).astype(int)

        # === Bollinger Band Squeeze ===
        bb_width = base["bb_width"]
        # Compare current BB width against historical 20th percentile (excluding current)
        # Rolling percentile of past 99 values, then compare current value to it
        bb_width_pct20 = bb_width.shift(1).rolling(99, min_periods=20).apply(
            lambda x: np.percentile(x, 20), raw=True
        )
        features["cross_bb_squeeze"] = (bb_width <= bb_width_pct20).astype(int)

        # === Trend Confirmation ===
        ema_short = ta.trend.ema_indicator(df["C"], window=8)
        ema_long = ta.trend.ema_indicator(df["C"], window=21)
        bullish_ema = ema_short > ema_long
        strong_trend = adx > 25

        features["cross_bullish_strong"] = (bullish_ema & strong_trend).astype(int)
        features["cross_bearish_strong"] = (~bullish_ema & strong_trend).astype(int)

        # === MACD-RSI Confluence ===
        macd = base["macd"]
        features["cross_bullish_confluence"] = ((macd > 0) & (rsi > 50) & (rsi < rsi_overbought)).astype(int)
        features["cross_bearish_confluence"] = ((macd < 0) & (rsi < 50) & (rsi > rsi_oversold)).astype(int)

        # === Divergence Detection ===
        price_hh = (df["H"] > df["H"].rolling(20).max().shift(1))
        rsi_hh = (rsi > rsi.rolling(20).max().shift(1))
        features["cross_bearish_divergence"] = (price_hh & ~rsi_hh).astype(int)

        price_ll = (df["L"] < df["L"].rolling(20).min().shift(1))
        rsi_ll = (rsi < rsi.rolling(20).min().shift(1))
        features["cross_bullish_divergence"] = (price_ll & ~rsi_ll).astype(int)

        # === Momentum-Volatility Score ===
        rsi_score = (rsi - 50) / 50
        adx_score = adx / 50
        vol_score = safe_divide(atr_pct, atr_pct.rolling(50).mean())
        momentum_vol_score = rsi_score * adx_score * vol_score
        features["cross_momentum_vol_score"] = momentum_vol_score

        # === Overbought/Oversold with Trend ===
        features["cross_overbought_uptrend"] = (
            (rsi > rsi_overbought) & (ema_short > ema_long) & (adx > 20)
        ).astype(int)
        features["cross_oversold_downtrend"] = (
            (rsi < rsi_oversold) & (ema_short < ema_long) & (adx > 20)
        ).astype(int)

        # === Stochastic-RSI Confluence ===
        stoch = base["stoch"]
        features["cross_stoch_rsi_overbought"] = ((stoch > 80) & (rsi > rsi_overbought)).astype(int)
        features["cross_stoch_rsi_oversold"] = ((stoch < 20) & (rsi < rsi_oversold)).astype(int)

        # === Confluence Score ===
        bullish_confluence = features["cross_bullish_confluence"]
        bullish_divergence = features["cross_bullish_divergence"]
        bearish_confluence = features["cross_bearish_confluence"]
        bearish_divergence = features["cross_bearish_divergence"]

        bullish_signals = bullish_confluence + bullish_divergence + (momentum_vol_score > 0).astype(int)
        bearish_signals = bearish_confluence + bearish_divergence + (momentum_vol_score < 0).astype(int)

        features["cross_bullish_count"] = bullish_signals
        features["cross_bearish_count"] = bearish_signals
        features["cross_signal_bias"] = bullish_signals - bearish_signals

        # === COT Positioning × Volatility Interaction ===
        # Extreme COT + low vol = explosive breakout potential
        cot_cols = [c for c in df.columns if c.startswith("macro_cot_")]
        atr_pct_rank = atr_pct.rolling(100, min_periods=50).rank(pct=True)

        for col in cot_cols:
            pair = col.replace("macro_", "")  # e.g. "cot_eurusd"
            net = df[col]

            # Inline z-score (from raw COT data, not pre-shifted)
            window = 52 * 5 * 24  # 52 weeks in H1 bars
            roll_mean = net.rolling(window, min_periods=window // 4).mean()
            roll_std = net.rolling(window, min_periods=window // 4).std().clip(lower=1e-6)
            zscore = (net - roll_mean) / roll_std

            # Positioning × Vol: extreme position + low vol = explosive
            inv_vol_rank = 1.0 / atr_pct_rank.clip(lower=0.01)
            features[f"cross_{pair}_vol_interaction"] = zscore * inv_vol_rank

            # Positioning Divergence: price momentum vs COT momentum
            price_mom = df["C"].pct_change(5 * 24) * 100  # 5-day price momentum
            cot_mom = net.pct_change(5 * 24)  # 5-day COT momentum
            # Normalize both to z-scores for comparability
            price_z = (price_mom - price_mom.rolling(500).mean()) / price_mom.rolling(500).std().clip(lower=1e-6)
            cot_z = (cot_mom - cot_mom.rolling(500).mean()) / cot_mom.rolling(500).std().clip(lower=1e-6)
            features[f"cross_{pair}_price_divergence"] = price_z - cot_z

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def _compute_base_indicators(self, df: pd.DataFrame) -> dict:
        """
        Berechnet Basis-Indikatoren IMMER neu (nicht geshiptet).

        KRITISCH: Diese Methode gibt NICHT geshiftete Werte zurück.
        Das ist wichtig weil:
        1. Bereits im DataFrame vorhandene Features sind geshiptet
        2. Wenn wir diese verwenden und am Ende nochmal shiften = Double-Shift
        3. Daher: Immer neu berechnen aus OHLC-Daten

        Returns:
            Dict mit Basis-Indikatoren (rsi, stoch, adx, macd, atr_pct, bb_width)
        """
        return {
            "rsi": ta.momentum.rsi(df["C"], window=14),
            "stoch": ta.momentum.StochasticOscillator(
                df["H"], df["L"], df["C"], window=14
            ).stoch(),
            "adx": ta.trend.adx(df["H"], df["L"], df["C"], window=14),
            "macd": safe_divide(
                ta.trend.MACD(df["C"]).macd_diff(),
                df["C"]
            ),
            "atr_pct": safe_divide(
                ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14),
                df["C"]
            ),
            "bb_width": ta.volatility.BollingerBands(df["C"], window=20).bollinger_wband(),
        }

    def get_feature_columns(self) -> List[str]:
        cols = [
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
        # COT Positioning × Volatility (dynamic per pair)
        for pair in ["cot_eurusd", "cot_usdjpy", "cot_gbpusd", "cot_usdcad",
                      "cot_audusd", "cot_usdchf", "cot_nzdusd"]:
            cols.append(f"cross_{pair}_vol_interaction")
            cols.append(f"cross_{pair}_price_divergence")
        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
        }


__all__ = ["CrossFeatureIndicators"]
