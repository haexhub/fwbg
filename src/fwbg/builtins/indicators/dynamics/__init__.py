"""
Dynamics Indicator Plugin.

Enthält:
- Indikator-Änderungen (RSI, ATR, ADX, etc.)
- Lag Features (vergangene Werte)
- Beschleunigungs-Features (2. Ableitung)

Dynamik-Features zeigen:
- Wie schnell ändern sich Indikatoren?
- Beschleunigt oder verlangsamt sich ein Trend?
- Was war der Zustand vor N Bars?
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features
from fwbg.core import register_indicator


@register_indicator("dynamics")
class DynamicsIndicators(BaseIndicator):
    """
    Dynamik-Features für Momentum und Volatility Changes.

    Features:
    - RSI Changes (4h, 8h, 24h)
    - ATR Changes (4h, 8h, 24h)
    - ADX Changes (4h, 8h, 24h)
    - MACD Changes
    - Stochastic Changes
    - Lag Features (vergangene Werte)
    - Beschleunigungs-Features
    """

    group = "dynamics"

    def compute(
        self,
        df: pd.DataFrame,
        lookbacks: List[int] = None,
        lag_periods: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Dynamik-Features.

        Args:
            df: DataFrame mit OHLC-Daten (und bereits berechneten Basis-Indikatoren)
            lookbacks: Lookback-Perioden für Changes (default: [4, 8, 24])
            lag_periods: Lag-Perioden (default: [4, 8, 24, 48])

        Returns:
            DataFrame mit Dynamics-Features
        """
        if lookbacks is None:
            lookbacks = [4, 8, 24]
        if lag_periods is None:
            lag_periods = [4, 8, 24, 48]

        features = {}

        # Berechne Basis-Indikatoren falls nicht vorhanden
        mom_rsi_14 = df.get("mom_rsi_14", ta.momentum.rsi(df["C"], window=14))
        trend_adx_14 = df.get("trend_adx_14", ta.trend.adx(df["H"], df["L"], df["C"], window=14))

        if "vol_atr_pct_14" in df.columns:
            vol_atr_pct_14 = df["vol_atr_pct_14"]
        else:
            atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
            vol_atr_pct_14 = atr / df["C"]

        if "vol_bb_wband_20" in df.columns:
            vol_bb_wband_20 = df["vol_bb_wband_20"]
        else:
            bb = ta.volatility.BollingerBands(df["C"], window=20)
            vol_bb_wband_20 = bb.bollinger_wband()

        if "mom_stoch_k_14" in df.columns:
            mom_stoch_k_14 = df["mom_stoch_k_14"]
        else:
            stoch = ta.momentum.StochasticOscillator(df["H"], df["L"], df["C"], window=14)
            mom_stoch_k_14 = stoch.stoch()

        if "trend_macd" in df.columns:
            trend_macd = df["trend_macd"]
        else:
            macd = ta.trend.MACD(df["C"])
            trend_macd = macd.macd_diff() / df["C"]

        # === RSI Changes ===
        for lookback in lookbacks:
            features[f"dyn_rsi14_chg_{lookback}h"] = mom_rsi_14 - mom_rsi_14.shift(lookback)
            features[f"dyn_rsi14_pct_{lookback}h"] = mom_rsi_14.pct_change(lookback) * 100

        # === ATR / Volatility Changes ===
        for lookback in lookbacks:
            features[f"dyn_atr_chg_{lookback}h"] = vol_atr_pct_14.pct_change(lookback) * 100
            features[f"dyn_bbwidth_chg_{lookback}h"] = vol_bb_wband_20.pct_change(lookback) * 100

        # === ADX Changes ===
        for lookback in lookbacks:
            features[f"dyn_adx_chg_{lookback}h"] = trend_adx_14 - trend_adx_14.shift(lookback)

        # === MACD Changes ===
        for lookback in [4, 8]:
            features[f"dyn_macd_chg_{lookback}h"] = trend_macd - trend_macd.shift(lookback)

        # === Stochastic Changes ===
        for lookback in [4, 8]:
            features[f"dyn_stoch_chg_{lookback}h"] = mom_stoch_k_14 - mom_stoch_k_14.shift(lookback)

        # === Lag Features ===
        for lag in lag_periods[:3]:  # [4, 8, 24]
            features[f"lag_rsi14_{lag}h"] = mom_rsi_14.shift(lag)
            features[f"lag_atr_{lag}h"] = vol_atr_pct_14.shift(lag)

        for lag in [4, 8]:
            features[f"lag_adx_{lag}h"] = trend_adx_14.shift(lag)

        for lag in lag_periods:  # [4, 8, 24, 48]
            features[f"lag_price_chg_{lag}h"] = (
                (df["C"] - df["C"].shift(lag)) / df["C"].shift(lag) * 100
            )

        # === Beschleunigungs-Features (2. Ableitung) ===
        features["accel_rsi"] = features["dyn_rsi14_chg_4h"] - features["dyn_rsi14_chg_4h"].shift(4)
        features["accel_atr"] = features["dyn_atr_chg_4h"] - features["dyn_atr_chg_4h"].shift(4)
        features["accel_adx"] = features["dyn_adx_chg_4h"] - features["dyn_adx_chg_4h"].shift(4)
        features["accel_price"] = features["lag_price_chg_4h"] - features["lag_price_chg_4h"].shift(4)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            # RSI Changes
            "dyn_rsi14_chg_4h", "dyn_rsi14_chg_8h", "dyn_rsi14_chg_24h",
            "dyn_rsi14_pct_4h", "dyn_rsi14_pct_8h", "dyn_rsi14_pct_24h",
            # ATR/BB Changes
            "dyn_atr_chg_4h", "dyn_atr_chg_8h", "dyn_atr_chg_24h",
            "dyn_bbwidth_chg_4h", "dyn_bbwidth_chg_8h", "dyn_bbwidth_chg_24h",
            # ADX Changes
            "dyn_adx_chg_4h", "dyn_adx_chg_8h", "dyn_adx_chg_24h",
            # MACD/Stoch Changes
            "dyn_macd_chg_4h", "dyn_macd_chg_8h",
            "dyn_stoch_chg_4h", "dyn_stoch_chg_8h",
            # Lag Features
            "lag_rsi14_4h", "lag_rsi14_8h", "lag_rsi14_24h",
            "lag_atr_4h", "lag_atr_8h", "lag_atr_24h",
            "lag_adx_4h", "lag_adx_8h",
            "lag_price_chg_4h", "lag_price_chg_8h",
            "lag_price_chg_24h", "lag_price_chg_48h",
            # Acceleration
            "accel_rsi", "accel_atr", "accel_adx", "accel_price",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "lookbacks": [4, 8, 24],
            "lag_periods": [4, 8, 24, 48],
        }


__all__ = ["DynamicsIndicators"]
