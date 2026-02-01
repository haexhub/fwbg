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

        # Berechne Basis-Indikatoren falls nicht vorhanden
        if "mom_rsi_14" not in df.columns:
            df["mom_rsi_14"] = ta.momentum.rsi(df["C"], window=14)

        if "trend_adx_14" not in df.columns:
            df["trend_adx_14"] = ta.trend.adx(df["H"], df["L"], df["C"], window=14)

        if "vol_atr_pct_14" not in df.columns:
            atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
            df["vol_atr_pct_14"] = atr / df["C"]

        if "vol_bb_wband_20" not in df.columns:
            bb = ta.volatility.BollingerBands(df["C"], window=20)
            df["vol_bb_wband_20"] = bb.bollinger_wband()

        if "mom_stoch_k_14" not in df.columns:
            stoch = ta.momentum.StochasticOscillator(df["H"], df["L"], df["C"], window=14)
            df["mom_stoch_k_14"] = stoch.stoch()

        if "trend_macd" not in df.columns:
            macd = ta.trend.MACD(df["C"])
            df["trend_macd"] = macd.macd_diff() / df["C"]

        # === RSI Changes ===
        for lookback in lookbacks:
            df[f"dyn_rsi14_chg_{lookback}h"] = (
                df["mom_rsi_14"] - df["mom_rsi_14"].shift(lookback)
            )
            df[f"dyn_rsi14_pct_{lookback}h"] = df["mom_rsi_14"].pct_change(lookback) * 100

        # === ATR / Volatility Changes ===
        for lookback in lookbacks:
            df[f"dyn_atr_chg_{lookback}h"] = df["vol_atr_pct_14"].pct_change(lookback) * 100
            df[f"dyn_bbwidth_chg_{lookback}h"] = (
                df["vol_bb_wband_20"].pct_change(lookback) * 100
            )

        # === ADX Changes ===
        for lookback in lookbacks:
            df[f"dyn_adx_chg_{lookback}h"] = (
                df["trend_adx_14"] - df["trend_adx_14"].shift(lookback)
            )

        # === MACD Changes ===
        for lookback in [4, 8]:
            df[f"dyn_macd_chg_{lookback}h"] = (
                df["trend_macd"] - df["trend_macd"].shift(lookback)
            )

        # === Stochastic Changes ===
        for lookback in [4, 8]:
            df[f"dyn_stoch_chg_{lookback}h"] = (
                df["mom_stoch_k_14"] - df["mom_stoch_k_14"].shift(lookback)
            )

        # === Lag Features ===
        for lag in lag_periods[:3]:  # [4, 8, 24]
            df[f"lag_rsi14_{lag}h"] = df["mom_rsi_14"].shift(lag)
            df[f"lag_atr_{lag}h"] = df["vol_atr_pct_14"].shift(lag)

        for lag in [4, 8]:
            df[f"lag_adx_{lag}h"] = df["trend_adx_14"].shift(lag)

        for lag in lag_periods:  # [4, 8, 24, 48]
            df[f"lag_price_chg_{lag}h"] = (
                (df["C"] - df["C"].shift(lag)) / df["C"].shift(lag) * 100
            )

        # === Beschleunigungs-Features (2. Ableitung) ===
        # RSI Beschleunigung
        if "dyn_rsi14_chg_4h" in df.columns:
            df["accel_rsi"] = (
                df["dyn_rsi14_chg_4h"] - df["dyn_rsi14_chg_4h"].shift(4)
            )

        # ATR Beschleunigung
        if "dyn_atr_chg_4h" in df.columns:
            df["accel_atr"] = df["dyn_atr_chg_4h"] - df["dyn_atr_chg_4h"].shift(4)

        # ADX Beschleunigung
        if "dyn_adx_chg_4h" in df.columns:
            df["accel_adx"] = df["dyn_adx_chg_4h"] - df["dyn_adx_chg_4h"].shift(4)

        # Price Momentum Beschleunigung
        if "lag_price_chg_4h" in df.columns:
            df["accel_price"] = (
                df["lag_price_chg_4h"] - df["lag_price_chg_4h"].shift(4)
            )

        return df

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
