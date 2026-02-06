"""
Volatility Indicator Plugin.

Enthält: ATR, Bollinger Bands, Keltner Channel, Donchian Channel.
"""
from typing import List
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features
from fwbg.core import register_indicator


@register_indicator("volatility")
class VolatilityIndicators(BaseIndicator):
    """
    Volatilitäts-Indikatoren für Trading-Strategien.

    Features:
    - ATR (7, 14, 21 Perioden) - als Prozent vom Preis
    - Bollinger Bands (20 Perioden) - pband, wband
    - Keltner Channel - pband, wband
    - Donchian Channel - pband, wband
    """

    group = "volatility"

    def compute(
        self,
        df: pd.DataFrame,
        atr_periods: List[int] = None,
        bb_period: int = 20,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet alle Volatilitäts-Indikatoren.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            atr_periods: ATR-Perioden (default: [7, 14, 21])
            bb_period: Bollinger Bands Periode (default: 20)

        Returns:
            DataFrame mit Volatilitäts-Features
        """
        if atr_periods is None:
            atr_periods = [7, 14, 21]

        features = {}

        # ATR (für interne Nutzung und als Feature)
        atr_14 = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        features["_atr"] = atr_14
        features["vol_atr"] = atr_14

        # ATR als Prozent vom Preis
        for period in atr_periods:
            atr = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=period
            )
            features[f"vol_atr_pct_{period}"] = atr / df["C"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["C"], window=bb_period)
        features[f"vol_bb_pband_{bb_period}"] = bb.bollinger_pband()
        features[f"vol_bb_wband_{bb_period}"] = bb.bollinger_wband()

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df["H"], df["L"], df["C"])
        features["vol_kc_pband"] = kc.keltner_channel_pband()
        features["vol_kc_wband"] = kc.keltner_channel_wband()

        # Donchian Channel
        dc = ta.volatility.DonchianChannel(df["H"], df["L"], df["C"])
        features["vol_dc_pband"] = dc.donchian_channel_pband()
        features["vol_dc_wband"] = dc.donchian_channel_wband()

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        """Gibt Liste aller Volatilitäts-Feature-Spalten zurück."""
        return [
            # ATR
            "vol_atr",
            "vol_atr_pct_7", "vol_atr_pct_14", "vol_atr_pct_21",
            # Bollinger
            "vol_bb_pband_20", "vol_bb_wband_20",
            # Keltner
            "vol_kc_pband", "vol_kc_wband",
            # Donchian
            "vol_dc_pband", "vol_dc_wband",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_periods": [7, 14, 21],
            "bb_period": 20,
        }


__all__ = ["VolatilityIndicators"]
