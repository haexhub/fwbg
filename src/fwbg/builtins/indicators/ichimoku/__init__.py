"""
Ichimoku Cloud Indicator Plugin.

Enthält alle Ichimoku-Komponenten:
- Tenkan-sen (Conversion Line)
- Kijun-sen (Base Line)
- Senkou Span A & B (Cloud/Kumo)
- Abgeleitete Features (Cloud Position, Thickness, Crosses)

Ichimoku ist ein komplettes Trading-System:
- Preis über Cloud = Bullish Bias
- Preis unter Cloud = Bearish Bias
- TK Cross = Signal
- Kijun als Support/Resistance
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


@register_indicator("ichimoku")
class IchimokuIndicators(BaseIndicator):
    """
    Ichimoku Cloud Features.

    Features:
    - Tenkan-sen (9-Periode Midpoint)
    - Kijun-sen (26-Periode Midpoint)
    - Senkou Span A (Cloud Leading Span A)
    - Senkou Span B (Cloud Leading Span B)
    - Cloud Position (Price relative to Cloud)
    - Cloud Thickness
    - TK Cross
    - Price-Kijun Distance
    """

    group = "ichimoku"

    def compute(
        self,
        df: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Ichimoku Features.

        Args:
            df: DataFrame mit OHLC-Daten
            tenkan_period: Tenkan-sen Periode (default: 9)
            kijun_period: Kijun-sen Periode (default: 26)
            senkou_b_period: Senkou Span B Periode (default: 52)

        Returns:
            DataFrame mit Ichimoku-Features
        """
        # Ichimoku Indicator aus ta-lib
        ichimoku = ta.trend.IchimokuIndicator(
            df["H"], df["L"],
            window1=tenkan_period,
            window2=kijun_period,
            window3=senkou_b_period
        )

        # Basis-Linien
        df["ichi_tenkan"] = ichimoku.ichimoku_conversion_line()
        df["ichi_kijun"] = ichimoku.ichimoku_base_line()
        df["ichi_senkou_a"] = ichimoku.ichimoku_a()
        df["ichi_senkou_b"] = ichimoku.ichimoku_b()

        # Cloud Top und Bottom
        cloud_top = df[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1)
        cloud_bottom = df[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1)

        # Cloud Thickness (normalisiert)
        df["ichi_cloud_thick"] = (cloud_top - cloud_bottom) / df["C"]

        # Cloud Position: Wo ist der Preis relativ zur Cloud (0-1)
        # < 0 = unter Cloud, > 1 = über Cloud, 0-1 = in Cloud
        df["ichi_cloud_pos"] = (df["C"] - cloud_bottom) / (cloud_top - cloud_bottom + 1e-10)

        # Preis über/unter Cloud
        df["ichi_above_cloud"] = (df["C"] > cloud_top).astype(int)
        df["ichi_below_cloud"] = (df["C"] < cloud_bottom).astype(int)
        df["ichi_in_cloud"] = (
            (df["C"] >= cloud_bottom) & (df["C"] <= cloud_top)
        ).astype(int)

        # TK Cross (Tenkan - Kijun, normalisiert)
        df["ichi_tk_cross"] = (df["ichi_tenkan"] - df["ichi_kijun"]) / df["C"]

        # TK Cross Direction Change (Signal)
        tk_bullish = (df["ichi_tenkan"] > df["ichi_kijun"]).astype(bool)
        tk_bullish_prev = tk_bullish.shift(1).fillna(False).astype(bool)
        df["ichi_tk_bullish_cross"] = (
            tk_bullish & ~tk_bullish_prev
        ).astype(int)
        df["ichi_tk_bearish_cross"] = (
            ~tk_bullish & tk_bullish_prev
        ).astype(int)

        # Price-Kijun Distance (Kijun als Support/Resistance)
        df["ichi_price_kijun"] = (df["C"] - df["ichi_kijun"]) / df["C"]

        # Kijun Flat (Ranging Market)
        kijun_change = df["ichi_kijun"].diff().abs()
        df["ichi_kijun_flat"] = (kijun_change < 0.0001 * df["C"]).astype(int)

        # Cloud Color (Bullish = Span A > Span B)
        df["ichi_bullish_cloud"] = (df["ichi_senkou_a"] > df["ichi_senkou_b"]).astype(int)

        # Cloud Color Change (Kumo Twist)
        cloud_bullish = (df["ichi_senkou_a"] > df["ichi_senkou_b"]).astype(bool)
        cloud_bullish_prev = cloud_bullish.shift(1).fillna(False).astype(bool)
        df["ichi_kumo_twist"] = (cloud_bullish != cloud_bullish_prev).astype(int)

        # Chikou Span (Lagging) - Preis vor 26 Perioden
        # Für Trading: aktueller Preis vs Preis vor 26 Perioden
        df["ichi_chikou_above"] = (df["C"] > df["C"].shift(kijun_period)).astype(int)

        # === Composite Signals ===
        # Strong Bullish: Above cloud + TK bullish + bullish cloud
        df["ichi_strong_bullish"] = (
            (df["ichi_above_cloud"] == 1) &
            (df["ichi_tk_cross"] > 0) &
            (df["ichi_bullish_cloud"] == 1)
        ).astype(int)

        # Strong Bearish: Below cloud + TK bearish + bearish cloud
        df["ichi_strong_bearish"] = (
            (df["ichi_below_cloud"] == 1) &
            (df["ichi_tk_cross"] < 0) &
            (df["ichi_bullish_cloud"] == 0)
        ).astype(int)

        # Neutral/Ranging: In cloud or conflicting signals
        df["ichi_neutral"] = (
            (df["ichi_in_cloud"] == 1) |
            (
                (df["ichi_above_cloud"] != df["ichi_bullish_cloud"]) &
                (df["ichi_below_cloud"] != (1 - df["ichi_bullish_cloud"]))
            )
        ).astype(int)

        # Distance to Cloud (für Entry Timing)
        dist_to_top = (cloud_top - df["C"]) / df["C"]
        dist_to_bottom = (df["C"] - cloud_bottom) / df["C"]
        df["ichi_dist_to_cloud"] = np.where(
            df["ichi_above_cloud"] == 1, dist_to_bottom,
            np.where(df["ichi_below_cloud"] == 1, -dist_to_top, 0)
        )

        return df

    def get_feature_columns(self) -> List[str]:
        return [
            # Base Lines (normalized)
            "ichi_tenkan", "ichi_kijun",
            "ichi_senkou_a", "ichi_senkou_b",
            # Cloud Features
            "ichi_cloud_thick", "ichi_cloud_pos",
            "ichi_above_cloud", "ichi_below_cloud", "ichi_in_cloud",
            # TK Features
            "ichi_tk_cross", "ichi_tk_bullish_cross", "ichi_tk_bearish_cross",
            # Price-Kijun
            "ichi_price_kijun", "ichi_kijun_flat",
            # Cloud Direction
            "ichi_bullish_cloud", "ichi_kumo_twist",
            # Chikou
            "ichi_chikou_above",
            # Composite
            "ichi_strong_bullish", "ichi_strong_bearish", "ichi_neutral",
            "ichi_dist_to_cloud",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
        }


__all__ = ["IchimokuIndicators"]
