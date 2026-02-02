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
        features = {}

        # Ichimoku Indicator aus ta-lib
        ichimoku = ta.trend.IchimokuIndicator(
            df["H"], df["L"],
            window1=tenkan_period,
            window2=kijun_period,
            window3=senkou_b_period
        )

        # Basis-Linien
        tenkan = ichimoku.ichimoku_conversion_line()
        kijun = ichimoku.ichimoku_base_line()
        senkou_a = ichimoku.ichimoku_a()
        senkou_b = ichimoku.ichimoku_b()

        features["ichi_tenkan"] = tenkan
        features["ichi_kijun"] = kijun
        features["ichi_senkou_a"] = senkou_a
        features["ichi_senkou_b"] = senkou_b

        # Cloud Top und Bottom
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        # Cloud Thickness (normalisiert)
        features["ichi_cloud_thick"] = (cloud_top - cloud_bottom) / df["C"]

        # Cloud Position
        features["ichi_cloud_pos"] = (df["C"] - cloud_bottom) / (cloud_top - cloud_bottom + 1e-10)

        # Preis über/unter Cloud
        above_cloud = (df["C"] > cloud_top).astype(int)
        below_cloud = (df["C"] < cloud_bottom).astype(int)
        features["ichi_above_cloud"] = above_cloud
        features["ichi_below_cloud"] = below_cloud
        features["ichi_in_cloud"] = ((df["C"] >= cloud_bottom) & (df["C"] <= cloud_top)).astype(int)

        # TK Cross
        tk_cross = (tenkan - kijun) / df["C"]
        features["ichi_tk_cross"] = tk_cross

        # TK Cross Direction Change
        tk_bullish = (tenkan > kijun).astype(bool)
        tk_bullish_prev = tk_bullish.shift(1).fillna(False).astype(bool)
        features["ichi_tk_bullish_cross"] = (tk_bullish & ~tk_bullish_prev).astype(int)
        features["ichi_tk_bearish_cross"] = (~tk_bullish & tk_bullish_prev).astype(int)

        # Price-Kijun Distance
        features["ichi_price_kijun"] = (df["C"] - kijun) / df["C"]

        # Kijun Flat
        kijun_change = kijun.diff().abs()
        features["ichi_kijun_flat"] = (kijun_change < 0.0001 * df["C"]).astype(int)

        # Cloud Color
        bullish_cloud = (senkou_a > senkou_b).astype(int)
        features["ichi_bullish_cloud"] = bullish_cloud

        # Kumo Twist
        cloud_bullish = (senkou_a > senkou_b).astype(bool)
        cloud_bullish_prev = cloud_bullish.shift(1).fillna(False).astype(bool)
        features["ichi_kumo_twist"] = (cloud_bullish != cloud_bullish_prev).astype(int)

        # Chikou Span
        features["ichi_chikou_above"] = (df["C"] > df["C"].shift(kijun_period)).astype(int)

        # === Composite Signals ===
        features["ichi_strong_bullish"] = (
            (above_cloud == 1) & (tk_cross > 0) & (bullish_cloud == 1)
        ).astype(int)

        features["ichi_strong_bearish"] = (
            (below_cloud == 1) & (tk_cross < 0) & (bullish_cloud == 0)
        ).astype(int)

        features["ichi_neutral"] = (
            (features["ichi_in_cloud"] == 1) |
            ((above_cloud != bullish_cloud) & (below_cloud != (1 - bullish_cloud)))
        ).astype(int)

        # Distance to Cloud
        dist_to_top = (cloud_top - df["C"]) / df["C"]
        dist_to_bottom = (df["C"] - cloud_bottom) / df["C"]
        features["ichi_dist_to_cloud"] = np.where(
            above_cloud == 1, dist_to_bottom,
            np.where(below_cloud == 1, -dist_to_top, 0)
        )

        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

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
