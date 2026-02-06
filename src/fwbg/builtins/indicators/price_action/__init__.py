"""
Price Action Indicator Plugin.

Enthält:
- Candle-basierte Features (Body Ratio, Range Position)
- Higher Highs / Lower Lows
- Gap Analysis
- Volume Features (falls verfügbar)

Price Action ist die Basis jeder technischen Analyse:
- Body Ratio zeigt Momentum
- Range Position zeigt wo Close relativ zum Range liegt
- HH/LL zeigt Trend-Struktur
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide
from fwbg.core import register_indicator


@register_indicator("price_action")
class PriceActionIndicators(BaseIndicator):
    """
    Price Action Features.

    Features:
    - Range Position (wo liegt Close im High-Low Range)
    - Body Ratio (Größe des Candle-Bodies)
    - Higher Highs / Lower Lows Counter
    - Gap Analysis
    - Volume-basierte Features (OBV, MFI - falls Volume verfügbar)
    """

    group = "price_action"

    def compute(
        self,
        df: pd.DataFrame,
        hh_ll_period: int = 5,
        compute_volume: bool = True,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Price Action Features.

        Args:
            df: DataFrame mit OHLC-Daten
            hh_ll_period: Periode für HH/LL Rolling Sum (default: 5)
            compute_volume: Berechne Volume Features falls Volume vorhanden

        Returns:
            DataFrame mit Price Action Features
        """
        features = {}

        # Bar Range (für safe_divide)
        bar_range = df["H"] - df["L"]

        # Range Position: Wo liegt Close im High-Low Range (0=Low, 1=High)
        features["pa_range_pos"] = safe_divide(df["C"] - df["L"], bar_range)

        # Body Ratio: Wie viel vom Range ist der Body (0=Doji, 1=Full Body)
        features["pa_body_ratio"] = safe_divide(abs(df["C"] - df["O"]), bar_range)

        # Body Direction: Bullish (+1) vs Bearish (-1)
        features["pa_body_dir"] = np.sign(df["C"] - df["O"])

        # Upper/Lower Shadow Ratio
        features["pa_upper_shadow"] = safe_divide(
            df["H"] - df[["C", "O"]].max(axis=1), bar_range
        )
        features["pa_lower_shadow"] = safe_divide(
            df[["C", "O"]].min(axis=1) - df["L"], bar_range
        )

        # Higher Highs / Lower Lows Counter
        features["pa_hh"] = (df["H"] > df["H"].shift(1)).astype(int).rolling(hh_ll_period).sum()
        features["pa_ll"] = (df["L"] < df["L"].shift(1)).astype(int).rolling(hh_ll_period).sum()

        # Higher Lows / Lower Highs (Trend-Struktur)
        features["pa_hl"] = (df["L"] > df["L"].shift(1)).astype(int).rolling(hh_ll_period).sum()
        features["pa_lh"] = (df["H"] < df["H"].shift(1)).astype(int).rolling(hh_ll_period).sum()

        # Trend Structure Score: HH+HL - LL-LH
        features["pa_trend_structure"] = (features["pa_hh"] + features["pa_hl"]) - (features["pa_ll"] + features["pa_lh"])

        # Gap Analysis
        gap = (df["O"] - df["C"].shift(1)) / df["C"].shift(1)
        features["pa_gap"] = gap
        features["pa_gap_abs"] = abs(gap)

        # Gap Direction (1=Gap Up, -1=Gap Down, 0=No significant gap)
        gap_threshold = 0.001  # 0.1%
        features["pa_gap_dir"] = np.where(
            gap > gap_threshold, 1,
            np.where(gap < -gap_threshold, -1, 0)
        )

        # Gap Fill: Wurde der Gap gefüllt?
        prev_close = df["C"].shift(1)
        features["pa_gap_filled"] = np.where(
            features["pa_gap_dir"] == 1, (df["L"] <= prev_close).astype(int),
            np.where(
                features["pa_gap_dir"] == -1, (df["H"] >= prev_close).astype(int),
                0
            )
        )

        # Consecutive Candles (Streak)
        bullish = (df["C"] > df["O"]).astype(int)
        bearish = (df["C"] < df["O"]).astype(int)

        # Bullish Streak
        bullish_streak = bullish.groupby((bullish != bullish.shift()).cumsum()).cumcount() + 1
        features["pa_bullish_streak"] = bullish_streak * bullish

        # Bearish Streak
        bearish_streak = bearish.groupby((bearish != bearish.shift()).cumsum()).cumcount() + 1
        features["pa_bearish_streak"] = bearish_streak * bearish

        # Range Expansion/Contraction
        current_range = df["H"] - df["L"]
        avg_range = current_range.rolling(20).mean()
        features["pa_range_expansion"] = safe_divide(current_range, avg_range)

        # Inside Bar (High < Previous High AND Low > Previous Low)
        features["pa_inside_bar"] = (
            (df["H"] < df["H"].shift(1)) & (df["L"] > df["L"].shift(1))
        ).astype(int)

        # Outside Bar (High > Previous High AND Low < Previous Low)
        features["pa_outside_bar"] = (
            (df["H"] > df["H"].shift(1)) & (df["L"] < df["L"].shift(1))
        ).astype(int)

        # === Volume Features (optional) ===
        if compute_volume:
            vol_col = "V" if "V" in df.columns else (
                "Volume" if "Volume" in df.columns else None
            )

            if vol_col:
                volume = df[vol_col]

                # On Balance Volume Change
                obv = ta.volume.on_balance_volume(df["C"], volume)
                features["vol_obv_change"] = obv.pct_change(periods=5)

                # Money Flow Index
                features["vol_mfi"] = ta.volume.money_flow_index(
                    df["H"], df["L"], df["C"], volume
                )

                # Volume Relative to Average
                vol_relative = safe_divide(volume, volume.rolling(20).mean())
                features["vol_relative"] = vol_relative

                # Volume Price Trend
                features["vol_price_trend"] = features["pa_body_dir"] * vol_relative

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            # Basic Candle Features
            "pa_range_pos", "pa_body_ratio", "pa_body_dir",
            "pa_upper_shadow", "pa_lower_shadow",
            # Trend Structure
            "pa_hh", "pa_ll", "pa_hl", "pa_lh", "pa_trend_structure",
            # Gap Features
            "pa_gap", "pa_gap_abs", "pa_gap_dir", "pa_gap_filled",
            # Streak Features
            "pa_bullish_streak", "pa_bearish_streak",
            # Range Features
            "pa_range_expansion", "pa_inside_bar", "pa_outside_bar",
            # Volume Features (optional)
            "vol_obv_change", "vol_mfi", "vol_relative", "vol_price_trend",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "hh_ll_period": 5,
            "compute_volume": True,
        }


__all__ = ["PriceActionIndicators"]
