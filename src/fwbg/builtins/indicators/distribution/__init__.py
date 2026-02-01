"""
Distribution Indicator Plugin.

Enthält:
- Skewness (Asymmetrie der Return-Verteilung)
- Kurtosis (Fat Tails)
- Z-Score Normalisierung dieser Features

Interpretation:
- Positive Skewness: Mehr extreme positive Returns
- Negative Skewness: Mehr extreme negative Returns (Crash-Risiko)
- Hohe Kurtosis: Fat Tails, mehr Extremereignisse
- Niedrige Kurtosis: Dünne Tails, weniger Extremereignisse
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


@register_indicator("distribution")
class DistributionIndicators(BaseIndicator):
    """
    Return-Verteilungs-Features.

    Features:
    - Rolling Skewness (20, 50, 100)
    - Rolling Kurtosis (20, 50, 100)
    - Z-Score normalisierte Versionen
    - Skewness/Kurtosis Änderungen
    """

    group = "distribution"

    def compute(
        self,
        df: pd.DataFrame,
        windows: List[int] = None,
        z_score_lookback: int = 200,
        compute_changes: bool = True,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Verteilungs-Features.

        Args:
            df: DataFrame mit OHLC-Daten
            windows: Rolling-Fenster für Skewness/Kurtosis (default: [20, 50, 100])
            z_score_lookback: Lookback für Z-Score Normalisierung
            compute_changes: Berechne Änderungen der Features

        Returns:
            DataFrame mit Distribution-Features
        """
        if windows is None:
            windows = [20, 50, 100]

        returns = df["C"].pct_change()

        for period in windows:
            # Rolling Skewness
            skew = returns.rolling(period).skew()
            df[f"dist_skew_{period}"] = skew

            # Rolling Kurtosis (Excess Kurtosis, 0 = Normal)
            kurt = returns.rolling(period).kurt()
            df[f"dist_kurt_{period}"] = kurt

        # Z-Score Normalisierung (relativ zur eigenen Historie)
        for period in windows:
            skew_col = f"dist_skew_{period}"
            kurt_col = f"dist_kurt_{period}"

            # Skewness Z-Score
            skew_mean = df[skew_col].rolling(z_score_lookback).mean()
            skew_std = df[skew_col].rolling(z_score_lookback).std()
            df[f"dist_skew_{period}_z"] = (df[skew_col] - skew_mean) / (skew_std + 1e-10)

            # Kurtosis Z-Score
            kurt_mean = df[kurt_col].rolling(z_score_lookback).mean()
            kurt_std = df[kurt_col].rolling(z_score_lookback).std()
            df[f"dist_kurt_{period}_z"] = (df[kurt_col] - kurt_mean) / (kurt_std + 1e-10)

        # === Änderungs-Features ===
        if compute_changes:
            # Skewness Change (Shift in Distribution)
            if "dist_skew_50" in df.columns:
                df["dist_skew_change_10"] = (
                    df["dist_skew_50"] - df["dist_skew_50"].shift(10)
                )
                df["dist_skew_change_20"] = (
                    df["dist_skew_50"] - df["dist_skew_50"].shift(20)
                )

            # Kurtosis Change (Tail-Risk Change)
            if "dist_kurt_50" in df.columns:
                df["dist_kurt_change_10"] = (
                    df["dist_kurt_50"] - df["dist_kurt_50"].shift(10)
                )
                df["dist_kurt_change_20"] = (
                    df["dist_kurt_50"] - df["dist_kurt_50"].shift(20)
                )

        # === Composite Features ===
        # Tail Risk Score: Kombiniert Kurtosis und negative Skewness
        if "dist_kurt_50" in df.columns and "dist_skew_50" in df.columns:
            # Hohe Kurtosis + negative Skewness = hohes Tail-Risk
            kurt_norm = df["dist_kurt_50"].clip(-3, 10) / 10  # Normalisiert auf ~0-1
            skew_contrib = (-df["dist_skew_50"]).clip(0, 3) / 3  # Nur negative Skewness
            df["dist_tail_risk"] = (kurt_norm + skew_contrib) / 2

        # Distribution Stability (Std der Skewness über Zeit)
        if "dist_skew_50" in df.columns:
            df["dist_stability"] = df["dist_skew_50"].rolling(50).std()

        return df

    def get_feature_columns(self) -> List[str]:
        return [
            # Raw Features
            "dist_skew_20", "dist_skew_50", "dist_skew_100",
            "dist_kurt_20", "dist_kurt_50", "dist_kurt_100",
            # Z-Score
            "dist_skew_20_z", "dist_skew_50_z", "dist_skew_100_z",
            "dist_kurt_20_z", "dist_kurt_50_z", "dist_kurt_100_z",
            # Changes
            "dist_skew_change_10", "dist_skew_change_20",
            "dist_kurt_change_10", "dist_kurt_change_20",
            # Composite
            "dist_tail_risk", "dist_stability",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "windows": [20, 50, 100],
            "z_score_lookback": 200,
            "compute_changes": True,
        }


__all__ = ["DistributionIndicators"]
