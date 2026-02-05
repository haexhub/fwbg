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

        features = {}
        returns = df["C"].pct_change()

        for period in windows:
            # Rolling Skewness
            features[f"dist_skew_{period}"] = returns.rolling(period).skew()
            # Rolling Kurtosis (Excess Kurtosis, 0 = Normal)
            features[f"dist_kurt_{period}"] = returns.rolling(period).kurt()

        # Z-Score Normalisierung (relativ zur eigenen Historie)
        for period in windows:
            skew = features[f"dist_skew_{period}"]
            kurt = features[f"dist_kurt_{period}"]

            # Skewness Z-Score
            skew_mean = skew.rolling(z_score_lookback).mean()
            skew_std = skew.rolling(z_score_lookback).std()
            features[f"dist_skew_{period}_z"] = (skew - skew_mean) / (skew_std + 1e-10)

            # Kurtosis Z-Score
            kurt_mean = kurt.rolling(z_score_lookback).mean()
            kurt_std = kurt.rolling(z_score_lookback).std()
            features[f"dist_kurt_{period}_z"] = (kurt - kurt_mean) / (kurt_std + 1e-10)

        # === Änderungs-Features ===
        if compute_changes and 50 in windows:
            skew_50 = features["dist_skew_50"]
            kurt_50 = features["dist_kurt_50"]

            features["dist_skew_change_10"] = skew_50 - skew_50.shift(10)
            features["dist_skew_change_20"] = skew_50 - skew_50.shift(20)
            features["dist_kurt_change_10"] = kurt_50 - kurt_50.shift(10)
            features["dist_kurt_change_20"] = kurt_50 - kurt_50.shift(20)

        # === Composite Features ===
        if 50 in windows:
            skew_50 = features["dist_skew_50"]
            kurt_50 = features["dist_kurt_50"]

            # Tail Risk Score: Kombiniert Kurtosis und negative Skewness
            kurt_norm = kurt_50.clip(-3, 10) / 10
            skew_contrib = (-skew_50).clip(0, 3) / 3
            features["dist_tail_risk"] = (kurt_norm + skew_contrib) / 2

            # Distribution Stability
            features["dist_stability"] = skew_50.rolling(50).std()

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        # At bar i, the model should use features from bar i-1, not bar i
        features_df = pd.DataFrame(features, index=df.index)
        for col in features_df.columns:
            features_df[col] = features_df[col].shift(1)

        return pd.concat([df, features_df], axis=1)

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
