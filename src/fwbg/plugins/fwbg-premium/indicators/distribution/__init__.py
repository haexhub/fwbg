"""
Distribution Indicator Plugin.

Enthält:
- Skewness (Asymmetrie der Return-Verteilung)
- Kurtosis (Fat Tails)
- Z-Score Normalisierung dieser Features
- Auto-Korrelation (Persistence/Mean-Reversion auf verschiedenen Lags)

Interpretation:
- Positive Skewness: Mehr extreme positive Returns
- Negative Skewness: Mehr extreme negative Returns (Crash-Risiko)
- Hohe Kurtosis: Fat Tails, mehr Extremereignisse
- Niedrige Kurtosis: Dünne Tails, weniger Extremereignisse
- Positive Auto-Korrelation: Trending/Persistence
- Negative Auto-Korrelation: Mean-Reversion
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide
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
    - Auto-Korrelation auf Lags 1, 5, 10, 20
    """

    name = "distribution"
    version = "2.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        windows: List[int] = None,
        z_score_lookback: int = 200,
        compute_changes: bool = True,
        autocorr_lags: List[int] = None,
        autocorr_window: int = 100,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Verteilungs-Features.

        Args:
            df: DataFrame mit OHLC-Daten
            windows: Rolling-Fenster für Skewness/Kurtosis (default: [20, 50, 100])
            z_score_lookback: Lookback für Z-Score Normalisierung
            compute_changes: Berechne Änderungen der Features
            autocorr_lags: Lags für Auto-Korrelation (default: [1, 5, 10, 20])
            autocorr_window: Rolling-Fenster für Auto-Korrelation (default: 100)

        Returns:
            DataFrame mit Distribution-Features
        """
        if windows is None:
            windows = [20, 50, 100]
        if autocorr_lags is None:
            autocorr_lags = [1, 5, 10, 20]

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
            features[f"dist_skew_{period}_z"] = safe_divide(skew - skew_mean, skew_std)

            # Kurtosis Z-Score
            kurt_mean = kurt.rolling(z_score_lookback).mean()
            kurt_std = kurt.rolling(z_score_lookback).std()
            features[f"dist_kurt_{period}_z"] = safe_divide(kurt - kurt_mean, kurt_std)

        # === Änderungs-Features ===
        if compute_changes and 50 in windows:
            skew_50 = features["dist_skew_50"]
            kurt_50 = features["dist_kurt_50"]

            features["dist_skew_change_10"] = skew_50 - skew_50.shift(10)
            features["dist_skew_change_20"] = skew_50 - skew_50.shift(20)
            features["dist_kurt_change_10"] = kurt_50 - kurt_50.shift(10)
            features["dist_kurt_change_20"] = kurt_50 - kurt_50.shift(20)

        # === Auto-Korrelation Features ===
        # Misst Persistence (positiv) vs. Mean-Reversion (negativ) auf verschiedenen Zeitskalen
        for lag in autocorr_lags:
            features[f"dist_autocorr_{lag}"] = returns.rolling(autocorr_window).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=True,
            )

        # Auto-Korrelation Änderung (Regime-Shift Indikator)
        if 1 in autocorr_lags:
            ac1 = features["dist_autocorr_1"]
            features["dist_autocorr_1_change"] = ac1 - ac1.shift(20)

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
        features_df = shift_features(features, df.index)

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
            # Auto-Korrelation
            "dist_autocorr_1", "dist_autocorr_5",
            "dist_autocorr_10", "dist_autocorr_20",
            "dist_autocorr_1_change",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "windows": [20, 50, 100],
            "z_score_lookback": 200,
            "compute_changes": True,
            "autocorr_lags": [1, 5, 10, 20],
            "autocorr_window": 100,
        }


__all__ = ["DistributionIndicators"]
