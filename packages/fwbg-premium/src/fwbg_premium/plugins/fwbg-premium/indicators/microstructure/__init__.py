"""
Microstructure Indicator Plugin.

Analysiert Intrabar-Dynamik und Marktmikrostruktur-Signale:
- Wick Imbalance: Verhältnis von Upper/Lower Wick
- Intrabar Bias: Open-to-Close Bewegung relativ zur Range
- Range over ATR: Normalisierte Volatilität
- Pressure Score: Kauf-/Verkaufsdruck basierend auf Kerzenstruktur
- Accumulation/Distribution Line: Volume-gewichteter Preistrend
- Chaikin Money Flow (CMF): Geldfluss-Indikator

Diese Features erfassen Informationen, die in OHLC-Daten
versteckt sind aber selten genutzt werden.
"""
import numpy as np
import pandas as pd
from typing import List

from fwbg_sdk import BaseIndicator, shift_features, safe_divide, register_indicator


@register_indicator("microstructure")
class MicrostructureIndicator(BaseIndicator):
    """
    Microstructure/Execution Layer Features.

    Berechnet Features basierend auf Kerzenstruktur:
    - Wick Imbalance: (UpperWick - LowerWick) / Range
    - Intrabar Bias: (Close - Open) / Range
    - Range/ATR Ratio: Normalisierte Bar-Größe
    - Pressure Score: sign(C-O) × body_ratio
    - Rolling Features: Akkumulierte Imbalances
    """

    name = "microstructure"
    version = "2.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        rolling_window: int = 5,
    ) -> pd.DataFrame:
        """
        Berechnet Microstructure-Features.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            atr_period: Periode für ATR-Berechnung
            rolling_window: Fenster für Rolling-Akkumulation

        Returns:
            DataFrame mit zusätzlichen Spalten
        """
        features = {}

        o = df["O"]
        h = df["H"]
        low = df["L"]
        c = df["C"]

        # Bar Range (vermeidet Division durch 0)
        bar_range = h - low
        bar_range_safe = bar_range.replace(0, np.nan)

        # Upper und Lower Wick
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - low
        body = np.abs(c - o)

        # --- Wick Imbalance ---
        # Positiv = mehr Upper Wick (Verkaufsdruck oben)
        # Negativ = mehr Lower Wick (Kaufdruck unten)
        features["micro_wick_imbalance"] = (upper_wick - lower_wick) / bar_range_safe

        # --- Intrabar Bias ---
        # Positiv = Close > Open (bullish)
        # Negativ = Close < Open (bearish)
        features["micro_intrabar_bias"] = (c - o) / bar_range_safe

        # --- Body Ratio ---
        # Wie viel der Range ist Body vs. Wick
        features["micro_body_ratio"] = body / bar_range_safe

        # --- ATR und Range/ATR ---
        tr = pd.concat([
            h - low,
            (h - c.shift(1)).abs(),
            (low - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        features["micro_range_over_atr"] = bar_range / atr

        # --- Pressure Score ---
        # Kombination aus Richtung und Body-Stärke
        direction = np.sign(c - o)
        body_ratio = body / bar_range_safe
        features["micro_pressure_score"] = direction * body_ratio

        # --- Rolling Imbalances ---
        features["micro_wick_imbalance_sum"] = (
            features["micro_wick_imbalance"].rolling(rolling_window).sum()
        )
        features["micro_intrabar_bias_sum"] = (
            features["micro_intrabar_bias"].rolling(rolling_window).sum()
        )
        features["micro_pressure_sum"] = (
            features["micro_pressure_score"].rolling(rolling_window).sum()
        )

        # --- Shadow Extremes ---
        # Maximale Wicks als Proxy für Liquiditäts-Absorption
        features["micro_upper_shadow_max"] = (
            (upper_wick / bar_range_safe).rolling(rolling_window).max()
        )
        features["micro_lower_shadow_max"] = (
            (lower_wick / bar_range_safe).rolling(rolling_window).max()
        )

        # --- Consistency Metrics ---
        # Wie konsistent ist die Richtung?
        direction_series = direction.rolling(rolling_window).mean()
        features["micro_direction_consistency"] = direction_series.abs()

        # --- Volume-weighted (falls V vorhanden) ---
        if "V" in df.columns and df["V"].notna().any() and (df["V"] > 0).any():
            v = df["V"]

            # Volume-gewichteter Pressure Score
            features["micro_vwap_pressure"] = (
                (features["micro_pressure_score"] * v).rolling(rolling_window).sum()
                / v.rolling(rolling_window).sum()
            )

            # Relative Volume (vs. rolling average)
            v_avg = v.rolling(rolling_window * 4).mean()
            features["micro_relative_volume"] = v / v_avg

            # --- Accumulation/Distribution Line ---
            # CLV (Close Location Value): Wo der Close innerhalb der Range liegt
            # CLV = ((C - L) - (H - C)) / (H - L) = (2C - L - H) / (H - L)
            clv = safe_divide(2 * c - low - h, bar_range)
            ad_flow = clv * v
            features["micro_ad_line"] = ad_flow.cumsum()
            # Normalisiert: A/D relativ zum Rolling-Mean (für Stationarität)
            ad_cumsum = features["micro_ad_line"]
            ad_mean = ad_cumsum.rolling(50).mean()
            ad_std = ad_cumsum.rolling(50).std()
            features["micro_ad_zscore"] = safe_divide(ad_cumsum - ad_mean, ad_std)

            # --- Chaikin Money Flow (CMF) ---
            # CMF = Sum(CLV * Volume, N) / Sum(Volume, N)
            for cmf_window in [10, 20]:
                features[f"micro_cmf_{cmf_window}"] = safe_divide(
                    ad_flow.rolling(cmf_window).sum(),
                    v.rolling(cmf_window).sum(),
                )
        else:
            # Fallback wenn kein Volume
            features["micro_vwap_pressure"] = features["micro_pressure_sum"] / rolling_window
            features["micro_relative_volume"] = 1.0
            # A/D und CMF brauchen Volume - nutze CLV als Proxy
            clv = safe_divide(2 * c - low - h, bar_range)
            features["micro_ad_line"] = clv.cumsum()
            ad_cumsum = features["micro_ad_line"]
            ad_mean = ad_cumsum.rolling(50).mean()
            ad_std = ad_cumsum.rolling(50).std()
            features["micro_ad_zscore"] = safe_divide(ad_cumsum - ad_mean, ad_std)
            for cmf_window in [10, 20]:
                features[f"micro_cmf_{cmf_window}"] = clv.rolling(cmf_window).mean()

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        """Gibt alle Feature-Spalten zurück."""
        return [
            "micro_wick_imbalance",
            "micro_intrabar_bias",
            "micro_body_ratio",
            "micro_range_over_atr",
            "micro_pressure_score",
            "micro_wick_imbalance_sum",
            "micro_intrabar_bias_sum",
            "micro_pressure_sum",
            "micro_upper_shadow_max",
            "micro_lower_shadow_max",
            "micro_direction_consistency",
            "micro_vwap_pressure",
            "micro_relative_volume",
            # Volume Flow
            "micro_ad_line",
            "micro_ad_zscore",
            "micro_cmf_10",
            "micro_cmf_20",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        """Default-Parameter."""
        return {
            "atr_period": 14,
            "rolling_window": 5,
        }


__all__ = ["MicrostructureIndicator"]
