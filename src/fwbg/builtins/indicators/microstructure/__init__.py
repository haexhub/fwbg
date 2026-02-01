"""
Microstructure Indicator Plugin.

Analysiert Intrabar-Dynamik und Marktmikrostruktur-Signale:
- Wick Imbalance: Verhältnis von Upper/Lower Wick
- Intrabar Bias: Open-to-Close Bewegung relativ zur Range
- Range over ATR: Normalisierte Volatilität
- Pressure Score: Kauf-/Verkaufsdruck basierend auf Kerzenstruktur

Diese Features erfassen Informationen, die in OHLC-Daten
versteckt sind aber selten genutzt werden.
"""
import numpy as np
import pandas as pd
from typing import List

from fwbg.plugins import BaseIndicator
from fwbg.core.registry import register_indicator


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

    group = "microstructure"

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
        o = df["O"]
        h = df["H"]
        l = df["L"]
        c = df["C"]

        # Bar Range (vermeidet Division durch 0)
        bar_range = h - l
        bar_range_safe = bar_range.replace(0, np.nan)

        # Upper und Lower Wick
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        body = np.abs(c - o)

        # --- Wick Imbalance ---
        # Positiv = mehr Upper Wick (Verkaufsdruck oben)
        # Negativ = mehr Lower Wick (Kaufdruck unten)
        df["micro_wick_imbalance"] = (upper_wick - lower_wick) / bar_range_safe

        # --- Intrabar Bias ---
        # Positiv = Close > Open (bullish)
        # Negativ = Close < Open (bearish)
        df["micro_intrabar_bias"] = (c - o) / bar_range_safe

        # --- Body Ratio ---
        # Wie viel der Range ist Body vs. Wick
        df["micro_body_ratio"] = body / bar_range_safe

        # --- ATR und Range/ATR ---
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        df["micro_range_over_atr"] = bar_range / atr

        # --- Pressure Score ---
        # Kombination aus Richtung und Body-Stärke
        direction = np.sign(c - o)
        body_ratio = body / bar_range_safe
        df["micro_pressure_score"] = direction * body_ratio

        # --- Rolling Imbalances ---
        df["micro_wick_imbalance_sum"] = (
            df["micro_wick_imbalance"].rolling(rolling_window).sum()
        )
        df["micro_intrabar_bias_sum"] = (
            df["micro_intrabar_bias"].rolling(rolling_window).sum()
        )
        df["micro_pressure_sum"] = (
            df["micro_pressure_score"].rolling(rolling_window).sum()
        )

        # --- Shadow Extremes ---
        # Maximale Wicks als Proxy für Liquiditäts-Absorption
        df["micro_upper_shadow_max"] = (
            (upper_wick / bar_range_safe).rolling(rolling_window).max()
        )
        df["micro_lower_shadow_max"] = (
            (lower_wick / bar_range_safe).rolling(rolling_window).max()
        )

        # --- Consistency Metrics ---
        # Wie konsistent ist die Richtung?
        direction_series = direction.rolling(rolling_window).mean()
        df["micro_direction_consistency"] = direction_series.abs()

        # --- Volume-weighted (falls V vorhanden) ---
        if "V" in df.columns and df["V"].notna().any() and (df["V"] > 0).any():
            v = df["V"]
            v_safe = v.replace(0, np.nan)

            # Volume-gewichteter Pressure Score
            df["micro_vwap_pressure"] = (
                (df["micro_pressure_score"] * v).rolling(rolling_window).sum()
                / v.rolling(rolling_window).sum()
            )

            # Relative Volume (vs. rolling average)
            v_avg = v.rolling(rolling_window * 4).mean()
            df["micro_relative_volume"] = v / v_avg
        else:
            # Fallback wenn kein Volume
            df["micro_vwap_pressure"] = df["micro_pressure_sum"] / rolling_window
            df["micro_relative_volume"] = 1.0

        # NaN-Behandlung für erste Perioden
        # (behalten wir bei, da Modelle damit umgehen können)

        return df

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
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        """Default-Parameter."""
        return {
            "atr_period": 14,
            "rolling_window": 5,
        }


__all__ = ["MicrostructureIndicator"]
