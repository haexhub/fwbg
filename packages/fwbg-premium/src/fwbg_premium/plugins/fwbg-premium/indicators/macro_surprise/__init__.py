"""
Macro Surprise / Information Flow Indicator Plugin.

Erkennt ungewöhnliche Preisbewegungen die auf neue Informationen hindeuten:
- Overnight Gaps: Gaps zwischen Sessions
- Session Returns: Intraday vs. Overnight Bewegungen
- Surprise Moves: Unerwartete Volatilität
- Gap Persistence: Werden Gaps gefüllt oder halten sie?

Diese Features erfassen "Information Arrival" - Momente
wo neue Informationen in den Markt kommen.
"""
import numpy as np
import pandas as pd
from typing import List

from fwbg_sdk import BaseIndicator, shift_features, register_indicator


@register_indicator("macro_surprise")
class MacroSurpriseIndicator(BaseIndicator):
    """
    Macro Surprise / Information Flow Features.

    Berechnet Features für ungewöhnliche Marktbewegungen:
    - Gap Analysis: Overnight/Session Gaps
    - Surprise Detection: Moves außerhalb erwarteter Range
    - Volatility Breaks: Unerwartete Vol-Spikes
    - Return Decomposition: Intraday vs. Overnight
    """

    name = "macro_surprise"
    version = "2.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        vol_lookback: int = 20,
        surprise_threshold: float = 2.0,
        gap_ma_period: int = 10,
    ) -> pd.DataFrame:
        """
        Berechnet Macro Surprise Features.

        Args:
            df: DataFrame mit OHLC-Daten
            vol_lookback: Lookback für Volatilitätsschätzung
            surprise_threshold: Std-Abweichungen für "Überraschung"
            gap_ma_period: Periode für Gap-Moving-Average

        Returns:
            DataFrame mit zusätzlichen Spalten
        """
        features = {}

        o = df["O"]
        h = df["H"]
        l = df["L"]
        c = df["C"]
        c_prev = c.shift(1)

        # === GAP ANALYSIS ===

        # Gap = Open vs. Previous Close
        gap = o - c_prev
        gap_pct = gap / c_prev

        features["macro_gap"] = gap
        features["macro_gap_pct"] = gap_pct

        # Gap relativ zur durchschnittlichen Range
        avg_range = (h - l).rolling(vol_lookback).mean()
        features["macro_gap_normalized"] = gap / avg_range

        # Gap Direction
        features["macro_gap_up"] = (gap > 0).astype(float)
        features["macro_gap_down"] = (gap < 0).astype(float)

        # Gap gefüllt? (Close kommt zurück zum vorherigen Close)
        gap_filled = np.where(
            gap > 0,
            c <= c_prev,  # Up-Gap gefüllt wenn Close <= prev Close
            c >= c_prev   # Down-Gap gefüllt wenn Close >= prev Close
        )
        features["macro_gap_filled"] = pd.Series(gap_filled, index=df.index).astype(float)

        # Gap Extension (Markt bewegt sich weiter in Gap-Richtung)
        gap_extended = np.where(
            gap > 0,
            c > o,  # Up-Gap extended wenn Close > Open
            c < o   # Down-Gap extended wenn Close < Open
        )
        features["macro_gap_extended"] = pd.Series(gap_extended, index=df.index).astype(float)

        # Rolling Gap Statistics
        features["macro_gap_avg"] = gap_pct.rolling(gap_ma_period).mean()
        features["macro_gap_std"] = gap_pct.rolling(gap_ma_period).std()

        # === RETURN DECOMPOSITION ===

        # Total Return
        total_return = (c - c_prev) / c_prev
        features["macro_total_return"] = total_return

        # Overnight Return (Gap)
        overnight_return = (o - c_prev) / c_prev
        features["macro_overnight_return"] = overnight_return

        # Intraday Return
        intraday_return = (c - o) / o
        features["macro_intraday_return"] = intraday_return

        # Return Ratio: Wie viel des Returns war Overnight vs. Intraday
        total_abs = total_return.abs()
        overnight_abs = overnight_return.abs()
        total_abs_safe = total_abs.replace(0, np.nan)
        features["macro_overnight_ratio"] = overnight_abs / total_abs_safe

        # === SURPRISE DETECTION ===

        # Expected Range (basierend auf historischer Vol)
        returns = c.pct_change()
        rolling_std = returns.rolling(vol_lookback).std()
        expected_move = rolling_std * c_prev

        # Actual Move
        actual_move = (h - l)

        # Surprise = Actual / Expected
        expected_move_safe = expected_move.replace(0, np.nan)
        features["macro_range_surprise"] = actual_move / expected_move_safe

        # Surprise Binary (Move > threshold * expected)
        features["macro_is_surprise"] = (
            actual_move > surprise_threshold * expected_move
        ).astype(float)

        # Return Surprise
        return_zscore = returns / rolling_std
        features["macro_return_zscore"] = return_zscore
        features["macro_return_surprise"] = (
            return_zscore.abs() > surprise_threshold
        ).astype(float)

        # === VOLATILITY BREAKS ===

        # Realized Vol vs. Expected Vol
        realized_vol = returns.abs().rolling(5).mean()
        expected_vol = realized_vol.rolling(vol_lookback).mean()
        expected_vol_safe = expected_vol.replace(0, np.nan)

        features["macro_vol_ratio"] = realized_vol / expected_vol_safe

        # Vol Spike Detection
        vol_std = realized_vol.rolling(vol_lookback).std()
        vol_zscore = (realized_vol - expected_vol) / vol_std
        features["macro_vol_zscore"] = vol_zscore

        # === STREAK / PERSISTENCE ===

        # Consecutive Gap Direction
        gap_direction = np.sign(gap)
        gap_streak = self._compute_streak(gap_direction)
        features["macro_gap_streak"] = gap_streak

        # Surprise Streak
        is_surprise = features["macro_is_surprise"]
        surprise_streak = self._compute_streak(is_surprise)
        features["macro_surprise_streak"] = surprise_streak

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def _compute_streak(self, series: pd.Series) -> pd.Series:
        """Berechnet aufeinanderfolgende gleiche Werte."""
        # Gruppenwechsel erkennen
        group_change = series != series.shift(1)
        group_id = group_change.cumsum()

        # Innerhalb jeder Gruppe zählen
        streak = series.groupby(group_id).cumcount() + 1

        # Bei 0 oder NaN ist Streak 0
        streak = streak.where(series != 0, 0)
        streak = streak.where(series.notna(), np.nan)

        return streak

    def get_feature_columns(self) -> List[str]:
        """Gibt alle Feature-Spalten zurück."""
        return [
            "macro_gap",
            "macro_gap_pct",
            "macro_gap_normalized",
            "macro_gap_up",
            "macro_gap_down",
            "macro_gap_filled",
            "macro_gap_extended",
            "macro_gap_avg",
            "macro_gap_std",
            "macro_total_return",
            "macro_overnight_return",
            "macro_intraday_return",
            "macro_overnight_ratio",
            "macro_range_surprise",
            "macro_is_surprise",
            "macro_return_zscore",
            "macro_return_surprise",
            "macro_vol_ratio",
            "macro_vol_zscore",
            "macro_gap_streak",
            "macro_surprise_streak",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        """Default-Parameter."""
        return {
            "vol_lookback": 20,
            "surprise_threshold": 2.0,
            "gap_ma_period": 10,
        }


__all__ = ["MacroSurpriseIndicator"]
