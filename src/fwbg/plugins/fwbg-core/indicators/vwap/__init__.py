"""
VWAP Indicator Plugin.

Berechnet den Volume-Weighted Average Price (VWAP) mit Session-Reset
und abgeleitete Mean-Reversion-Features:
- VWAP-Abweichung (normalisiert)
- Rolling Z-Score der Abweichung
- VWAP-Bands (±1σ, ±2σ) basierend auf volume-gewichteter Standardabweichung
- Bandposition als normalisiertes Signal

Volume-Fallback: Wenn V-Spalte fehlt oder leer, wird equal-weight (typical price)
verwendet.
"""
from typing import List, Union
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide, EPSILON


@register_indicator("vwap")
class VwapIndicator(BaseIndicator):
    """
    VWAP (Volume-Weighted Average Price) mit Session-Reset.

    Berechnet VWAP anhand der Typical Price und Volume (oder equal-weight Fallback).
    Der VWAP wird zu Beginn jeder Session zurückgesetzt (session_start_hour).

    Features:
    - vwap: VWAP-Wert
    - vwap_deviation: Normalisierte Abweichung vom VWAP
    - vwap_zscore_N: Rolling Z-Score der Abweichung (Mean-Reversion-Signal)
    - vwap_upper_N, vwap_lower_N: VWAP-Bands (±N × volume-gew. Std)
    - vwap_band_pos: Bandposition innerhalb ±1σ (0=unten, 1=oben)
    - vwap_above: Binary ob Preis über VWAP liegt
    """

    name = "vwap"
    version = "1.0.0"
    benefits_from_stationary = False

    def compute(
        self,
        df: pd.DataFrame,
        session_start_hour: int = 9,
        zscore_windows: List[int] = None,
        band_multipliers: List[Union[int, float]] = None,
        **params,
    ) -> pd.DataFrame:
        """
        Berechnet VWAP und abgeleitete Mean-Reversion-Features.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C) und optional V
            session_start_hour: Stunde des VWAP-Resets (0-23), z.B. 9 für US, 8 für EU
            zscore_windows: Fenstergröße für Rolling Z-Score (default: [20, 50])
            band_multipliers: Std-Multiplikatoren für VWAP-Bands (default: [1.0, 2.0])

        Returns:
            DataFrame mit VWAP-Feature-Spalten
        """
        if zscore_windows is None:
            zscore_windows = [20, 50]
        if band_multipliers is None:
            band_multipliers = [1.0, 2.0]

        features = {}

        h = df["H"]
        lo = df["L"]
        c = df["C"]

        # Typical Price: Referenzpunkt für VWAP
        tp = (h + lo + c) / 3.0

        # Gewichte: Volume wenn verfügbar, sonst equal-weight
        has_volume = (
            "V" in df.columns
            and df["V"].notna().any()
            and (df["V"] > 0).any()
        )
        if has_volume:
            w = df["V"].copy().fillna(0.0).clip(lower=0.0)
            w = w.where(w > 0, EPSILON)
        else:
            w = pd.Series(1.0, index=df.index)

        # Session-ID: neue Session wenn hour == session_start_hour
        # und die vorherige Bar eine andere Stunde hatte
        hours = pd.Series(df.index.hour, index=df.index)
        prev_hour = hours.shift(1)
        session_boundary = (hours == session_start_hour) & (prev_hour != session_start_hour)
        session_boundary.iloc[0] = True
        session_id = session_boundary.cumsum()

        # Kumulativer VWAP pro Session via groupby + cumsum
        tp_w = tp * w
        cum_tp_w = tp_w.groupby(session_id).transform("cumsum")
        cum_w = w.groupby(session_id).transform("cumsum")
        vwap = safe_divide(cum_tp_w, cum_w)
        features["vwap"] = vwap

        # Normalisierte Abweichung: (Close - VWAP) / VWAP
        deviation = safe_divide(c - vwap, vwap.abs())
        features["vwap_deviation"] = deviation

        # Rolling Z-Score der Abweichung
        for window in zscore_windows:
            min_p = max(window // 2, 2)
            roll_mean = deviation.rolling(window, min_periods=min_p).mean()
            roll_std = deviation.rolling(window, min_periods=min_p).std()
            features[f"vwap_zscore_{window}"] = safe_divide(deviation - roll_mean, roll_std)

        # Volume-gewichtete Standardabweichung für VWAP-Bands
        dev_sq_w = ((tp - vwap) ** 2) * w
        cum_dev_sq_w = dev_sq_w.groupby(session_id).transform("cumsum")
        vwap_variance = safe_divide(cum_dev_sq_w, cum_w)
        vwap_std = np.sqrt(vwap_variance.clip(lower=0))

        # VWAP-Bands
        for mult in band_multipliers:
            mult_label = int(mult) if mult == int(mult) else str(mult).replace(".", "_")
            features[f"vwap_upper_{mult_label}"] = vwap + mult * vwap_std
            features[f"vwap_lower_{mult_label}"] = vwap - mult * vwap_std

        # Bandposition: 0=unteres ±1σ-Band, 0.5=VWAP, 1=oberes ±1σ-Band
        lower_1 = vwap - vwap_std
        band_width_1 = 2 * vwap_std
        features["vwap_band_pos"] = safe_divide(c - lower_1, band_width_1)

        # Binary: Preis über VWAP
        features["vwap_above"] = (c > vwap).astype(float)

        # CRITICAL: Alle Features um 1 Bar shiftenim um Lookahead Bias zu verhindern
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        """Alle VWAP-Feature-Spalten (basierend auf Default-Parametern)."""
        return [
            "vwap",
            "vwap_deviation",
            "vwap_zscore_20",
            "vwap_zscore_50",
            "vwap_upper_1",
            "vwap_lower_1",
            "vwap_upper_2",
            "vwap_lower_2",
            "vwap_band_pos",
            "vwap_above",
        ]

    def get_signal_columns(self) -> List[str]:
        return ["vwap_above"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "session_start_hour": 9,
            "zscore_windows": [20, 50],
            "band_multipliers": [1.0, 2.0],
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "session_start_hour": {
                "type": "int",
                "default": 9,
                "description": "Stunde des Tages (0-23) bei der VWAP zurückgesetzt wird. "
                               "9 für US-Märkte (09:00 Uhr ET), 8 für EU-Märkte.",
                "min": 0,
                "max": 23,
                "step": 1,
            },
            "zscore_windows": {
                "type": "list[int]",
                "default": [20, 50],
                "description": "Rollende Fenstergröße für Z-Score der VWAP-Abweichung. "
                               "Kleinere Fenster (20) reagieren schneller auf Extremwerte, "
                               "größere (50) glätten mehr Noise heraus.",
                "min": 5,
                "max": 500,
            },
            "band_multipliers": {
                "type": "list[float]",
                "default": [1.0, 2.0],
                "description": "Multiplikatoren für VWAP-Bands in Einheiten der "
                               "volume-gewichteten Standardabweichung. 1.0 = ±1σ, 2.0 = ±2σ. "
                               "Preise außerhalb ±2σ sind Mean-Reversion-Signale.",
                "min": 0.5,
                "max": 5.0,
            },
        }


__all__ = ["VwapIndicator"]
