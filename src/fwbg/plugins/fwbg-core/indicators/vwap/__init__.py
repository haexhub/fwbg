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

from fwbg_sdk import (
    BaseIndicator,
    register_indicator,
    shift_features,
    safe_divide,
    EPSILON,
)


def _mult_label(mult: Union[int, float]) -> str:
    """Column suffix of a band multiplier: 1.0 -> "1", 1.5 -> "1_5"."""
    return str(int(mult)) if mult == int(mult) else str(mult).replace(".", "_")


def _session_keys(
    index: pd.DatetimeIndex, session_start_hour: int, session_tz: str
) -> np.ndarray:
    """
    Session label of every bar: the wall-clock date of the last
    ``session_start_hour`` o'clock at or before the bar, in ``session_tz``.

    A naive index is read as UTC (the loader's convention). Every bar of a new
    session start gets a new label even when the bar at exactly
    ``session_start_hour`` is missing, daylight saving time is followed, and
    bars of one day or coarser get a label each (the VWAP restarts every bar).
    """
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert(session_tz).tz_localize(None)
    return (local - pd.Timedelta(hours=session_start_hour)).normalize().to_numpy()


@register_indicator("vwap")
class VwapIndicator(BaseIndicator):
    """
    VWAP (Volume-Weighted Average Price) mit Session-Reset.

    Berechnet VWAP anhand der Typical Price und Volume (oder equal-weight Fallback).
    Der VWAP wird zu Beginn jeder Session zurückgesetzt: um session_start_hour Uhr
    in der Zeitzone session_tz (Default UTC; naive Indizes gelten als UTC). Bei
    Tages- oder gröberen Bars beginnt jede Bar eine neue Session, VWAP ist dann
    der Typical Price der Bar.

    Die Bands nutzen die volume-gewichtete Standardabweichung der Typical Prices
    der Session bis zur Bar: sqrt(sum(w*tp^2)/sum(w) - vwap^2).

    Features:
    - vwap: VWAP-Wert
    - vwap_deviation: Normalisierte Abweichung vom VWAP
    - vwap_zscore_N: Rolling Z-Score der Abweichung (Mean-Reversion-Signal)
    - vwap_upper_N, vwap_lower_N: VWAP-Bands (±N × volume-gew. Std)
    - vwap_band_pos: Bandposition innerhalb ±1σ (0=unten, 1=oben)
    - vwap_above: Binary ob Preis über VWAP liegt
    """

    name = "vwap"
    version = "1.1.0"
    benefits_from_stationary = False

    def compute(
        self,
        df: pd.DataFrame,
        session_start_hour: int = 9,
        session_tz: str = "UTC",
        zscore_windows: List[int] = None,
        band_multipliers: List[Union[int, float]] = None,
        **params,
    ) -> pd.DataFrame:
        """
        Berechnet VWAP und abgeleitete Mean-Reversion-Features.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C) und optional V
            session_start_hour: Stunde des VWAP-Resets (0-23) in session_tz
            session_tz: IANA-Zeitzone von session_start_hour (Default UTC), z.B.
                "America/New_York" für US-Märkte; folgt der Sommerzeit
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
        has_volume = "V" in df.columns and df["V"].notna().any() and (df["V"] > 0).any()
        if has_volume:
            w = df["V"].copy().fillna(0.0).clip(lower=0.0)
            w = w.where(w > 0, EPSILON)
        else:
            w = pd.Series(1.0, index=df.index)

        # Session-ID: neue Session bei jedem neuen session_start_hour (Ortszeit)
        session_id = pd.Series(
            _session_keys(df.index, session_start_hour, session_tz), index=df.index
        )

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
            features[f"vwap_zscore_{window}"] = safe_divide(
                deviation - roll_mean, roll_std
            )

        # Volume-gewichtete Standardabweichung der Typical Prices der Session bis
        # zur Bar: E[tp^2] - E[tp]^2 (gegen die laufende Session-Basis zentriert,
        # damit die Differenz bei hohen Preisen numerisch stabil bleibt)
        base = tp.groupby(session_id).transform("first")
        centered = tp - base
        mean_c = safe_divide(
            (centered * w).groupby(session_id).transform("cumsum"), cum_w
        )
        mean_sq_c = safe_divide(
            ((centered**2) * w).groupby(session_id).transform("cumsum"), cum_w
        )
        vwap_std = np.sqrt((mean_sq_c - mean_c**2).clip(lower=0))

        # VWAP-Bands
        for mult in band_multipliers:
            mult_label = _mult_label(mult)
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

    def get_feature_columns(self, params=None) -> List[str]:
        """Feature columns compute() produces for ``params`` (defaults when None)."""
        params = params or {}
        windows = params.get("zscore_windows") or [20, 50]
        multipliers = params.get("band_multipliers") or [1.0, 2.0]
        cols = ["vwap", "vwap_deviation"]
        cols += [f"vwap_zscore_{w}" for w in windows]
        for mult in multipliers:
            label = _mult_label(mult)
            cols += [f"vwap_upper_{label}", f"vwap_lower_{label}"]
        cols += ["vwap_band_pos", "vwap_above"]
        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "session_start_hour": 9,
            "session_tz": "UTC",
            "zscore_windows": [20, 50],
            "band_multipliers": [1.0, 2.0],
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "session_start_hour": {
                "type": "int",
                "default": 9,
                "description": "Stunde des Tages (0-23) in session_tz, bei der VWAP "
                "zurückgesetzt wird. Bei Tages- oder gröberen Bars beginnt jede Bar "
                "eine neue Session.",
                "min": 0,
                "max": 23,
                "step": 1,
            },
            "session_tz": {
                "type": "string",
                "default": "UTC",
                "description": "IANA-Zeitzone von session_start_hour, z.B. "
                "America/New_York (US-Märkte) oder Europe/Berlin. Naive Zeitstempel "
                "gelten als UTC.",
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
