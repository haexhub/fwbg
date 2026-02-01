"""
Fractional Differentiation Preprocessor Plugin.

Fractional Differentiation nach López de Prado:
- Macht Zeitreihen stationär unter Beibehaltung von Memory
- d=0: Keine Transformation (original)
- d=1: Volle Differentiation (wie pct_change, verliert Memory)
- d=0.3-0.5: Optimal für Trading (stationär + Memory)
"""
from typing import List
import numpy as np
import pandas as pd
from scipy.special import gamma

from fwbg.plugins import BasePreprocessor
from fwbg.core import register_preprocessor


def _get_weights(d: float, size: int) -> np.ndarray:
    """Berechnet Gewichte für Fractional Differentiation."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def _frac_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Wendet Fractional Differentiation auf eine Serie an.

    Args:
        series: Input-Serie
        d: Differentiation-Exponent (0-1)
        threshold: Minimum-Gewicht für Cutoff

    Returns:
        Transformierte Serie
    """
    if d == 0:
        return series

    # Gewichte berechnen (mit Cutoff für Effizienz)
    weights = _get_weights(d, len(series))
    weights = weights[np.abs(weights) > threshold]
    width = len(weights)

    result = pd.Series(index=series.index, dtype=float)

    for i in range(width - 1, len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i - width + 1:i + 1].values)

    return result


def _find_optimal_d(
    series: pd.Series,
    max_d: float = 1.0,
    p_threshold: float = 0.05
) -> float:
    """
    Findet optimales d für Stationarität via ADF-Test.

    Sucht das kleinste d bei dem die Serie stationär wird.
    """
    from statsmodels.tsa.stattools import adfuller

    for d in np.arange(0.1, max_d + 0.1, 0.1):
        diff_series = _frac_diff(series, d)
        diff_series = diff_series.dropna()

        if len(diff_series) < 100:
            continue

        try:
            adf_result = adfuller(diff_series)
            p_value = adf_result[1]

            if p_value < p_threshold:
                return round(d, 1)
        except Exception:
            continue

    return 0.5  # Default wenn kein optimales d gefunden


@register_preprocessor("fractional_diff")
class FractionalDiffPreprocessor(BasePreprocessor):
    """
    Fractional Differentiation Preprocessor.

    Macht OHLC-Spalten stationär unter Beibehaltung von Memory.
    """

    order = 10  # Früh ausführen

    def transform(
        self,
        df: pd.DataFrame,
        auto_d: bool = True,
        default_d: float = 0.4,
        columns: List[str] = None,
        **params
    ) -> pd.DataFrame:
        """
        Wendet Fractional Differentiation auf DataFrame an.

        Args:
            df: Input DataFrame mit OHLC-Daten
            auto_d: Automatische d-Optimierung via ADF-Test
            default_d: Default d-Wert wenn auto_d=False
            columns: Spalten zu transformieren (default: O, H, L, C)

        Returns:
            Transformierter DataFrame
        """
        if columns is None:
            columns = ["O", "H", "L", "C"]

        # Nur vorhandene Spalten transformieren
        columns = [c for c in columns if c in df.columns]

        if not columns:
            return df

        # Optimales d finden
        if auto_d:
            d = _find_optimal_d(df["C"])
        else:
            d = default_d

        # Spalten transformieren
        for col in columns:
            df[col] = _frac_diff(df[col], d)

        # NaN am Anfang durch Dropna entfernen
        first_valid = df[columns[0]].first_valid_index()
        if first_valid is not None:
            df = df.loc[first_valid:]

        # Metadata speichern
        df.attrs["frac_diff_d"] = d

        return df

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "auto_d": True,
            "default_d": 0.4,
            "columns": ["O", "H", "L", "C"],
        }


__all__ = ["FractionalDiffPreprocessor"]
