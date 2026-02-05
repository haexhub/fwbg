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


def _frac_diff(series: pd.Series, d: float, threshold: float = 1e-5, max_window: int = 500) -> pd.Series:
    """
    Wendet Fractional Differentiation auf eine Serie an.

    Args:
        series: Input-Serie
        d: Differentiation-Exponent (0-1)
        threshold: Minimum-Gewicht für Cutoff
        max_window: Maximale Anzahl von Gewichten (verhindert Memory-Probleme bei langen Serien)

    Returns:
        Transformierte Serie
    """
    if d == 0:
        return series

    # Gewichte berechnen (limitiert auf max_window für Performance)
    # Nach López de Prado: Gewichte werden nach ~100-500 Bars vernachlässigbar klein
    window_size = min(max_window, len(series))
    weights = _get_weights(d, window_size)
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

    Folgt sklearn fit/transform Pattern um Lookahead Bias zu verhindern:
    - fit() lernt optimales d NUR von Train-Daten
    - transform() wendet das gelernte d an (auf Train/Test/OOS)
    """

    order = 10  # Früh ausführen

    def fit(
        self,
        df: pd.DataFrame,
        auto_d: bool = True,
        default_d: float = 0.4,
        columns: List[str] = None,
        **params
    ) -> "FractionalDiffPreprocessor":
        """
        Lernt optimales d von Train-Daten.

        WICHTIG: Diese Methode MUSS auf Train-Daten aufgerufen werden,
        um Lookahead Bias zu verhindern!

        Args:
            df: Train DataFrame mit OHLC-Daten
            auto_d: Automatische d-Optimierung via ADF-Test (NUR auf Train-Daten!)
            default_d: Default d-Wert wenn auto_d=False
            columns: Spalten zu transformieren (default: O, H, L, C)

        Returns:
            self (fitted preprocessor)
        """
        if columns is None:
            columns = ["O", "H", "L", "C"]

        # Nur vorhandene Spalten speichern
        self.columns_ = [c for c in columns if c in df.columns]

        if not self.columns_:
            self.d_ = 0.0  # Keine Transformation
            self.fitted_ = True
            return self

        # Optimales d finden (NUR auf Train-Daten!)
        if auto_d:
            self.d_ = _find_optimal_d(df["C"])
        else:
            self.d_ = default_d

        self.fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.DataFrame:
        """
        Wendet Fractional Differentiation mit gelerntem d an.

        Diese Methode verwendet das in fit() gelernte d und kann
        auf Train, Test und OOS-Daten angewendet werden.

        Args:
            df: Input DataFrame mit OHLC-Daten

        Returns:
            Transformierter DataFrame

        Raises:
            RuntimeError: Wenn fit() noch nicht aufgerufen wurde
        """
        if not self.fitted_:
            raise RuntimeError(
                "FractionalDiffPreprocessor: fit() must be called before transform()"
            )

        if not self.columns_ or self.d_ == 0.0:
            return df

        # DataFrame kopieren um Original nicht zu verändern
        df = df.copy()

        # Spalten transformieren mit gelerntem d
        for col in self.columns_:
            if col in df.columns:
                df[col] = _frac_diff(df[col], self.d_)

        # NaN am Anfang entfernen (durch die Gewichtung entstehen NaNs)
        first_valid = df[self.columns_[0]].first_valid_index()
        if first_valid is not None:
            df = df.loc[first_valid:]

        # Metadata speichern
        df.attrs["frac_diff_d"] = self.d_

        return df

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "auto_d": True,
            "default_d": 0.4,
            "columns": ["O", "H", "L", "C"],
        }


__all__ = ["FractionalDiffPreprocessor"]
