"""
Fractional Differentiation Preprocessor Plugin.

Fractional Differentiation nach López de Prado:
- Macht Zeitreihen stationär unter Beibehaltung von Memory
- d=0: Keine Transformation (original)
- d=1: Volle Differentiation (wie pct_change, verliert Memory)
- d=0.3-0.5: Optimal für Trading (stationär + Memory)
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
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

    Verwendet vektorisierte Convolution für maximale Performance.

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

    # Vektorisierte Convolution statt Python-Loop
    # np.convolve mit 'valid' Mode gibt nur vollständig überlappende Ergebnisse
    values = series.values
    convolved = np.convolve(values, weights[::-1], mode='valid')

    # Result-Array mit NaNs für den Anfang (wo keine vollständige Convolution möglich)
    result = np.full(len(series), np.nan)
    result[width - 1:] = convolved

    return pd.Series(result, index=series.index)


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
class FractionalDiffPreprocessor(BasePlugin):
    """
    Fractional Differentiation Preprocessor.

    Macht OHLC-Spalten stationär unter Beibehaltung von Memory.

    Folgt sklearn fit/transform Pattern um Lookahead Bias zu verhindern:
    - fit() lernt optimales d NUR von Train-Daten UND speichert History
    - execute() wendet das gelernte d an (auf Train/Test/OOS)

    WICHTIG: Für Val/Test-Daten wird die in fit() gespeicherte History
    verwendet, um NaNs am Anfang zu vermeiden. Das ist kein Lookahead Bias,
    da die History aus den TRAIN-Daten stammt (zeitlich VOR Val/Test).
    """

    # Required BasePlugin class attributes
    name = "fractional_diff"
    version = "2.0.0"
    phase = PluginPhase.PREPROCESSING
    stateful = True
    cacheable = False

    order = 10  # Früh ausführen

    # Max window für frac_diff (bestimmt wie viel History wir brauchen)
    MAX_WINDOW = 500

    def __init__(self) -> None:
        """Initialize plugin instance state."""
        super().__init__()
        self.d_: float = 0.0
        self.history_: pd.DataFrame | None = None
        self.train_end_idx_ = None
        self.columns_: List[str] = []

    def validate(self) -> bool:
        """
        Validate that the plugin is properly configured.

        Returns:
            True if valid, False otherwise
        """
        return True

    def fit(
        self,
        ctx: PipelineContext,
        auto_d: bool = True,
        default_d: float = 0.4,
        columns: List[str] = None,
        **params
    ) -> None:
        """
        Lernt optimales d von Train-Daten und speichert History für spätere transforms.

        WICHTIG: Diese Methode MUSS auf Train-Daten aufgerufen werden,
        um Lookahead Bias zu verhindern!

        Args:
            ctx: PipelineContext mit Train-Daten
            auto_d: Automatische d-Optimierung via ADF-Test (NUR auf Train-Daten!)
            default_d: Default d-Wert wenn auto_d=False
            columns: Spalten zu transformieren (default: O, H, L, C)
        """
        df = ctx.df

        if columns is None:
            columns = ["O", "H", "L", "C"]

        # Nur vorhandene Spalten speichern
        self.columns_ = [c for c in columns if c in df.columns]

        if not self.columns_:
            self.d_ = 0.0  # Keine Transformation
            self.history_ = None
            self._fitted = True
            return

        # Optimales d finden (NUR auf Train-Daten!)
        if auto_d:
            self.d_ = _find_optimal_d(df["C"])
        else:
            self.d_ = default_d

        # History speichern: Die letzten MAX_WINDOW Zeilen des Train-DataFrames
        # Diese werden bei execute() von Val/Test-Daten prepended,
        # um NaNs am Anfang zu vermeiden (kein Lookahead - das sind TRAIN-Daten!)
        history_size = min(self.MAX_WINDOW, len(df))
        self.history_ = df.iloc[-history_size:].copy()

        # Speichere auch den letzten Index des Train-DataFrames
        # um später zu erkennen ob wir Train oder Val/Test transformieren
        self.train_end_idx_ = df.index[-1]

        self._fitted = True

    def execute(
        self,
        ctx: PipelineContext,
        **params
    ) -> PipelineContext:
        """
        Wendet Fractional Differentiation mit gelerntem d an.

        Diese Methode verwendet das in fit() gelernte d und kann
        auf Train, Test und OOS-Daten angewendet werden.

        Für Val/Test-Daten (zeitlich nach Train) wird die in fit() gespeicherte
        History prepended, um NaNs am Anfang zu vermeiden. Das ist KEIN
        Lookahead Bias, da die History aus TRAIN-Daten stammt!

        Args:
            ctx: PipelineContext mit DataFrame

        Returns:
            Updated PipelineContext mit transformiertem DataFrame

        Raises:
            RuntimeError: Wenn fit() noch nicht aufgerufen wurde
        """
        if not self._fitted:
            raise RuntimeError(
                "FractionalDiffPreprocessor: fit() must be called before execute()"
            )

        df = ctx.df

        if not self.columns_ or self.d_ == 0.0:
            return ctx

        # Prüfe ob das Train-Daten oder Val/Test-Daten sind
        # Train-Daten: Erster Index <= train_end_idx (überlappend oder identisch)
        # Val/Test-Daten: Erster Index > train_end_idx (zeitlich danach)
        is_train_data = df.index[0] <= self.train_end_idx_

        if is_train_data:
            # Train-Daten: Normal transformieren, NaNs am Anfang entfernen
            df = df.copy()

            for col in self.columns_:
                if col in df.columns:
                    df[col] = _frac_diff(df[col], self.d_)

            # NaN am Anfang entfernen (nur bei Train-Daten!)
            first_valid = df[self.columns_[0]].first_valid_index()
            if first_valid is not None:
                df = df.loc[first_valid:]

        else:
            # Val/Test-Daten: History prependen um NaNs zu vermeiden
            # Die History enthält die letzten N Zeilen der TRAIN-Daten
            if self.history_ is not None:
                # Kombiniere History + Val/Test-Daten
                combined = pd.concat([self.history_, df], axis=0)

                # Transformieren
                combined_transformed = combined.copy()
                for col in self.columns_:
                    if col in combined_transformed.columns:
                        combined_transformed[col] = _frac_diff(combined_transformed[col], self.d_)

                # Nur den ursprünglichen Val/Test-Teil zurückgeben (ohne History)
                # WICHTIG: Keine NaN-Entfernung hier - alle Zeilen sind valide!
                df = combined_transformed.loc[df.index].copy()
            else:
                # Kein History vorhanden - Fallback auf normale Transformation
                df = df.copy()
                for col in self.columns_:
                    if col in df.columns:
                        df[col] = _frac_diff(df[col], self.d_)

        # Metadata speichern
        df.attrs["frac_diff_d"] = self.d_

        # Update context with transformed DataFrame
        ctx.df = df
        return ctx

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "auto_d": True,
            "default_d": 0.4,
            "columns": ["O", "H", "L", "C"],
        }


__all__ = ["FractionalDiffPreprocessor"]
