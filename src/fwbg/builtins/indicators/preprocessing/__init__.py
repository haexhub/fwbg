"""
Preprocessing Modul für OHLC-Daten.

Enthält:
- Fractional Differentiation (Lopez de Prado)
- Log-Returns Transformation
- Z-Score Normalisierung
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from fwbg.plugins.indicator import safe_divide


def get_frac_diff_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
    """
    Berechnet die Gewichte für Fractional Differentiation.

    Args:
        d: Differenzierungsordnung (0 < d < 1)
        size: Maximale Anzahl der Gewichte
        threshold: Abbruchschwelle für kleine Gewichte

    Returns:
        Array mit Gewichten (absteigend)
    """
    weights = [1.0]
    for k in range(1, size):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights[::-1])


def frac_diff(series: np.ndarray, d: float, threshold: float = 1e-4) -> np.ndarray:
    """
    Wendet Fractional Differentiation auf eine Zeitreihe an.

    Args:
        series: Input-Zeitreihe
        d: Differenzierungsordnung (0 < d < 1)
        threshold: Abbruchschwelle für Gewichte

    Returns:
        Fractionally differenzierte Serie (mit NaN am Anfang)
    """
    weights = get_frac_diff_weights(d, len(series), threshold)
    width = len(weights)

    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights, series[i - width + 1:i + 1])

    return result


def find_min_d_for_stationarity(
    series: np.ndarray,
    max_d: float = 1.0,
    p_value_threshold: float = 0.05,
    step: float = 0.05
) -> float:
    """
    Findet den minimalen d-Wert der Stationarität erreicht (ADF-Test).

    Args:
        series: Input-Zeitreihe (ohne NaN)
        max_d: Maximaler d-Wert zum Testen
        p_value_threshold: p-Wert Schwelle für Stationarität
        step: Schrittgröße für d-Suche

    Returns:
        Optimaler d-Wert
    """
    series_clean = series[~np.isnan(series)]
    if len(series_clean) < 100:
        return 0.5

    for d in np.arange(step, max_d + step, step):
        diff_series = frac_diff(series_clean, d)
        diff_clean = diff_series[~np.isnan(diff_series)]

        if len(diff_clean) < 50:
            continue

        try:
            adf_result = adfuller(diff_clean, maxlag=1)
            p_value = adf_result[1]

            if p_value < p_value_threshold:
                return round(d, 2)
        except Exception:
            continue

    return max_d


def apply_frac_diff_preprocessing(
    df: pd.DataFrame,
    use_auto_d: bool = True,
    default_d: float = 0.4
) -> tuple:
    """
    Wendet Fractional Differentiation als Preprocessing auf OHLC-Daten an.

    Args:
        df: DataFrame mit OHLC Daten
        use_auto_d: Automatische d-Optimierung via ADF-Test
        default_d: Standard d-Wert wenn auto_d=False

    Returns:
        Tuple (DataFrame mit transformierten OHLC, d-Wert)
    """
    df["_original_close"] = df["C"].copy()

    if use_auto_d:
        sample_size = min(5000, len(df))
        sample = df["C"].values[-sample_size:]
        d = find_min_d_for_stationarity(sample)
    else:
        d = default_d

    for col in ["O", "H", "L", "C"]:
        if col in df.columns:
            df[col] = frac_diff(df[col].values, d)

    first_valid = df["C"].first_valid_index()
    if first_valid is not None:
        df = df.loc[first_valid:]

    return df, d


def apply_log_returns_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformiert OHLC-Daten zu Log-Returns.

    Args:
        df: DataFrame mit OHLC Daten

    Returns:
        DataFrame mit transformierten OHLC
    """
    df["_original_close"] = df["C"].copy()

    for col in ["O", "H", "L", "C"]:
        if col in df.columns:
            df[col] = np.log(df[col] / df[col].shift(1))

    df = df.iloc[1:]

    return df


def apply_normalize_preprocessing(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Normalisiert OHLC-Daten mit Rolling Z-Score.

    Args:
        df: DataFrame mit OHLC Daten
        window: Rolling-Window für Mean/Std Berechnung

    Returns:
        DataFrame mit normalisierten OHLC
    """
    if "_original_close" not in df.columns:
        df["_original_close"] = df["C"].copy()

    for col in ["O", "H", "L", "C"]:
        if col in df.columns:
            rolling_mean = df[col].rolling(window).mean()
            rolling_std = df[col].rolling(window).std()
            df[col] = safe_divide(df[col] - rolling_mean, rolling_std)

    df = df.iloc[window:]

    return df


def apply_preprocessing(df: pd.DataFrame, preprocessing_params) -> tuple:
    """
    Wendet alle konfigurierten Preprocessing-Schritte auf OHLC-Daten an.

    Reihenfolge:
    1. Fractional Differentiation (falls aktiviert)
    2. Log-Returns (falls aktiviert)
    3. Normalisierung (falls aktiviert)

    Args:
        df: DataFrame mit OHLC Daten
        preprocessing_params: PreprocessingParams Objekt

    Returns:
        Tuple (DataFrame, preprocessing_info dict)
    """
    info = {"applied": [], "frac_diff_d": None}

    # 1. Fractional Differentiation
    if preprocessing_params.fractional_differentiation:
        df, d = apply_frac_diff_preprocessing(
            df,
            use_auto_d=preprocessing_params.frac_diff_auto_d,
            default_d=preprocessing_params.frac_diff_default_d
        )
        info["applied"].append("fractional_differentiation")
        info["frac_diff_d"] = d

    # 2. Log-Returns
    if preprocessing_params.log_returns:
        df = apply_log_returns_preprocessing(df)
        info["applied"].append("log_returns")

    # 3. Normalisierung
    if preprocessing_params.normalize:
        df = apply_normalize_preprocessing(df, preprocessing_params.normalize_window)
        info["applied"].append(f"normalize_{preprocessing_params.normalize_window}")

    return df, info


__all__ = [
    "get_frac_diff_weights",
    "frac_diff",
    "find_min_d_for_stationarity",
    "apply_frac_diff_preprocessing",
    "apply_log_returns_preprocessing",
    "apply_normalize_preprocessing",
    "apply_preprocessing",
]
