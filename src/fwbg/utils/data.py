"""
Daten-Utility-Funktionen für FWBG.
"""
import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Bereinigt DataFrame von NaN/Inf-Werten.

    Args:
        df: Input DataFrame
        fill_value: Wert zum Ersetzen von NaN (default: 0.0)

    Returns:
        Bereinigter DataFrame
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(fill_value)
    return df


def clean_array(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Bereinigt NumPy-Array von NaN/Inf-Werten.

    Args:
        arr: Input Array
        fill_value: Wert zum Ersetzen von NaN/Inf (default: 0.0)

    Returns:
        Bereinigtes Array
    """
    arr = np.where(np.isinf(arr), np.nan, arr)
    arr = np.nan_to_num(arr, nan=fill_value)
    return arr


__all__ = ["clean_dataframe", "clean_array"]
