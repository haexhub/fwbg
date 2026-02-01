"""
Makro-Daten-Loader für Trading Bot.

Standalone-Funktionen zum Laden von Makro-Indikatoren und Zinsdaten,
unabhängig vom Optimizer-Modul.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# Default Konfiguration
DEFAULT_DATA_PATH = os.environ.get("DATA_PATH", "data")

DEFAULT_MACRO_INDICATORS = {
    "VIX": "vix",
    "HYG": "hyg",
    "GOLD_FUT": "gold_fut",
    "TLT": "tlt",
    "DXY": "dxy",
    "TNX": "tnx",
    "IRX": "irx",
    "SPY": "spy",
    "TIP": "tip",
}

DEFAULT_LOOKBACKS_HOURS = [6, 12, 24, 48, 72]
DEFAULT_LOOKBACKS_DAYS = [5, 10, 20]


def load_macro_csv(path: str) -> Optional[pd.DataFrame]:
    """Lädt eine Makro-CSV Datei."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df
    except Exception:
        return None


def load_macro_indicators(
    df: pd.DataFrame,
    data_path: str = None,
    indicators: Dict[str, str] = None,
    lookbacks_hours: List[int] = None,
    lookbacks_days: List[int] = None,
) -> pd.DataFrame:
    """
    Lädt alle Makro-Indikatoren und fügt sie zum DataFrame hinzu.

    Args:
        df: DataFrame mit OHLC-Daten und DatetimeIndex
        data_path: Pfad zu Makro-CSV Dateien
        indicators: Dict von Dateiname -> Prefix
        lookbacks_hours: Stunden-Lookbacks für Changes
        lookbacks_days: Tages-Lookbacks für Changes

    Returns:
        DataFrame mit Makro-Features
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if indicators is None:
        indicators = DEFAULT_MACRO_INDICATORS
    if lookbacks_hours is None:
        lookbacks_hours = DEFAULT_LOOKBACKS_HOURS
    if lookbacks_days is None:
        lookbacks_days = DEFAULT_LOOKBACKS_DAYS

    df["_date"] = df.index.date

    for filename, prefix in indicators.items():
        macro_path = f"{data_path}/{filename}.csv"
        macro_df = load_macro_csv(macro_path)
        if macro_df is not None:
            try:
                macro_lookup = macro_df["Close"].to_dict()

                col_name = f"macro_{prefix}"
                df[col_name] = df["_date"].map(lambda d: macro_lookup.get(pd.Timestamp(d), np.nan))
                df[col_name] = df[col_name].ffill()

                # Stunden-basierte Lookbacks
                for lb_h in lookbacks_hours:
                    df[f"{col_name}_chg_{lb_h}h"] = df[col_name].pct_change(lb_h) * 100

                # Tages-basierte Lookbacks
                for lb_d in lookbacks_days:
                    df[f"{col_name}_chg_{lb_d}d"] = df[col_name].pct_change(24 * lb_d) * 100

            except Exception:
                pass

    df = df.drop(columns=["_date"], errors="ignore")

    # === ABGELEITETE FEATURES ===
    # Yield Curve
    if "macro_tnx" in df.columns and "macro_irx" in df.columns:
        df["macro_yield_curve"] = df["macro_tnx"] - df["macro_irx"]

    # Risk-Off Composite
    if all(c in df.columns for c in ["macro_vix", "macro_gold_fut", "macro_tlt"]):
        df["macro_riskoff_composite"] = (
            df["macro_vix"].pct_change(24) +
            df["macro_gold_fut"].pct_change(24) +
            df["macro_tlt"].pct_change(24)
        ) / 3

    # Inflation-Erwartungen
    if "macro_tip" in df.columns and "macro_tlt" in df.columns:
        df["macro_inflation_expect"] = df["macro_tip"] - df["macro_tlt"]

    return df


def load_interest_rates(
    df: pd.DataFrame,
    data_path: str = None,
) -> pd.DataFrame:
    """
    Lädt Fed und ECB Zinsdaten.

    Args:
        df: DataFrame mit OHLC-Daten
        data_path: Pfad zu Zins-CSV Dateien

    Returns:
        DataFrame mit Zins-Features
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    for rate_name, rate_file in [("fed", "FED_RATE.csv"), ("ecb", "ECB_RATE.csv")]:
        rate_path = f"{data_path}/{rate_file}"
        if os.path.exists(rate_path):
            try:
                rate_df = pd.read_csv(rate_path, parse_dates=["Date"], index_col="Date")
                rate_series = rate_df["Rate"].reindex(df.index, method="ffill")
                df[f"macro_{rate_name}_rate"] = rate_series
                for lb in [30, 90, 180]:
                    df[f"macro_{rate_name}_chg_{lb}d"] = df[f"macro_{rate_name}_rate"].diff(24 * lb)
            except Exception:
                pass

    # Zinsdifferenz EUR/USD
    if "macro_fed_rate" in df.columns and "macro_ecb_rate" in df.columns:
        df["macro_rate_diff_usd_eur"] = df["macro_fed_rate"] - df["macro_ecb_rate"]

    return df


__all__ = [
    "load_macro_indicators",
    "load_interest_rates",
    "load_macro_csv",
]
