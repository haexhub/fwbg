"""
Daten laden und Makro-Indikatoren integrieren
"""
import os
import pandas as pd
import numpy as np

from .config import (
    TARGET_TZ, DATA_PATH, MACRO_INDICATORS,
    LOOKBACKS_HOURS, LOOKBACKS_DAYS
)


def load_data_aligned(path, is_sentiment=False):
    """Lädt OHLC-Daten aus CSV mit Zeitzone-Alignment."""
    try:
        df_raw = pd.read_csv(path)
        start = 1 if str(df_raw.iloc[0, 0]).isdigit() else 0
        if len(df_raw.columns) >= start + 5:
            df = df_raw.iloc[
                :, [start, start + 1, start + 2, start + 3, start + 4]
            ].copy()
            df.columns = ["T", "O", "H", "L", "C"]
        else:
            df = df_raw.iloc[:, [start, start + 1]].copy()
            df.columns = ["T", "C"]
            df["O"] = df["H"] = df["L"] = df["C"]
        df["T"] = pd.to_datetime(df["T"])
        if is_sentiment:
            if df["T"].dt.tz is None:
                df["T"] = df["T"].dt.tz_localize("UTC")
            df["T"] = df["T"].dt.tz_convert(TARGET_TZ)
        else:
            # Keine TZ-Lokalisierung - behandle Daten als naive Timestamps
            # Das vermeidet DST-Probleme (ambiguous/nonexistent times)
            pass
        # Stelle sicher dass der Index keine TZ hat
        if df["T"].dt.tz is not None:
            df["T"] = df["T"].dt.tz_localize(None)
        return df.set_index("T")
    except Exception as e:
        print(f"Fehler beim Laden von {path}: {e}")
        return None


def load_macro_csv(path):
    """
    Lädt eine Makro-CSV-Datei mit flexibler Spalten-Erkennung.
    Unterstützt: DATE, Datetime, Time als Index-Spalte.
    """
    if not os.path.exists(path):
        return None

    try:
        raw_df = pd.read_csv(path, nrows=1)
        cols = list(raw_df.columns)

        # Finde Datums-Spalte (case-insensitive)
        date_col = None
        for candidate in ["DATE", "Datetime", "datetime", "Time", "time", "Date"]:
            if candidate in cols:
                date_col = candidate
                break

        if not date_col:
            return None

        macro_df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
        return macro_df
    except Exception:
        return None


def load_macro_indicators(df):
    """Lädt alle Makro-Indikatoren und fügt sie zum DataFrame hinzu."""
    df["_date"] = df.index.date

    for filename, prefix in MACRO_INDICATORS.items():
        macro_path = f"{DATA_PATH}/{filename}.csv"
        macro_df = load_macro_csv(macro_path)
        if macro_df is not None:
            try:
                macro_lookup = macro_df["Close"].to_dict()

                col_name = f"macro_{prefix}"
                df[col_name] = df["_date"].map(lambda d: macro_lookup.get(pd.Timestamp(d), np.nan))
                df[col_name] = df[col_name].ffill()

                # Stunden-basierte Lookbacks
                for lb_h in LOOKBACKS_HOURS:
                    df[f"{col_name}_chg_{lb_h}h"] = df[col_name].pct_change(lb_h) * 100

                # Tages-basierte Lookbacks
                for lb_d in LOOKBACKS_DAYS:
                    df[f"{col_name}_chg_{lb_d}d"] = df[col_name].pct_change(24 * lb_d) * 100

            except Exception:
                pass

    df = df.drop(columns=["_date"], errors="ignore")

    # === ABGELEITETE FEATURES ===
    # Yield Curve
    if "macro_tnx" in df.columns and "macro_irx" in df.columns:
        df["macro_yield_curve_10y_3m"] = df["macro_tnx"] - df["macro_irx"]
    if "macro_tnx" in df.columns and "macro_fvx" in df.columns:
        df["macro_yield_curve_10y_5y"] = df["macro_tnx"] - df["macro_fvx"]

    # VIX/VVIX Ratio
    if "macro_vix" in df.columns and "macro_vvix" in df.columns:
        df["macro_vix_vvix_ratio"] = df["macro_vix"] / (df["macro_vvix"] + 1e-10)

    # Risk On/Off Ratios
    if "macro_spx" in df.columns and "macro_tlt" in df.columns:
        df["macro_risk_ratio_spx_tlt"] = df["macro_spx"] / (df["macro_tlt"] + 1e-10)
    if "macro_hyg" in df.columns and "macro_lqd" in df.columns:
        df["macro_credit_spread_proxy"] = df["macro_hyg"] / (df["macro_lqd"] + 1e-10)

    # Small Cap vs Large Cap
    if "macro_russell" in df.columns and "macro_spx" in df.columns:
        df["macro_smallcap_ratio"] = df["macro_russell"] / (df["macro_spx"] + 1e-10)

    # Tech vs Defensive
    if "macro_xlk" in df.columns and "macro_xlu" in df.columns:
        df["macro_tech_defensive_ratio"] = df["macro_xlk"] / (df["macro_xlu"] + 1e-10)

    return df


def load_interest_rates(df):
    """Lädt Fed und ECB Zinsdaten."""
    for rate_name, rate_file in [("fed", "FED_RATE.csv"), ("ecb", "ECB_RATE.csv")]:
        rate_path = f"{DATA_PATH}/{rate_file}"
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
