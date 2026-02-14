"""
Konfiguration für den Optimizer
"""
import os
import numpy as np

DATA_PATH = "./data/forexsb"

CORR_THRESHOLD = 0.75  # Für Asset-Währungskorrelation (Portfolio-Diversifikation)
TARGET_TZ = "Europe/Berlin"
MIN_TRADES = 50  # Minimum Trades für statistische Signifikanz

# Timeframe-abhängige Parameter
TIMEFRAME = os.environ.get("TIMEFRAME", "HOUR")

TIMEFRAME_CONFIG = {
    "HOUR": {"bars_per_hour": 1, "window_size": 35000, "oos_size": 4000},
    "HOUR_SWING": {"bars_per_hour": 1, "window_size": 35000, "oos_size": 4000},
    "MINUTE_15": {"bars_per_hour": 4, "window_size": 50000, "oos_size": 8000},
    "MINUTE_5": {"bars_per_hour": 12, "window_size": 80000, "oos_size": 16000},
    "MINUTE_1": {"bars_per_hour": 60, "window_size": 100000, "oos_size": 20000},
    "DAY": {"bars_per_hour": 1/24, "window_size": 2000, "oos_size": 500},
}

tf_cfg = TIMEFRAME_CONFIG.get(TIMEFRAME, TIMEFRAME_CONFIG["HOUR"])
WINDOW_SIZE = tf_cfg["window_size"]
OOS_SIZE = tf_cfg["oos_size"]
WALK_FORWARD_FOLDS = 8


# Grid-Konfiguration pro Klasse (in Spread-Vielfachen)
# TP und SL sind jetzt symmetrisch - KI entscheidet ob RRR > 1 oder < 1 besser ist
# Kleine TP + große SL = hohe Winrate, niedriges RRR (Scalping-Stil)
# Große TP + kleine SL = niedrige Winrate, hohes RRR (Trend-Stil)
#
# HINWEIS: Diese Grids sind die Defaults. Für Swing-Trading oder andere Stile
# verwende Strategy-Dateien in strategies/*.json (z.B. swing_trading.json)

CLASS_GRIDS = {
    # === SCALPING GRIDS (Standard) ===
    "FOREX": {
        "tp": [15, 20, 25, 30, 40, 50, 60, 80],
        "sl": [15, 20, 25, 30, 40, 50, 60, 80],
        "ct": [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70],
    },
    "INDEX": {
        "tp": [20, 30, 50, 70, 100, 150],
        "sl": [20, 30, 50, 70, 100, 150],
        "ct": [0.50, 0.52, 0.55, 0.60, 0.65, 0.70],
    },
    "COMMODITY": {
        "tp": [20, 30, 40, 60, 80, 100],
        "sl": [20, 30, 40, 60, 80, 100],
        "ct": [0.50, 0.52, 0.55, 0.58, 0.62, 0.65, 0.70],
    },
    "CRYPTO": {
        "tp": [20, 30, 50, 80, 120, 200],
        "sl": [20, 30, 50, 80, 120, 200],
        "ct": [0.50, 0.52, 0.55, 0.60, 0.65, 0.70],
    },
    "TEST": {
        "tp": [20, 40],
        "sl": [20, 40],
        "ct": [0.50, 0.55, 0.60],
    },
}

# === SWING TRADING GRIDS ===
# Für längerfristige Trades (Tage bis Wochen)
# TP bis 1000x Spread = bei EURUSD (0.0002 Spread) ca. 200 Pips = 2% Bewegung
SWING_GRIDS = {
    "FOREX": {
        "tp": [100, 150, 200, 300, 500, 750, 1000],  # Bis 1000x Spread
        "sl": [50, 75, 100, 150, 200, 300],          # Engere SL für besseres RRR
        "ct": [0.55, 0.60, 0.65, 0.70, 0.75],        # Höhere CT für Qualität
    },
    "INDEX": {
        "tp": [100, 200, 300, 500, 750, 1000],
        "sl": [50, 100, 150, 200, 300],
        "ct": [0.55, 0.60, 0.65, 0.70, 0.75],
    },
    "COMMODITY": {
        "tp": [100, 200, 300, 500, 750, 1000],
        "sl": [50, 100, 150, 200, 300],
        "ct": [0.55, 0.60, 0.65, 0.70, 0.75],
    },
    "CRYPTO": {
        "tp": [200, 400, 600, 1000, 1500, 2000],     # Crypto braucht größere Range
        "sl": [100, 200, 300, 500],
        "ct": [0.55, 0.60, 0.65, 0.70],
    },
    "TEST": {
        "tp": [100, 200],
        "sl": [50, 100],
        "ct": [0.55, 0.60],
    },
}

# Makro-Indikatoren für Daily-Daten
MACRO_INDICATORS = {
    "VIX_DAY": "vix",
    "VVIX_DAY": "vvix",
    "SKEW_DAY": "skew",
    "VXN_DAY": "vxn",
    "TNX_DAY": "tnx",
    "TYX_DAY": "tyx",
    "FVX_DAY": "fvx",
    "IRX_DAY": "irx",
    "DXY_DAY": "dxy",
    "GOLD_FUT_DAY": "gold_fut",
    "OIL_FUT_DAY": "oil",
    "SILVER_FUT_DAY": "silver_fut",
    "SPX_DAY": "spx",
    "NASDAQ_DAY": "nasdaq",
    "DOW_DAY": "dow",
    "RUSSELL_DAY": "russell",
    "NIKKEI_DAY": "nikkei",
    "HANGSENG_DAY": "hangseng",
    "FTSE_DAY": "ftse_idx",
    "DAX_IDX_DAY": "dax_idx",
    "XLF_DAY": "xlf",
    "XLE_DAY": "xle",
    "XLK_DAY": "xlk",
    "XLU_DAY": "xlu",
    "XLP_DAY": "xlp",
    "TLT_DAY": "tlt",
    "HYG_DAY": "hyg",
    "LQD_DAY": "lqd",
}

# Lookback-Perioden
LOOKBACKS_HOURS = [1, 2, 4, 8, 12, 24]
LOOKBACKS_DAYS = [2, 5, 10, 20, 60]

# Abgeleitete Makro-Features (Spreads & Ratios)
MACRO_DERIVED_FEATURES = [
    {"name": "macro_yield_curve_10y_3m", "op": "subtract", "a": "macro_tnx", "b": "macro_irx"},
    {"name": "macro_yield_curve_10y_5y", "op": "subtract", "a": "macro_tnx", "b": "macro_fvx"},
    {"name": "macro_vix_vvix_ratio", "op": "ratio", "a": "macro_vix", "b": "macro_vvix"},
    {"name": "macro_risk_ratio_spx_tlt", "op": "ratio", "a": "macro_spx", "b": "macro_tlt"},
    {"name": "macro_credit_spread_proxy", "op": "ratio", "a": "macro_hyg", "b": "macro_lqd"},
    {"name": "macro_smallcap_ratio", "op": "ratio", "a": "macro_russell", "b": "macro_spx"},
    {"name": "macro_tech_defensive_ratio", "op": "ratio", "a": "macro_xlk", "b": "macro_xlu"},
]

# Zinsdaten
INTEREST_RATES = [
    {"name": "fed", "file": "FED_RATE.csv", "lookbacks_days": [30, 90, 180]},
    {"name": "ecb", "file": "ECB_RATE.csv", "lookbacks_days": [30, 90, 180]},
]

# Zinsdifferenzen
INTEREST_RATE_DIFFS = [
    {"name": "macro_rate_diff_usd_eur", "a": "macro_fed_rate", "b": "macro_ecb_rate"},
]


def compute_macro_derived(df, config=None):
    """Compute derived macro features from config (subtract/ratio operations)."""
    for spec in (config or MACRO_DERIVED_FEATURES):
        a, b = spec["a"], spec["b"]
        if a in df.columns and b in df.columns:
            if spec["op"] == "subtract":
                df[spec["name"]] = df[a] - df[b]
            elif spec["op"] == "ratio":
                df[spec["name"]] = df[a] / (df[b] + 1e-10)
    return df


def compute_interest_rates(df, data_path):
    """Load interest rate CSVs and compute rate diffs from config."""
    import os
    import pandas as pd

    for rate_cfg in INTEREST_RATES:
        rate_path = os.path.join(data_path, rate_cfg["file"])
        if not os.path.exists(rate_path):
            continue
        try:
            rate_df = pd.read_csv(rate_path, parse_dates=["Date"], index_col="Date")
            rate_series = rate_df["Rate"].reindex(df.index, method="ffill")
            col_name = f"macro_{rate_cfg['name']}_rate"
            df[col_name] = rate_series
            for lb in rate_cfg["lookbacks_days"]:
                df[f"macro_{rate_cfg['name']}_chg_{lb}d"] = df[col_name].diff(24 * lb)
        except Exception:
            pass

    for diff_cfg in INTEREST_RATE_DIFFS:
        a, b = diff_cfg["a"], diff_cfg["b"]
        if a in df.columns and b in df.columns:
            df[diff_cfg["name"]] = df[a] - df[b]

    return df


def convert_numpy(obj):
    """Konvertiert numpy-Typen zu Python-nativen Typen für JSON-Serialisierung."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


