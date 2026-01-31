"""
Konfiguration für den Optimizer
"""
import os
import numpy as np

# Zielordner für diesen Account
ACCOUNT_NAME = os.environ.get("ACCOUNT_NAME", "main_demo")
DATA_PATH = "./data/forexsb"
BASE_PATH = f"accounts/{ACCOUNT_NAME}"
EXPORT_FILE = f"{BASE_PATH}/assets.json"
PLOT_PATH = f"{BASE_PATH}/plots"

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


# Asset-Klassifizierung und Spread-Kosten
# Spreads from IG Demo Account (Jan 2026)
ASSET_CONFIG = {
    # FOREX - Majors
    "EURUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00018, "currency": ["EUR", "USD"]},      # 1.8 pips
    "GBPUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00058, "currency": ["GBP", "USD"]},      # 5.8 pips
    "USDJPY": {"class": "FOREX", "point": 0.01, "spread": 0.060, "currency": ["USD", "JPY"]},          # 6.0 pips
    "USDCHF": {"class": "FOREX", "point": 0.0001, "spread": 0.00029, "currency": ["USD", "CHF"]},      # 2.9 pips
    "USDCAD": {"class": "FOREX", "point": 0.0001, "spread": 0.00035, "currency": ["USD", "CAD"]},      # 3.5 pips
    "AUDUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["AUD", "USD"]},      # 3.0 pips
    "NZDUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00093, "currency": ["NZD", "USD"]},      # 9.3 pips
    # FOREX - Crosses
    "EURGBP": {"class": "FOREX", "point": 0.0001, "spread": 0.00040, "currency": ["EUR", "GBP"]},      # 4.0 pips
    "EURCAD": {"class": "FOREX", "point": 0.0001, "spread": 0.00117, "currency": ["EUR", "CAD"]},      # 11.7 pips
    "EURCHF": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["EUR", "CHF"]},      # 3.0 pips
    "EURNZD": {"class": "FOREX", "point": 0.0001, "spread": 0.00180, "currency": ["EUR", "NZD"]},      # 18.0 pips
    # Indices
    "DAX": {"class": "INDEX", "point": 1.0, "spread": 7.0, "currency": ["EUR"]},                       # 7.0 points
    "DOW30": {"class": "INDEX", "point": 1.0, "spread": 4.8, "currency": ["USD"]},                     # 4.8 points
    "SPX500": {"class": "INDEX", "point": 0.1, "spread": 0.6, "currency": ["USD"]},                    # 0.6 points (6 pips)
    "NAS100": {"class": "INDEX", "point": 0.1, "spread": 2.0, "currency": ["USD"]},                    # ~2.0 points (estimated)
    "FTSE100": {"class": "INDEX", "point": 1.0, "spread": 4.0, "currency": ["GBP"]},                   # 4.0 points
    # Commodities
    "XAUUSD": {"class": "COMMODITY", "point": 0.1, "spread": 0.60, "currency": ["USD"]},               # ~0.60 USD (estimated)
    "GOLD": {"class": "COMMODITY", "point": 0.1, "spread": 0.60, "currency": ["USD"]},                 # ~0.60 USD (estimated)
    "XAGUSD": {"class": "COMMODITY", "point": 0.01, "spread": 0.040, "currency": ["USD"]},             # ~0.04 USD (estimated)
    "SILVER": {"class": "COMMODITY", "point": 0.01, "spread": 0.040, "currency": ["USD"]},             # ~0.04 USD (estimated)
    "BRENT": {"class": "COMMODITY", "point": 0.01, "spread": 0.078, "currency": ["USD"]},              # 7.8 cents
    # Crypto
    "BTCUSD": {"class": "CRYPTO", "point": 1.0, "spread": 581.0, "currency": ["USD"]},                 # 581 USD
    "ETHUSD": {"class": "CRYPTO", "point": 0.1, "spread": 4.0, "currency": ["USD"]},                   # ~4.0 USD (estimated)
    # Test
    "TESTUSD": {"class": "TEST", "point": 0.0001, "spread": 0.00018, "currency": ["USD"]},
}

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

# Feature-Gruppen für systematische Grid-Search
# Jede Gruppe wird separat mit allen TP/SL/CT Kombinationen getestet
FEATURE_GROUPS = {
    "trend": {
        "name": "Trend Indikatoren",
        "prefixes": ["trend_", "ichi_"],
        "description": "ADX, EMA, SMA, MACD, CCI, Aroon, Ichimoku, Efficiency Ratio",
    },
    "momentum": {
        "name": "Momentum Indikatoren",
        "prefixes": ["mom_"],
        "description": "RSI, Stochastic, Williams %R, ROC, Ultimate Oscillator",
    },
    "volatility": {
        "name": "Volatilität Indikatoren",
        "prefixes": ["vol_"],
        "description": "Bollinger Bands, Keltner, Donchian, ATR",
    },
    "price_action": {
        "name": "Price Action",
        "prefixes": ["pa_"],
        "description": "Range Position, Higher Highs/Lower Lows, Body Ratio, Gaps",
    },
    "time": {
        "name": "Zeit Features",
        "prefixes": ["time_", "season_"],
        "description": "Stunde, Wochentag, Monat, Quartal, Saisonalität",
    },
    "macro": {
        "name": "Makro Indikatoren",
        "prefixes": ["macro_"],
        "description": "VIX, Yields, DXY, Indices, Commodities, Sectors",
    },
    "dynamics": {
        "name": "Dynamik & Lags",
        "prefixes": ["dyn_", "lag_", "accel_"],
        "description": "Indikator-Änderungen, Lags, Beschleunigung",
    },
    "mtf": {
        "name": "Multi-Timeframe",
        "prefixes": ["mtf_"],
        "description": "H4 aggregierte Features",
    },
    "cross": {
        "name": "Cross-Indikator",
        "prefixes": ["cross_"],
        "description": "Kombinierte Signale aus mehreren Indikatoren",
    },
    "distribution": {
        "name": "Verteilungs-Features",
        "prefixes": ["dist_"],
        "description": "Rolling Skewness und Kurtosis der Returns",
    },
    "fft": {
        "name": "Fourier-Analyse",
        "prefixes": ["fft_"],
        "description": "FFT-basierte Zykluserkennung: Dominante Frequenz, Spektrale Energie/Entropie",
    },
    "regime": {
        "name": "Regime-Indikatoren",
        "prefixes": ["regime_"],
        "description": "Hurst-Exponent (Markt-Charakter: trending/mean-reverting/random)",
    },
    "event": {
        "name": "Event-Features",
        "prefixes": ["event_"],
        "description": "Time-Since-Event: Bars seit High/Low, EMA-Cross, RSI-Extrem, Vol-Spike",
    },
    "structure": {
        "name": "Struktur-Features",
        "prefixes": ["path_", "fractal_", "convex_", "structure_"],
        "description": "Path Efficiency, Fractal Dimension, Convexity, VWAP Distance",
    },
    "correlation": {
        "name": "Korrelations-Features",
        "prefixes": ["corr_", "lead_lag_", "vix_lead_"],
        "description": "Korrelationsstabilität, Decoupling, Lead-Lag Momentum",
    },
    "risk": {
        "name": "Risk/Tail-Risk Features",
        "prefixes": ["risk_"],
        "description": "Drawdown State, CVaR, Vol-of-Vol, Crash Probability",
    },
    # Kombinationen
    "trend_momentum": {
        "name": "Trend + Momentum",
        "prefixes": ["trend_", "ichi_", "mom_"],
        "description": "Klassische technische Analyse Kombination",
    },
    "macro_vol": {
        "name": "Makro + Volatilität",
        "prefixes": ["macro_", "vol_"],
        "description": "Fundamentale + Volatilitäts-basierte Signale",
    },
    "full_technical": {
        "name": "Volle technische Analyse",
        "prefixes": ["trend_", "ichi_", "mom_", "vol_", "pa_"],
        "description": "Alle technischen Indikatoren ohne Makro/Zeit",
    },
}

# Standard Feature-Gruppen die getestet werden (kann überschrieben werden)
# Alle Gruppen als Default - in Strategy-File kann eingeschränkt werden
DEFAULT_FEATURE_GROUPS = list(FEATURE_GROUPS.keys())


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


