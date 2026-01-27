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

CORR_THRESHOLD = 0.75
RELEVANCE_THRESHOLD = 0.02  # 2% Hürde
TARGET_TZ = "Europe/Berlin"
MIN_TRADES = 50  # Minimum Trades für statistische Signifikanz (realistisch bei sequentiellem Trading)
FEATURE_STABILITY_MIN = 3  # Feature muss in min. 3 von 5 Folds relevant sein

# Timeframe-abhängige Parameter
TIMEFRAME = os.environ.get("TIMEFRAME", "HOUR")

TIMEFRAME_CONFIG = {
    "HOUR": {"bars_per_hour": 1, "max_trade_bars": 48, "window_size": 35000, "oos_size": 4000},
    "MINUTE_15": {"bars_per_hour": 4, "max_trade_bars": 96, "window_size": 50000, "oos_size": 8000},
    "MINUTE_5": {"bars_per_hour": 12, "max_trade_bars": 144, "window_size": 80000, "oos_size": 16000},
    "MINUTE_1": {"bars_per_hour": 60, "max_trade_bars": 240, "window_size": 100000, "oos_size": 20000},
    "DAY": {"bars_per_hour": 1/24, "max_trade_bars": 20, "window_size": 2000, "oos_size": 500},
}

tf_cfg = TIMEFRAME_CONFIG.get(TIMEFRAME, TIMEFRAME_CONFIG["HOUR"])
WINDOW_SIZE = tf_cfg["window_size"]
OOS_SIZE = tf_cfg["oos_size"]
MAX_TRADE_BARS = tf_cfg["max_trade_bars"]
WALK_FORWARD_FOLDS = 8

# Regime-Filter Thresholds
VIX_HIGH = 25
ADX_MIN = 20

# Asset-Klassifizierung und Spread-Kosten (verdoppelt für konservative Simulation)
ASSET_CONFIG = {
    # FOREX - Majors
    "EURUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00020, "currency": ["EUR", "USD"]},
    "GBPUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00024, "currency": ["GBP", "USD"]},
    "USDJPY": {"class": "FOREX", "point": 0.01, "spread": 0.020, "currency": ["USD", "JPY"]},
    "USDCHF": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["USD", "CHF"]},
    "USDCAD": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["USD", "CAD"]},
    "AUDUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00024, "currency": ["AUD", "USD"]},
    "NZDUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["NZD", "USD"]},
    # FOREX - Crosses
    "EURGBP": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["EUR", "GBP"]},
    "EURCAD": {"class": "FOREX", "point": 0.0001, "spread": 0.00040, "currency": ["EUR", "CAD"]},
    "EURCHF": {"class": "FOREX", "point": 0.0001, "spread": 0.00036, "currency": ["EUR", "CHF"]},
    "EURNZD": {"class": "FOREX", "point": 0.0001, "spread": 0.00050, "currency": ["EUR", "NZD"]},
    # Indizes
    "DAX": {"class": "INDEX", "point": 1.0, "spread": 3.0, "currency": ["EUR"]},
    "DOW30": {"class": "INDEX", "point": 1.0, "spread": 4.0, "currency": ["USD"]},
    "SPX500": {"class": "INDEX", "point": 0.1, "spread": 1.0, "currency": ["USD"]},
    "NAS100": {"class": "INDEX", "point": 0.1, "spread": 2.0, "currency": ["USD"]},
    "FTSE100": {"class": "INDEX", "point": 1.0, "spread": 3.0, "currency": ["GBP"]},
    # Commodities
    "XAUUSD": {"class": "COMMODITY", "point": 0.1, "spread": 0.60, "currency": ["USD"]},
    "GOLD": {"class": "COMMODITY", "point": 0.1, "spread": 0.60, "currency": ["USD"]},  # Andere Zeitzone, 2h mehr/Tag
    "XAGUSD": {"class": "COMMODITY", "point": 0.01, "spread": 0.040, "currency": ["USD"]},
    "SILVER": {"class": "COMMODITY", "point": 0.01, "spread": 0.040, "currency": ["USD"]},  # Andere Zeitzone
    "BRENT": {"class": "COMMODITY", "point": 0.01, "spread": 0.060, "currency": ["USD"]},
    # Crypto
    "BTCUSD": {"class": "CRYPTO", "point": 1.0, "spread": 100.0, "currency": ["USD"]},
    "ETHUSD": {"class": "CRYPTO", "point": 0.1, "spread": 4.0, "currency": ["USD"]},
    # Test
    "TESTUSD": {"class": "TEST", "point": 0.0001, "spread": 0.00020, "currency": ["USD"]},
}

# Grid-Konfiguration pro Klasse (in Spread-Vielfachen)
# TP und SL sind jetzt symmetrisch - KI entscheidet ob RRR > 1 oder < 1 besser ist
# Kleine TP + große SL = hohe Winrate, niedriges RRR (Scalping-Stil)
# Große TP + kleine SL = niedrige Winrate, hohes RRR (Trend-Stil)
CLASS_GRIDS = {
    "FOREX": {
        "tp": [15, 20, 25, 30, 40, 50, 60, 80],   # Symmetrisch mit SL
        "sl": [15, 20, 25, 30, 40, 50, 60, 80],   # Symmetrisch mit TP
        "ct": [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70],  # CT ab 0.50
    },
    "INDEX": {
        "tp": [20, 30, 50, 70, 100, 150],
        "sl": [20, 30, 50, 70, 100, 150],         # Symmetrisch
        "ct": [0.50, 0.52, 0.55, 0.60, 0.65, 0.70],
    },
    "COMMODITY": {
        "tp": [20, 30, 40, 60, 80, 100],
        "sl": [20, 30, 40, 60, 80, 100],          # Symmetrisch
        "ct": [0.50, 0.52, 0.55, 0.58, 0.62, 0.65, 0.70],
    },
    "CRYPTO": {
        "tp": [20, 30, 50, 80, 120, 200],         # Größere Range wegen hoher Volatilität
        "sl": [20, 30, 50, 80, 120, 200],
        "ct": [0.50, 0.52, 0.55, 0.60, 0.65, 0.70],
    },
    "TEST": {
        "tp": [20, 40],
        "sl": [20, 40],
        "ct": [0.50, 0.55, 0.60],
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
        "description": "ADX, EMA, SMA, MACD, CCI, Aroon, Ichimoku",
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
# Weniger Gruppen = schnellere Optimierung
DEFAULT_FEATURE_GROUPS = ["trend_momentum", "macro_vol", "full_technical"]


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


def get_asset_config(sym):
    """Holt Asset-Konfiguration oder Default-Werte."""
    cfg = ASSET_CONFIG.get(sym, {"class": "FOREX", "point": 0.0001, "spread": 0.00020, "currency": ["USD"]})
    return cfg["class"], cfg["point"], cfg["spread"], cfg["currency"]
