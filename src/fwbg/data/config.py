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
