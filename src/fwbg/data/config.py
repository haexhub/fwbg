"""
Konfiguration für den Optimizer
"""
import os
import numpy as np

from fwbg_sdk.enums import Timeframe

DATA_PATH = None

CORR_THRESHOLD = 0.75  # Für Asset-Währungskorrelation (Portfolio-Diversifikation)
TARGET_TZ = "Europe/Berlin"
MIN_TRADES = 50  # Minimum Trades für statistische Signifikanz

# Timeframe-abhängige Parameter. Direkt auf die kanonische Langform normalisieren:
# TIMEFRAME speist Datei-Globbing (cli/main.py, cli/_analyze.py) und Results-
# Metadaten — eine Legacy-Schreibweise aus der Umgebung ("HOUR") würde sonst an
# kanonisch benannten Dateien (EURUSD_HOUR_1.csv) vorbei globben.
TIMEFRAME = Timeframe.from_str(os.environ.get("TIMEFRAME", "HOUR")).canonical

# Walk-Forward-Ziel-Dimensionen je Timeframe (nach kanonischem Namen, siehe
# Timeframe.canonical). window_size = Trainingsfenster, oos_size = Out-of-Sample-
# Balken je Fold. Beides sind Zielwerte, die das adaptive Fold-Sizing in
# optimization/process.py nach unten an die tatsächlich vorhandene Historie
# anpasst. bars_per_hour speist die Jahres-Hochrechnung (test_period_years).
TIMEFRAME_CONFIG = {
    "MINUTE_1":  {"bars_per_hour": 60,     "window_size": 100000, "oos_size": 20000},
    "MINUTE_5":  {"bars_per_hour": 12,     "window_size": 80000,  "oos_size": 16000},
    "MINUTE_15": {"bars_per_hour": 4,      "window_size": 50000,  "oos_size": 8000},
    "MINUTE_30": {"bars_per_hour": 2,      "window_size": 40000,  "oos_size": 6000},
    "HOUR_1":    {"bars_per_hour": 1,      "window_size": 35000,  "oos_size": 4000},
    "HOUR_2":    {"bars_per_hour": 0.5,    "window_size": 20000,  "oos_size": 3000},
    "HOUR_4":    {"bars_per_hour": 0.25,   "window_size": 12000,  "oos_size": 2000},
    "DAY_1":     {"bars_per_hour": 1 / 24, "window_size": 2000,   "oos_size": 500},
    "WEEK_1":    {"bars_per_hour": 1 / 168, "window_size": 500,   "oos_size": 100},
}


def resolve_tf_config(timeframe: "str | Timeframe") -> dict:
    """Fold-Sizing-Parameter für einen Timeframe in beliebiger Schreibweise.

    Normalisiert über :meth:`Timeframe.from_str` auf den kanonischen Namen und
    schlägt darüber die Ziel-Dimensionen nach. Ersetzt den früheren stillen
    Fallback auf die HOUR-Werte, der DAY_1/HOUR_4/MINUTE_30 fälschlich auf
    Stunden-Dimensionen (oos_size=4000) abbildete.
    """
    return TIMEFRAME_CONFIG[Timeframe.from_str(timeframe).canonical]


RESAMPLE_FROM = None  # Set by CLI when loading lower-TF files for resampling

tf_cfg = resolve_tf_config(TIMEFRAME)
WINDOW_SIZE = tf_cfg["window_size"]
OOS_SIZE = tf_cfg["oos_size"]
WALK_FORWARD_FOLDS = 8  # Default, overridden by strategy.validation.folds


def convert_numpy(obj):
    """Konvertiert numpy-Typen zu Python-nativen Typen für JSON-Serialisierung.

    Ersetzt inf/nan durch None (nicht JSON-serialisierbar).
    """
    import math
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        return None if math.isinf(val) or math.isnan(val) else val
    elif isinstance(obj, float):
        return None if math.isinf(obj) or math.isnan(obj) else obj
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj
