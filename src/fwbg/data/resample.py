"""Timeframe hierarchy, resampling utilities, and source-file fallback logic."""
import glob as _glob
import os
from pathlib import Path

import pandas as pd

# Ordered from lowest to highest resolution
TIMEFRAME_ORDER = [
    "MINUTE_1", "MINUTE_5", "MINUTE_15", "MINUTE_30",
    "HOUR", "HOUR_4", "DAY",
]

# Pandas resample rule for each timeframe
RESAMPLE_RULE: dict[str, str] = {
    "MINUTE_1": "1min",
    "MINUTE_5": "5min",
    "MINUTE_15": "15min",
    "MINUTE_30": "30min",
    "HOUR": "1h",
    "HOUR_4": "4h",
    "DAY": "1D",
}


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a higher timeframe."""
    rule = RESAMPLE_RULE.get(target_tf)
    if not rule:
        return df

    agg = {"O": "first", "H": "max", "L": "min", "C": "last"}
    if "V" in df.columns:
        agg["V"] = "sum"

    return df.resample(rule).agg(agg).dropna(subset=["O"])


def parse_symbol_timeframe(stem: str) -> tuple[str, str] | None:
    """Parse a filename stem like 'ASX200_MINUTE_15' into ('ASX200', 'MINUTE_15')."""
    for tf in sorted(TIMEFRAME_ORDER, key=len, reverse=True):
        suffix = f"_{tf}"
        if stem.endswith(suffix):
            symbol = stem[: -len(suffix)]
            if symbol:
                return symbol, tf
    return None


def find_fallback_files(
    data_path: str, target_tf: str
) -> tuple[list[str], str | None]:
    """Find CSV files for a lower timeframe when target_tf files don't exist.

    Returns (files, source_tf) where source_tf is the timeframe of the found
    files, or ([], None) if no fallback is available.
    """
    if target_tf not in TIMEFRAME_ORDER:
        return [], None

    target_idx = TIMEFRAME_ORDER.index(target_tf)

    # Try each lower timeframe, lowest first (most granular = best source)
    for tf in TIMEFRAME_ORDER[:target_idx]:
        files = sorted(_glob.glob(os.path.join(data_path, f"*_{tf}.csv")))
        if files:
            return files, tf

    return [], None
