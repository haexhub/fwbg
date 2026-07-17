"""Timeframe hierarchy, resampling utilities, and source-file fallback logic.

Alle Timeframe-Bezeichnungen laufen über die zentrale :class:`~fwbg_sdk.enums.Timeframe`
(Single Source of Truth). Dateinamen und Vergleiche verwenden die kanonische
Langform (``HOUR_1``, ``DAY_1`` …); ältere Kurzformen (``HOUR``, ``DAY``) werden
beim Parsen toleriert und auf die kanonische Form normalisiert.
"""
import glob as _glob
import os

import pandas as pd

from fwbg_sdk.enums import Timeframe

# Ordered from lowest to highest resolution (canonical names)
TIMEFRAME_ORDER = [tf.canonical for tf in sorted(Timeframe, key=lambda t: t.minutes)]

# Pandas resample rule per canonical timeframe (kept for reference; resample_ohlcv
# resolves via the enum so it also accepts legacy/short spellings).
RESAMPLE_RULE: dict[str, str] = {tf.canonical: tf.resample_rule for tf in Timeframe}

# Legacy short forms still found in older on-disk filenames, longest-first so
# canonical suffixes (``HOUR_1``) win over their short alias (``HOUR``).
_PARSE_SUFFIXES = sorted(
    set(TIMEFRAME_ORDER) | {"HOUR", "DAY", "WEEK", "MINUTE"},
    key=len,
    reverse=True,
)


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a higher timeframe.

    Accepts any timeframe spelling (canonical, short, enum value); an unknown
    value leaves the frame unchanged.
    """
    try:
        rule = Timeframe.from_str(target_tf).resample_rule
    except ValueError:
        return df

    agg = {"O": "first", "H": "max", "L": "min", "C": "last"}
    if "V" in df.columns:
        agg["V"] = "sum"

    return df.resample(rule).agg(agg).dropna(subset=["O"])


def parse_symbol_timeframe(stem: str) -> tuple[str, str] | None:
    """Parse a filename stem like ``'ASX200_MINUTE_15'`` into ``('ASX200', 'MINUTE_15')``.

    Recognises both canonical (``HOUR_1``) and legacy short (``HOUR``) suffixes and
    always returns the canonical timeframe form.
    """
    for suffix_tf in _PARSE_SUFFIXES:
        suffix = f"_{suffix_tf}"
        if stem.endswith(suffix):
            symbol = stem[: -len(suffix)]
            if symbol:
                return symbol, Timeframe.from_str(suffix_tf).canonical
    return None


def find_fallback_files(
    data_path: str, target_tf: str
) -> tuple[list[str], str | None]:
    """Find CSV files for a lower timeframe when target_tf files don't exist.

    Returns (files, source_tf) where source_tf is the canonical timeframe of the
    found files, or ([], None) if no fallback is available.
    """
    try:
        target = Timeframe.from_str(target_tf).canonical
    except ValueError:
        return [], None

    target_idx = TIMEFRAME_ORDER.index(target)

    # Try each lower timeframe, lowest first (most granular = best source)
    for tf in TIMEFRAME_ORDER[:target_idx]:
        files = sorted(_glob.glob(os.path.join(data_path, f"*_{tf}.csv")))
        if files:
            return files, tf

    return [], None
