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

TIMEFRAME_ATTR = "fwbg_timeframe"

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


def bar_timing(df: pd.DataFrame) -> tuple[pd.Timedelta | None, int]:
    """Return the bar length and the first row where it is known.

    Loaders provide the nominal timeframe. For frames without it, fix the
    estimate as soon as two intervals are observed. Taking the shorter interval
    tolerates one leading gap and later gaps cannot revise the estimate.
    """
    timeframe = df.attrs.get(TIMEFRAME_ATTR)
    if timeframe is not None:
        return pd.Timedelta(minutes=Timeframe.from_str(timeframe).minutes), 0
    index = df.index
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        step = min(index[1] - index[0], index[2] - index[1])
        if step > pd.Timedelta(0):
            return step, 2
    return None, len(df)


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a higher timeframe.

    Accepts any timeframe spelling (canonical, short, enum value); an unknown
    value leaves the frame unchanged.
    """
    try:
        timeframe = Timeframe.from_str(target_tf)
        rule = timeframe.resample_rule
    except ValueError:
        return df

    agg = {"O": "first", "H": "max", "L": "min", "C": "last"}
    if "V" in df.columns:
        agg["V"] = "sum"

    # Every bar is labelled with its start time, like the native bars. pandas
    # defaults the weekly rule ("W-SUN") to right-closed/right-labelled, which
    # stamps a week with the Sunday that ends it; W1 bars start on Monday.
    if timeframe is Timeframe.W1:
        resampler = df.resample("W-MON", closed="left", label="left")
    else:
        resampler = df.resample(rule)
    result = resampler.agg(agg).dropna(subset=["O"])
    result.attrs[TIMEFRAME_ATTR] = timeframe.canonical
    return result


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


def find_fallback_files(data_path: str, target_tf: str) -> tuple[list[str], str | None]:
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
