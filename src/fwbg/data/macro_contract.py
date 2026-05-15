"""Release-date contract for external macro / COT data.

Any DataFrame stored under ``data/<source>/`` that participates in feature
merges MUST include a ``release_date`` column (datetime, ideally tz-aware
UTC).  The merger refuses to attach a row whose ``release_date`` is greater
than the target bar's timestamp.

Why: report-date (when the data describes) and release-date (when the data
became public) differ.  COT reports cover Tuesday but publish Friday; OECD
monthly stats cover a month but publish weeks later.  Joining on report-date
leaks future information into backtests.

See ``docs/plans/2026-05-15-cot-macro-audit.md`` for the audit that produced
this contract.
"""
from __future__ import annotations

import pandas as pd

RELEASE_COL = "release_date"


def assert_release_dates_present(df: pd.DataFrame, source_name: str) -> None:
    """Validate that ``df`` carries a usable ``release_date`` column."""
    if RELEASE_COL not in df.columns:
        raise ValueError(
            f"{source_name}: missing required column '{RELEASE_COL}'. "
            f"Every macro dataset must declare when its rows became public."
        )
    if df[RELEASE_COL].isna().any():
        raise ValueError(f"{source_name}: '{RELEASE_COL}' contains NaT.")
    if not pd.api.types.is_datetime64_any_dtype(df[RELEASE_COL]):
        raise ValueError(f"{source_name}: '{RELEASE_COL}' must be datetime.")


def merge_respect_release(
    bars: pd.DataFrame,
    macro: pd.DataFrame,
    source_name: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Asof-join ``macro`` onto ``bars`` respecting ``release_date``.

    For each bar at time ``t``, attach the macro row with the largest
    ``release_date <= t``.  Bars before the first release receive NaN.

    Args:
        bars: DataFrame whose index is the bar timestamp.
        macro: DataFrame containing ``release_date`` and ``value_cols``.
        source_name: Tag used in error messages.
        value_cols: Names of columns to copy from ``macro`` onto ``bars``.

    Returns:
        ``bars`` with ``value_cols`` attached.  The auxiliary
        ``release_date`` column is **not** propagated.
    """
    assert_release_dates_present(macro, source_name)
    missing = [c for c in value_cols if c not in macro.columns]
    if missing:
        raise ValueError(
            f"{source_name}: value_cols not found in macro DataFrame: {missing}"
        )

    index_name = bars.index.name or "_bar_ts"
    bars_sorted = bars.sort_index()
    left = bars_sorted.reset_index().rename(columns={bars_sorted.index.name or "index": index_name})

    macro_sorted = macro[[RELEASE_COL] + value_cols].sort_values(RELEASE_COL)

    merged = pd.merge_asof(
        left,
        macro_sorted,
        left_on=index_name,
        right_on=RELEASE_COL,
        direction="backward",
    )
    merged = merged.drop(columns=[RELEASE_COL])
    merged = merged.set_index(index_name)
    merged.index.name = bars.index.name
    return merged
