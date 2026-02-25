"""Shared retest logic for ORB and PDHL indicators.

Both strategies trade the same pattern:
1. Define a reference range (ORB: first N bars, PDHL: previous day)
2. Detect breakout beyond range boundary
3. Wait for price to depart from entry zone
4. Wait for price to return to entry zone (retest)
5. Fire signal once per group (session/day)

This module provides the shared computation so both indicators use
identical signal logic, just with different reference ranges.
"""
from typing import Dict, Optional

import numpy as np


def compute_break_state(
    above: np.ndarray,
    below: np.ndarray,
    nan_guard: np.ndarray,
    group_ids: np.ndarray,
    n: int,
    session_mask: np.ndarray = None,
) -> tuple:
    """Compute break detection and post-breakout state arrays.

    above/below are pre-computed boolean arrays indicating where price
    crossed the range boundary.  The caller decides how to compute them
    (e.g. native-bar Close vs resampled Close, with or without threshold).

    nan_guard: array used only for NaN checks (bars without reference data).
    group_ids: session or day IDs — state resets at each group boundary.

    When session_mask is None, breaks are detected on all bars.
    When provided, only session bars can trigger breakouts.

    Returns (high_break, low_break, post_bull, post_bear,
             broke_high_arr, broke_low_arr).
    """
    high_break = np.zeros(n)
    low_break = np.zeros(n)
    post_bull = np.zeros(n)
    post_bear = np.zeros(n)
    broke_high_arr = np.zeros(n, dtype=bool)
    broke_low_arr = np.zeros(n, dtype=bool)

    prev_group_id = -1
    broke_high = False
    broke_low = False

    for i in range(n):
        if group_ids[i] != prev_group_id:
            prev_group_id = group_ids[i]
            broke_high = False
            broke_low = False

        if np.isnan(nan_guard[i]):
            continue

        in_session = session_mask[i] if session_mask is not None else True
        if in_session:
            if above[i] and not broke_high:
                high_break[i] = 1.0
                broke_high = True
            if below[i] and not broke_low:
                low_break[i] = 1.0
                broke_low = True

        broke_high_arr[i] = broke_high
        broke_low_arr[i] = broke_low

        if broke_high:
            post_bull[i] = 1.0
        if broke_low:
            post_bear[i] = 1.0

    return high_break, low_break, post_bull, post_bear, broke_high_arr, broke_low_arr


def apply_breakout_threshold(
    close: np.ndarray,
    range_high: np.ndarray,
    range_low: np.ndarray,
    range_size: np.ndarray,
    breakout_threshold: float = 0.0,
    breakout_threshold_abs: float = 0.0,
) -> tuple:
    """Compute above/below breakout arrays with configurable threshold.

    breakout_threshold: minimum distance as fraction of range (e.g. 0.05 = 5%).
    breakout_threshold_abs: minimum distance in absolute terms (pips/points).
    The effective threshold is max(pct * range, abs).

    Returns (above, below) boolean arrays.
    """
    if breakout_threshold > 0 or breakout_threshold_abs > 0:
        pct_dist = breakout_threshold * range_size
        dist = np.maximum(pct_dist, breakout_threshold_abs)
        above = close > range_high + dist
        below = close < range_low - dist
    else:
        above = close > range_high
        below = close < range_low
    return above, below


def compute_retest_signals(
    close: np.ndarray,
    high: Optional[np.ndarray],
    low: Optional[np.ndarray],
    range_high: np.ndarray,
    range_low: np.ndarray,
    group_ids: np.ndarray,
    broke_high_arr: np.ndarray,
    broke_low_arr: np.ndarray,
    entry_bull: np.ndarray,
    entry_bear: np.ndarray,
    half_band: np.ndarray,
    n: int,
    min_retracement: float = 0.0,
    session_mask: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Compute retest signals using shared logic.

    Callers pre-compute entry levels and zone widths:
    - entry_bull / entry_bear: center of the retest zone per bar
    - half_band: half-width of the zone around entry

    ORB computes: entry = midpoint, half_band = zone_width/2 * range
    PDHL computes: entry = boundary - rl * range, half_band = atr_width * max(range, atr) / 2

    min_retracement: minimum fraction of range that must be retraced
    (checked via High/Low) before retest fires.  0 = disabled.

    session_mask: if provided, retest signals only fire during session bars.

    Returns dict with 'retest_bull' and 'retest_bear' arrays.
    """
    near_entry_bull = (close >= entry_bull - half_band) & (close <= entry_bull + half_band)
    near_entry_bear = (close >= entry_bear - half_band) & (close <= entry_bear + half_band)

    range_size = range_high - range_low
    retrace_bull_threshold = range_high - min_retracement * range_size
    retrace_bear_threshold = range_low + min_retracement * range_size

    retest_bull = np.zeros(n)
    retest_bear = np.zeros(n)

    prev_group_id = -1
    retested_bull = False
    retested_bear = False
    retracement_ok_bull = False
    retracement_ok_bear = False
    departed_bull = False
    departed_bear = False

    for i in range(n):
        if group_ids[i] != prev_group_id:
            prev_group_id = group_ids[i]
            retested_bull = False
            retested_bear = False
            retracement_ok_bull = False
            retracement_ok_bear = False
            departed_bull = False
            departed_bear = False

        if np.isnan(range_high[i]):
            continue

        # Track retracement via High/Low (cumulative per group)
        if min_retracement > 0 and low is not None:
            if broke_high_arr[i] and not retracement_ok_bull:
                if low[i] <= retrace_bull_threshold[i]:
                    retracement_ok_bull = True
            if broke_low_arr[i] and not retracement_ok_bear:
                if high[i] >= retrace_bear_threshold[i]:
                    retracement_ok_bear = True
        else:
            retracement_ok_bull = True
            retracement_ok_bear = True

        # Departure: price must move ABOVE (bull) / BELOW (bear) the
        # entry band before a retest can fire.
        if broke_high_arr[i] and not departed_bull:
            if close[i] > entry_bull[i] + half_band[i]:
                departed_bull = True
        if broke_low_arr[i] and not departed_bear:
            if close[i] < entry_bear[i] - half_band[i]:
                departed_bear = True

        # Session filter for signal firing
        in_session = session_mask[i] if session_mask is not None else True
        if not in_session:
            continue

        # Bull retest: departed + retraced + in zone + price above range low
        if departed_bull and retracement_ok_bull and not retested_bull \
                and near_entry_bull[i] and close[i] > range_low[i]:
            retest_bull[i] = 1.0
            retested_bull = True
        # Bear retest: departed + retraced + in zone + price below range high
        if departed_bear and retracement_ok_bear and not retested_bear \
                and near_entry_bear[i] and close[i] < range_high[i]:
            retest_bear[i] = 1.0
            retested_bear = True

    return {
        "retest_bull": retest_bull,
        "retest_bear": retest_bear,
    }
