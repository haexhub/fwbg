"""
Opening Range Breakout (ORB) Indicator Plugin.

Berechnet Features basierend auf dem Opening Range Konzept:
- Session ORB: Range/Position/Breakout für konfigurierbare Session-Stunden. Die Range
  ist das Zeitfenster [Open, Open + range_bars * Bar-Dauer); Features gelten bis zum
  Session-Ende (``session_end_hour``, Default 24 = lokale Mitternacht) und sind danach NaN
  — eine Range wird nie in den nächsten Tag oder über das Wochenende getragen.
- Retest-Signale bei konfigurierbaren Retracement-Levels (rl{N}_ Prefix)
- Session-Zeiten: ohne ``session_timezone`` sind ``sessions`` Stunden der Index-Uhr
  (kein DST). Mit ``session_timezone`` (z.B. "America/New_York", ``session_minute=30``)
  wird der Open pro Tag in lokaler Zeit berechnet und nach UTC umgerechnet (DST-korrekt);
  ein naiver Index gilt dann als UTC. Bars sind mit ihrer Open-Zeit gelabelt.

Spalten-Konvention (wie ibs/zscore/gotobi): Signal- (``_breakout_up/down``,
``_retest_bull/bear``) und Struktur-Spalten (``_range``, ``_sl_dist``, ``_or_*``) sind
NICHT geshiftet: sie tragen den Wert der eigenen Bar und sind für eine Entscheidung am
Bar-Close gedacht (Entry im Backtest am nächsten Open). Sie stehen in
``get_feature_columns()``, sind aber über ``get_signal_columns()`` als Entscheidungs-
Spalten markiert und gehören nicht als Modell-Features in eine ML-Pipeline
(Lookahead um eine Bar); alle übrigen Features sind per ``shift_features`` geshiftet.

Timeframe-Kompatibilität: Intraday (M1-H4). Auf DAY-Bars → NaN.

Spalten-Namensformat (via orb_col()):
  rb{N}_[cf{N}_][prb{N}_]orb_s{HH}_{feature}

Prefix-Abkürzungen:
  rb  = range bars      — Anzahl Bars für die Opening Range
  cf  = carry forward    — Tage ohne Breakout, Range weiter tragen (nur wenn aktiv)
  prb = pre-range bars   — Bars vor Session-Start in Range einbeziehen (nur wenn aktiv)
  rl  = retracement level — Retracement-Level für Retest-Signale (z.B. rl50, rl382)

cf/prb erscheinen nur im Prefix wenn sie konfiguriert sind (Wert != 0 oder Liste).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import (
    BaseIndicator,
    register_indicator,
    shift_features,
    safe_divide,
    rl_tag,
)
from fwbg_sdk.retest import (
    apply_breakout_threshold,
    compute_break_state,
    compute_retest_signals,
)


# ── Naming helpers ──────────────────────────────────────────────────────────


def orb_col(rb: int, cf, prb, session: int, feature: str) -> str:
    """Build ORB column name.

    Prefix segments:
      rb  = range bars      — number of bars defining the opening range
      cf  = carry forward    — days to carry range when no breakout (optional)
      prb = pre-range bars   — bars before session start included in range (optional)

    cf/prb are optional — pass None to omit from the prefix:
      orb_col(1, None, None, 8, "range") -> rb1_orb_s08_range
      orb_col(1, 0, None, 8, "range")    -> rb1_cf0_orb_s08_range
      orb_col(1, 0, 2, 8, "range")       -> rb1_cf0_prb2_orb_s08_range
    """
    parts = [f"rb{rb}"]
    if cf is not None:
        parts.append(f"cf{cf}")
    if prb is not None:
        parts.append(f"prb{prb}")
    parts.append(f"orb_s{session:02d}_{feature}")
    return "_".join(parts)


ORB_BASE_FEATURES = [
    "range",
    "position",
    "breakout_up",
    "breakout_down",
    "breakout_dist",
    "range_vs_atr",
    "poc_dist",
    "sl_dist",
    "or_high",
    "or_low",
    "or_midpoint",
    "post_bull",
    "post_bear",
]

ORB_SIGNAL_SUFFIXES = ("_breakout_up", "_breakout_down", "_retest_bull", "_retest_bear")

# Structural columns: computed from the completed opening range, not forward-looking.
# These must NOT be shifted — they're available as soon as the range period ends
# and are used by the exit strategy for SL/TP distances and structural price levels.
# Shifting them breaks entry_delay=0 (signal fires but distances are NaN).
ORB_STRUCTURAL_SUFFIXES = ("_sl_dist", "_or_high", "_or_low", "_or_midpoint", "_range")


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR via ta-lib, returns raw ATR (not percent)."""
    return ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)


def _median_step_ns(index: pd.DatetimeIndex) -> int:
    """Typical bar spacing in nanoseconds (median index difference, >= 1 minute fallback)."""
    if len(index) >= 2:
        step = index.to_series().diff().median()
        if pd.notna(step) and step > pd.Timedelta(0):
            return int(step.as_unit("ns").value)
    return int(pd.Timedelta(minutes=1).value)


@dataclass(frozen=True)
class _SessionWindows:
    """Per-bar session membership for one session open.

    ts_ns: bar timestamps as int64 ns (UTC when a session timezone is used,
        otherwise the index wall clock).
    open_ns: open of the session that belongs to the bar's local calendar day.
    session_id: 0 outside the session window, otherwise ordinal of the local
        calendar day + 1 (one id per session occurrence).
    runs: (session_id, first_pos, end_pos) of every session occurrence.
    """

    ts_ns: np.ndarray
    open_ns: np.ndarray
    session_id: np.ndarray
    runs: List[tuple]


def _session_windows(
    index: pd.DatetimeIndex,
    session_hour: int,
    session_minute: int,
    session_end_hour: float,
    session_timezone: Optional[str],
) -> _SessionWindows:
    """Assign every bar to the session [open, end) of its local calendar day.

    The open is ``session_hour:session_minute`` and the end ``session_end_hour``
    (clock hours, at most 24 = local midnight) of the same local day.  With a
    ``session_timezone`` the bar timestamps are read as UTC (or converted to UTC
    when the index is tz-aware) and the open/end are localised per day, so the
    UTC time of a session follows daylight saving time.  Without a timezone the
    index wall clock is used as is.
    """
    if session_timezone is None:
        wall = index.tz_localize(None) if index.tz is not None else index
        ts_ns = wall.as_unit("ns").asi8
        day_ns = wall.normalize().as_unit("ns").asi8
    else:
        utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        ts_ns = utc.as_unit("ns").asi8
        local = utc.tz_convert(session_timezone)
        day_ns = local.tz_localize(None).normalize().as_unit("ns").asi8

    uniq_days, inverse = np.unique(day_ns, return_inverse=True)
    days = pd.DatetimeIndex(uniq_days.astype("datetime64[ns]"))
    open_wall = days + pd.Timedelta(hours=session_hour, minutes=session_minute)
    end_wall = days + pd.Timedelta(hours=session_end_hour)
    if session_timezone is not None:
        # First occurrence for repeated wall times, shift forward over gaps.
        first = np.ones(len(days), dtype=bool)
        open_wall = open_wall.tz_localize(
            session_timezone, ambiguous=first, nonexistent="shift_forward"
        ).tz_convert("UTC")
        end_wall = end_wall.tz_localize(
            session_timezone, ambiguous=first, nonexistent="shift_forward"
        ).tz_convert("UTC")
    open_ns = open_wall.as_unit("ns").asi8[inverse]
    end_ns = end_wall.as_unit("ns").asi8[inverse]
    in_session = (ts_ns >= open_ns) & (ts_ns < end_ns)
    session_id = np.where(in_session, inverse + 1, 0).astype(np.int64)
    return _SessionWindows(
        ts_ns=ts_ns,
        open_ns=open_ns,
        session_id=session_id,
        runs=_session_runs(session_id),
    )


def _session_runs(session_id: np.ndarray) -> List[tuple]:
    """(session_id, first_pos, end_pos) for every contiguous run with id > 0."""
    n = len(session_id)
    if n == 0:
        return []
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(session_id) != 0) + 1, [n]))
    return [
        (int(session_id[a]), int(a), int(b))
        for a, b in zip(bounds[:-1], bounds[1:])
        if session_id[a] > 0
    ]


def _breakout_flags(
    close: np.ndarray,
    range_high: Union[np.ndarray, float],
    range_low: Union[np.ndarray, float],
    session_ok: np.ndarray,
    effective_min: Optional[np.ndarray],
    breakout_threshold: float,
    breakout_threshold_abs: float,
) -> tuple:
    """Breakout test shared by the carry-forward check and the signal columns.

    session_ok marks the bars on which a breakout may count (after the range
    window, outside pre-range bars); bars whose range is below ``effective_min``
    never count.  Returns (above, below) boolean arrays.
    """
    rng = range_high - range_low
    ok = session_ok
    if effective_min is not None:
        ok = ok & ~(rng < effective_min)
    above, below = apply_breakout_threshold(
        close,
        range_high,
        range_low,
        rng,
        breakout_threshold,
        breakout_threshold_abs,
    )
    return above & ok, below & ok


def _session_orb_features(
    df: pd.DataFrame,
    sessions: List[int],
    range_bars: int,
    atr: pd.Series,
    enable_retracement: bool = False,
    carry_forward_days: int = 0,
    pre_range_bars: int = 0,
    candle_span: str = "hl",
    breakout_threshold: float = 0.0,
    breakout_threshold_abs: float = 0.0,
    retracement_levels: Union[float, List[float]] = 0.5,
    min_retracement: float = 0.0,
    min_range_height: float = 0.0,
    min_range_height_atr: float = 0.0,
    session_end_hour: float = 24.0,
    session_minute: int = 0,
    session_timezone: Optional[str] = None,
    windows: Optional[Dict[int, _SessionWindows]] = None,
    bar_step_ns: Optional[int] = None,
) -> tuple[Dict[str, Union[pd.Series, np.ndarray]], List[dict]]:
    """
    Compute session-specific ORB features.

    For each configured session hour the opening range is the time window
    ``[open, open + range_bars * bar_step)`` of that day's session (bars are
    labelled by their open time; a missing bar shrinks the range, it does not
    stretch it to later bars).  Features are live from the end of the range
    window until ``session_end_hour``; outside the session they are NaN, so a
    range never reaches into the next day (or over a weekend).

    carry_forward_days: if no breakout occurs during a session, carry the range
        forward to the next N session occurrences. 0 = no carry (default).
    pre_range_bars: include the N bar periods before the session open in the
        range calculation, expanding the opening range window backward. 0 = disabled.

    Returns (features_dict, range_zones) where range_zones is a list of dicts
    describing each session's opening range rectangle for chart visualization.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    zones: List[dict] = []

    n = len(df)
    if bar_step_ns is None:
        bar_step_ns = _median_step_ns(df.index)
    range_dur_ns = range_bars * bar_step_ns
    pre_dur_ns = pre_range_bars * bar_step_ns

    close_vals = df["C"].values
    atr_vals = atr.values
    if min_range_height > 0 or min_range_height_atr > 0:
        atr_min = min_range_height_atr * atr_vals if min_range_height_atr > 0 else 0
        effective_min: Optional[np.ndarray] = np.maximum(
            min_range_height, atr_min
        ) + np.zeros(n)
    else:
        effective_min = None

    for session_hour in sessions:
        prefix = f"orb_s{session_hour:02d}"

        win = (windows or {}).get(session_hour)
        if win is None:
            win = _session_windows(
                df.index,
                session_hour,
                session_minute,
                session_end_hour,
                session_timezone,
            )
        sid_vals = win.session_id
        ts_ns = win.ts_ns

        # Session ID: one id per session occurrence, 0 outside any session window.
        session_id = pd.Series(sid_vals, index=df.index)
        in_session_vals = sid_vals > 0
        runs = win.runs

        # Opening range window by time: [open, open + range_bars * step).
        in_range_window_vals = in_session_vals & (ts_ns < win.open_ns + range_dur_ns)
        range_mask = pd.Series(in_range_window_vals, index=df.index)
        if candle_span == "hl":
            # H/L of all bars in the range window
            or_high = df["H"].where(range_mask).groupby(session_id).transform("max")
            or_low = df["L"].where(range_mask).groupby(session_id).transform("min")
        else:
            # Body: max/min of O,C across all bars in range (ignoring wicks)
            body_high = np.maximum(df["O"], df["C"])
            body_low = np.minimum(df["O"], df["C"])
            or_high = body_high.where(range_mask).groupby(session_id).transform("max")
            or_low = body_low.where(range_mask).groupby(session_id).transform("min")

        # --- pre_range_bars: expand range to include pre-session bars ---
        # Pre-range bars are the bars in [open - pre_range_bars * step, open).
        pre_mask_vals = np.zeros(n, dtype=bool)
        pre_bounds: Dict[int, tuple] = {}
        if pre_range_bars > 0:
            arr_h = or_high.values.copy()
            arr_l = or_low.values.copy()
            for sid, a, b in runs:
                open_ns = int(win.open_ns[a])
                pre_end = int(np.searchsorted(ts_ns, open_ns, side="left"))
                pre_start = int(
                    np.searchsorted(ts_ns, open_ns - pre_dur_ns, side="left")
                )
                if pre_start >= pre_end:
                    continue
                pre_bounds[sid] = (pre_start, pre_end)
                pre_mask_vals[pre_start:pre_end] = True
                if candle_span == "hl":
                    pre_h = df["H"].values[pre_start:pre_end].max()
                    pre_l = df["L"].values[pre_start:pre_end].min()
                else:
                    pre_o = df["O"].values[pre_start:pre_end]
                    pre_c = df["C"].values[pre_start:pre_end]
                    pre_h = max(pre_o.max(), pre_c.max())
                    pre_l = min(pre_o.min(), pre_c.min())
                cur_h = arr_h[a]
                cur_l = arr_l[a]
                if not np.isnan(cur_h):
                    arr_h[a:b] = max(cur_h, pre_h)
                    arr_l[a:b] = min(cur_l, pre_l)

            or_high = pd.Series(arr_h, index=df.index)
            or_low = pd.Series(arr_l, index=df.index)

        # --- carry_forward_days: carry range from no-breakout sessions ---
        carried_session_ids = set()
        if carry_forward_days > 0:
            arr_h = or_high.values.copy()
            arr_l = or_low.values.copy()

            carry_remaining = 0
            carry_h = None
            carry_l = None

            for sid, a, b in runs:
                carried = carry_remaining > 0
                if carried:
                    # Use carried range instead of this session's own range
                    arr_h[a:b] = carry_h
                    arr_l[a:b] = carry_l
                    carried_session_ids.add(sid)
                    cur_h, cur_l = carry_h, carry_l
                else:
                    cur_h, cur_l = arr_h[a], arr_l[a]
                    if np.isnan(cur_h):
                        continue

                # Same breakout test as the signal columns; a carried range
                # is pre-established, so its session has no range period.
                session_ok = ~pre_mask_vals[a:b]
                if not carried:
                    session_ok = session_ok & ~in_range_window_vals[a:b]
                up, down = _breakout_flags(
                    close_vals[a:b],
                    cur_h,
                    cur_l,
                    session_ok,
                    None if effective_min is None else effective_min[a:b],
                    breakout_threshold,
                    breakout_threshold_abs,
                )
                had_breakout = bool(up.any() or down.any())

                if carried:
                    carry_remaining = 0 if had_breakout else carry_remaining - 1
                elif not had_breakout:
                    carry_remaining = carry_forward_days
                    carry_h = cur_h
                    carry_l = cur_l

            or_high = pd.Series(arr_h, index=df.index)
            or_low = pd.Series(arr_l, index=df.index)

        # --- Collect range zones for chart visualization ---
        for sid, a, b in runs:
            if sid in carried_session_ids:
                continue
            h_val = or_high.values[a]
            l_val = or_low.values[a]
            if np.isnan(h_val) or np.isnan(l_val):
                continue
            zone_start = pre_bounds[sid][0] if sid in pre_bounds else a
            range_end = int(
                np.searchsorted(ts_ns, int(win.open_ns[a]) + range_dur_ns, side="left")
            )
            zone_end = max(a, min(range_end - 1, n - 1))
            # Use or_high/or_low which already respect candle_span
            zones.append(
                {
                    "start_ts": int(df.index[zone_start].timestamp() * 1000),
                    "end_ts": int(df.index[zone_end].timestamp() * 1000),
                    "high": float(h_val),
                    "low": float(l_val),
                    "session": f"s{session_hour:02d}",
                }
            )

        # Valid: inside the session, after the opening range window.
        # For carried sessions, range is pre-established → no range period masking
        in_range_period = in_range_window_vals
        if carried_session_ids:
            in_range_period = in_range_period & ~np.isin(
                sid_vals, list(carried_session_ids)
            )
        valid_vals = in_session_vals & ~in_range_period
        valid = pd.Series(valid_vals, index=df.index)

        or_range = or_high - or_low
        or_midpoint = (or_high + or_low) / 2

        # Signal-valid: additionally suppress signals during pre-range bars
        # (they are visually inside the new range zone — no trades allowed)
        # and, via _breakout_flags, for ranges below the minimum height.
        # Low-volatility ranges don't produce reliable breakout/retest signals.
        # Effective minimum = max(absolute, atr_multiple * ATR).
        signal_valid_vals = valid_vals & ~pre_mask_vals
        if effective_min is not None:
            signal_valid_vals = signal_valid_vals & ~(or_range.values < effective_min)
        signal_valid = pd.Series(signal_valid_vals, index=df.index)

        features[f"{prefix}_range"] = or_range.where(valid, np.nan)
        features[f"{prefix}_position"] = safe_divide(df["C"] - or_low, or_range).where(
            valid, np.nan
        )

        or_high_vals = or_high.values
        or_low_vals = or_low.values
        or_range_vals = or_range.values

        # Breakout signal: persistent 1 from the first bar closing above/below
        # the range until end of session.  This allows the simulation to enter
        # a trade even when an earlier trade blocked the initial breakout bar
        # (e.g. previous day's trade still open at breakout time).
        # Applies breakout_threshold / breakout_threshold_abs: effective = max(pct*range, abs).
        above_masked, below_masked = _breakout_flags(
            close_vals,
            or_high_vals,
            or_low_vals,
            valid_vals & ~pre_mask_vals,
            effective_min,
            breakout_threshold,
            breakout_threshold_abs,
        )
        above_valid = pd.Series(above_masked.astype(np.int8), index=df.index)
        below_valid = pd.Series(below_masked.astype(np.int8), index=df.index)
        ever_above = above_valid.groupby(session_id).cummax()
        ever_below = below_valid.groupby(session_id).cummax()
        features[f"{prefix}_breakout_up"] = ever_above.astype(float).where(
            signal_valid, np.nan
        )
        features[f"{prefix}_breakout_down"] = ever_below.astype(float).where(
            signal_valid, np.nan
        )

        # Breakout distance beyond the range edge, signed and normalized by
        # range size: > 0 above the high ((close - or_high) / range), < 0 below
        # the low (-(or_low - close) / range), 0 inside the range.
        dist_above = (df["C"] - or_high).clip(lower=0)
        dist_below = (or_low - df["C"]).clip(lower=0)
        breakout_dist = safe_divide(dist_above - dist_below, or_range)
        features[f"{prefix}_breakout_dist"] = breakout_dist.where(signal_valid, np.nan)

        features[f"{prefix}_range_vs_atr"] = safe_divide(or_range, atr).where(
            valid, np.nan
        )

        # POC proxy: normalized distance from close to ORB midpoint
        features[f"{prefix}_poc_dist"] = safe_divide(df["C"] - or_midpoint, atr).where(
            valid, np.nan
        )

        # SL distance: full range (breakout entry at boundary → SL at opposite boundary)
        # sl_mult in exit strategy controls buffer: 1.0 = exact boundary, >1.0 = beyond
        # When body range is 0 (O==C doji), sl_dist is invalid → NaN
        sl_dist = or_range.where(valid & (or_range > 0), np.nan)
        features[f"{prefix}_sl_dist"] = sl_dist

        # OR structural levels: absolute price levels for anchoring SL/TP.
        # The exit strategy can reference these via sl_level (suffix match)
        # to place SL at a fixed structural level regardless of entry price.
        range_valid = valid & (or_range > 0)
        features[f"{prefix}_or_high"] = or_high.where(range_valid, np.nan)
        features[f"{prefix}_or_low"] = or_low.where(range_valid, np.nan)
        features[f"{prefix}_or_midpoint"] = or_midpoint.where(range_valid, np.nan)

        # Post-breakout state via shared break detection (with threshold).
        # Only count breakouts on signal-valid bars (after range + pre-range period).
        _, _, post_bull_arr, post_bear_arr, broke_high_arr, broke_low_arr = (
            compute_break_state(
                above_masked,
                below_masked,
                or_high_vals,
                sid_vals,
                n,
            )
        )
        features[f"{prefix}_post_bull"] = pd.Series(
            post_bull_arr,
            index=df.index,
        ).where(signal_valid, np.nan)
        features[f"{prefix}_post_bear"] = pd.Series(
            post_bear_arr,
            index=df.index,
        ).where(signal_valid, np.nan)

        if enable_retracement:
            # Retest signals per retracement level
            rl_list = (
                [retracement_levels]
                if isinstance(retracement_levels, (int, float))
                else list(retracement_levels)
            )

            for rl in rl_list:
                rl_pfx = f"{rl_tag(rl)}_"
                entry_bull = or_high_vals - rl * or_range_vals
                entry_bear = or_low_vals + rl * or_range_vals

                retest_result = compute_retest_signals(
                    close=close_vals,
                    high=df["H"].values,
                    low=df["L"].values,
                    range_high=or_high_vals,
                    range_low=or_low_vals,
                    group_ids=sid_vals,
                    broke_high_arr=broke_high_arr,
                    broke_low_arr=broke_low_arr,
                    entry_bull=entry_bull,
                    entry_bear=entry_bear,
                    n=n,
                    min_retracement=min_retracement,
                )
                features[f"{prefix}_{rl_pfx}retest_bull"] = pd.Series(
                    retest_result["retest_bull"],
                    index=df.index,
                ).where(signal_valid, np.nan)
                features[f"{prefix}_{rl_pfx}retest_bear"] = pd.Series(
                    retest_result["retest_bear"],
                    index=df.index,
                ).where(signal_valid, np.nan)

                # SL: entry to opposite boundary
                rl_sl = ((1 - rl) * or_range).where(valid & (or_range > 0), np.nan)
                features[f"{prefix}_{rl_pfx}sl_dist"] = rl_sl

    return features, zones


@register_indicator("opening_range")
class OpeningRangeIndicator(BaseIndicator):
    """
    Opening Range Breakout Features.

    Session ORB — für konfigurierbare Session-Stunden (forward-filled),
    mit Retest-Signalen bei konfigurierbaren Retracement-Levels.
    """

    name = "opening_range"
    version = "1.1.0"
    benefits_from_stationary = False
    group = "session"

    def compute(
        self,
        df: pd.DataFrame,
        range_bars: Union[int, List[int]] = 1,
        atr_period: int = 14,
        sessions: List[int] = None,
        enable_retracement: bool = True,
        carry_forward_days: Union[int, List[int]] = 0,
        pre_range_bars: Union[int, List[int]] = 0,
        candle_span: str = "hl",
        breakout_threshold: float = 0.0,
        breakout_threshold_abs: float = 0.0,
        retracement_levels: Union[float, List[float]] = 0.5,
        min_retracement: float = 0.0,
        min_range_height: float = 0.0,
        min_range_height_atr: float = 0.0,
        session_end_hour: float = 24.0,
        session_minute: int = 0,
        session_timezone: Optional[str] = None,
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DatetimeIndex haben")

        # Skip daily data — ORB is intraday only
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                return df

        if sessions is None:
            sessions = [8, 9, 14, 15]
        else:
            sessions = [int(s) for s in sessions]

        # Normalisiere Parameter zu Listen
        self._validate_session_params(
            sessions, session_end_hour, session_minute, session_timezone
        )
        bar_step_ns = _median_step_ns(df.index)
        windows = {
            h: _session_windows(
                df.index, h, session_minute, session_end_hour, session_timezone
            )
            for h in sessions
        }

        rb_list = [range_bars] if isinstance(range_bars, int) else list(range_bars)
        cf_list = (
            [carry_forward_days]
            if isinstance(carry_forward_days, int)
            else list(carry_forward_days)
        )
        prb_list = (
            [pre_range_bars]
            if isinstance(pre_range_bars, int)
            else list(pre_range_bars)
        )

        features: Dict[str, Union[pd.Series, np.ndarray]] = {}
        all_zones: List[dict] = []
        atr = _compute_atr(df, atr_period)

        # cf/prb only in prefix when active (non-zero or multiple values)
        include_cf = len(cf_list) > 1 or any(v != 0 for v in cf_list)
        include_prb = len(prb_list) > 1 or any(v != 0 for v in prb_list)

        for rb in rb_list:
            for cf in cf_list:
                for prb in prb_list:
                    pfx = f"rb{rb}_"
                    if include_cf:
                        pfx += f"cf{cf}_"
                    if include_prb:
                        pfx += f"prb{prb}_"
                    session_features, session_zones = _session_orb_features(
                        df,
                        sessions,
                        rb,
                        atr,
                        enable_retracement,
                        cf,
                        prb,
                        candle_span,
                        breakout_threshold,
                        breakout_threshold_abs,
                        retracement_levels,
                        min_retracement,
                        min_range_height,
                        min_range_height_atr,
                        session_end_hour,
                        session_minute,
                        session_timezone,
                        windows,
                        bar_step_ns,
                    )
                    features.update(
                        {f"{pfx}{k}": v for k, v in session_features.items()}
                    )
                    all_zones.extend(session_zones)

        self._range_zones = all_zones

        if not features:
            return df

        self._feature_columns = list(features.keys())
        self._signal_columns = [
            k for k in features if any(k.endswith(s) for s in ORB_SIGNAL_SUFFIXES)
        ]

        # Signal columns are NOT shifted: they are event-based (fire at bar
        # close) and simulate_pro_trade already enters at idx+1 which provides
        # the correct 1-bar delay.  Shifting them would add a redundant second
        # bar of delay.
        # Structural columns are NOT shifted: they represent completed-range
        # levels (or_high, or_low, sl_dist, etc.) needed by the exit strategy
        # at the breakout bar itself (especially with entry_delay=0).
        # A list in feature order, not a set: the column layout must not depend
        # on PYTHONHASHSEED. orb_based refuses several *_sl_dist candidates
        # unless exit_params names one, but other consumers still see the order.
        no_shift = [
            k
            for k in features
            if any(k.endswith(s) for s in ORB_SIGNAL_SUFFIXES + ORB_STRUCTURAL_SUFFIXES)
        ]
        shifted = {k: v for k, v in features.items() if k not in no_shift}
        features_df = shift_features(shifted, df.index)
        for k in no_shift:
            features_df[k] = features[k]
        return pd.concat([df, features_df], axis=1)

    @staticmethod
    def _validate_session_params(
        sessions: List[int],
        session_end_hour: float,
        session_minute: int,
        session_timezone: Optional[str],
    ) -> None:
        if session_timezone is not None:
            try:
                ZoneInfo(session_timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"Unknown session_timezone: {session_timezone!r}"
                ) from exc
        if not 0 <= session_minute < 60:
            raise ValueError(f"session_minute must be in 0..59, got {session_minute}")
        for h in sessions:
            if not 0 <= h <= 23:
                raise ValueError(f"session hour must be in 0..23, got {h}")
            if not h + session_minute / 60 < session_end_hour <= 24:
                raise ValueError(
                    f"session_end_hour ({session_end_hour}) must be after the session "
                    f"open {h:02d}:{session_minute:02d} and at most 24"
                )

    # Sessions covered by the orb_scalping pipeline (data-local hours; UTC for UTC data):
    # 0=Nikkei/ASX200 open, 1=Nikkei/HK50 morning, 2=All Asia morning,
    # 5=Nikkei/HK50 afternoon open, 6=DAX pre-market, 7=Xetra/DAX open,
    # 8=London open, 12=NY pre-dawn, 13=NY pre-market, 14=approaching NYSE open
    _PIPELINE_SESSIONS = [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]

    @staticmethod
    def _rl_prefixes(retracement_levels) -> List[str]:
        """Build rl{N}_ prefixes from retracement_levels param."""
        rl_list = (
            [retracement_levels]
            if isinstance(retracement_levels, (int, float))
            else list(retracement_levels)
        )
        return [f"{rl_tag(rl)}_" for rl in rl_list]

    def _resolve_col_params(self, params=None):
        """Merge defaults with optional params and normalise list params.

        Returns (rb_list, cf_list, prb_list, include_cf, include_prb,
                 sessions, rl_pfxs) — mirrors the normalisation in compute().
        rl_pfxs is empty when retracement is disabled (no retest columns).
        """
        d = {**self.get_default_params(), **(params or {})}
        rb_raw = d["range_bars"]
        cf_raw = d["carry_forward_days"]
        prb_raw = d["pre_range_bars"]
        rb_list = [rb_raw] if isinstance(rb_raw, (int, float)) else list(rb_raw)
        cf_list = [cf_raw] if isinstance(cf_raw, (int, float)) else list(cf_raw)
        prb_list = [prb_raw] if isinstance(prb_raw, (int, float)) else list(prb_raw)
        include_cf = len(cf_list) > 1 or any(v != 0 for v in cf_list)
        include_prb = len(prb_list) > 1 or any(v != 0 for v in prb_list)
        sessions = d.get("sessions") or [8, 9, 14, 15]
        rl_pfxs = (
            self._rl_prefixes(d["retracement_levels"])
            if d.get("enable_retracement", True)
            else []
        )
        return rb_list, cf_list, prb_list, include_cf, include_prb, sessions, rl_pfxs

    def get_feature_columns(self, params=None) -> List[str]:
        if self._feature_columns and not params:
            return self._feature_columns
        rb_list, cf_list, prb_list, include_cf, include_prb, sessions, rl_pfxs = (
            self._resolve_col_params(params)
        )
        cols = []
        for rb in rb_list:
            for cf_val in cf_list:
                for prb_val in prb_list:
                    cf = cf_val if include_cf else None
                    prb = prb_val if include_prb else None
                    for h in sessions:
                        for feat in ORB_BASE_FEATURES:
                            cols.append(orb_col(rb, cf, prb, h, feat))
                        for rl_pfx in rl_pfxs:
                            cols.append(orb_col(rb, cf, prb, h, f"{rl_pfx}retest_bull"))
                            cols.append(orb_col(rb, cf, prb, h, f"{rl_pfx}retest_bear"))
                            cols.append(orb_col(rb, cf, prb, h, f"{rl_pfx}sl_dist"))
        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        if self._signal_columns and not params:
            return self._signal_columns
        rb_list, cf_list, prb_list, include_cf, include_prb, sessions, rl_pfxs = (
            self._resolve_col_params(params)
        )
        signals = []
        for rb in rb_list:
            for cf_val in cf_list:
                for prb_val in prb_list:
                    cf = cf_val if include_cf else None
                    prb = prb_val if include_prb else None
                    for h in sessions:
                        signals.extend(
                            [
                                orb_col(rb, cf, prb, h, "breakout_up"),
                                orb_col(rb, cf, prb, h, "breakout_down"),
                            ]
                        )
                        for rl_pfx in rl_pfxs:
                            signals.extend(
                                [
                                    orb_col(rb, cf, prb, h, f"{rl_pfx}retest_bull"),
                                    orb_col(rb, cf, prb, h, f"{rl_pfx}retest_bear"),
                                ]
                            )
        return signals

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "range_bars": 1,
            "atr_period": 14,
            "sessions": [0, 1, 2, 5, 6, 7, 8, 12, 13, 14],
            "enable_retracement": True,
            "carry_forward_days": 0,
            "pre_range_bars": 0,
            "candle_span": "hl",
            "breakout_threshold": 0.0,
            "breakout_threshold_abs": 0.0,
            "retracement_levels": 0.5,
            "min_retracement": 0.0,
            "min_range_height": 0.0,
            "min_range_height_atr": 0.0,
            "session_end_hour": 24.0,
            "session_minute": 0,
            "session_timezone": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "range_bars": {
                "type": "list[int]",
                "default": 1,
                "description": "Number of bars defining the opening range (rb = range bars). At M15: 1 bar = 15min range, 2 bars = 30min range. Column format: rb{N}_[cf{N}_][prb{N}_]orb_s{HH}_{feature}. Can be a list (e.g. [1, 2]) to compute features for multiple range sizes.",
                "min": 1,
                "max": 12,
                "step": 1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing the opening range size.",
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "sessions": {
                "type": "list[int]",
                "default": [8, 9, 14, 15],
                "description": "Data-local hours for session-specific ORB features. Each session produces features that persist until the next occurrence of that session hour.",
                "min": 0,
                "max": 23,
                "choices": [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15],
                "choice_labels": {
                    "0": "00 – Tokyo open (09:00 JST)",
                    "1": "01 – Hong Kong morning",
                    "2": "02 – Asia morning",
                    "5": "05 – Hong Kong afternoon",
                    "6": "06 – DAX pre-market",
                    "7": "07 – Xetra / DAX open (summer)",
                    "8": "08 – London open (winter) / Xetra (winter)",
                    "9": "09 – London morning",
                    "12": "12 – NY pre-dawn",
                    "13": "13 – NY pre-market (NYSE open in summer: 13:30)",
                    "14": "14 – NYSE open (winter: 14:30)",
                    "15": "15 – NYSE first hour (winter) / second hour (summer)",
                },
            },
            "enable_retracement": {
                "type": "bool",
                "default": True,
                "description": "Enable retest zone features and retest signals at configured retracement levels.",
            },
            "carry_forward_days": {
                "type": "list[int]",
                "default": 0,
                "description": "Carry forward days (cf): if no breakout occurs during a session, carry the ORB range to the next N sessions. 0 = disabled. Only adds cf{N}_ prefix when active.",
                "min": 0,
                "max": 5,
                "step": 1,
            },
            "pre_range_bars": {
                "type": "list[int]",
                "default": 0,
                "description": "Pre-range bars (prb): number of bars before session start to include in range calculation. 0 = disabled. Only adds prb{N}_ prefix when active.",
                "min": 0,
                "max": 8,
                "step": 1,
            },
            "candle_span": {
                "type": "choice",
                "default": "hl",
                "description": "Vertical extent of candles used for range. 'hl': full candle including wicks (H/L). 'body': candle body only (max/min of O/C, ignoring wicks).",
                "choices": ["hl", "body"],
            },
            "breakout_threshold": {
                "type": "float",
                "default": 0.0,
                "description": "Minimum distance as fraction of range for breakout. E.g. 0.05 = close must be at least 5% of range beyond boundary. 0 = disabled.",
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
            },
            "breakout_threshold_abs": {
                "type": "float",
                "default": 0.0,
                "description": "Minimum distance in absolute terms (pips/points) for breakout. Effective threshold = max(pct * range, abs). 0 = disabled.",
                "min": 0.0,
                "max": 100.0,
                "step": 1.0,
            },
            "retracement_levels": {
                "type": "list[float]",
                "default": 0.5,
                "description": (
                    "Retracement fraction(s) of the OR range for entry level. "
                    "0.5 = midpoint. 0 = at boundary. 0.382 = shallow retrace. "
                    "Always generates rl{N}_ prefixed columns (e.g. rl50_retest_bull). "
                    "Pass a list to precompute multiple variants for grid search."
                ),
                "min": 0.0,
                "max": 0.9,
                "step": 0.1,
            },
            "min_retracement": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum retracement of OR range (checked via H/L) before "
                    "retest signal fires. 0.3 = price must retrace at least 30% from "
                    "breakout boundary. 0 = disabled."
                ),
                "min": 0.0,
                "max": 0.9,
                "step": 0.1,
            },
            "min_range_height": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum opening range height in absolute terms (pips/points). "
                    "Sessions with range below this threshold produce no signals. "
                    "Effective minimum = max(absolute, ATR multiple). 0 = disabled."
                ),
                "min": 0.0,
                "max": 1000.0,
                "step": 1.0,
            },
            "min_range_height_atr": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum opening range height as ATR multiple. "
                    "E.g. 0.5 = range must be at least 50% of ATR. "
                    "Effective minimum = max(absolute, ATR multiple). 0 = disabled."
                ),
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
            },
            "session_end_hour": {
                "type": "float",
                "default": 24.0,
                "description": (
                    "Clock hour (session time, 24 = midnight) at which each session ends. "
                    "Ranges, breakouts and retests only exist between the session open and "
                    "this hour of the same day; afterwards the session columns are NaN. "
                    "Must be after the latest configured session open."
                ),
                "min": 1.0,
                "max": 24.0,
                "step": 1.0,
            },
            "session_minute": {
                "type": "int",
                "default": 0,
                "description": (
                    "Minute of the session open (e.g. 30 with sessions=[9] and "
                    "session_timezone='America/New_York' = NYSE cash open 09:30). "
                    "Applies to all sessions; column names keep the hour only (orb_s09)."
                ),
                "min": 0,
                "max": 59,
                "step": 1,
            },
            "session_timezone": {
                "type": "string",
                "default": None,
                "description": (
                    "IANA timezone in which the session hours are given, e.g. "
                    "'America/New_York' or 'Europe/London'. The open is computed per day "
                    "in that timezone and converted to UTC, so it follows daylight saving "
                    "time; naive index timestamps are read as UTC. None = hours are read "
                    "directly on the index clock (no DST handling)."
                ),
            },
        }

    def get_column_group_labels(self) -> dict:
        # Hours are index-clock hours (UTC for UTC data) without DST handling.
        # Market opens move by one hour with daylight saving time; use
        # session_timezone / session_minute for an exact local open.
        return {
            "s00": "00:00 — Tokyo open (09:00 JST); ASX open in southern winter only",
            "s01": "01:00 — Hong Kong morning",
            "s02": "02:00 — Asia morning",
            "s05": "05:00 — Hong Kong afternoon",
            "s06": "06:00 — DAX pre-market",
            "s07": "07:00 — Xetra / DAX open in European summer, London open in BST",
            "s08": "08:00 — London / Xetra open in European winter (GMT/CET)",
            "s09": "09:00 — London morning",
            "s12": "12:00 — NY pre-dawn",
            "s13": "13:00 — NY pre-market (NYSE opens 13:30 in US summer)",
            "s14": "14:00 — 30 min before the NYSE open in US winter (14:30), 30 min after it in summer",
            "s15": "15:00 — 30 min after the NYSE open in US winter, 90 min after it in summer",
        }


__all__ = ["OpeningRangeIndicator"]
