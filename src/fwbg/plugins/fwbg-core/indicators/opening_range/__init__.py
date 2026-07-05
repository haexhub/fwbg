"""
Opening Range Breakout (ORB) Indicator Plugin.

Berechnet Features basierend auf dem Opening Range Konzept:
- Session ORB: Range/Position/Breakout für konfigurierbare Session-Stunden (forward-filled)
- Retest-Signale bei konfigurierbaren Retracement-Levels (rl{N}_ Prefix)

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
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide, rl_tag
from fwbg_sdk.retest import apply_breakout_threshold, compute_break_state, compute_retest_signals


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
    "range", "position", "breakout_up", "breakout_down",
    "breakout_dist",
    "range_vs_atr", "poc_dist", "sl_dist",
    "post_bull", "post_bear",
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
) -> tuple[Dict[str, Union[pd.Series, np.ndarray]], List[dict]]:
    """
    Compute session-specific ORB features.

    For each configured session hour, compute the opening range at that hour
    and forward-fill until the next occurrence of that session hour.

    carry_forward_days: if no breakout occurs during a session, carry the range
        forward to the next N session occurrences. 0 = no carry (default).
    pre_range_bars: include N bars before the session start in the range
        calculation, expanding the opening range window backward. 0 = disabled.

    Returns (features_dict, range_zones) where range_zones is a list of dicts
    describing each session's opening range rectangle for chart visualization.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    zones: List[dict] = []

    hours = pd.Series(df.index.hour, index=df.index)
    prev_hour = hours.shift(1)

    for session_hour in sessions:
        prefix = f"orb_s{session_hour:02d}"

        # Boolean Series: is this bar within the session hour?
        is_session = hours == session_hour

        # Session start: first bar of a new session_hour block
        session_start = is_session & (prev_hour != session_hour)

        # Session ID: increments at each session start, groups all bars until next start
        session_id = session_start.cumsum()

        # Count ALL bars from session start (0-indexed), crossing hour boundaries.
        # Old code used is_session filter which limited counting to one hour.
        bar_in_block = pd.Series(1, index=df.index).groupby(session_id).cumsum() - 1

        # Opening range computation — uses absolute bar count from session start
        in_session = session_id > 0
        if candle_span == "hl":
            # H/L of all bars in the range window
            range_mask = in_session & (bar_in_block < range_bars)
            or_high = df["H"].where(range_mask).groupby(session_id).transform("max")
            or_low = df["L"].where(range_mask).groupby(session_id).transform("min")
        else:
            # Body: max/min of O,C across all bars in range (ignoring wicks)
            range_mask = in_session & (bar_in_block < range_bars)
            body_high = np.maximum(df["O"], df["C"])
            body_low = np.minimum(df["O"], df["C"])
            or_high = body_high.where(range_mask).groupby(session_id).transform("max")
            or_low = body_low.where(range_mask).groupby(session_id).transform("min")

        # --- pre_range_bars: expand range to include pre-session bars ---
        if pre_range_bars > 0:
            start_positions = np.where(session_start.values)[0]
            arr_h = or_high.values.copy()
            arr_l = or_low.values.copy()
            sid_vals = session_id.values

            for pos in start_positions:
                sid = sid_vals[pos]
                if sid == 0:
                    continue
                pre_start = max(0, pos - pre_range_bars)
                if pre_start < pos:
                    if candle_span == "hl":
                        pre_h = df["H"].values[pre_start:pos].max()
                        pre_l = df["L"].values[pre_start:pos].min()
                    else:
                        pre_o = df["O"].values[pre_start:pos]
                        pre_c = df["C"].values[pre_start:pos]
                        pre_h = max(pre_o.max(), pre_c.max())
                        pre_l = min(pre_o.min(), pre_c.min())
                    sess_mask = sid_vals == sid
                    cur_h = arr_h[sess_mask][0]
                    cur_l = arr_l[sess_mask][0]
                    if not np.isnan(cur_h):
                        arr_h[sess_mask] = max(cur_h, pre_h)
                        arr_l[sess_mask] = min(cur_l, pre_l)

            or_high = pd.Series(arr_h, index=df.index)
            or_low = pd.Series(arr_l, index=df.index)

        # --- carry_forward_days: carry range from no-breakout sessions ---
        carried_session_ids = set()
        if carry_forward_days > 0:
            unique_sids = sorted(session_id.unique())
            arr_h = or_high.values.copy()
            arr_l = or_low.values.copy()
            sid_vals = session_id.values
            close_vals = df["C"].values

            carry_remaining = 0
            carry_h = None
            carry_l = None

            for sid in unique_sids:
                if sid == 0:
                    continue

                sess_mask = sid_vals == sid

                if carry_remaining > 0:
                    # Use carried range instead of this session's own range
                    arr_h[sess_mask] = carry_h
                    arr_l[sess_mask] = carry_l
                    carried_session_ids.add(sid)

                    # Check breakout against carried range
                    sess_close = close_vals[sess_mask]
                    had_breakout = (sess_close > carry_h).any() or (sess_close < carry_l).any()

                    if had_breakout:
                        carry_remaining = 0
                    else:
                        carry_remaining -= 1
                else:
                    # Use own range, check for breakout
                    own_h = arr_h[sess_mask][0]
                    own_l = arr_l[sess_mask][0]

                    if not np.isnan(own_h):
                        sess_close = close_vals[sess_mask]
                        had_breakout = (sess_close > own_h).any() or (sess_close < own_l).any()

                        if not had_breakout:
                            carry_remaining = carry_forward_days
                            carry_h = own_h
                            carry_l = own_l

            or_high = pd.Series(arr_h, index=df.index)
            or_low = pd.Series(arr_l, index=df.index)

        # --- Collect range zones for chart visualization ---
        start_positions = np.where(session_start.values)[0]
        for pos in start_positions:
            sid = session_id.values[pos]
            if sid == 0:
                continue
            if sid in carried_session_ids:
                continue
            h_val = or_high.values[pos]
            l_val = or_low.values[pos]
            if np.isnan(h_val) or np.isnan(l_val):
                continue
            zone_start = max(0, pos - pre_range_bars) if pre_range_bars > 0 else pos
            zone_end = min(pos + range_bars - 1, len(df) - 1)
            # Use or_high/or_low which already respect candle_span
            zone_h = float(h_val)
            zone_l = float(l_val)
            zones.append({
                "start_ts": int(df.index[zone_start].timestamp() * 1000),
                "end_ts": int(df.index[zone_end].timestamp() * 1000),
                "high": zone_h,
                "low": zone_l,
                "session": f"s{session_hour:02d}",
            })

        # Valid: after first session, not within the opening range bars themselves
        # For carried sessions, range is pre-established → no range period masking
        in_range_period = in_session & (bar_in_block < range_bars)
        if carried_session_ids:
            is_carried = session_id.isin(carried_session_ids)
            in_range_period = in_range_period & ~is_carried
        valid = (session_id > 0) & ~in_range_period

        # Signal-valid: additionally suppress signals during pre-range bars.
        # Pre-range bars belong to the previous session group (valid=True there)
        # but visually they're inside the new range zone — no trades allowed.
        if pre_range_bars > 0:
            signal_valid = valid.copy()
            _start_positions = np.where(session_start.values)[0]
            for pos in _start_positions:
                pre_start = max(0, pos - pre_range_bars)
                if pre_start < pos:
                    signal_valid.iloc[pre_start:pos] = False
        else:
            signal_valid = valid

        or_range = or_high - or_low
        or_midpoint = (or_high + or_low) / 2

        # Suppress signals for sessions where range is below minimum height.
        # Low-volatility ranges don't produce reliable breakout/retest signals.
        # Effective minimum = max(absolute, atr_multiple * ATR).
        if min_range_height > 0 or min_range_height_atr > 0:
            abs_min = min_range_height
            atr_min = min_range_height_atr * atr if min_range_height_atr > 0 else 0
            effective_min = np.maximum(abs_min, atr_min)
            range_too_small = or_range < effective_min
            signal_valid = signal_valid & ~range_too_small

        features[f"{prefix}_range"] = or_range.where(valid, np.nan)
        features[f"{prefix}_position"] = safe_divide(
            df["C"] - or_low, or_range
        ).where(valid, np.nan)

        # Breakout signal: persistent 1 from the first bar closing above/below
        # the range until end of session.  This allows the simulation to enter
        # a trade even when an earlier trade blocked the initial breakout bar
        # (e.g. previous day's trade still open at breakout time).
        # Applies breakout_threshold / breakout_threshold_abs: effective = max(pct*range, abs).
        if breakout_threshold > 0 or breakout_threshold_abs > 0:
            pct_dist = breakout_threshold * or_range
            bo_dist = np.maximum(pct_dist, breakout_threshold_abs)
            above_valid = ((df["C"] > or_high + bo_dist) & signal_valid).astype(np.int8)
            below_valid = ((df["C"] < or_low - bo_dist) & signal_valid).astype(np.int8)
        else:
            above_valid = ((df["C"] > or_high) & signal_valid).astype(np.int8)
            below_valid = ((df["C"] < or_low) & signal_valid).astype(np.int8)
        ever_above = above_valid.groupby(session_id).cummax()
        ever_below = below_valid.groupby(session_id).cummax()
        features[f"{prefix}_breakout_up"] = (
            ever_above.astype(float).where(signal_valid, np.nan)
        )
        features[f"{prefix}_breakout_down"] = (
            ever_below.astype(float).where(signal_valid, np.nan)
        )

        # Breakout distance: how far price is from the range edge,
        # normalized by range size. 0 = at edge, 0.5 = half a range away.
        # For longs: (close - or_high) / range, for shorts: (or_low - close) / range.
        # We take the max of both (whichever side is active).
        dist_above = (df["C"] - or_high).clip(lower=0)
        dist_below = (or_low - df["C"]).clip(lower=0)
        breakout_dist = safe_divide(
            np.maximum(dist_above, dist_below), or_range
        )
        features[f"{prefix}_breakout_dist"] = breakout_dist.where(signal_valid, np.nan)

        features[f"{prefix}_range_vs_atr"] = safe_divide(or_range, atr).where(
            valid, np.nan
        )

        # POC proxy: normalized distance from close to ORB midpoint
        features[f"{prefix}_poc_dist"] = safe_divide(
            df["C"] - or_midpoint, atr
        ).where(valid, np.nan)

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
        close_vals = df["C"].values
        or_high_vals = or_high.values
        or_low_vals = or_low.values
        or_range_vals = or_range.values
        signal_valid_vals = signal_valid.values
        sid_vals = session_id.values

        above_raw, below_raw = apply_breakout_threshold(
            close_vals, or_high_vals, or_low_vals, or_range_vals,
            breakout_threshold, breakout_threshold_abs,
        )
        # Mask out range-period + pre-range bars
        above_masked = above_raw & signal_valid_vals
        below_masked = below_raw & signal_valid_vals

        _, _, post_bull_arr, post_bear_arr, broke_high_arr, broke_low_arr = \
            compute_break_state(
                above_masked, below_masked, or_high_vals,
                sid_vals, len(df),
            )
        features[f"{prefix}_post_bull"] = pd.Series(
            post_bull_arr, index=df.index,
        ).where(signal_valid, np.nan)
        features[f"{prefix}_post_bear"] = pd.Series(
            post_bear_arr, index=df.index,
        ).where(signal_valid, np.nan)

        if enable_retracement:
            # Retest signals per retracement level
            rl_list = [retracement_levels] if isinstance(retracement_levels, (int, float)) else list(retracement_levels)

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
                    n=len(df),
                    min_retracement=min_retracement,
                )
                features[f"{prefix}_{rl_pfx}retest_bull"] = pd.Series(
                    retest_result["retest_bull"], index=df.index,
                ).where(signal_valid, np.nan)
                features[f"{prefix}_{rl_pfx}retest_bear"] = pd.Series(
                    retest_result["retest_bear"], index=df.index,
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
    version = "1.0.0"
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
        rb_list = [range_bars] if isinstance(range_bars, int) else list(range_bars)
        cf_list = [carry_forward_days] if isinstance(carry_forward_days, int) else list(carry_forward_days)
        prb_list = [pre_range_bars] if isinstance(pre_range_bars, int) else list(pre_range_bars)

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
                        df, sessions, rb, atr, enable_retracement,
                        cf, prb, candle_span,
                        breakout_threshold, breakout_threshold_abs,
                        retracement_levels, min_retracement,
                        min_range_height, min_range_height_atr,
                    )
                    features.update({f"{pfx}{k}": v for k, v in session_features.items()})
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
        signal_keys = set(self._signal_columns)
        structural_keys = {
            k for k in features
            if any(k.endswith(s) for s in ORB_STRUCTURAL_SUFFIXES)
        }
        no_shift = signal_keys | structural_keys
        shifted = {k: v for k, v in features.items() if k not in no_shift}
        features_df = shift_features(shifted, df.index)
        for k in no_shift:
            features_df[k] = features[k]
        return pd.concat([df, features_df], axis=1)

    # UTC sessions covered by the orb_scalping pipeline:
    # 0=Nikkei/ASX200 open, 1=Nikkei/HK50 morning, 2=All Asia morning,
    # 5=Nikkei/HK50 afternoon open, 6=DAX pre-market, 7=Xetra/DAX open,
    # 8=London open, 12=NY pre-dawn, 13=NY pre-market, 14=approaching NYSE open
    _PIPELINE_SESSIONS = [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]

    @staticmethod
    def _rl_prefixes(retracement_levels) -> List[str]:
        """Build rl{N}_ prefixes from retracement_levels param."""
        rl_list = [retracement_levels] if isinstance(retracement_levels, (int, float)) else list(retracement_levels)
        return [f"{rl_tag(rl)}_" for rl in rl_list]

    def _resolve_col_params(self, params=None):
        """Merge defaults with optional params and normalise list params.

        Returns (rb_list, cf_val_or_None, prb_val_or_None, include_cf, include_prb,
                 sessions, rl_pfxs) — mirrors the normalisation in compute().
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
        rl_pfxs = self._rl_prefixes(d["retracement_levels"])
        return rb_list, cf_list, prb_list, include_cf, include_prb, sessions, rl_pfxs

    def get_feature_columns(self, params=None) -> List[str]:
        if self._feature_columns and not params:
            return self._feature_columns
        rb_list, cf_list, prb_list, include_cf, include_prb, sessions, rl_pfxs = self._resolve_col_params(params)
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
        rb_list, cf_list, prb_list, include_cf, include_prb, sessions, rl_pfxs = self._resolve_col_params(params)
        signals = []
        for rb in rb_list:
            for cf_val in cf_list:
                for prb_val in prb_list:
                    cf = cf_val if include_cf else None
                    prb = prb_val if include_prb else None
                    for h in sessions:
                        signals.extend([
                            orb_col(rb, cf, prb, h, "breakout_up"),
                            orb_col(rb, cf, prb, h, "breakout_down"),
                        ])
                        for rl_pfx in rl_pfxs:
                            signals.extend([
                                orb_col(rb, cf, prb, h, f"{rl_pfx}retest_bull"),
                                orb_col(rb, cf, prb, h, f"{rl_pfx}retest_bear"),
                            ])
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
                    "0": "00 – ASX / Nikkei Open",
                    "1": "01 – Nikkei / HK50 Morning",
                    "2": "02 – Asia Morning",
                    "5": "05 – HK50 Afternoon",
                    "6": "06 – DAX Pre-Market",
                    "7": "07 – Xetra / DAX Open",
                    "8": "08 – London Open",
                    "9": "09 – London",
                    "12": "12 – NY Pre-Dawn",
                    "13": "13 – NY Pre-Market",
                    "14": "14 – NYSE Open",
                    "15": "15 – NYSE",
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
        }

    def get_column_group_labels(self) -> dict:
        return {
            "s00": "Nikkei / ASX200 Open — 00:00 UTC",
            "s01": "Nikkei / HK50 Morning — 01:00 UTC",
            "s02": "All Asia Morning — 02:00 UTC",
            "s05": "Nikkei / HK50 Afternoon — 05:00 UTC",
            "s06": "DAX Pre-Market — 06:00 UTC",
            "s07": "Xetra / DAX Open — 07:00 UTC",
            "s08": "London Open — 08:00 UTC",
            "s09": "London Morning — 09:00 UTC",
            "s12": "NY Pre-Dawn — 12:00 UTC",
            "s13": "NY Pre-Market — 13:00 UTC",
            "s14": "NYSE Pre-Open — 14:00 UTC",
            "s15": "NYSE Open — 15:00 UTC",
        }


__all__ = ["OpeningRangeIndicator"]
