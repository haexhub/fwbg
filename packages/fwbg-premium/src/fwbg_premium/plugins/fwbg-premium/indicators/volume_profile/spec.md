# Plugin Spec — volume_profile

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes previous-session Volume Profile levels (POC, VAH, VAL) and emits ATR-normalized distance features plus inside-Value-Area, POC relative position, and Value Area width ratio.

## Summary

For each intraday bar, builds a per-session volume histogram (bar-based OHLCV approximation, with TPO fallback when volume is missing) using n_levels price buckets over the session H-L range, extracts POC (highest-volume level) and Value Area High/Low by expanding outward from POC until value_area_pct of total volume is covered, and assigns the PREVIOUS session's profile to bars of the current session. Emits six features: ATR-normalized signed distances from close to POC/VAH/VAL, a binary inside-Value-Area flag, POC relative position within the prior session range, and Value Area width as a fraction of that range. Skips daily-or-slower timeframes (median bar spacing >= 20h) and requires a DatetimeIndex.

## Inputs

- ohlcv: DataFrame with columns H, L, C and optional V, indexed by DatetimeIndex at intraday frequency

## Parameters

- `atr_period` (int, default=14): ATR period used to normalize the POC/VAH/VAL distance features.
- `n_levels` (int, default=50): Number of price buckets in the session volume histogram; higher values give a more precise POC at higher computation cost.
- `value_area_pct` (float, default=0.7): Fraction of total session volume that defines the Value Area, expanded outward from the POC until reached (standard: 70%).

## Outputs

- vp_poc_dist: (close - prev_session_POC) / ATR
- vp_vah_dist: (close - prev_session_VAH) / ATR
- vp_val_dist: (close - prev_session_VAL) / ATR
- vp_inside_va: 1.0 if close is between prev VAL and VAH, 0.0 otherwise, NaN when profile is unavailable
- vp_poc_rel_pos: (POC - session_low) / session_range within the previous session (0=bottom, 1=top)
- vp_va_width_ratio: (VAH - VAL) / session_range of the previous session

## Acceptance Criteria

- AC-001: Given an intraday DataFrame with H, L, C, V and a DatetimeIndex spanning at least two calendar days, compute returns the input DataFrame with the six vp_* feature columns appended.
- AC-002: Bars of the first calendar day in the input have NaN for all vp_* features, since no previous-session profile exists.
- AC-003: From the second calendar day onward, all bars of a given day share the same POC/VAH/VAL values (derived exclusively from the immediately previous session), guaranteeing no lookahead bias.
- AC-004: vp_inside_va equals 1.0 when close is within [VAL, VAH] of the previous session, 0.0 when outside, and NaN when the previous-session profile is not available.
- AC-005: vp_poc_rel_pos and vp_va_width_ratio lie in [0, 1] for well-formed sessions with a non-zero H-L range.
- AC-006: If the median bar spacing is >= 20 hours (daily or slower data), compute returns the input DataFrame unchanged (no vp_* columns added).
- AC-007: If V is missing from the DataFrame or contains no positive values, the profile is built using TPO (1 unit per bar) instead of volume.
- AC-008: get_feature_columns returns exactly ['vp_poc_dist','vp_vah_dist','vp_val_dist','vp_inside_va','vp_poc_rel_pos','vp_va_width_ratio'].
- AC-009: A DataFrame whose index is not a DatetimeIndex raises ValueError.

## Edge Cases

- Session with a single bar (len(day_df) < 2): session is skipped in profile construction, so no profile is stored for that date and the following day's bars keep NaN for vp_* features.
- Session where session_high == session_low (zero range): POC=VAH=VAL are all set to the mid price and session_high/low collapse to that mid; downstream vp_poc_rel_pos and vp_va_width_ratio yield 0 or NaN via safe_divide.
- Bar with H == L (single-price bar): its entire volume is placed at the histogram bucket containing the close, rather than distributed across levels.
- Bar with non-positive volume when use_volume is True: volume is treated as 1.0 for that bar.
- V column absent from df or all V <= 0: use_volume becomes False and the profile falls back to TPO (1 unit per bar).
- Session with total histogram volume of 0 after aggregation: POC/VAH/VAL all collapse to the session mid price.
- Value Area expansion when both up and down neighboring buckets are empty: the expansion loop breaks early, so the Value Area may cover less than value_area_pct of total volume.
- Value Area expansion when lo == 0 and hi < n_levels-1 with dn_vol > up_vol: implementation falls through to expanding upward instead of stopping, since only the lo > 0 branch is guarded.
- Non-intraday data (median bar spacing >= 20h): compute short-circuits and returns df unchanged, without vp_* columns.
- Input DataFrame with no DatetimeIndex: raises ValueError.

## Assumptions

- DataFrame index is a DatetimeIndex; calendar-day boundaries (index.date) define session boundaries.
- OHLCV columns are named H, L, C, and optionally V (uppercase).
- For intraday data the median inter-bar gap is < 20 hours; the plugin treats anything at or above that as daily and no-ops.
- safe_divide handles division by zero by producing NaN or a defined fallback, so zero-range sessions do not raise.

## Needs Clarification

- [NEEDS CLARIFICATION: Should the fall-through branch in the Value Area expansion loop (lo == 0, hi < n_levels-1, dn_vol > up_vol expanding up anyway) be considered intended behavior or a bug worth flagging in invariants?]
- [NEEDS CLARIFICATION: Is calendar-day segmentation (index.date) the intended session definition, or should exchange session boundaries be used for markets that span midnight?]
