# Plugin Spec — opening_range

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Computes session-anchored opening-range breakout features and optional retest signals at configurable retracement levels for one or more intraday session hours.

## Summary

For each configured session hour, defines an opening range as the time window `[open, open + range_bars * bar_step)` of that day's session (bar_step = nominal timeframe or the shorter of the first two observed intervals; bars are labelled by open time, so a missing bar shrinks the range instead of pulling in a later bar; optionally expanded by the `pre_range_bars` bar periods before the open and/or carried forward from prior no-breakout sessions). Range/position/breakout/structural features are live from the end of the range window until `session_end_hour` of the same day (default 24 = local midnight) and NaN afterwards, so a range never reaches into the next day or over a weekend, and breakouts/retests only exist inside the session. Sessions are index-clock hours by default (no DST); with `session_timezone` (e.g. `America/New_York`, `session_minute=30`) the open is computed per day in that timezone and converted to UTC, so it follows daylight saving time (naive index = UTC). Supports absolute and percent breakout thresholds, minimum range-height filters (absolute or ATR-based), and per-retracement-level retest signals with a minimum retracement requirement. Emits column names via `orb_col(rb, cf, prb, session, feature)` with `rb{N}_[cf{N}_][prb{N}_]orb_s{HH}_{feature}` format; `cf`/`prb` segments only appear when configured. Signal columns and structural columns (`_or_high`, `_or_low`, `_or_midpoint`, `_range`, `_sl_dist`) are deliberately not shifted so the exit strategy can consume them at entry_delay=0; all other features are shifted by one bar to avoid lookahead. Like the other unshifted indicators (ibs, zscore, gotobi) the unshifted signal columns are also listed by `get_feature_columns()` and identified through `get_signal_columns()`: they are decision-time columns (signal on the bar's own close, entry at the next open) and must not be used as model features in an ML pipeline, which would see the bar's own close.

## Inputs

- OHLC DataFrame with DatetimeIndex (columns O, H, L, C); intraday timeframes only — daily bars (bar duration >= 20h) are passed through unchanged

## Parameters

- `range_bars` (list[int], default=1): Number of bars defining the opening range (rb). Accepts an int or list of ints — one feature set is emitted per value.
- `atr_period` (int, default=14): ATR window (in bars) used to normalize range and POC distance.
- `sessions` (list[int], default=[0, 1, 2, 5, 6, 7, 8, 12, 13, 14]): Hours (0-23) of the session opens, on the index clock or in `session_timezone`; each produces its own feature set.
- `enable_retracement` (bool, default=True): If True, emit retest_bull/retest_bear/sl_dist columns for each retracement level.
- `carry_forward_days` (list[int], default=0): Number of subsequent sessions to reuse a range whose session had no breakout (cf). 0 disables. Accepts int or list; `cf{N}_` prefix appears only when active.
- `pre_range_bars` (list[int], default=0): Number of bar periods before the session open to fold into the range (prb). 0 disables. Accepts int or list; `prb{N}_` prefix appears only when active. Pre-range bars are also excluded from signal-valid mask.
- `candle_span` (choice, default='hl'): Range extent source: 'hl' uses full candle high/low (including wicks); 'body' uses max/min of open/close only.
- `breakout_threshold` (float, default=0): Minimum breakout distance as fraction of range. Effective threshold = max(pct * range, breakout_threshold_abs). 0 disables.
- `breakout_threshold_abs` (float, default=0): Minimum breakout distance in absolute price units. Combined with breakout_threshold via max(). 0 disables.
- `retracement_levels` (list[float], default=0.5): Retracement fraction(s) of the range for retest entry level(s) (0=boundary, 0.5=midpoint). Each level produces its own rl{tag}_ prefixed columns via rl_tag().
- `min_retracement` (float, default=0): Minimum retracement of the range (checked via H/L after breakout) required before a retest signal may fire. 0 disables.
- `min_range_height` (float, default=0): Minimum opening-range height in absolute price units; sessions with a smaller range have all signals suppressed. Effective minimum = max(absolute, ATR-multiple).
- `min_range_height_atr` (float, default=0): Minimum opening-range height as multiple of ATR; combined with min_range_height via max().
- `session_end_hour` (float, default=24.0): Clock hour (session time, 24 = midnight) at which each session ends; must be after the latest session open. Everything after it, up to the next day's range, is NaN.
- `session_minute` (int, default=0): Minute of the session open (applies to all sessions; e.g. `sessions=[9]`, `session_minute=30`, `session_timezone="America/New_York"` = NYSE cash open). Column names keep the hour only.
- `session_timezone` (str or None, default=None): IANA timezone of the session hours. None = hours are read on the index clock (legacy, no DST handling).

## Outputs

- Per (range_bars, carry_forward_days, pre_range_bars, session_hour) combination: rb{N}_[cf{N}_][prb{N}_]orb_s{HH}_range
- ..._position
- ..._breakout_up
- ..._breakout_down
- ..._breakout_dist — signed distance beyond the range edge in range units: > 0 above the high, < 0 below the low, 0 inside
- ..._range_vs_atr
- ..._poc_dist
- ..._sl_dist
- ..._or_high
- ..._or_low
- ..._or_midpoint
- ..._post_bull
- ..._post_bear
- When enable_retracement=True, per retracement level rl: ..._rl{tag}_retest_bull, ..._rl{tag}_retest_bear, ..._rl{tag}_sl_dist
- Side-effect: self._range_zones populated with per-session opening-range rectangles ({start_ts, end_ts, high, low, session}) for chart visualization

## Acceptance Criteria

- AC-001: On a DataFrame whose bar duration is >= 20 hours (daily data), compute() returns the input DataFrame unchanged (no ORB columns added).
- AC-002: Raises ValueError when the DataFrame index is not a pd.DatetimeIndex.
- AC-003: For each (rb, cf, prb, session_hour) combination, emits all ORB_BASE_FEATURES columns and, when enable_retracement=True, retest_bull/retest_bear/sl_dist columns for each retracement level; column names follow the orb_col() format.
- AC-004: The `cf{N}_` prefix segment appears only when carry_forward_days is a multi-element list or contains a non-zero value; likewise for `prb{N}_` and pre_range_bars.
- AC-005: Non-signal, non-structural feature columns are shifted by one bar via shift_features() (no lookahead); columns ending in ORB_SIGNAL_SUFFIXES (_breakout_up, _breakout_down, _retest_bull, _retest_bear) and ORB_STRUCTURAL_SUFFIXES (_sl_dist, _or_high, _or_low, _or_midpoint, _range) are NOT shifted.
- AC-006: Within the opening-range window itself (`[open, open + range_bars * bar_step)`), non-carried sessions produce NaN for feature values (valid mask excludes the range-formation window); bars outside `[open, session_end)` of their local day are NaN too.
- AC-013: The range is defined by bar timestamps, not row counts: a missing bar inside the window leaves a smaller range and never pulls in later bars.
- AC-014: With `session_timezone`, the session open/end are localised per local day, so the UTC time of the same session shifts with daylight saving time.
- AC-015: The carry-forward "no breakout" test is the signal breakout test (threshold, min range height, post-range bars only).
- AC-007: Pre-range bars are excluded from signal_valid so no breakout/retest/dist signals fire on bars before the session start even though they belong to the previous session group.
- AC-008: When breakout_threshold or breakout_threshold_abs is > 0, breakout signals fire only when close exceeds range boundary by max(pct * range, abs).
- AC-009: Breakout signals (`_breakout_up`/`_breakout_down`) are persistent (cummax within session_id) — once fired they remain 1 until the session ends.
- AC-010: When min_range_height or min_range_height_atr is > 0, sessions whose range is below max(absolute, atr_multiple * ATR) have all signal columns suppressed (NaN) but non-signal columns still compute where valid.
- AC-011: get_feature_columns()/get_signal_columns() with no cached state reproduce the full column list (including `_or_high/_or_low/_or_midpoint`, and no retest columns when `enable_retracement=False`) from the (merged default + supplied) parameters, matching what compute() would emit.
- AC-012: After compute(), self._range_zones is populated with one dict per session opening range (excluding carried sessions and NaN ranges) containing start_ts, end_ts, high, low, session in the expected schema.

## Edge Cases

- Daily or higher timeframe input (bar duration >= 20h) — returns df unchanged.
- Invalid `session_end_hour` (not after a session open, or > 24), unknown `session_timezone`, or `session_minute` outside 0-59 — raises ValueError.
- DataFrame without a DatetimeIndex — raises ValueError.
- sessions=None handling: compute()'s Python signature has sessions=None as its default; when called with sessions=None (e.g. a direct Python call bypassing the plugin system), the code substitutes [8, 9, 14, 15]. When invoked via the plugin system using get_default_params(), sessions is [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]. The same [8, 9, 14, 15] fallback applies inside get_feature_columns() / _resolve_col_params() when sessions is absent from the passed params dict.
- or_range == 0 (doji session where O==C in body mode, or H==L) — sl_dist, or_high/or_low/or_midpoint set to NaN via range_valid guard.
- safe_divide is used for all divisions (position, breakout_dist, range_vs_atr, poc_dist) to avoid ZeroDivisionError.
- carry_forward_days > 0 with a session that never breaks out — its range is reused for up to N subsequent sessions (calendar days that have a session; Friday carries to the next day with bars); those sessions are marked as carried and excluded from range_zones. Without carry, nothing of a session is live after its end (Friday range is NaN on Monday before Monday's own range).
- pre_range_bars extending before start of DataFrame — clamped via max(0, pos - pre_range_bars).
- range_bars extending beyond DataFrame end — zone_end clamped via min(pos + range_bars - 1, len(df) - 1).
- Scalar vs list parameters — range_bars/carry_forward_days/pre_range_bars/retracement_levels accept either; single scalar values do not add cf/prb prefix unless non-zero.
- min_range_height / min_range_height_atr suppress signals but leave structural/statistic features intact.
- ATR is the Wilder ATR (`fwbg.utils.indicator_math.wilder_atr`) and NaN until `atr_period` bars exist. With min_range_height_atr > 0 a bar whose ATR is still unknown cannot meet the floor, so its signal columns are NaN (earlier versions read the warm-up ATR as 0.0 and let the range pass).
- Retracement levels list is normalized to a list even when a single float is passed; rl_tag() drives the prefix (e.g., 0.5 -> rl50).

## Assumptions

- Input DataFrame columns O, H, L, C exist and are numeric.
- Without `session_timezone` DatetimeIndex hours are interpreted on the index clock (data-local hours) with no timezone conversion; with it, a naive index is read as UTC.
- Bars are labelled by their open time.
- fwbg_sdk.retest.apply_breakout_threshold / compute_break_state / compute_retest_signals implement the shared breakout/retest state machine correctly.
- ta.volatility.average_true_range is available and returns raw ATR values aligned to df's index.
