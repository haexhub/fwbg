# Plugin Spec — opening_range

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes session-anchored opening-range breakout features and optional retest signals at configurable retracement levels for one or more intraday session hours.

## Summary

For each configured session hour, defines an opening range from the first `range_bars` bars (optionally expanded by `pre_range_bars` pre-session bars and/or carried forward from prior no-breakout sessions), then forward-fills range/position/breakout/structural features until the next occurrence of that session hour. Supports absolute and percent breakout thresholds, minimum range-height filters (absolute or ATR-based), and per-retracement-level retest signals with a minimum retracement requirement. Emits column names via `orb_col(rb, cf, prb, session, feature)` with `rb{N}_[cf{N}_][prb{N}_]orb_s{HH}_{feature}` format; `cf`/`prb` segments only appear when configured. Signal columns and structural columns (`_or_high`, `_or_low`, `_or_midpoint`, `_range`, `_sl_dist`) are deliberately not shifted so the exit strategy can consume them at entry_delay=0; all other features are shifted by one bar to avoid lookahead.

## Inputs

- OHLC DataFrame with DatetimeIndex (columns O, H, L, C); intraday timeframes only — daily bars (median diff >= 20h) are passed through unchanged

## Parameters

- `range_bars` (list[int], default=1): Number of bars defining the opening range (rb). Accepts an int or list of ints — one feature set is emitted per value.
- `atr_period` (int, default=14): ATR window (in bars) used to normalize range and POC distance.
- `sessions` (list[int], default=[0, 1, 2, 5, 6, 7, 8, 12, 13, 14]): Data-local hours (0-23) for which session-specific ORB features are computed and forward-filled until the next occurrence of that hour.
- `enable_retracement` (bool, default=True): If True, emit retest_bull/retest_bear/sl_dist columns for each retracement level.
- `carry_forward_days` (list[int], default=0): Number of subsequent sessions to reuse a range whose session had no breakout (cf). 0 disables. Accepts int or list; `cf{N}_` prefix appears only when active.
- `pre_range_bars` (list[int], default=0): Number of bars before session start to fold into the range (prb). 0 disables. Accepts int or list; `prb{N}_` prefix appears only when active. Pre-range bars are also excluded from signal-valid mask.
- `candle_span` (choice, default='hl'): Range extent source: 'hl' uses full candle high/low (including wicks); 'body' uses max/min of open/close only.
- `breakout_threshold` (float, default=0): Minimum breakout distance as fraction of range. Effective threshold = max(pct * range, breakout_threshold_abs). 0 disables.
- `breakout_threshold_abs` (float, default=0): Minimum breakout distance in absolute price units. Combined with breakout_threshold via max(). 0 disables.
- `retracement_levels` (list[float], default=0.5): Retracement fraction(s) of the range for retest entry level(s) (0=boundary, 0.5=midpoint). Each level produces its own rl{tag}_ prefixed columns via rl_tag().
- `min_retracement` (float, default=0): Minimum retracement of the range (checked via H/L after breakout) required before a retest signal may fire. 0 disables.
- `min_range_height` (float, default=0): Minimum opening-range height in absolute price units; sessions with a smaller range have all signals suppressed. Effective minimum = max(absolute, ATR-multiple).
- `min_range_height_atr` (float, default=0): Minimum opening-range height as multiple of ATR; combined with min_range_height via max().

## Outputs

- Per (range_bars, carry_forward_days, pre_range_bars, session_hour) combination: rb{N}_[cf{N}_][prb{N}_]orb_s{HH}_range
- ..._position
- ..._breakout_up
- ..._breakout_down
- ..._breakout_dist
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

- AC-001: On a DataFrame whose median index spacing is >= 20 hours (daily data), compute() returns the input DataFrame unchanged (no ORB columns added).
- AC-002: Raises ValueError when the DataFrame index is not a pd.DatetimeIndex.
- AC-003: For each (rb, cf, prb, session_hour) combination, emits all ORB_BASE_FEATURES columns and, when enable_retracement=True, retest_bull/retest_bear/sl_dist columns for each retracement level; column names follow the orb_col() format.
- AC-004: The `cf{N}_` prefix segment appears only when carry_forward_days is a multi-element list or contains a non-zero value; likewise for `prb{N}_` and pre_range_bars.
- AC-005: Non-signal, non-structural feature columns are shifted by one bar via shift_features() (no lookahead); columns ending in ORB_SIGNAL_SUFFIXES (_breakout_up, _breakout_down, _retest_bull, _retest_bear) and ORB_STRUCTURAL_SUFFIXES (_sl_dist, _or_high, _or_low, _or_midpoint, _range) are NOT shifted.
- AC-006: Within the opening-range period itself (bars 0..range_bars-1 of a session), non-carried sessions produce NaN for feature values (valid mask excludes the range-formation window).
- AC-007: Pre-range bars are excluded from signal_valid so no breakout/retest/dist signals fire on bars before the session start even though they belong to the previous session group.
- AC-008: When breakout_threshold or breakout_threshold_abs is > 0, breakout signals fire only when close exceeds range boundary by max(pct * range, abs).
- AC-009: Breakout signals (`_breakout_up`/`_breakout_down`) are persistent (cummax within session_id) — once fired they remain 1 until the session ends.
- AC-010: When min_range_height or min_range_height_atr is > 0, sessions whose range is below max(absolute, atr_multiple * ATR) have all signal columns suppressed (NaN) but non-signal columns still compute where valid.
- AC-011: get_feature_columns()/get_signal_columns() with no cached state reproduce the full column list from the (merged default + supplied) parameters, matching what compute() would emit.
- AC-012: After compute(), self._range_zones is populated with one dict per session opening range (excluding carried sessions and NaN ranges) containing start_ts, end_ts, high, low, session in the expected schema.

## Edge Cases

- Daily or higher timeframe input (median diff >= 20h) — returns df unchanged.
- DataFrame without a DatetimeIndex — raises ValueError.
- Empty session list defaulting: when sessions=None, defaults to [8, 9, 14, 15] inside _session_orb_features (compute() also defaults to [8,9,14,15] when sessions is None).
- or_range == 0 (doji session where O==C in body mode, or H==L) — sl_dist, or_high/or_low/or_midpoint set to NaN via range_valid guard.
- safe_divide is used for all divisions (position, breakout_dist, range_vs_atr, poc_dist) to avoid ZeroDivisionError.
- carry_forward_days > 0 with a session that never breaks out — its range is reused for up to N subsequent sessions; those sessions are marked as carried and excluded from range_zones.
- pre_range_bars extending before start of DataFrame — clamped via max(0, pos - pre_range_bars).
- range_bars extending beyond DataFrame end — zone_end clamped via min(pos + range_bars - 1, len(df) - 1).
- Scalar vs list parameters — range_bars/carry_forward_days/pre_range_bars/retracement_levels accept either; single scalar values do not add cf/prb prefix unless non-zero.
- min_range_height / min_range_height_atr suppress signals but leave structural/statistic features intact.
- Retracement levels list is normalized to a list even when a single float is passed; rl_tag() drives the prefix (e.g., 0.5 -> rl50).

## Assumptions

- Input DataFrame columns O, H, L, C exist and are numeric.
- DatetimeIndex hours are interpreted in the data's own timezone (data-local hours) — the plugin does not convert timezones.
- fwbg_sdk.retest.apply_breakout_threshold / compute_break_state / compute_retest_signals implement the shared breakout/retest state machine correctly.
- ta.volatility.average_true_range is available and returns raw ATR values aligned to df's index.
