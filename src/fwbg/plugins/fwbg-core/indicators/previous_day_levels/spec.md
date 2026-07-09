# Plugin Spec — previous_day_levels

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes intraday features (ATR-normalized distances, position, break/retest signals, SL distance, range expansion) relative to previous day's high/low across candle-span, scope and mode variants.

## Summary

Intraday-only indicator that derives previous-day high/low reference levels and emits a family of features per (candle_span × range_scope × break_mode) variant: ATR-normalized distances to PDH/PDL, position within the PDH-PDL range, midpoint distance, range-vs-ATR, first-break detection with post-break state, retest signals and stop-loss distance per retracement level (rl{N}_ prefix), rolling MA of position, and a day-range-expansion flag. Supports session vs 24h scopes, wick vs body ranges, weekend skipping, breakout thresholds (fraction and absolute), resample-based (e.g. hourly) breakout confirmation, and returns the input df unchanged for daily-timeframe data. Features are shifted by one bar via shift_features to avoid lookahead.

## Inputs

- df with DatetimeIndex and OHLC columns O/H/L/C

## Parameters

- `atr_period` (int, default=14): ATR period for normalizing distances.
- `ma_period` (int, default=20): Rolling window for position moving average.
- `enable_retest` (bool, default=True): Enable PDH/PDL retest signals at the configured retracement level(s).
- `retracement_levels` (list[float], default=0.5): Retracement fraction(s) of the PDH-PDL range used as retest entry level; scalar or list. Always produces rl{N}_ prefixed columns.
- `session_start_hour` (int, default=7): UTC hour when trading session starts.
- `session_end_hour` (int, default=21): UTC hour when trading session ends (wraps around midnight when start >= end).
- `candle_span` (choice, default='hl'): Vertical extent of candles used for range: 'hl' (full H/L including wicks) or 'body' (max/min of O/C).
- `range_scope` (list[string], default=['session']): Which bars are included in range computation: 'session' (session hours only, ses_ prefix) or 'all' (24h, all_ prefix). List to precompute multiple.
- `break_modes` (list[string], default=['all_hours']): Break-detection timing modes: 'all_hours' (24/7, no prefix) or 'session_only' (sesbrk_ prefix, only session bars trigger).
- `retest_modes` (list[string], default=['all_hours']): Retest signal timing modes: 'all_hours' (24/7, no prefix) or 'session_only' (sesret_ prefix, session bars only).
- `skip_weekends` (bool, default=True): Skip Saturday/Sunday when computing previous day levels (Monday uses Friday's range). Set False for 24/7 markets.
- `min_sl_atr_mult` (float, default=0): Minimum SL distance as a multiple of ATR; SL becomes max(range-based, min_sl_atr_mult * ATR). 0 = no floor.
- `resample_tf` (string, default=None): Optional resample timeframe (e.g. '1h', '4h') for body-range computation and breakout confirmation using resampled O/C. None = native bars.
- `min_retracement` (float, default=0.3): Minimum fraction of previous-day range (checked via H/L) that price must retrace before a retest signal fires. 0 disables the gate.
- `breakout_threshold` (float, default=0): Minimum distance beyond PDH/PDL as fraction of range for a breakout. 0 disables.
- `breakout_threshold_abs` (float, default=0): Minimum absolute distance beyond PDH/PDL for a breakout; effective threshold is max(pct * range, abs). 0 disables.

## Outputs

- <span>_<scope>_<break>_pdl_high_dist
- <span>_<scope>_<break>_pdl_low_dist
- <span>_<scope>_<break>_pdl_position
- <span>_<scope>_<break>_pdl_range_vs_atr
- <span>_<scope>_<break>_pdl_above_high
- <span>_<scope>_<break>_pdl_below_low
- <span>_<scope>_<break>_pdl_midpoint_dist
- <span>_<scope>_<break>_pdl_high_break
- <span>_<scope>_<break>_pdl_low_break
- <span>_<scope>_<break>_pdl_post_bull
- <span>_<scope>_<break>_pdl_post_bear
- <span>_<scope>_<break>_pdl_range_position_ma
- <span>_<scope>_<break>_pdl_day_range_expanding
- <span>_<scope>_<break>_<retest>_rl{N}_pdl_retest_bull
- <span>_<scope>_<break>_<retest>_rl{N}_pdl_retest_bear
- <span>_<scope>_<break>_<retest>_rl{N}_pdl_sl_dist

## Acceptance Criteria

- AC-001: Returns df unchanged when the median bar interval is >= 20 hours (daily timeframe).
- AC-002: Raises ValueError when df.index is not a DatetimeIndex.
- AC-003: Raises ValueError when range_scope contains a value outside {'session','all'}.
- AC-004: For each (candle_span, range_scope, break_mode) variant, emits the 13 base pdl_* feature columns prefixed with <span>_<scope>_<break_pfx>.
- AC-005: For each retracement level in retracement_levels and each retest_mode, emits pdl_retest_bull, pdl_retest_bear (when enable_retest=True) and pdl_sl_dist columns prefixed with <span>_<scope>_<break_pfx><retest_pfx>rl{N}_.
- AC-006: pdl_high_dist = (PDH - close) / ATR and pdl_low_dist = (close - PDL) / ATR, using a safe ATR floor of 1.0 where ATR <= EPSILON.
- AC-007: pdl_position = (close - PDL) / (PDH - PDL) with NaN where |PDH - PDL| <= EPSILON.
- AC-008: pdl_above_high and pdl_below_low are 0/1 floats indicating close strictly above PDH or below PDL.
- AC-009: pdl_midpoint_dist normalizes distance from close to (PDH+PDL)/2 by ATR.
- AC-010: pdl_high_break / pdl_low_break mark the first bar of each day where the breakout condition holds (with optional session mask when break_mode='session_only').
- AC-011: pdl_post_bull / pdl_post_bear stay set for all bars after the first break within the day.
- AC-012: pdl_sl_dist = max((1 - rl) * pd_range, min_sl_atr_mult * ATR) when min_sl_atr_mult > 0, else (1 - rl) * pd_range.
- AC-013: pdl_range_position_ma is a rolling mean of pdl_position over ma_period bars with min_periods=1.
- AC-014: pdl_day_range_expanding is 1.0 when the current-day range (per candle_span/scope) exceeds the previous-day range.
- AC-015: All feature values are NaN on bars where the previous day's H/L is unavailable (e.g. first day, or after weekend when skip_weekends=True and prior data is missing).
- AC-016: When skip_weekends=True, Saturday/Sunday PDH/PDL are forward-filled from the last trading day (so Monday references Friday).
- AC-017: When resample_tf is set, breakout detection is anchored at the last bar of each resampled period (end-of-bar timestamp) to avoid intra-period lookahead.
- AC-018: When resample_tf is set and candle_span='body', the PDH/PDL range is computed from resampled O/C rather than native bars.
- AC-019: Breakout thresholds combine multiplicatively and absolutely via offset = max(breakout_threshold * range, breakout_threshold_abs).
- AC-020: Features are shifted by one bar via shift_features before being concatenated into the returned DataFrame (no lookahead).
- AC-021: get_feature_columns and get_signal_columns return the exact set of columns produced by compute for the given params (via _default_fallback_columns when not yet computed).

## Edge Cases

- Daily or higher-timeframe input (median bar diff >= 20h): compute returns the original df with no columns added.
- First day(s) of data have no previous-day reference: all pdl_* features are NaN on those bars.
- Session that wraps midnight (session_start_hour >= session_end_hour): day grouping uses a (24 - session_start_hour) offset so the trading day aligns to session start.
- skip_weekends=True with Monday bars: PDH/PDL are taken from Friday via ffill of trading days only.
- ATR <= EPSILON: safe_atr falls back to 1.0 to avoid division blow-up in distance features.
- |PDH - PDL| <= EPSILON: pdl_position is NaN for those bars (safe_range = NaN).
- range_scope contains an unknown value: raises ValueError listing invalid entries.
- Multiple retracement_levels supplied: each generates its own rl{N}_-prefixed retest/SL columns.
- enable_retest=False: pdl_retest_bull / pdl_retest_bear are not emitted, but pdl_sl_dist is still emitted per rl.
- resample_tf on a df with a single bar: bar_duration falls back to 0 (no offset), avoiding IndexError.
- break_mode='session_only': breakouts occurring outside session hours are ignored via the session mask.
- retest_mode='session_only': retest signals are gated by the session mask.
- candle_span='body': range uses max/min of (O, C), ignoring wicks.
- breakout_threshold and breakout_threshold_abs both > 0: effective threshold is the elementwise maximum of the two.

## Assumptions

- df is an intraday OHLC frame with columns O, H, L, C and a DatetimeIndex (verified for datetime index; OHLC columns assumed present).
- Session hours are interpreted in the timezone of df.index (typically UTC per param docs).
- shift_features shifts by one bar to enforce the no-lookahead invariant required for indicators.
- compute_break_state, compute_retest_signals, apply_breakout_threshold, EPSILON, and rl_tag are provided by fwbg_sdk / fwbg_sdk.retest and behave as their names suggest.

## Needs Clarification

- [NEEDS CLARIFICATION: Behavior when required OHLC columns (O/H/L/C) are missing is not explicitly handled — presumably raises a KeyError but is not asserted in code.]
- [NEEDS CLARIFICATION: resample_tf with candle_span='hl' is documented as 'no effect on range' but the code still uses resampled O/C for breakout confirmation — confirm this is intentional for hl mode.]
- [NEEDS CLARIFICATION: Interaction of session_start_hour >= session_end_hour with skip_weekends=True (does the offset day grouping keep the Friday->Monday ffill semantics correct?) is not covered by explicit tests in-source.]
