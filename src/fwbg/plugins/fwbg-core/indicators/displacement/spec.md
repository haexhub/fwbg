# Plugin Spec — displacement

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes breakout/displacement quality features (body ratio, body/ATR, wick ratios, FVG formation, consecutive same-direction candles, range expansion, and close position) from OHLC bars.

## Summary

Derives eight per-bar price-action features that quantify candle conviction and breakout quality: body-to-range ratio, body-to-ATR impulse magnitude, upper/lower wick ratios, current-bar FVG formation flag, signed run-length of same-direction candles, current range vs. rolling average range, and close position within the bar. All feature columns are shifted by one bar via shift_features to avoid lookahead.

## Inputs

- df with OHLC columns 'O', 'H', 'L', 'C'

## Parameters

- `atr_period` (int, default=14): ATR period for normalizing body size (body / ATR impulse magnitude).
- `range_avg_period` (int, default=20): Rolling window length for the average candle range used in range-expansion computation.

## Outputs

- disp_body_ratio
- disp_body_atr
- disp_upper_wick_ratio
- disp_lower_wick_ratio
- disp_fvg_formed
- disp_consecutive_dir
- disp_range_expansion
- disp_close_position

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame concatenated with all eight disp_* feature columns listed in get_feature_columns().
- AC-002: All feature columns are shifted by one bar (via shift_features) so values at bar i depend only on data up to and including bar i-1, eliminating lookahead.
- AC-003: disp_body_ratio equals |C - O| / (H - L) for bars where range > EPSILON, else NaN.
- AC-004: disp_body_atr equals |C - O| divided by a rolling mean of true range over atr_period bars (min_periods=1); when ATR <= EPSILON the divisor falls back to 1.0.
- AC-005: disp_upper_wick_ratio equals (H - max(O, C)) / (H - L) and disp_lower_wick_ratio equals (min(O, C) - L) / (H - L) for valid ranges.
- AC-006: disp_fvg_formed is 1.0 at bar i when H[i-2] < L[i] (bullish gap) or L[i-2] > H[i] (bearish gap), else 0.0; the first two bars are always 0.0.
- AC-007: disp_consecutive_dir accumulates signed sign(C - O) while the direction matches the previous bar, resets to the current signed direction on direction change, and resets to 0 on a doji (C == O).
- AC-008: disp_range_expansion equals current (H - L) divided by a rolling mean of (H - L) over range_avg_period bars (min_periods=1); avg <= EPSILON falls back to 1.0.
- AC-009: disp_close_position equals (C - L) / (H - L) for valid ranges (1 = close at high, 0 = close at low).
- AC-010: get_signal_columns() returns ['disp_fvg_formed'].
- AC-011: get_default_params() returns {'atr_period': 14, 'range_avg_period': 20} and get_param_schema() declares int bounds atr_period in [2, 100] and range_avg_period in [5, 100].

## Edge Cases

- Bars with H == L (zero range) produce NaN for the range-normalized features (body_ratio, wick ratios, close_position) via the safe_range mask.
- The first bar's previous close is set to its own close, so its true range collapses to (H - L); ATR is well-defined from bar 0 due to min_periods=1.
- For n < 2 bars, disp_fvg_formed is all zeros (the loop starts at i=2).
- Doji bars (C == O) reset disp_consecutive_dir to 0 and do not extend a run.
- When rolling ATR or average range is <= EPSILON, the divisor is replaced by 1.0 to avoid division blow-ups.
- shift_features shifts all outputs by one bar, so the first row of every disp_* column is NaN after compute().

## Assumptions

- Input DataFrame contains uppercase OHLC columns named 'O', 'H', 'L', 'C'.
- fwbg_sdk.shift_features shifts feature columns by exactly one bar and aligns them to df.index.
- EPSILON is a small positive constant used consistently for safe-divide guards.
