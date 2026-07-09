# Plugin Spec — macd

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes MACD line, signal line, histogram and derived sign/distance/flip features (all normalized by close) from configurable fast, slow and signal EMA periods.

## Summary

MACD momentum/trend indicator built on ta.trend.MACD over the close price. Emits six close-normalized features: macd_line (fast EMA − slow EMA), macd_signal (signal-period EMA of MACD), macd_hist (MACD − signal), macd_above_zero (sign of MACD line), macd_dist_zero (absolute MACD normalized by close), and macd_hist_flip (1.0 when histogram sign changes vs. prior bar). All features are shifted by one bar via shift_features to prevent lookahead. Divisions use safe_divide. Signal columns are macd_above_zero and macd_hist_flip; no overlay columns.

## Inputs

- df["C"]: close price series used for all EMA computations and as the normalization denominator

## Parameters

- `fast_period` (int, default=12): Fast EMA period for MACD line calculation.
- `slow_period` (int, default=26): Slow EMA period for MACD line calculation.
- `signal_period` (int, default=9): Signal line EMA period.

## Outputs

- macd_line: (fast EMA − slow EMA) of close, divided by close
- macd_signal: signal-period EMA of MACD line, divided by close
- macd_hist: MACD − signal (macd_diff from ta), divided by close
- macd_above_zero: np.sign of the raw MACD line (+1/0/-1)
- macd_dist_zero: absolute value of raw MACD line, divided by close
- macd_hist_flip: 1.0 when np.sign(macd_hist) differs from its prior-bar sign, else 0.0

## Acceptance Criteria

- AC-001: compute() returns the original df concatenated with the six declared feature columns.
- AC-002: get_feature_columns() returns exactly [macd_hist, macd_signal, macd_line, macd_above_zero, macd_dist_zero, macd_hist_flip].
- AC-003: get_signal_columns() returns [macd_above_zero, macd_hist_flip]; get_overlay_columns() returns [].
- AC-004: MACD/signal/histogram are computed via ta.trend.MACD using the provided fast_period, slow_period and signal_period.
- AC-005: macd_line, macd_signal, macd_hist and macd_dist_zero are normalized by dividing by df['C'] using safe_divide.
- AC-006: macd_above_zero equals np.sign of the raw (pre-normalization) MACD line.
- AC-007: macd_hist_flip equals 1.0 on bars where np.sign(macd_hist) differs from np.sign(macd_hist.shift(1)), else 0.0.
- AC-008: All emitted features are shifted by one bar via shift_features so that row i uses only information from bars ≤ i-1 (no lookahead).
- AC-009: get_default_params() returns {fast_period: 12, slow_period: 26, signal_period: 9}.
- AC-010: get_param_schema() declares int types with the ranges fast_period∈[2,100], slow_period∈[2,200], signal_period∈[2,50], step 1, and non-empty descriptions.
- AC-011: get_column_group_labels() returns {'macd': 'MACD'}.

## Edge Cases

- First slow_period+signal_period-1 bars: ta.trend.MACD produces NaNs for the warm-up region, which propagate through safe_divide and the shift, yielding NaN feature values there.
- macd_hist_flip on the very first bar: np.sign(macd_hist.shift(1)) is NaN, so the (sign != NaN) comparison evaluates to True and casts to 1.0 (before the one-bar shift).
- Rows where df['C'] is zero: safe_divide handles the division without raising, avoiding inf/NaN blowups in the normalized features.
- macd_above_zero uses np.sign, so bars where the raw MACD line is exactly 0 yield 0 (not +1 or -1).
- One-bar shift means the first output row after shifting is NaN for all emitted features regardless of warm-up.

## Assumptions

- Input DataFrame contains a numeric 'C' (close) column.
- ta.trend.MACD is available and behaves per the ta library's documented semantics.
- shift_features applies a uniform one-bar forward shift to every provided feature series to enforce no-lookahead.
