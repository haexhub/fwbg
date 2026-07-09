# Plugin Spec — supertrend

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes ATR-based Supertrend trend direction (+1/-1), flip events, and the Supertrend price level from OHLC bars.

## Summary

Standard Supertrend indicator built on the ATR of high/low/close. Maintains sticky upper/lower bands and flips direction when close crosses the active band; emits the current direction (+1 up, -1 down), a flip flag (1 on direction change), and the Supertrend price line as an overlay. Features are shifted by one bar via shift_features to prevent lookahead. A sibling class SupertrendMTFIndicator in the same package (registered as "supertrend_mtf") applies the same core to a rolling OHLC aggregation over d1_bars base bars and additionally exposes distance-to-line in ATR units; this spec documents the primary "supertrend" registration only.

## Inputs

- df with columns H (high), L (low), C (close)

## Parameters

- `period` (int, default=14): ATR lookback period. Lower values = more responsive but noisier.
- `multiplier` (float, default=3): ATR multiplier for band width. Higher values = fewer trend flips.

## Outputs

- st_direction (signal, +1/-1)
- st_flip (signal, 1.0 on direction change else 0.0)
- _st_line (overlay, Supertrend price level)

## Acceptance Criteria

- AC-001: compute() returns the original df concatenated with columns st_direction, st_flip, and _st_line.
- AC-002: st_direction takes only the values +1 (uptrend) or -1 (downtrend).
- AC-003: st_flip equals 1.0 on bars where st_direction differs from the previous bar and 0.0 otherwise.
- AC-004: Direction flips from +1 to -1 when close falls below the active lower band, and from -1 to +1 when close rises above the active upper band.
- AC-005: The final upper band is non-increasing while price stays above the previous lower band, and the final lower band is non-decreasing while price stays below the previous upper band (sticky-band behavior).
- AC-006: All emitted feature columns are shifted by one bar via shift_features so bar i's features depend only on data up to and including bar i-1.
- AC-007: get_feature_columns() returns ['st_direction', 'st_flip']; get_signal_columns() returns the same; get_overlay_columns() returns ['_st_line'].
- AC-008: get_default_params() returns {'period': 14, 'multiplier': 3.0} and get_param_schema() declares matching bounds (period 2..500 step 1, multiplier 0.5..20.0 step 0.5).

## Edge Cases

- First bar (i=0): direction initialized to +1 and supertrend_line initialized to 0.0 before the loop starts at i=1.
- Warmup window of `period` bars: ATR from ta.volatility.average_true_range is NaN, which propagates into upper/lower bands and thus into st_line during warmup.
- After shift_features, the first row of every emitted feature is NaN regardless of input length.
- Constant price series: ATR collapses to 0, upper and lower bands coincide with hl2, and direction remains at its initialized value (+1) with no flips.
- Very short input (len(close) < 2): the direction/line loop body does not execute; outputs remain at their initialization values.

## Assumptions

- Input df uses uppercase OHLC column names H, L, C as consumed by compute().
- The ta package's average_true_range implementation is used verbatim (Wilder-style smoothing with the given window).

## Needs Clarification

- [NEEDS CLARIFICATION: The package registers a second indicator 'supertrend_mtf' (SupertrendMTFIndicator) with distinct params (adds d1_bars), outputs (st_d1_direction, st_d1_dist_atr, _st_d1_line), and version 1.1.0 — confirm whether it should be documented as a separate PluginSpec or folded into this one.]
