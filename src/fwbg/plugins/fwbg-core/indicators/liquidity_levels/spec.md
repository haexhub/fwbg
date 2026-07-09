# Plugin Spec — liquidity_levels

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects equal-high/equal-low liquidity pools from clustered swing points and flags stop-hunt sweeps where price briefly exceeds the level and reverses.

## Summary

Scans swing highs/lows within a rolling lookback window, groups them into clusters using an ATR-scaled tolerance, and emits per-bar features describing the ATR-normalized distance to the nearest equal-high above and equal-low below the close, the touch counts of those clusters, boolean active flags, and sweep-up/sweep-down indicators when the current bar wicks through a level and closes back on the origin side. Outputs are shifted by one bar to prevent lookahead.

## Inputs

- H
- L
- C

## Parameters

- `swing_lookback` (int, default=10): N-bar lookback for swing high/low detection.
- `tolerance_atr_mult` (float, default=0.2): How close swing highs/lows must be (in ATR units) to count as equal.
- `atr_period` (int, default=14): ATR period for normalizing distances and tolerance.
- `min_touches` (int, default=2): Minimum swing points at similar price to form a liquidity level.
- `lookback_window` (int, default=200): Bars to look back for swing points when building liquidity levels.

## Outputs

- liq_eqh_dist
- liq_eql_dist
- liq_eqh_count
- liq_eql_count
- liq_eqh_active
- liq_eql_active
- liq_sweep_up
- liq_sweep_down

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame with the eight liq_* feature columns appended.
- AC-002: Feature columns are shifted by one bar via shift_features so values at bar i depend only on data up to bar i-1 (no lookahead).
- AC-003: Swing highs/lows are identified as bars whose H (respectively L) equals the max (min) over a symmetric window of size 2*swing_lookback+1 centered on the bar.
- AC-004: Equal-level clusters are formed from swing prices within tolerance = ATR * tolerance_atr_mult and require at least min_touches members.
- AC-005: Only swing points strictly before the current bar and within the last lookback_window bars are considered when building levels for bar i.
- AC-006: liq_eqh_dist is the ATR-normalized distance from close to the nearest equal-high cluster above the close; liq_eql_dist is the ATR-normalized distance from close to the nearest equal-low cluster below the close.
- AC-007: liq_eqh_count / liq_eql_count report the touch count of that nearest cluster; liq_eqh_active / liq_eql_active are 1.0 when such a cluster exists and 0.0 otherwise.
- AC-008: liq_sweep_up is 1.0 when H exceeds the nearest equal-high above but C closes back below it; liq_sweep_down is 1.0 when L pierces the nearest equal-low below but C closes back above it.
- AC-009: ATR is computed as a rolling mean of true range with min_periods=1 and atr_period window, with a floor of 1.0 substituted when ATR falls below EPSILON to avoid division blow-ups.
- AC-010: get_feature_columns() returns exactly the eight liq_* columns; get_signal_columns() returns the four active/sweep flags.

## Edge Cases

- Bars before index swing_lookback*2 receive default values (NaN for distances, 0.0 for counts/flags) because the swing search and level-building loop does not execute for them.
- When no swing highs/lows exist in the lookback window (sh_mask/sl_mask all False), distance stays NaN and count/active stay 0.0 for that bar.
- When qualifying clusters exist but none lie above (resp. below) the current close, no eqh (resp. eql) feature is populated for that bar.
- Empty input to _find_equal_levels returns an empty cluster list, so an empty DataFrame yields the eight columns filled with their default NaN/0.0 values.
- ATR values at or below EPSILON are replaced with 1.0 to prevent divide-by-zero in the tolerance and distance normalization.
- Sweep flags require both the wick condition (H>level or L<level) and the close-back condition (C on the origin side); a full-body breakout past the level does not set the sweep flag.

## Assumptions

- Input DataFrame has columns named 'H', 'L', 'C' with numeric OHLC data.
- df.index is compatible with pd.concat alignment used by shift_features.
- EPSILON imported from fwbg_sdk is a small positive floor used to guard ATR-based divisions.
