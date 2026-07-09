# Plugin Spec — supply_demand_flip

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects support/resistance zones from swing highs/lows, flips their polarity on breakout, and emits nearest-flip-zone distance, strength, and touch-count features for both bull and bear sides.

## Summary

Finds N-bar swing highs and lows as candidate resistance/support zones. When close breaks a zone by more than an ATR-scaled width, the zone flips polarity (resistance→bull support, support→bear resistance) and is added to a bounded list of active flip zones. For each bar, reports whether a bull/bear flip zone is active, the ATR-normalized distance to the nearest one on each side, the breakout strength recorded at flip time, and the number of subsequent touches. Zones expire after `zone_expiry` bars, and the active set is capped at `max_active_zones`. All feature columns are shifted by one bar to prevent lookahead.

## Inputs

- OHLC columns H, L, C on the input DataFrame

## Parameters

- `swing_lookback` (int, default=10): N-bar lookback for swing high/low detection.
- `zone_atr_width` (float, default=0.3): Zone width as fraction of ATR.
- `atr_period` (int, default=14): ATR period for normalizing distances and zone width.
- `max_active_zones` (int, default=20): Maximum active flip zones to track.
- `zone_expiry` (int, default=200): Bars after which a zone expires.

## Outputs

- sdf_bull_active
- sdf_bear_active
- sdf_bull_dist
- sdf_bear_dist
- sdf_bull_strength
- sdf_bear_strength
- sdf_bull_touches
- sdf_bear_touches

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame concatenated with the eight sdf_* feature columns listed in _FEATURES.
- AC-002: All emitted feature columns are shifted by one bar via shift_features so bar i's features never depend on data from bar i or later.
- AC-003: Swing highs/lows are identified using a symmetric N-bar window (swing_lookback bars on each side) and only in the interior range [lookback, n - lookback).
- AC-004: A resistance zone flips to a bull flip zone when close exceeds level + zone_atr_width * ATR; a support zone flips to a bear flip zone when close falls below level - zone_atr_width * ATR.
- AC-005: Recorded flip strength equals |close - level| / ATR at the flip bar; touches increment whenever |close - level| <= zone_atr_width * ATR while the flip zone survives.
- AC-006: Zones (resistance, support, and flip) older than zone_expiry bars are dropped; the active flip-zone list is truncated to the last max_active_zones after each bar.
- AC-007: For each bar, sdf_bull_dist/sdf_bear_dist report the ATR-normalized distance to the nearest flip zone on the corresponding side (bull below price, bear above price); sdf_bull_active/sdf_bear_active are 1.0 when such a zone exists and 0.0 otherwise.
- AC-008: sdf_bull_strength/sdf_bear_strength/sdf_bull_touches/sdf_bear_touches carry the strength and touch count of that nearest flip zone (NaN for strength/dist when no zone is active, 0 for active and touches).
- AC-009: get_feature_columns() returns the eight sdf_* columns; get_signal_columns() returns [sdf_bull_active, sdf_bear_active].

## Edge Cases

- When ATR is <= EPSILON, current_atr falls back to 1.0 so zone_width and distances remain finite.
- Series shorter than 2 * swing_lookback + 1 yields no swing points, hence no flip zones; active flags stay 0 and dist/strength stay NaN.
- If no bull (or bear) flip zone is nearer than +inf on a bar, that side's active flag is 0.0 and its dist/strength remain NaN while touches remain 0.
- A flip zone is dropped mid-life if price re-crosses it in the opposite direction (bull zone with close < level - zone_width, bear zone with close > level + zone_width).
- flip_zones is truncated to the last max_active_zones entries every bar, so older zones can be evicted even before zone_expiry.
- compute() returns df unchanged only in the (unreachable) case where _compute_sdf_features returns an empty dict; otherwise it always concatenates the eight feature columns.

## Assumptions

- Input DataFrame contains uppercase OHLC columns H, L, C (matches the fwbg convention).
- shift_features applies a one-bar shift to every returned column to enforce no-lookahead.
- EPSILON from fwbg_sdk is a small positive constant used to guard the ATR fallback.
