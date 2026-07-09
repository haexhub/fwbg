# Plugin Spec — support_resistance

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects clustered S/R zones on H1 and D1 timeframes from multi-period swings and emits ATR-normalized distance, strength, trend-class, and S/R×trend interaction features.

## Summary

Detects swing highs/lows across multiple periods, clusters them into S/R zones on both an H1 view and a scaled D1 view (via d1_bars multiplier on the same price series), and computes ATR-normalized distances to nearest support/resistance, zone touch-counts (strength), in-zone flags, and flip-zone (support-resistance overlap) flags for each timeframe. Adds a Rayner-Teo trend classifier (-3..+3) from MA20/50/200 alignment plus MA-alignment score, pullback depth, price-vs-MA distances, and a trend-break flag triggered when price violates the last swing structure against the trend. Combines these into interaction features (at-support-in-uptrend, at-resistance-in-downtrend, at-support/resistance-in-range, range-width/position, breakout-up/down, at-flipped-support/resistance). All features are ATR-normalized where applicable and shifted by one bar via shift_features to prevent lookahead.

## Inputs

- H
- L
- C

## Parameters

- `swing_periods` (list[int], default=[5, 10, 20]): Periods used for strict swing-high/low detection; each period p confirms a swing at index i by checking window [i-2p, i] with the candidate at the midpoint.
- `lookback` (int, default=200): How many bars back (per timeframe) to collect swing levels when building the current zone set.
- `cluster_threshold` (float, default=1.5): ATR multiplier used both to cluster nearby swing levels into a single zone and to decide whether a support and resistance overlap into a flip zone.
- `atr_period` (int, default=14): Rolling window (in bars) for the ATR used to normalize distances and set clustering/proximity thresholds.
- `ma_periods` (list[int], default=[20, 50, 200]): Three moving-average periods (short, medium, long) that feed the Rayner-Teo trend classifier and the price_vs_maX features.
- `zone_proximity_atr_mult` (float, default=0.5): ATR multiplier defining how close (in ATRs) price must be to a zone to count as 'in' the zone; 2x this value defines the 'near' band used by interaction features.
- `d1_bars` (int, default=24): Number of H1 bars per D1 bar; used to scale swing_periods, lookback and ATR window to approximate a daily view on the same H1 price series.

## Outputs

- sr_dist_nearest_support
- sr_dist_nearest_resistance
- sr_support_strength
- sr_resistance_strength
- sr_in_support_zone
- sr_in_resistance_zone
- sr_nearest_is_flip_zone
- sr_d1_dist_nearest_support
- sr_d1_dist_nearest_resistance
- sr_d1_support_strength
- sr_d1_resistance_strength
- sr_d1_in_support_zone
- sr_d1_in_resistance_zone
- sr_d1_nearest_is_flip_zone
- sr_trend_class
- sr_pullback_depth
- sr_ma_alignment
- sr_price_vs_ma20
- sr_price_vs_ma50
- sr_price_vs_ma200
- sr_trend_break
- sr_at_support_in_uptrend
- sr_at_resistance_in_downtrend
- sr_at_support_in_range
- sr_at_resistance_in_range
- sr_range_width
- sr_range_position
- sr_breakout_up
- sr_breakout_down
- sr_at_flipped_support
- sr_at_flipped_resistance

## Acceptance Criteria

- AC-001: compute() returns the original DataFrame concatenated with exactly the 31 feature columns listed by get_feature_columns() (7 H1 S/R + 7 D1 S/R + 7 trend + 10 interaction).
- AC-002: All produced feature columns are passed through shift_features so that the value at bar i reflects information available up to bar i-1 (no lookahead).
- AC-003: Support/resistance distance features are ATR-normalized: sr_dist_nearest_support/resistance are expressed in ATR units and are NaN when no zone exists on the correct side of price.
- AC-004: sr_support_strength / sr_resistance_strength equal the touch-count (cluster size) of the nearest respective zone, or 0 if none.
- AC-005: sr_in_support_zone / sr_in_resistance_zone are 1.0 when the nearest zone distance is below zone_proximity_atr_mult (in ATR units), else 0.0.
- AC-006: sr_nearest_is_flip_zone is 1.0 when the nearest support or resistance zone has a counterpart of the opposite type within cluster_threshold*ATR, else 0.0.
- AC-007: D1 features (sr_d1_*) mirror the H1 semantics but are computed with swing_periods scaled by d1_bars, lookback scaled by d1_bars, and ATR windowed over atr_period*d1_bars bars.
- AC-008: sr_trend_class ∈ {-3,-2,-1,0,+1,+2,+3} following Rayner-Teo MA alignment (ma20>ma50>ma200 bullish, reversed bearish, otherwise 0) with the sub-tier depending on close vs ma20/ma50.
- AC-009: sr_ma_alignment ∈ {-1.0, -0.5, 0.0, 0.5, 1.0} computed as (bull_score - bear_score)/2 over the two ordering checks (ma20 vs ma50, ma50 vs ma200).
- AC-010: sr_pullback_depth is the ATR-normalized distance from the running max close (in uptrend) or running min close (in downtrend) since the last trend-sign change, and 0.0 when trend_class is 0.
- AC-011: sr_price_vs_ma{20,50,200} = (close - ma) / atr when atr > EPSILON, else 0.0.
- AC-012: sr_trend_break is -1.0 when in an uptrend and close falls below the last confirmed swing low (using the middle swing_period), +1.0 in the mirror bearish case, else 0.0.
- AC-013: sr_at_support_in_uptrend / sr_at_resistance_in_downtrend / sr_at_support_in_range / sr_at_resistance_in_range are 1.0 only when the nearest zone is within 2*zone_proximity_atr_mult ATR and the trend sign matches, else 0.0.
- AC-014: sr_range_width = dist_support + dist_resistance (in ATRs) and sr_range_position = dist_support / (dist_support + dist_resistance) clipped to [0,1], both NaN when either side has no zone.
- AC-015: sr_breakout_up is 1.0 on the bar where the previous bar was near resistance and the current bar has no resistance above; sr_breakout_down is the mirror for support.
- AC-016: sr_at_flipped_support / sr_at_flipped_resistance are 1.0 when price is within zone_proximity_atr_mult ATR of a former resistance/support respectively (H1 only); D1 does not populate flipped-* features.
- AC-017: get_default_params() returns {swing_periods:[5,10,20], lookback:200, cluster_threshold:1.5, atr_period:14, ma_periods:[20,50,200], zone_proximity_atr_mult:0.5, d1_bars:24}.
- AC-018: When atr <= EPSILON the code substitutes 1.0 to avoid divide-by-zero in distance and price-vs-MA computations.

## Edge Cases

- Very short DataFrame (n < 2 * min(swing_periods)): no swings are detected, all zone features fall back to NaN distances / 0 strengths / 0 flags.
- No valid swings found at all: _cluster_levels is called with np.array([np.nan]), producing no zones, so distance features are NaN and strengths are 0.
- ATR collapses to ~0 (constant price series): code substitutes 1.0 for atr in normalizations and for the recent-ATR clustering threshold, avoiding divide-by-zero.
- All bars produce only supports or only resistances: the missing side yields NaN distance and the range_width/range_position features are NaN for those bars.
- Overlapping support and resistance clusters: _find_zones merges them into a 'both' (flip) zone; per-bar _compute_sr_features re-detects overlap via cluster_threshold*ATR to set sr_nearest_is_flip_zone.
- Trend-sign flip between consecutive bars resets the running swing_high/swing_low used for pullback_depth so depth restarts at 0 after the flip.
- First bar of the series: np.roll wraparound for TR and for prev_in_res/prev_in_sup is explicitly overwritten (tr[0] = H-L, prev_in_*[0] = 0) so no wraparound leakage.
- d1_bars * lookback or d1_bars * atr_period may exceed n; rolling uses min_periods=1 so D1 features degrade gracefully to whatever history is available.

## Assumptions

- The input DataFrame uses uppercase OHLC column names 'H', 'L', 'C' (only these three are read).
- Bars are H1-spaced so that d1_bars=24 is the intended H1-to-D1 scaling; the D1 view is computed on the H1 price series with scaled periods, not on resampled daily bars.
- ma_periods is a length-3 list in ascending order (short, medium, long); trend classification and the sr_price_vs_ma{20,50,200} column names both assume this shape.
- swing_periods has at least one element and its middle element is used for the trend-break swing detection.

## Needs Clarification

- [NEEDS CLARIFICATION: Whether sr_at_flipped_support / sr_at_flipped_resistance are intentionally H1-only (D1 sr_d1_* features skip them) — the code gates flip computation on prefix == 'sr'.]
- [NEEDS CLARIFICATION: Whether the D1 view should ever be computed on true daily-resampled bars rather than scaled H1 windows; current implementation is documented as intentional to avoid rolling-max plateaus but this is a modeling choice worth confirming.]
