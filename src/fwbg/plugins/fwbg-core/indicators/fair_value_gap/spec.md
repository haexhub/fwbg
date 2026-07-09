# Plugin Spec — fair_value_gap

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects 3-candle Fair Value Gap imbalance zones and emits ATR-normalized features for active gaps, nearest-gap distance/size, in-gap membership, active count, and engulfing confirmation.

## Summary

Scans the OHLC series for Smart Money Concepts Fair Value Gaps — bullish when H[i-2] < L[i] and bearish when L[i-2] > H[i]. Maintains a rolling set of active (unfilled, within-lookback) gaps: a bullish gap is filled when a subsequent low breaks below its bottom, a bearish gap when a subsequent high breaks above its top. An active gap is flagged "confirmed" once an engulfing candle of the matching direction reaches into the gap. Per bar, emits ten features derived from the nearest active gap on each side (ATR-normalized distance and size), whether the current close sits inside any active gap, the count of active gaps, and confirmation flags. All features are shifted by one bar via shift_features to prevent lookahead bias.

## Inputs

- df with OHLC columns O, H, L, C
- param atr_period (int) — ATR window for normalizing distances/sizes
- param lookback (int) — max bar age before an unfilled FVG is discarded

## Parameters

- `atr_period` (int, default=14): ATR lookback period used to normalize FVG distances and gap sizes into ATR units for scale-independence across price levels and instruments.
- `lookback` (int, default=100): Maximum number of bars an unfilled FVG remains active; older gaps are discarded from the active set.

## Outputs

- fvg_bull_active
- fvg_bear_active
- fvg_bull_dist
- fvg_bear_dist
- fvg_bull_size
- fvg_bear_size
- fvg_in_gap
- fvg_count
- fvg_bull_confirmed
- fvg_bear_confirmed

## Acceptance Criteria

- AC-001: Bullish FVG is recorded at bar i when H[i-2] < L[i]; bearish FVG at bar i when L[i-2] > H[i].
- AC-002: A bullish active FVG is removed once lows[i] <= fvg.bottom; a bearish active FVG is removed once highs[i] >= fvg.top.
- AC-003: An active FVG whose age (i - creation_bar) exceeds the lookback parameter is dropped from the active set.
- AC-004: fvg_bull_active / fvg_bear_active are 1.0 on bars where at least one active gap of that direction lies ahead of the current close (positive ATR-normalized distance to gap midpoint), else 0.0.
- AC-005: fvg_bull_dist / fvg_bear_dist report the ATR-normalized distance from close to the midpoint of the nearest active gap on that side; NaN when no such gap exists.
- AC-006: fvg_bull_size / fvg_bear_size report (top - bottom) / ATR of the same nearest active gap; NaN when no such gap exists.
- AC-007: fvg_in_gap is 1.0 on bars where close lies within [bottom, top] of any active FVG (bullish or bearish), else 0.0.
- AC-008: fvg_count equals the number of active FVGs on that bar after fill/lookback pruning and inclusion of newly detected gaps.
- AC-009: fvg_bull_confirmed / fvg_bear_confirmed flip to 1.0 on a bar when an engulfing candle of the matching direction reaches into an active gap of that type; remain 1.0 for as long as that confirmed gap stays active.
- AC-010: All returned feature columns are shifted by one bar via shift_features so that row i's features depend only on data through bar i-1.
- AC-011: ATR is computed from true range with a rolling mean of window atr_period (min_periods=1); when ATR <= EPSILON the denominator falls back to 1.0.
- AC-012: get_feature_columns returns exactly the ten _FEATURES names in order; get_signal_columns returns fvg_bull_active, fvg_bear_active, fvg_in_gap, fvg_bull_confirmed, fvg_bear_confirmed.

## Edge Cases

- DataFrames with fewer than 3 rows produce no FVG detections; features are all zeros/NaN as initialized and then shifted.
- First bar's true range is set to H[0]-L[0] (np.roll wrap is overwritten) so ATR is well-defined from bar 0.
- ATR near zero is clamped to 1.0 via the EPSILON guard, preventing division by zero in distance/size normalization.
- When no active bullish (or bearish) gap lies ahead of close, the corresponding *_active is 0, *_dist and *_size are NaN.
- A gap that is both filled and confirmed on the same bar is still removed by the fill check before feature computation.
- Distance uses (close - mid) for bullish and (mid - close) for bearish and only considers gaps with positive distance, so a gap the price has moved past on its own side is not reported as 'nearest'.
- in_gap can be set by either a bullish or bearish active gap containing the close; fvg_count includes both directions.
- The one-bar shift by shift_features means the very first row of features is NaN/undefined and the last computed bar's values are dropped from the shifted frame.

## Assumptions

- Input df has uppercase OHLC columns O, H, L, C as used by compute().
- shift_features shifts every feature series by one bar to enforce no-lookahead and returns a DataFrame aligned to df.index.
- EPSILON is a small positive constant used only to guard the ATR denominator.
