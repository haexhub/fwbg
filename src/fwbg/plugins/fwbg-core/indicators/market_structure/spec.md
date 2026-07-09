# Plugin Spec — market_structure

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects Break-of-Structure (BOS) and Change-of-Character (CHOCH) events from rolling swing highs/lows and emits trend state plus ATR-normalised distance features.

## Summary

ICT/SMC market-structure indicator that flags bullish/bearish BOS when close breaks the prior rolling swing high/low, marks CHOCH as the first opposite-direction BOS after a trend, maintains a +1/0/-1 trend state, and outputs ATR-normalised distances to the last BOS levels and current swing extremes, plus a linear CHOCH recency score.

## Inputs

- df[H]
- df[L]
- df[C]

## Parameters

- `swing_lookback` (int, default=20): Rolling window (bars) used to compute the swing high/low whose break triggers a BOS.
- `choch_lookback` (int, default=50): Number of bars over which the CHOCH recency feature decays linearly from 1 to 0.
- `atr_period` (int, default=14): ATR lookback period used to normalise the BOS-level and swing-extreme distance features.

## Outputs

- ms_bos_bull
- ms_bos_bear
- ms_choch_bull
- ms_choch_bear
- ms_trend
- ms_bull_bos_dist
- ms_bear_bos_dist
- ms_swing_high_dist
- ms_swing_low_dist
- ms_choch_recency

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame concatenated with exactly the 10 feature columns listed in _FEATURES.
- AC-002: ms_bos_bull is 1.0 on bars where close exceeds the prior-bar rolling swing high over swing_lookback, else 0.0.
- AC-003: ms_bos_bear is 1.0 on bars where close is below the prior-bar rolling swing low over swing_lookback, else 0.0.
- AC-004: ms_choch_bull is 1.0 only on a bullish BOS bar that occurs while current_trend is negative; ms_choch_bear is 1.0 only on a bearish BOS bar while current_trend is positive.
- AC-005: ms_trend is +1 after any bullish BOS, -1 after any bearish BOS, and 0 until the first BOS occurs.
- AC-006: ms_bull_bos_dist equals (close - last_bull_bos_level) / ATR and ms_bear_bos_dist equals (last_bear_bos_level - close) / ATR, both NaN until the first respective BOS has occurred.
- AC-007: ms_swing_high_dist and ms_swing_low_dist are non-negative ATR-normalised distances (clamped at 0) to the prior-bar rolling swing high/low.
- AC-008: ms_choch_recency is 0 before the first CHOCH, then decays linearly from 1.0 at the CHOCH bar to 0.0 at choch_lookback bars later and stays at 0 thereafter.
- AC-009: All feature columns are shifted by one bar via shift_features before being returned, so bar i's features depend only on data up to bar i-1.
- AC-010: get_feature_columns() returns the 10 _FEATURES; get_signal_columns() returns ['ms_bos_bull', 'ms_bos_bear', 'ms_choch_bull', 'ms_choch_bear'].
- AC-011: get_default_params() returns {'swing_lookback': 20, 'choch_lookback': 50, 'atr_period': 14}.

## Edge Cases

- First bar (i=0): all signal/trend columns are 0 and distance columns are NaN because the loop starts at i=1.
- ATR value at bar i is replaced by 1.0 whenever it is <= EPSILON, avoiding divide-by-zero in distance normalisation.
- Before any bullish (bearish) BOS has occurred, last_bull_bos_level (last_bear_bos_level) is NaN, so ms_bull_bos_dist (ms_bear_bos_dist) stays NaN.
- When the close equals the prior swing high or low exactly (no strict break), no BOS is emitted (comparisons are strict > and <).
- If a bullish and bearish BOS condition both hold on the same bar, both bos flags are set and current_trend ends at -1 because the bearish branch runs second.
- True range on bar 0 is set to high[0] - low[0] (np.roll wrap is overwritten) so ATR is defined from the first bar.
- Rolling swing_high/swing_low use min_periods=1, so early bars use whatever history is available rather than producing NaN.
- CHOCH recency saturates: for ages beyond choch_lookback the min() clamp keeps recency at exactly 0.

## Assumptions

- _none_
