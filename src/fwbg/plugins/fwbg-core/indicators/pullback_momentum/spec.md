# Plugin Spec — pullback_momentum

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Emits long/short entries when an EMA-trending market breaks a rolling swing, retraces by a minimum fraction of the impulse, then breaks the local lower-high or higher-low.

## Summary

Three-step trend/pullback/momentum-break entry indicator. Uses an EMA trend filter, a rolling swing-high/low break-of-structure, and a local-high/low break after a minimum Fibonacci-style retracement to fire entries. Manages independent long and short state machines and emits entry flags, structural stop distances (with an ATR buffer), current pullback depth, in-pullback flags, and trend-ok flags.

## Inputs

- H
- L
- C

## Parameters

- `ema_period` (int, default=200): EMA period for the trend filter. Long entries require close > EMA, short entries require close < EMA.
- `swing_lookback` (int, default=20): Rolling window used to detect swing highs/lows; a BOS fires when close breaks the rolling extreme of this window.
- `min_pullback_pct` (float, default=0.382): Minimum pullback depth as a fraction of the impulse range (0.382 = 38.2% Fibonacci retracement).
- `local_high_lookback` (int, default=3): Bars to look back when identifying the local Lower High (long) or Higher Low (short) that price must break to trigger entry.
- `sl_buffer_mult` (float, default=0.25): ATR multiplier added to the structural SL distance as a noise buffer below the pullback low (long) or above the pullback high (short).
- `atr_period` (int, default=14): ATR lookback period used to compute the SL buffer.

## Outputs

- tpm_entry_long
- tpm_entry_short
- tpm_sl_dist_long
- tpm_sl_dist_short
- tpm_pullback_pct
- tpm_in_pullback_long
- tpm_in_pullback_short
- tpm_trend_ok_long
- tpm_trend_ok_short

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame concatenated with all nine feature columns listed in _FEATURES.
- AC-002: All feature columns are shifted by one bar via shift_features before being returned (no lookahead).
- AC-003: tpm_entry_long is 1 on bars where: the long state machine is in PULLBACK, current pullback fraction >= min_pullback_pct, close > EMA, and close breaks the max of highs over the prior local_high_lookback bars; 0 otherwise.
- AC-004: tpm_entry_short is 1 on bars where: the short state machine is in PULLBACK, current pullback fraction >= min_pullback_pct, close < EMA, and close breaks the min of lows over the prior local_high_lookback bars; 0 otherwise.
- AC-005: Long PULLBACK state is entered when close breaks the previous rolling swing_high AND close > EMA; the impulse top is set to that prior swing_high and the impulse base to the prior swing_low.
- AC-006: Short PULLBACK state is entered when close breaks the previous rolling swing_low AND close < EMA; the impulse top is the prior swing_high and the impulse base is the prior swing_low.
- AC-007: A long setup is invalidated (state reset to SCANNING) when close falls below the impulse swing low; a short setup is invalidated when close rises above the impulse swing high.
- AC-008: While waiting for a pullback, a fresh BOS in the same direction refreshes the impulse levels and the running pullback extreme.
- AC-009: tpm_sl_dist_long at an entry bar equals (close - running pullback low) + atr * sl_buffer_mult, floored to atr * 0.5 when non-positive; NaN otherwise. tpm_sl_dist_short is the mirror using (pullback high - close).
- AC-010: tpm_pullback_pct records the current retracement depth (clipped to [0, 2]) on bars where no entry fires while a setup is active; 0 on other bars.
- AC-011: tpm_in_pullback_long / tpm_in_pullback_short are 1 while the corresponding state machine is in PULLBACK and no entry fires on that bar, 0 otherwise (including on the entry bar itself).
- AC-012: tpm_trend_ok_long is 1 iff EMA is valid and close > EMA; tpm_trend_ok_short is 1 iff EMA is valid and close < EMA.
- AC-013: Rolling swing extremes use only past bars (min_periods=1) and the state machine references prev_sh / prev_sl (index i-1), preserving no-lookahead.
- AC-014: get_default_params() returns the six documented parameters with the stated defaults; get_param_schema() exposes min/max/step for each.

## Edge Cases

- Bars where the EMA is still warming up (NaN) produce no entries and set both trend_ok flags to 0.
- ATR values of NaN are coerced to 0; when atr <= EPSILON on a bar, an internal floor of 1e-5 is used so sl_buffer computations remain finite.
- Impulse range <= EPSILON leaves pullback fraction at 0 for that bar, preventing division-by-zero and blocking entry.
- When the computed structural SL distance is non-positive, it is replaced by atr * 0.5 so tpm_sl_dist_* is always positive on entry bars.
- First bar (i=0) is skipped by the loop; all features on that bar retain their initial values (zeros or NaN for sl_dist_*).
- Long and short state machines run independently on the same bar, so both in_pullback_long and in_pullback_short can be active simultaneously.
- A new BOS during an active pullback refreshes the impulse rather than firing a duplicate entry or resetting to SCANNING.
- pullback_pct is clipped to [0, 2] to bound the feature when retracements overshoot the impulse range.

## Assumptions

- Input DataFrame has columns H (high), L (low), C (close) with a monotonically increasing time index.
- shift_features shifts every feature column by one bar to enforce the SDK's no-lookahead invariant before concatenation.
- ta.trend.ema_indicator and ta.volatility.average_true_range are the canonical EMA/ATR implementations used across the SDK.
