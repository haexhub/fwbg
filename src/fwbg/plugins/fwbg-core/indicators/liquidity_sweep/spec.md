# Plugin Spec — liquidity_sweep

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects bullish/bearish liquidity sweep (stop-hunt) zones where price wicks beyond a recent swing extreme and closes back inside, and emits ATR-normalised features about active zones.

## Summary

Scans each bar for a wick that pierces the prior rolling swing low (bullish sweep) or swing high (bearish sweep) with a close back on the original side. Each detected sweep creates a zone bounded by the wick extreme and the close; zones expire after `zone_lookback` bars or when price re-sweeps through the wick extreme. Per bar, the indicator exposes ATR-normalised distance to the nearest bull/bear zone midpoint, zone size, in-zone flags, active flags, and a linear recency score (1=just formed, 0=expired). Features are shifted by one bar to prevent lookahead.

## Inputs

- df[H]
- df[L]
- df[C]

## Parameters

- `swing_lookback` (int, default=20): Bars to look back for identifying recent swing highs/lows via a rolling max/min; this level is what a sweep must pierce.
- `zone_lookback` (int, default=50): Maximum bars a detected sweep zone stays active; also used as the denominator for the linear recency score.
- `atr_period` (int, default=14): ATR lookback period used to normalise zone distances and sizes.

## Outputs

- lsw_bull_active
- lsw_bear_active
- lsw_bull_dist
- lsw_bear_dist
- lsw_bull_size
- lsw_bear_size
- lsw_bull_in_zone
- lsw_bear_in_zone
- lsw_bull_recency
- lsw_bear_recency

## Acceptance Criteria

- AC-001: Registers under the name 'liquidity_sweep' via @register_indicator and exposes version '1.0.0'.
- AC-002: compute() returns the input DataFrame concatenated with the 10 feature columns listed in _FEATURES.
- AC-003: Detects a bullish sweep at bar i when lows[i] < swing_low[i-1] and closes[i] > swing_low[i-1]; the resulting zone has zone_bottom = lows[i] (wick) and zone_top = closes[i].
- AC-004: Detects a bearish sweep at bar i when highs[i] > swing_high[i-1] and closes[i] < swing_high[i-1]; the resulting zone has zone_bottom = closes[i] and zone_top = highs[i] (wick).
- AC-005: swing_high and swing_low are the rolling max/min over swing_lookback bars of highs and lows respectively, with min_periods=1, and the value from bar i-1 is used to evaluate the sweep at bar i.
- AC-006: A zone is dropped once its age (i - zone['bar']) reaches zone_lookback, or when a bull zone sees lows[i] < zone_bottom, or when a bear zone sees highs[i] > zone_top.
- AC-007: lsw_bull_active / lsw_bear_active are 1.0 iff there exists an active zone of that type whose midpoint lies on the far side of the current close (d > 0), else 0.0.
- AC-008: lsw_bull_dist and lsw_bear_dist are the ATR-normalised distance from close to the nearest such zone midpoint; unset when no qualifying zone exists (remains NaN).
- AC-009: lsw_bull_size / lsw_bear_size are (zone_top - zone_bottom) / current_atr for the same nearest zone selected for the distance feature.
- AC-010: lsw_bull_in_zone / lsw_bear_in_zone are 1.0 when the current close lies within any active zone of that type, else 0.0.
- AC-011: lsw_bull_recency / lsw_bear_recency equal 1 - min(i - last_bar, zone_lookback) / zone_lookback using the most recent active zone's bar, and are 0.0 when no such zone exists.
- AC-012: ATR normalisation substitutes 1.0 whenever atr[i] <= EPSILON to avoid division blow-ups.
- AC-013: All feature columns are shifted by one bar via shift_features(...) before being returned, preventing lookahead.

## Edge Cases

- First bar (i=0) is skipped by the main loop, so all features at index 0 are their initial values (0.0 for active/in_zone/recency flags, NaN for dist/size) prior to the one-bar shift.
- ATR is near zero (flat market): current_atr falls back to 1.0 so distances and sizes remain finite instead of exploding.
- No qualifying zone on the far side of close: nearest_*_dist stays at np.inf, active flag remains 0.0, and dist/size stay NaN for that bar.
- Simultaneous bullish and bearish sweep on the same bar: both zones are appended and both sides' features are populated independently.
- Zone immediately invalidated by a subsequent wick through its extreme is dropped and no longer contributes to active/dist/size (but still influences recency until age >= zone_lookback).
- Constant / very short input series where no wick ever exceeds the swing extreme: all outputs remain in their initial states (zeros and NaNs) prior to shifting.

## Assumptions

- Input DataFrame contains columns 'H', 'L', 'C' as float-typed OHLC data indexed by time.
- EPSILON, shift_features, BaseIndicator, and register_indicator behave per the fwbg_sdk contract (shift_features shifts features by one bar and aligns to df.index).
- Recency semantics are 'most recent active zone', not 'most recent sweep ever' — once all zones of a side expire or invalidate, recency returns to 0.
