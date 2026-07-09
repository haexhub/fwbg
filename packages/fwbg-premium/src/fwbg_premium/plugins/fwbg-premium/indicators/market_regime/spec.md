# Plugin Spec — market_regime

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes a composite risk-on/risk-off regime score from VIX, credit-spread (HYG/LQD), equity (SPX) momentum, and treasury (TLT) flight macro series, plus binary risk-on/off flags.

## Summary

Combines up to four macro components — inverted VIX z-score, HYG/LQD credit-spread z-score, SPX 20-day percentage change, and inverted TLT 10-day percentage change — into an averaged composite z-score, and emits binary risk_on (>0.5) and risk_off (<-0.5) flags. All emitted features are shifted by one bar to avoid lookahead.

## Inputs

- macro_vix (optional)
- macro_hyg (optional, paired with macro_lqd)
- macro_lqd (optional, paired with macro_hyg)
- macro_spx (optional)
- macro_tlt (optional)

## Parameters

- `window` (int, default=50): Rolling window length in days used for the VIX and credit-spread z-score mean/std (scaled by bars_per_day internally). Also acts as min_periods for the rolling stats.
- `bars_per_day` (int, default=24): Number of bars per trading day; used to convert the day-based window and momentum lookbacks (20d for SPX, 10d for TLT) into bar counts.

## Outputs

- regime_vix_zscore
- regime_credit_zscore
- regime_equity_momentum
- regime_treasury_flight
- regime_risk_composite
- regime_risk_on
- regime_risk_off

## Acceptance Criteria

- AC-001: When macro_vix is present, regime_vix_zscore = -(vix - rolling_mean) / rolling_std over a window of window*bars_per_day bars (min_periods=window), with std floored at 1e-6.
- AC-002: When macro_hyg and macro_lqd are both present, regime_credit_zscore is the rolling z-score of the HYG/(LQD+1e-10) ratio over window*bars_per_day bars (min_periods=window), std floored at 1e-6.
- AC-003: When macro_spx is present, regime_equity_momentum = pct_change over 20*bars_per_day bars, expressed as a percentage (multiplied by 100).
- AC-004: When macro_tlt is present, regime_treasury_flight = -pct_change over 10*bars_per_day bars, expressed as a percentage (inverted so that rising TLT is risk-off).
- AC-005: regime_risk_composite is the row-wise mean of the available components whose feature name ends in 'zscore' (i.e., the VIX and credit z-scores only).
- AC-006: regime_risk_on is 1.0 where regime_risk_composite > 0.5 and 0.0 otherwise (float dtype).
- AC-007: regime_risk_off is 1.0 where regime_risk_composite < -0.5 and 0.0 otherwise (float dtype).
- AC-008: All emitted feature columns are shifted by one bar via shift_features before being concatenated back onto the input DataFrame, ensuring no lookahead.
- AC-009: get_feature_columns() returns the fixed list of seven regime_* column names regardless of which macro inputs are available.
- AC-010: validate() returns True unconditionally.

## Edge Cases

- None of macro_vix, macro_hyg/macro_lqd, macro_spx, macro_tlt columns present → compute() returns the original DataFrame unchanged (no features added, no composite/flag columns).
- Only momentum inputs present (macro_spx and/or macro_tlt, but neither VIX nor HYG+LQD) → no z-score components exist, so regime_risk_composite, regime_risk_on, and regime_risk_off are not emitted even though momentum columns are.
- Only one of macro_hyg or macro_lqd is present → the credit-spread component is skipped entirely.
- Rolling std of VIX or credit ratio is zero → clipped to 1e-6 to prevent divide-by-zero in the z-score.
- macro_lqd contains zero values → denominator is offset by +1e-10 in the HYG/LQD ratio to avoid division by zero.
- Fewer than `window` observations of VIX or the credit ratio → z-score is NaN until min_periods=window is reached.
- Fewer than 20*bars_per_day (SPX) or 10*bars_per_day (TLT) bars of history → the corresponding momentum feature is NaN at the head of the series.

## Assumptions

- Macro columns, when present, are aligned to the same bar index as df and are numeric.
- bars_per_day accurately reflects the bar frequency of df so that day-based windows/lookbacks are meaningfully scaled.
- The one-bar shift applied by shift_features is sufficient to eliminate lookahead for the macro series used.

## Needs Clarification

- [NEEDS CLARIFICATION: Contract-level min/max/step for `window` and `bars_per_day` are not encoded in the source (no get_param_schema); acceptable ranges are unspecified.]
