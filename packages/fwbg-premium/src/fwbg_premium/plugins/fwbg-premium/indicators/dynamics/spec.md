# Plugin Spec — dynamics

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes momentum/volatility change, lag, and acceleration (2nd-derivative) features over configurable lookback and lag horizons from RSI, ATR, ADX, BB-width, MACD, Stochastic, and price.

## Summary

Derives dynamics features describing how fast base indicators (RSI, ATR%, ADX, BB width, MACD, Stochastic %K) and price change across configurable lookbacks, plus lag (past values) and acceleration (second-derivative) features. Consumes precomputed base indicator columns when present or falls back to computing them from OHLC via `ta`. Applies `shift_features` to prevent lookahead bias.

## Inputs

- df['C']
- df['H']
- df['L']
- optional df['mom_rsi_14']
- optional df['trend_adx_14']
- optional df['vol_atr_pct_14']
- optional df['vol_bb_wband_20']
- optional df['mom_stoch_k_14']
- optional df['trend_macd']

## Parameters

- `lookbacks` (list[int], default=[4, 8, 24]): Lookback horizons (in bars) used for change/percent-change features on RSI, ATR%, BB-width, and ADX.
- `lag_periods` (list[int], default=[4, 8, 24, 48]): Lag horizons (in bars) used for lag features; first three drive RSI/ATR lags, all four drive price-change lags.

## Outputs

- dyn_rsi14_chg_4h
- dyn_rsi14_chg_8h
- dyn_rsi14_chg_24h
- dyn_rsi14_pct_4h
- dyn_rsi14_pct_8h
- dyn_rsi14_pct_24h
- dyn_atr_chg_4h
- dyn_atr_chg_8h
- dyn_atr_chg_24h
- dyn_bbwidth_chg_4h
- dyn_bbwidth_chg_8h
- dyn_bbwidth_chg_24h
- dyn_adx_chg_4h
- dyn_adx_chg_8h
- dyn_adx_chg_24h
- dyn_macd_chg_4h
- dyn_macd_chg_8h
- dyn_stoch_chg_4h
- dyn_stoch_chg_8h
- lag_rsi14_4h
- lag_rsi14_8h
- lag_rsi14_24h
- lag_atr_4h
- lag_atr_8h
- lag_atr_24h
- lag_adx_4h
- lag_adx_8h
- lag_price_chg_4h
- lag_price_chg_8h
- lag_price_chg_24h
- lag_price_chg_48h
- accel_rsi
- accel_atr
- accel_adx
- accel_price

## Acceptance Criteria

- AC-001: Returns the original df concatenated with all 35 feature columns listed in get_feature_columns().
- AC-002: For each lookback L in lookbacks, produces dyn_rsi14_chg_Lh (absolute diff), dyn_rsi14_pct_Lh (safe percent change *100), dyn_atr_chg_Lh, dyn_bbwidth_chg_Lh, and dyn_adx_chg_Lh.
- AC-003: Produces dyn_macd_chg_{4h,8h} and dyn_stoch_chg_{4h,8h} using fixed lookbacks [4, 8] regardless of the lookbacks parameter.
- AC-004: Uses precomputed columns mom_rsi_14, trend_adx_14, vol_atr_pct_14, vol_bb_wband_20, mom_stoch_k_14, trend_macd when present; otherwise computes them from OHLC using ta (RSI14, ADX14, ATR14/close, BB wband window=20, StochK14, MACD diff / close).
- AC-005: Produces lag_rsi14_{Lh} and lag_atr_{Lh} for L in lag_periods[:3] and lag_adx_{4h,8h}; lag_price_chg_{Lh} for every L in lag_periods as safe percent change *100.
- AC-006: accel_rsi, accel_atr, accel_adx, accel_price are second differences computed as (feature - feature.shift(4)) over the corresponding *_chg_4h / lag_price_chg_4h series.
- AC-007: All divisions use safe_divide to avoid inf/NaN blow-ups from zero denominators.
- AC-008: All feature columns are shifted by 1 via shift_features before being concatenated, guaranteeing no lookahead bias.

## Edge Cases

- lookbacks or lag_periods passed as None: defaults [4, 8, 24] and [4, 8, 24, 48] are applied.
- Base indicator columns already present in df: those columns are reused verbatim instead of being recomputed.
- Zero or near-zero denominators in percent-change / lag_price_chg / atr_chg / bbwidth_chg: safe_divide prevents inf/NaN propagation.
- Leading rows shorter than the largest lookback/lag/accel offset: the corresponding shifted values are NaN.
- Feature column names always match get_feature_columns() regardless of the values in lookbacks (which only cover 4/8/24) — passing lookbacks outside {4,8,24} would generate columns not listed in get_feature_columns().

## Assumptions

- Input DataFrame provides OHLC columns named 'H', 'L', 'C' (uppercase).
- The `ta` library is available for fallback indicator computation.
- shift_features shifts every provided feature series by 1 bar and returns a DataFrame aligned to df.index.
- safe_divide returns 0 (or NaN) rather than inf where the denominator is 0.
- get_feature_columns() reflects the default lookbacks/lag_periods; changing those params can produce a different set of output columns.

## Needs Clarification

- [NEEDS CLARIFICATION: Should get_feature_columns() dynamically reflect non-default lookbacks/lag_periods, or is it fixed to the defaults by design?]
- [NEEDS CLARIFICATION: Is passing lookbacks values outside {4, 8, 24} (which would produce columns beyond the fixed get_feature_columns list) considered supported?]
