# Plugin Spec — volatility

**Kind**: indicator  •  **Version**: 3.0.0

## Capability

Computes OHLC volatility features: ATR (raw and %-of-price), Bollinger/Keltner/Donchian %B and bandwidth, Garman-Klass/Parkinson/Yang-Zhang estimators, compression ranks, and RV-vs-VIX.

## Summary

Consolidates classical range/band volatility measures (ATR, Bollinger, Keltner, Donchian) with statistically efficient OHLC estimators (Garman-Klass, Parkinson, Yang-Zhang), adds percentile-ranked compression detection, and — when a macro_vix column is present — realized-vs-implied volatility comparisons. All outputs are shifted by 1 bar for no-lookahead safety.

## Inputs

- df['O'] — open price series
- df['H'] — high price series
- df['L'] — low price series
- df['C'] — close price series
- df['macro_vix'] — optional VIX series; when present enables vol_rv_iv_spread and vol_rv_iv_ratio features

## Parameters

- `atr_periods` (list[int], default=[7, 14, 21]): Periods for ATR calculation, emitted as percentage of price (vol_atr_pct_{p}). Shorter periods capture recent volatility spikes; longer periods yield a smoother baseline.
- `bb_period` (int, default=20): Lookback period for Bollinger Bands SMA and standard-deviation envelope; also drives Bollinger %B and bandwidth (squeeze) features.
- `vol_est_windows` (list[int], default=[20, 50]): Rolling window sizes for Garman-Klass, Parkinson, and Yang-Zhang OHLC volatility estimators. Shorter windows react faster to regime changes; longer windows produce more stable estimates.
- `compression_lookback` (int, default=100): Lookback in bars for rolling percentile-rank computation used by vol_atr_pct_*_rank, vol_bb_wband_*_rank, and the vol_compression flag. Consumed via **params — not declared in get_default_params/get_param_schema.
- `rv_window` (int, default=20): Base window (in days) for realized-volatility calculation; the effective rolling window is rv_window*24 bars, then annualized by sqrt(252*24)*100. Consumed via **params — not declared in get_default_params/get_param_schema.

## Outputs

- vol_atr — raw ATR(14)
- vol_atr_pct_{p} for each p in atr_periods — ATR as fraction of close
- vol_bb_pband_{bb_period}, vol_bb_wband_{bb_period} — Bollinger %B and bandwidth
- vol_kc_pband, vol_kc_wband — Keltner Channel %B and bandwidth
- vol_dc_pband, vol_dc_wband — Donchian Channel %B and bandwidth
- vol_gk_{w}, vol_parkinson_{w}, vol_yz_{w} for each w in vol_est_windows — OHLC volatility estimators
- vol_yz_atr_ratio — Yang-Zhang(20) divided by ATR%(14), emitted only when 20 ∈ vol_est_windows
- vol_atr_pct_{p}_rank for each p in atr_periods — rolling percentile rank over compression_lookback bars
- vol_bb_wband_{bb_period}_rank — rolling percentile rank of BB bandwidth
- vol_compression — 1.0 when both vol_atr_pct_14_rank and vol_bb_wband_{bb_period}_rank are below 0.20, else 0.0
- vol_rv_{rv_window} — annualized realized volatility as percentage
- vol_rv_iv_spread, vol_rv_iv_ratio — only when df['macro_vix'] exists

## Acceptance Criteria

- AC-001: Returns the original DataFrame concatenated with the volatility feature columns listed by get_feature_columns().
- AC-002: All emitted feature columns are shifted by one bar via shift_features() before being returned, preventing lookahead bias.
- AC-003: ATR-percent features are computed as safe_divide(ATR, df['C']) so a zero close cannot raise or return inf.
- AC-004: vol_compression is 1.0 exactly when vol_atr_pct_14_rank < 0.20 AND vol_bb_wband_{bb_period}_rank < 0.20, else 0.0.
- AC-005: vol_rv_iv_spread and vol_rv_iv_ratio are emitted only when df contains a 'macro_vix' column; otherwise they are omitted from the output.
- AC-006: vol_yz_atr_ratio is emitted only when 20 is present in vol_est_windows.
- AC-007: vol_rv_{rv_window} is annualized realized volatility as a percentage, computed from log-returns rolled over rv_window*24 bars and scaled by sqrt(252*24)*100.
- AC-008: Garman-Klass, Parkinson, and Yang-Zhang estimators clip negative variance to zero before taking the square root, so outputs are non-negative.

## Edge Cases

- df has no 'macro_vix' column → vol_rv_iv_spread / vol_rv_iv_ratio are silently omitted; get_feature_columns() still advertises them.
- vol_est_windows does not contain 20 → vol_yz_atr_ratio is silently omitted; get_feature_columns() still advertises it.
- Divisions inside estimators use EPSILON in the denominator, so zero highs/lows/opens/closes do not raise; safe_divide is used for the ATR-percent and ratio features.
- Insufficient history for the rolling windows → early rows are NaN according to each estimator's min_periods (half the window for OHLC estimators, compression_lookback//2 for percentile ranks).
- compression_lookback and rv_window are only accepted via **params — passing them under any other name has no effect and defaults (100, 20) are used silently.
- get_feature_columns() returns a fixed list keyed to default parameters; when atr_periods, bb_period, or vol_est_windows are changed, actually-produced columns diverge from the advertised list.

## Assumptions

- Input df is indexed by time and contains OHLC columns named exactly 'O', 'H', 'L', 'C'.
- Bars are 1-hour resolution — the rv_window*24 factor and sqrt(252*24) annualization assume 24 bars per trading day.
- macro_vix, when present, is already expressed in percentage points comparable to the annualized realized-vol output.

## Needs Clarification

- [NEEDS CLARIFICATION: compression_lookback and rv_window are consumed via **params but do not appear in get_default_params() or get_param_schema(); confirm whether they should be promoted to declared params or intentionally kept undocumented.]
- [NEEDS CLARIFICATION: get_feature_columns() hard-codes the default column set (e.g. vol_atr_pct_7/14/21, vol_gk_20/50); confirm whether it should be derived dynamically from the configured atr_periods / vol_est_windows / bb_period.]
