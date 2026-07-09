# Plugin Spec — microstructure

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Derives intrabar candle-microstructure and volume-flow features (wick imbalance, intrabar bias, body ratio, range/ATR, pressure, rolling accumulations, A/D line, CMF) from OHLC(+V) bars.

## Summary

Computes a suite of microstructure/execution-layer features from OHLC(+V) bars: wick and body ratios, intrabar directional bias, ATR-normalized range, a signed pressure score, rolling sums/extremes of these across a configurable window, direction consistency, and (when volume is available) VWAP-weighted pressure, relative volume, an Accumulation/Distribution line with a 50-bar z-score, and Chaikin Money Flow over 10 and 20 bars. Falls back to volume-free proxies when V is missing/zero. All features are shifted by one bar to prevent lookahead.

## Inputs

- OHLC columns O, H, L, C on the input DataFrame
- Optional volume column V (used only when present, non-all-NaN, and has at least one positive value)

## Parameters

- `atr_period` (int, default=14): Rolling window length for the ATR used to normalize bar range in micro_range_over_atr.
- `rolling_window` (int, default=5): Window used for rolling sums (wick imbalance, intrabar bias, pressure), shadow max extremes, direction consistency, VWAP-weighted pressure, and (×4) the relative-volume baseline.

## Outputs

- micro_wick_imbalance
- micro_intrabar_bias
- micro_body_ratio
- micro_range_over_atr
- micro_pressure_score
- micro_wick_imbalance_sum
- micro_intrabar_bias_sum
- micro_pressure_sum
- micro_upper_shadow_max
- micro_lower_shadow_max
- micro_direction_consistency
- micro_vwap_pressure
- micro_relative_volume
- micro_ad_line
- micro_ad_zscore
- micro_cmf_10
- micro_cmf_20

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame concatenated with all 17 feature columns declared in get_feature_columns().
- AC-002: All feature columns are shifted by one bar via shift_features(), so row i contains only information from bars <= i-1.
- AC-003: Divisions that could hit a zero bar range use bar_range with zeros replaced by NaN (or safe_divide), producing NaN rather than inf/errors on doji bars.
- AC-004: micro_wick_imbalance equals (upper_wick - lower_wick) / (H - L); micro_intrabar_bias equals (C - O) / (H - L); micro_body_ratio equals |C - O| / (H - L) (pre-shift).
- AC-005: micro_range_over_atr equals (H - L) divided by a rolling mean of true range over atr_period bars (pre-shift).
- AC-006: micro_pressure_score equals sign(C - O) * body_ratio (pre-shift).
- AC-007: micro_wick_imbalance_sum, micro_intrabar_bias_sum, and micro_pressure_sum are rolling sums of the corresponding per-bar features over rolling_window bars (pre-shift).
- AC-008: micro_upper_shadow_max and micro_lower_shadow_max are rolling maxima of upper_wick / range and lower_wick / range over rolling_window bars (pre-shift).
- AC-009: micro_direction_consistency equals |rolling_mean(sign(C-O), rolling_window)| (pre-shift).
- AC-010: When V is present, non-all-NaN, and has any positive value: micro_vwap_pressure is a rolling volume-weighted sum of pressure_score / rolling volume sum; micro_relative_volume equals V divided by a rolling mean over rolling_window * 4 bars; micro_ad_line is the cumulative sum of CLV * V where CLV = (2C - L - H)/(H - L); micro_ad_zscore is the 50-bar z-score of micro_ad_line; micro_cmf_10 and micro_cmf_20 equal sum(CLV*V, N) / sum(V, N) for N in {10, 20}.
- AC-011: When V is absent or non-usable: micro_vwap_pressure equals micro_pressure_sum / rolling_window; micro_relative_volume equals 1.0; micro_ad_line equals cumulative sum of CLV; micro_ad_zscore is its 50-bar z-score; micro_cmf_{10,20} equal rolling means of CLV over 10 and 20 bars.
- AC-012: get_feature_columns() returns exactly the 17 column names listed and matches the columns compute() actually appends.
- AC-013: get_default_params() returns {'atr_period': 14, 'rolling_window': 5}.
- AC-014: The plugin is registered under the name 'microstructure' via @register_indicator and exposes class attribute name == 'microstructure', version == '2.0.0'.

## Edge Cases

- Doji bars where H == L: bar_range zeros are replaced with NaN so wick_imbalance, intrabar_bias, body_ratio, pressure_score, shadow maxima, and vwap_pressure yield NaN rather than inf on those bars.
- First atr_period-1 bars produce NaN for micro_range_over_atr (rolling ATR not yet warmed up).
- First rolling_window-1 bars produce NaN for all rolling sum/max/mean features; first rolling_window*4-1 bars produce NaN for micro_relative_volume; first 49 bars produce NaN for micro_ad_zscore; first (N-1) bars produce NaN for micro_cmf_N.
- Missing V column, all-NaN V, or V with no positive values triggers the fallback branch (constant micro_relative_volume=1.0, CLV-only A/D line and CMF, pressure-derived micro_vwap_pressure).
- A single-row DataFrame yields NaN for all rolling/ATR/z-score features (the per-bar ratios themselves are still computed unless the row is a doji).
- One-bar lookahead shift means the very first output row is all NaN for every feature column, regardless of parameters.
- safe_divide handles zero denominators in CLV and A/D z-score computations without raising.

## Assumptions

- _none_
