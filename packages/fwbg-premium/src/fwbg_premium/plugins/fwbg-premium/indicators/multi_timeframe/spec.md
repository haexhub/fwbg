# Plugin Spec — multi_timeframe

**Kind**: indicator  •  **Version**: 3.0.0

## Capability

Computes higher-timeframe (H4, D1, W1, Y1) trend, range, momentum, volatility, alignment, and support/resistance features from H1 OHLC bars.

## Summary

Aggregates H1 OHLC data into synthetic H4/D1/W1/Y1 timeframes via rolling windows and emits ~30 features covering EMA distances, range positions, ADX/RSI/ATR/Bollinger %B on H4, trend strength on D1/W1, 52-week range and 200d EMA distance for Y1, sign-based trend alignment scores across timeframes, H1/H4 volatility ratio, RSI divergence, and distance/breakout flags relative to the previous D1 high/low. All feature columns are shifted by one bar via `shift_features` to prevent lookahead bias.

## Inputs

- df with H1 OHLC columns: O (open), H (high), L (low), C (close)

## Parameters

- `h4_bars` (int, default=4): Number of H1 bars aggregated into one H4 candle (used for H4 rolling highs/lows and EMA window scaling).
- `d1_bars` (int, default=24): Number of H1 bars aggregated into one D1 candle (used for D1 rolling windows, EMA scaling, prev-day levels, and 200d EMA).
- `w1_bars` (int, default=120): Number of H1 bars aggregated into one W1 candle (default 5*24=120; used for W1 rolling windows, EMA scaling, and 52-week range).
- `ema_periods` (list[int], default=[20, 50]): EMA periods (in native timeframe units) used to emit mtf_{h4,d1,w1}_ema{period}_dist features on each higher timeframe.
- `include_yearly` (bool, default=True): If True, additionally compute Y1 features: 200-day EMA distance, 52-week range position, and 52-week high/low distances.

## Outputs

- mtf_h4_trend
- mtf_h4_range_pos
- mtf_h4_ema20_dist
- mtf_h4_ema50_dist
- mtf_h4_adx
- mtf_h4_rsi
- mtf_h4_atr_pct
- mtf_h4_bb_pband
- mtf_d1_range_pos
- mtf_d1_ema20_dist
- mtf_d1_ema50_dist
- mtf_d1_trend_strength
- mtf_w1_range_pos
- mtf_w1_ema20_dist
- mtf_w1_ema50_dist
- mtf_w1_trend_strength
- mtf_y1_ema200d_dist (only when include_yearly=True)
- mtf_y1_52w_range_pos (only when include_yearly=True)
- mtf_y1_52w_high_dist (only when include_yearly=True)
- mtf_y1_52w_low_dist (only when include_yearly=True)
- mtf_trend_alignment_h1h4
- mtf_trend_alignment_h4d1
- mtf_trend_alignment_d1w1
- mtf_consensus
- mtf_trend_strength
- mtf_vol_ratio_h1h4
- mtf_rsi_divergence
- mtf_d1_above_prev_high
- mtf_d1_below_prev_low
- mtf_d1_dist_to_high
- mtf_d1_dist_to_low

## Acceptance Criteria

- AC-001: Returns the input DataFrame concatenated with a feature-columns block on the same index; original OHLC columns are preserved.
- AC-002: All feature columns are shifted by one bar via shift_features before being returned so that row i's features depend only on data up to and including row i-1 (no lookahead).
- AC-003: All divisions (range-position, EMA distance, ATR%, volatility ratio, distance-to-high/low) go through safe_divide, so zero denominators do not raise.
- AC-004: H4 block emits mtf_h4_trend, mtf_h4_range_pos, mtf_h4_ema{p}_dist for each p in ema_periods, mtf_h4_adx, mtf_h4_rsi, mtf_h4_atr_pct, mtf_h4_bb_pband using H4-scaled windows built from h4_bars-length rolling H/L extremes.
- AC-005: D1 block emits mtf_d1_range_pos, mtf_d1_ema{p}_dist for each p in ema_periods, and mtf_d1_trend_strength as the pct_change of a 20*d1_bars EMA over d1_bars bars, scaled by 100.
- AC-006: W1 block emits mtf_w1_range_pos, mtf_w1_ema{p}_dist for each p in ema_periods, and mtf_w1_trend_strength as the pct_change of a 20*w1_bars EMA over w1_bars bars, scaled by 100.
- AC-007: When include_yearly=True, additionally emits mtf_y1_ema200d_dist (using a 200*d1_bars EMA) and mtf_y1_52w_range_pos, mtf_y1_52w_high_dist, mtf_y1_52w_low_dist computed over a rolling 52*w1_bars window with min_periods=w1_bars. When include_yearly=False, these four columns are absent from the output DataFrame entirely. Note: get_feature_columns() always lists all four mtf_y1_* columns regardless of include_yearly, so its output is inconsistent with the actual DataFrame when include_yearly=False.
- AC-008: Trend-alignment features mtf_trend_alignment_h1h4, mtf_trend_alignment_h4d1, mtf_trend_alignment_d1w1 are int {0,1} indicators of matching signs between successive-timeframe EMA-distance trends (H1 uses EMA21).
- AC-009: mtf_consensus is 1 iff the H1, H4-ema20, D1-ema20, and W1-ema20 trend signs all agree, and mtf_trend_strength equals the sum of the three pairwise alignment flags (integer in 0..3).
- AC-010: mtf_vol_ratio_h1h4 equals safe_divide(H1 14-bar ATR%, H4 14-bar ATR%) and mtf_rsi_divergence equals H1 14-period RSI minus H4 (14*h4_bars)-period RSI.
- AC-011: mtf_d1_above_prev_high and mtf_d1_below_prev_low are int {0,1} flags comparing close to the previous-day (d1_bars-shifted) rolling high/low, and mtf_d1_dist_to_high, mtf_d1_dist_to_low are their normalized distances.

## Edge Cases

- Early rows produce NaN feature values while the largest rolling / EMA windows are still filling (particularly the 52*w1_bars yearly window and 200*d1_bars EMA).
- When include_yearly=False, the four mtf_y1_* columns are not created, so downstream consumers must not assume their presence unconditionally despite them appearing in get_feature_columns().
- Zero-width ranges (e.g. flat H4/D1/W1/Y1 highs equal to lows) do not raise because range-position and distance features go through safe_divide.
- For yearly high/low series, min_periods=w1_bars means values appear well before the full 52*w1_bars window is filled, so the 52-week features are approximations until the full window is available.
- mtf_consensus and each pairwise alignment flag treat sign(0) as 0, so a trend of exactly zero on any timeframe is considered aligned only with other exactly-zero trends.

## Assumptions

- Input DataFrame uses hourly (H1) bars and exposes OHLC as columns named exactly O, H, L, C.
- h4_bars, d1_bars, w1_bars are chosen consistently (e.g. d1_bars = 6*h4_bars, w1_bars = 5*d1_bars) so the aggregated windows match real calendar timeframes.
