# Plugin Spec — cross_features

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes cross-indicator interaction, divergence, confluence, and COT-positioning×volatility features, recomputing base indicators inline to avoid double-shift.

## Summary

Recomputes core indicators (RSI, Stochastic, ADX, MACD, ATR%, Bollinger width) from raw OHLC and combines them into conditional, interaction, divergence, and confluence features, plus per-pair COT positioning × volatility interactions when macro_cot_* columns are provided. All emitted features are shifted by one bar to eliminate lookahead bias.

## Inputs

- df: DataFrame with OHLC columns H, L, C (used both for indicator recomputation and divergence/price-momentum logic)
- df: optional macro_cot_<pair> columns (raw, unshifted COT net-positioning series) for COT interaction features

## Parameters

- `rsi_overbought` (float, default=70): RSI level above which the market is treated as overbought for conditional / confluence / stoch-RSI features.
- `rsi_oversold` (float, default=30): RSI level below which the market is treated as oversold for conditional / confluence / stoch-RSI features.

## Outputs

- cross_rsi_high_rising
- cross_rsi_low_falling
- cross_rsi_high_falling
- cross_rsi_low_rising
- cross_vol_trend
- cross_expanding_trend
- cross_contracting
- cross_bb_squeeze
- cross_bullish_strong
- cross_bearish_strong
- cross_bullish_confluence
- cross_bearish_confluence
- cross_bearish_divergence
- cross_bullish_divergence
- cross_momentum_vol_score
- cross_overbought_uptrend
- cross_oversold_downtrend
- cross_stoch_rsi_overbought
- cross_stoch_rsi_oversold
- cross_bullish_count
- cross_bearish_count
- cross_signal_bias
- cross_<pair>_vol_interaction (one per macro_cot_<pair> column in input)
- cross_<pair>_price_divergence (one per macro_cot_<pair> column in input)

## Acceptance Criteria

- AC-001: Recomputes base indicators (rsi, stoch, adx, macd, atr_pct, bb_width) from raw OHLC on every call rather than reusing pre-shifted columns from the input DataFrame, to prevent double-shift when run after other indicator modules
- AC-002: Emits RSI conditional flags (cross_rsi_high_rising, cross_rsi_low_falling, cross_rsi_high_falling, cross_rsi_low_rising) comparing RSI vs. rsi_overbought/rsi_oversold and its 4-bar change
- AC-003: Emits volatility-trend interaction features (cross_vol_trend, cross_expanding_trend, cross_contracting) from ATR% change and ADX
- AC-004: Emits cross_bb_squeeze flag when current Bollinger band width is at or below the rolling 20th percentile of the prior 99 bars (min_periods=20, shifted by 1)
- AC-005: Emits trend-confirmation flags (cross_bullish_strong, cross_bearish_strong) from an EMA(8) vs EMA(21) crossover combined with ADX > 25
- AC-006: Emits MACD-RSI confluence flags (cross_bullish_confluence, cross_bearish_confluence) requiring MACD sign, RSI vs 50, and RSI within the non-extreme range
- AC-007: Emits divergence flags (cross_bearish_divergence, cross_bullish_divergence) comparing 20-bar price highs/lows against 20-bar RSI highs/lows
- AC-008: Emits cross_momentum_vol_score as the product of normalized RSI, ADX and ATR%-vs-50-bar-mean
- AC-009: Emits overbought/oversold-with-trend flags (cross_overbought_uptrend, cross_oversold_downtrend) combining RSI extremes, EMA(8) vs EMA(21) direction, and ADX > 20
- AC-010: Emits stochastic-RSI confluence flags (cross_stoch_rsi_overbought, cross_stoch_rsi_oversold) requiring Stoch > 80 / < 20 alongside RSI extremes
- AC-011: Emits composite counts cross_bullish_count, cross_bearish_count, and cross_signal_bias summing confluence, divergence, and momentum_vol_score sign
- AC-012: For every macro_cot_* column present in the input, emits cross_<pair>_vol_interaction (COT z-score times inverse ATR%-rank) and cross_<pair>_price_divergence (price 5-day-momentum z-score minus COT 5-day-momentum z-score)
- AC-013: Applies shift_features on the full feature dict before returning so no feature at bar i depends on data at bar i+1
- AC-014: Uses safe_divide for MACD-diff/close, ATR/close, and ATR%-vs-mean ratios to avoid division-by-zero

## Edge Cases

- When run after other indicator modules that already produced shifted columns, base indicators are recomputed from raw OHLC to avoid double-shift
- Bollinger squeeze percentile uses min_periods=20 over a 99-bar window and is shifted by 1, so it is NaN until at least 20 prior bars are available
- Rolling 50/100/500-bar statistics (ATR% mean, ATR% rank, price/COT z-scores) yield NaN early in the series until enough history accumulates
- safe_divide guards MACD/close, ATR/close, and ATR%-vs-mean against zero denominators; rolling std for COT and price/COT z-scores is clipped at 1e-6
- If no macro_cot_* columns are present in the input, no cross_<pair>_vol_interaction / cross_<pair>_price_divergence features are produced, even though get_feature_columns lists them
- atr_pct_rank is clipped at a floor of 0.01 before inversion so extremely low volatility bars do not produce infinite interaction values
- All produced features are shifted by 1 bar before being concatenated back, so the first row is NaN for every emitted feature

## Assumptions

- _none_
