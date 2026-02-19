# Cross Features

Combines multiple technical indicators into interaction, confluence, and divergence features that capture multi-dimensional market states no single indicator can express.

## Concept

Individual indicators like RSI, ADX, or MACD each measure one dimension of market behavior. However, the most actionable signals emerge when multiple indicators align or diverge. For example, an RSI above 70 alone is far less meaningful than RSI above 70 *and still rising* -- or RSI above 70 while price makes a new high but RSI does not (bearish divergence). Cross features formalize these multi-indicator relationships into numerical features.

The plugin computes conditional boolean features (e.g. "RSI overbought AND rising"), continuous interaction terms (e.g. volatility change multiplied by trend strength), and composite scores that aggregate several signals into a single directional bias. It also detects classical divergences between price and RSI, Bollinger Band squeezes, and MACD-RSI confluence zones.

When COT (Commitment of Traders) macro data is available, the plugin further generates cross-asset features that combine speculative positioning with volatility regime. Extreme positioning paired with low volatility often precedes explosive breakouts -- a relationship that ML models can learn to exploit.

## Features

| Feature | Description |
|---------|-------------|
| `cross_rsi_high_rising` | Binary: RSI > overbought threshold AND RSI increased over last 4 bars |
| `cross_rsi_low_falling` | Binary: RSI < oversold threshold AND RSI decreased over last 4 bars |
| `cross_rsi_high_falling` | Binary: RSI > overbought threshold AND RSI decreased over last 4 bars |
| `cross_rsi_low_rising` | Binary: RSI < oversold threshold AND RSI increased over last 4 bars |
| `cross_vol_trend` | Continuous: ATR percentage change (4-bar) multiplied by ADX / 100 |
| `cross_expanding_trend` | Binary: ATR rising AND ADX > 25 (strong trend with expanding volatility) |
| `cross_contracting` | Binary: ATR falling > 5% AND ADX < 20 (low trend with contracting volatility) |
| `cross_bb_squeeze` | Binary: Bollinger Band width at or below its rolling 20th percentile |
| `cross_bullish_strong` | Binary: EMA(8) > EMA(21) AND ADX > 25 |
| `cross_bearish_strong` | Binary: EMA(8) < EMA(21) AND ADX > 25 |
| `cross_bullish_confluence` | Binary: MACD > 0 AND 50 < RSI < overbought |
| `cross_bearish_confluence` | Binary: MACD < 0 AND oversold < RSI < 50 |
| `cross_bearish_divergence` | Binary: Price makes new 20-bar high but RSI does not |
| `cross_bullish_divergence` | Binary: Price makes new 20-bar low but RSI does not |
| `cross_momentum_vol_score` | Continuous: RSI score * ADX score * volatility score composite |
| `cross_overbought_uptrend` | Binary: RSI overbought AND EMA(8) > EMA(21) AND ADX > 20 |
| `cross_oversold_downtrend` | Binary: RSI oversold AND EMA(8) < EMA(21) AND ADX > 20 |
| `cross_stoch_rsi_overbought` | Binary: Stochastic > 80 AND RSI > overbought |
| `cross_stoch_rsi_oversold` | Binary: Stochastic < 20 AND RSI < oversold |
| `cross_bullish_count` | Integer (0-3): Count of active bullish signals |
| `cross_bearish_count` | Integer (0-3): Count of active bearish signals |
| `cross_signal_bias` | Integer: Bullish count minus bearish count (directional bias) |
| `cross_{pair}_vol_interaction` | Continuous: COT z-score * inverse ATR percentile rank (per currency pair) |
| `cross_{pair}_price_divergence` | Continuous: Price momentum z-score minus COT momentum z-score (per currency pair) |

COT-based features are generated dynamically for each available pair: `cot_eurusd`, `cot_usdjpy`, `cot_gbpusd`, `cot_usdcad`, `cot_audusd`, `cot_usdchf`, `cot_nzdusd`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rsi_overbought` | `float` | `70` | RSI level above which the market is considered overbought |
| `rsi_oversold` | `float` | `30` | RSI level below which the market is considered oversold |

## Usage Notes

- All base indicators (RSI, ADX, ATR, MACD, Stochastic, Bollinger Bands) are recomputed internally from raw OHLC data to avoid double-shift issues when this plugin runs after other indicator plugins.
- All features are shifted by 1 bar to prevent lookahead bias.
- COT cross features require `macro_cot_*` columns to be present in the DataFrame; if absent, those features are simply skipped.
- The Bollinger Band squeeze uses a rolling 99-bar window to compute the 20th percentile of historical band width, requiring at least 20 bars of warmup.
- The `cross_momentum_vol_score` uses a 50-bar rolling mean for volatility normalization, so it needs at least 50 bars of data before producing meaningful values.
- `benefits_from_stationary: false` -- features are already relative or binary.
