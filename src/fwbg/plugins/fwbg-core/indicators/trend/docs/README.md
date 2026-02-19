# Trend Indicators

Comprehensive trend-following features that measure the direction, strength, and persistence of price movements using a combination of classic technical indicators and adaptive filters.

## Concept

Trend indicators form the backbone of most systematic trading strategies. They answer the fundamental question: *is the market moving directionally, and how strongly?* This plugin combines multiple approaches to trend measurement -- from smoothed averages (EMA/SMA) to directional movement analysis (ADX) and adaptive momentum filters (Kaufman Efficiency Ratio, Supertrend) -- to give ML models a multi-dimensional view of trend state.

The mathematical foundations span several domains. Moving average distances measure price deviation from equilibrium levels, normalized as a percentage of price for cross-asset comparability. ADX applies Wilder's directional movement system to quantify trend strength on a 0-100 scale regardless of direction. CCI measures deviation from a statistical mean relative to average deviation. The Efficiency Ratio captures the signal-to-noise ratio of price movement -- a perfectly trending market has an ER near 1.0, while a choppy market approaches 0.

For ML models, these features provide complementary information. Short-period indicators (ADX-7, EMA-8) capture emerging trends and reversals, while long-period indicators (EMA-200, SMA-200) capture macro regime positioning. The Supertrend provides a clean binary trend signal with built-in noise filtering, and its flip detection marks regime transitions that often precede sustained moves.

## Features

| Feature | Description |
|---------|-------------|
| `trend_adx_7` | Average Directional Index (7-period). Measures short-term trend strength on a 0-100 scale. Values above 25 suggest a trending market, below 20 a ranging market. |
| `trend_adx_14` | Average Directional Index (14-period). The classic Wilder setting, balancing responsiveness and smoothness. |
| `trend_adx_21` | Average Directional Index (21-period). Smoother trend strength reading, less reactive to short-term swings. |
| `trend_ema_dist_8` | Percentage distance of close price from the 8-period EMA. Captures immediate price momentum relative to very short-term trend. |
| `trend_ema_dist_21` | Percentage distance from the 21-period EMA. Measures short-term trend displacement, roughly one trading month. |
| `trend_ema_dist_50` | Percentage distance from the 50-period EMA. Intermediate trend positioning. |
| `trend_ema_dist_100` | Percentage distance from the 100-period EMA. Medium-term trend positioning. |
| `trend_ema_dist_200` | Percentage distance from the 200-period EMA. Long-term trend positioning, widely watched institutional level. |
| `trend_sma_dist_20` | Percentage distance from the 20-period SMA. Equal-weighted short-term mean reversion anchor. |
| `trend_sma_dist_50` | Percentage distance from the 50-period SMA. Classic institutional trend level. |
| `trend_sma_dist_200` | Percentage distance from the 200-period SMA. The "golden cross / death cross" reference level. |
| `trend_macd` | MACD histogram value (MACD line minus signal line), normalized by close price. Measures short-term trend momentum acceleration. |
| `trend_macd_signal` | MACD signal line value, normalized by close price. Provides the smoothed trend direction reference. |
| `trend_cci_14` | Commodity Channel Index (14-period). Measures how far price deviates from its statistical mean. Values beyond +/-100 indicate strong trend. |
| `trend_cci_20` | Commodity Channel Index (20-period). Smoother CCI with less noise. |
| `trend_aroon_up` | Aroon Up (25-period). Measures how recently the highest high occurred within the lookback window (0-100). Values near 100 indicate a recent new high. |
| `trend_aroon_down` | Aroon Down (25-period). Measures how recently the lowest low occurred (0-100). Values near 100 indicate a recent new low. |
| `trend_er_10` | Kaufman Efficiency Ratio (10-period). Signal-to-noise ratio of price movement. 1.0 = perfectly trending, 0.0 = perfectly choppy. |
| `trend_er_20` | Kaufman Efficiency Ratio (20-period). Medium-term trend efficiency. |
| `trend_er_50` | Kaufman Efficiency Ratio (50-period). Long-term trend efficiency. |
| `trend_er_10_chg` | 5-bar change in the 10-period Efficiency Ratio. Captures transitions from choppy to trending regimes (and vice versa). |
| `trend_er_20_chg` | 10-bar change in the 20-period Efficiency Ratio. Captures slower regime transitions. |
| `trend_supertrend` | Supertrend direction signal. +1 for uptrend, -1 for downtrend. ATR-based trend filter that is less noisy than Parabolic SAR. |
| `trend_supertrend_flip` | Supertrend flip detector. 1.0 when the Supertrend direction just changed, 0.0 otherwise. Marks potential trend reversal points. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adx_periods` | list[int] | [7, 14, 21] | Periods for ADX calculation. ADX measures trend strength on a 0-100 scale regardless of direction. Shorter periods react faster to trend changes, longer periods smooth out noise. Min: 2, Max: 500. |
| `ema_periods` | list[int] | [8, 21, 50, 100, 200] | Periods for EMA distance features. Measures how far the current price deviates from each EMA as a percentage. Short EMAs (8, 21) capture immediate momentum, long EMAs (100, 200) capture macro trend positioning. Min: 2, Max: 1000. |
| `sma_periods` | list[int] | [20, 50, 200] | Periods for SMA distance features. Similar to EMA distances but with equal weighting of all bars in the window. Classic levels like 50 and 200 are widely watched by institutional traders. Min: 2, Max: 1000. |
| `supertrend_period` | int | 14 | ATR lookback period for the Supertrend indicator. Controls sensitivity of the ATR-based trend-following bands. Lower values make Supertrend more responsive but noisier. Min: 2, Max: 500. |
| `supertrend_multiplier` | float | 3.0 | ATR multiplier for Supertrend band width. Higher values create wider bands requiring larger moves to trigger trend flips, reducing whipsaws but increasing lag. Min: 0.5, Max: 20.0, Step: 0.5. |

## Usage Notes

- This plugin benefits from stationarity transformations applied beforehand (`benefits_from_stationary: true` in manifest).
- All features are shifted by 1 bar to prevent lookahead bias. The value at bar `t` reflects information available at bar `t-1`.
- EMA and SMA distance features are normalized by close price (`(close - MA) / close`), making them comparable across assets with different price scales.
- MACD values are also price-normalized for cross-asset comparability.
- The longest lookback period required depends on the configured parameters. With defaults, the 200-period SMA/EMA requires at least 200 bars of warm-up data before producing valid values.
- ADX values are bounded (0-100) and already stationary. CCI is unbounded but mean-reverting. EMA/SMA distances are inherently stationary as percentages.
- The Supertrend indicator uses an internal loop and cannot be vectorized, but computation is fast since it only produces a direction signal (+1/-1).
