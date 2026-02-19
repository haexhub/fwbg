# Momentum Indicators

Oscillator-based features that measure the speed, magnitude, and persistence of price changes to identify overbought/oversold conditions and momentum shifts.

## Concept

Momentum indicators quantify the rate of change and relative position of prices within recent ranges. Unlike trend indicators that focus on direction, momentum oscillators focus on *exhaustion* and *acceleration* -- they detect when a move is losing steam or when price has reached extreme levels relative to its recent history. This makes them particularly valuable for anticipating reversals and timing entries within an established trend.

The mathematical foundations of these oscillators vary but share common principles. RSI compares the average magnitude of up moves versus down moves over a window, producing a bounded 0-100 reading. The Stochastic Oscillator measures where the current close falls within the recent high-low range. Williams %R is conceptually identical to the Stochastic but inverted (0 to -100 scale). The Ultimate Oscillator combines three different timeframes into a single reading to reduce false signals. Rate of Change (ROC) is the simplest momentum measure -- just the percentage price change over N bars.

For ML models, momentum features serve as mean-reversion signals and regime classifiers. When RSI reaches extreme values (above 70 or below 30), it suggests stretched conditions that may revert. The multi-period approach (RSI-7, RSI-14, RSI-21) allows models to detect divergences across timeframes -- for example, short-term RSI may be oversold while long-term RSI remains neutral, suggesting a pullback within a larger uptrend. Stochastic crossovers (%K crossing %D) and Williams %R extremes provide complementary overbought/oversold readings.

## Features

| Feature | Description |
|---------|-------------|
| `mom_rsi_7` | Relative Strength Index (7-period). Fast RSI that reacts quickly to price changes. Range: 0-100. Above 70 = overbought, below 30 = oversold. |
| `mom_rsi_14` | Relative Strength Index (14-period). The classic Wilder RSI period, balanced between sensitivity and smoothness. |
| `mom_rsi_21` | Relative Strength Index (21-period). Smoother RSI that filters out short-term noise and captures larger momentum swings. |
| `mom_stoch_k_14` | Stochastic %K (14-period). Measures where close price is relative to the 14-bar high-low range. Range: 0-100. |
| `mom_stoch_d_14` | Stochastic %D (14-period). Signal line (smoothed %K) for the 14-period Stochastic. Crossovers of %K above/below %D generate signals. |
| `mom_stoch_k_21` | Stochastic %K (21-period). Longer lookback range position, smoother than the 14-period variant. |
| `mom_stoch_d_21` | Stochastic %D (21-period). Signal line for the 21-period Stochastic. |
| `mom_williams_14` | Williams %R (14-period). Inverted Stochastic oscillator, range: 0 to -100. Near 0 = overbought, near -100 = oversold. |
| `mom_williams_21` | Williams %R (21-period). Longer lookback variant with smoother readings. |
| `mom_uo` | Ultimate Oscillator. Combines momentum across three timeframes (7, 14, 28 periods) into a single 0-100 reading, reducing false signals from single-period oscillators. |
| `mom_roc_5` | Rate of Change (5-period). Percentage price change over 5 bars. Captures very short-term momentum. |
| `mom_roc_10` | Rate of Change (10-period). Percentage price change over 10 bars. Short-term swing momentum. |
| `mom_roc_20` | Rate of Change (20-period). Percentage price change over 20 bars. Medium-term momentum, roughly one trading month. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rsi_periods` | list[int] | [7, 14, 21] | Periods for RSI calculation. RSI oscillates 0-100 measuring the speed and magnitude of price changes. 14 is the classic Wilder period; shorter periods (7) are more sensitive to recent moves, longer periods (21) smoother. Min: 2, Max: 500. |
| `stoch_periods` | list[int] | [14, 21] | Lookback periods for Stochastic Oscillator (%K and %D). Measures where price closed relative to its high-low range over N bars. Shorter periods react faster to price swings, longer periods filter out noise. Min: 2, Max: 500. |
| `williams_periods` | list[int] | [14, 21] | Lookback periods for Williams %R. Similar to Stochastic but inverted (0 to -100 scale). Identifies overbought (near 0) and oversold (near -100) conditions within the given lookback window. Min: 2, Max: 500. |
| `roc_periods` | list[int] | [5, 10, 20] | Periods for Rate of Change calculation. Measures the percentage change in price over N bars. Short periods (5) capture immediate momentum, longer periods (20) capture swing momentum. Min: 1, Max: 500. |

## Usage Notes

- This plugin does not benefit from stationarity transformations (`benefits_from_stationary: false`), as its features are already bounded or percentage-based.
- All features are shifted by 1 bar to prevent lookahead bias.
- RSI, Stochastic, and Williams %R are bounded oscillators (inherently stationary). ROC is unbounded but typically mean-reverting.
- The Ultimate Oscillator uses fixed internal periods (7, 14, 28) and is not configurable through parameters.
- The longest lookback required with default parameters is 21 bars (RSI-21, Stochastic-21, Williams-21), plus additional warm-up bars for the internal EMA smoothing.
- Multi-period features (e.g., RSI at 7, 14, 21) enable ML models to detect cross-timeframe divergences, which are strong reversal signals.
