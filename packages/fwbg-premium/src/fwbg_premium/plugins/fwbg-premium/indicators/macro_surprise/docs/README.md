# Macro Surprise

Detects unusual price movements that signal new information arrival, including gap analysis, surprise moves, volatility breaks, and return decomposition into overnight vs. intraday components.

## Concept

Markets move on information. When new, unexpected information arrives -- economic data releases, geopolitical events, central bank decisions -- it manifests as abnormal price behavior: gaps between sessions, outsized intraday ranges, or returns far exceeding what recent volatility would predict. The Macro Surprise plugin systematically quantifies these "information events" and their characteristics.

Gap analysis decomposes the open-to-previous-close relationship, measuring gap size, direction, and whether gaps get filled or extended. Gap persistence patterns carry predictive power: institutional-driven gaps tend to extend, while retail-driven gaps tend to fill. The return decomposition separates total returns into overnight and intraday components, revealing whether moves are driven by off-hours information flow or intraday price discovery.

Surprise detection compares actual price ranges and returns to their statistically expected values based on recent volatility. A range that is 3x the expected move, or a return that is 2+ standard deviations from the mean, is flagged as a surprise. Volatility break detection identifies periods where realized volatility suddenly exceeds its recent average. Together with streak features that track consecutive gaps or surprises, these features give ML models the ability to detect and react to information regimes -- periods of elevated news flow versus calm markets.

## Features

| Feature | Description |
|---------|-------------|
| `macro_gap` | Raw gap: open minus previous close (absolute) |
| `macro_gap_pct` | Gap as percentage of previous close |
| `macro_gap_normalized` | Gap divided by average true range over vol_lookback bars |
| `macro_gap_up` | Binary: gap is positive (open > previous close) |
| `macro_gap_down` | Binary: gap is negative (open < previous close) |
| `macro_gap_filled` | Binary: close returns to or past previous close level, filling the gap |
| `macro_gap_extended` | Binary: close extends beyond open in the gap direction |
| `macro_gap_avg` | Rolling mean of gap percentage over gap_ma_period bars |
| `macro_gap_std` | Rolling standard deviation of gap percentage over gap_ma_period bars |
| `macro_total_return` | Total bar return: (close - prev_close) / prev_close |
| `macro_overnight_return` | Overnight return component: (open - prev_close) / prev_close |
| `macro_intraday_return` | Intraday return component: (close - open) / open |
| `macro_overnight_ratio` | Fraction of total absolute return attributable to the overnight component |
| `macro_range_surprise` | Actual high-low range divided by expected range from historical volatility |
| `macro_is_surprise` | Binary: actual range exceeds surprise_threshold * expected range |
| `macro_return_zscore` | Return z-score: bar return divided by rolling standard deviation |
| `macro_return_surprise` | Binary: absolute return z-score exceeds surprise_threshold |
| `macro_vol_ratio` | Realized 5-bar volatility divided by expected volatility (vol_lookback rolling mean) |
| `macro_vol_zscore` | Z-score of realized volatility relative to its rolling distribution |
| `macro_gap_streak` | Consecutive bars with same gap direction (resets on direction change) |
| `macro_surprise_streak` | Consecutive bars flagged as surprises |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vol_lookback` | `int` | `20` | Lookback period for historical volatility estimation and average range |
| `surprise_threshold` | `float` | `2.0` | Number of standard deviations for a move to be classified as a "surprise" |
| `gap_ma_period` | `int` | `10` | Rolling window for gap moving average and standard deviation |

## Usage Notes

- Gap analysis works best on instruments with distinct session boundaries (e.g. forex pairs with weekend gaps, or equity indices with overnight closes). On 24-hour instruments with no real session breaks, gap features will be small and less informative.
- The `macro_overnight_ratio` can produce NaN when total return is zero (no price change). These are left as NaN and should be handled by downstream imputation.
- Surprise detection is relative to recent volatility: a 2% move is a surprise in low-vol environments but normal in high-vol ones. The `surprise_threshold` parameter controls this sensitivity.
- Realized volatility for `macro_vol_ratio` uses a 5-bar fast window versus the vol_lookback slow window, making it responsive to sudden volatility shifts.
- Streak features reset to 0 when the tracked condition is not met (e.g. gap direction changes or no surprise).
- All features are shifted by 1 bar to prevent lookahead bias.
- `benefits_from_stationary: false` -- most features are already ratios, z-scores, or binary flags.
