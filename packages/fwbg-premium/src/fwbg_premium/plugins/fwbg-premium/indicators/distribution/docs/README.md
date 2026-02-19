# Distribution

Measures the statistical shape of the return distribution using rolling skewness, kurtosis, autocorrelation, and derived tail-risk scores.

## Concept

Financial returns are not normally distributed. They exhibit fat tails (kurtosis), asymmetry (skewness), and serial dependence (autocorrelation). These higher-order statistical properties change over time and carry predictive information about future market behavior. Periods of high kurtosis signal elevated tail risk, while shifts in skewness can precede trend changes or crash events.

Rolling skewness and kurtosis, computed across multiple window lengths, capture the evolving shape of the return distribution. Z-score normalization places current values in historical context -- a kurtosis of 5 means something different in a market that typically shows kurtosis of 2 versus one that typically shows 6. Change features (deltas of skewness and kurtosis) detect regime shifts as they happen.

Autocorrelation features measure serial dependence at various lags. Positive autocorrelation indicates trending/persistence behavior, while negative autocorrelation indicates mean-reversion. Changes in autocorrelation structure can signal transitions between trending and range-bound regimes. ML models benefit from these features because they encode information about market microstructure that simple price-based indicators miss.

## Features

| Feature | Description |
|---------|-------------|
| `dist_skew_20` | Rolling skewness of returns over 20 bars |
| `dist_skew_50` | Rolling skewness of returns over 50 bars |
| `dist_skew_100` | Rolling skewness of returns over 100 bars |
| `dist_kurt_20` | Rolling excess kurtosis of returns over 20 bars (0 = normal) |
| `dist_kurt_50` | Rolling excess kurtosis of returns over 50 bars |
| `dist_kurt_100` | Rolling excess kurtosis of returns over 100 bars |
| `dist_skew_20_z` | Z-score of 20-bar skewness relative to its own rolling history |
| `dist_skew_50_z` | Z-score of 50-bar skewness relative to its own rolling history |
| `dist_skew_100_z` | Z-score of 100-bar skewness relative to its own rolling history |
| `dist_kurt_20_z` | Z-score of 20-bar kurtosis relative to its own rolling history |
| `dist_kurt_50_z` | Z-score of 50-bar kurtosis relative to its own rolling history |
| `dist_kurt_100_z` | Z-score of 100-bar kurtosis relative to its own rolling history |
| `dist_skew_change_10` | 10-bar change in 50-bar skewness |
| `dist_skew_change_20` | 20-bar change in 50-bar skewness |
| `dist_kurt_change_10` | 10-bar change in 50-bar kurtosis |
| `dist_kurt_change_20` | 20-bar change in 50-bar kurtosis |
| `dist_tail_risk` | Composite tail-risk score combining high kurtosis and negative skewness (0-1 range) |
| `dist_stability` | Rolling standard deviation of 50-bar skewness over 50 bars (distribution stability) |
| `dist_autocorr_1` | Rolling autocorrelation at lag 1 (short-term serial dependence) |
| `dist_autocorr_5` | Rolling autocorrelation at lag 5 |
| `dist_autocorr_10` | Rolling autocorrelation at lag 10 |
| `dist_autocorr_20` | Rolling autocorrelation at lag 20 (longer-term serial dependence) |
| `dist_autocorr_1_change` | 20-bar change in lag-1 autocorrelation (regime shift indicator) |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windows` | `List[int]` | `[20, 50, 100]` | Rolling window sizes for skewness and kurtosis computation |
| `z_score_lookback` | `int` | `200` | Lookback period for z-score normalization of skewness/kurtosis |
| `compute_changes` | `bool` | `True` | Whether to compute change (delta) features for skewness and kurtosis |
| `autocorr_lags` | `List[int]` | `[1, 5, 10, 20]` | Lag values for autocorrelation computation |
| `autocorr_window` | `int` | `100` | Rolling window size for autocorrelation calculation |

## Usage Notes

- The largest window (100) plus the z-score lookback (200) means approximately 300 bars are needed before all features produce meaningful values.
- Change features are only computed when window 50 is included in the `windows` parameter. If you customize windows and exclude 50, the `dist_skew_change_*`, `dist_kurt_change_*`, `dist_tail_risk`, and `dist_stability` features will not be generated.
- Autocorrelation computation uses `pd.Series.autocorr()` inside a rolling apply, which can be computationally expensive on very large DataFrames.
- The `dist_autocorr_1_change` feature is only computed when lag 1 is included in `autocorr_lags`.
- All features are shifted by 1 bar to prevent lookahead bias.
- `benefits_from_stationary: false` -- the plugin computes returns internally from close prices.
