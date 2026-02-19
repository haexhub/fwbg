# Regime

Detects market regimes using statistical measures of price persistence, complexity, and random-walk deviation: the Hurst exponent, Shannon entropy, and Lo-MacKinlay variance ratio.

## Concept

Financial markets do not behave uniformly over time. They alternate between trending periods (where momentum strategies thrive), mean-reverting periods (where contrarian strategies work), and random-walk periods (where neither approach has an edge). Identifying the current regime is critical for strategy selection and risk management.

The Hurst exponent, computed via Rescaled Range (R/S) analysis, quantifies the persistence of price movements. A Hurst value above 0.5 indicates trending/persistent behavior (positive autocorrelation), exactly 0.5 indicates a random walk, and below 0.5 indicates mean-reverting behavior (negative autocorrelation). The plugin computes rolling Hurst at multiple time scales (100, 200, 500 bars) to capture regime characteristics at different horizons. The divergence between short-term (100) and long-term (500) Hurst values serves as a regime-shift early warning signal.

Shannon entropy of the return distribution measures market complexity and predictability. High entropy indicates a wide, uniform distribution of returns (high uncertainty, difficult to predict), while low entropy indicates concentrated, structured return patterns (more predictable, better for ML models). The variance ratio test (Lo & MacKinlay) provides a complementary perspective: VR > 1 signals positive serial correlation (momentum), VR = 1 confirms a random walk, and VR < 1 signals negative serial correlation (mean reversion). Together, these three orthogonal measures give ML models a rich characterization of the current market microstate.

## Features

| Feature | Description |
|---------|-------------|
| `regime_hurst_100` | Rolling Hurst exponent (100-bar window). Short-term persistence measure |
| `regime_hurst_200` | Rolling Hurst exponent (200-bar window). Medium-term persistence measure |
| `regime_hurst_500` | Rolling Hurst exponent (500-bar window). Long-term persistence measure |
| `regime_hurst_100_chg` | Change in Hurst(100) over 24 bars. Detects short-term regime shifts |
| `regime_hurst_200_chg` | Change in Hurst(200) over 48 bars. Detects medium-term regime shifts |
| `regime_hurst_divergence` | Hurst(100) - Hurst(500). Short-term vs. long-term persistence divergence. Positive = short-term more trending than long-term |
| `regime_entropy_50` | Rolling Shannon entropy (50-bar window). Short-term market complexity |
| `regime_entropy_100` | Rolling Shannon entropy (100-bar window). Medium-term market complexity |
| `regime_entropy_100_chg` | Change in entropy(100) over 24 bars. Falling entropy = increasing predictability |
| `regime_vr_100_5` | Variance ratio (100-bar window, lag 5). >1 = momentum, =1 = random walk, <1 = mean reversion |
| `regime_vr_100_10` | Variance ratio (100-bar window, lag 10). Same interpretation at longer lag |
| `regime_vr_200_5` | Variance ratio (200-bar window, lag 5). More stable momentum/reversion signal |
| `regime_vr_200_10` | Variance ratio (200-bar window, lag 10). Long-window, long-lag regime measure |
| `regime_vr_deviation` | VR(100,5) - 1.0. Centered deviation from random walk. Positive = momentum, negative = mean reversion |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hurst_windows` | List[int] | [100, 200, 500] | Rolling window sizes for Hurst exponent computation |
| `step` | int | 10 | Step size for rolling computations. Larger values speed up calculation at the cost of granularity (values between steps are forward-filled) |
| `entropy_windows` | List[int] | [50, 100] | Rolling window sizes for Shannon entropy computation |
| `vr_windows` | List[int] | [100, 200] | Rolling window sizes for variance ratio computation |
| `vr_lags` | List[int] | [5, 10] | Lag periods for the variance ratio test. Each (window, lag) combination produces a separate feature |

## Usage Notes

- All features are shifted by 1 bar to prevent lookahead bias.
- The Hurst exponent uses the original (non-differenced) close price when available via `_original_close` column. This is important when fractional differencing is active, since Hurst should be computed on the original price series.
- The `step` parameter introduces a computation/granularity trade-off. With `step=10`, Hurst and entropy are only computed every 10th bar and forward-filled in between. This dramatically reduces computation time for large datasets.
- Hurst computation requires `max_lag * 2` data points per window, so very small windows may default to 0.5 (random walk assumption).
- The `manifest.json` sets `benefits_from_stationary: false` since the plugin uses log-returns internally and Hurst is computed on original prices.
- Shannon entropy uses 10 histogram bins by default (hardcoded). The entropy value depends on the number of bins -- it is not directly comparable across different bin settings.
- Variance ratio features centered at 1.0: the `regime_vr_deviation` feature subtracts 1.0 for easier interpretation (positive = momentum, negative = mean reversion).
