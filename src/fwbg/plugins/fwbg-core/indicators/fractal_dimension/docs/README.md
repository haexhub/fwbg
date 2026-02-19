# Fractal Dimension (Higuchi)

Measures the complexity and roughness of a price series using the Higuchi Fractal Dimension algorithm, classifying market regimes as trending, random, or mean-reverting.

## Concept

The fractal dimension quantifies how "rough" or "complex" a time series is. For a price series, the Higuchi Fractal Dimension (HFD) produces a value between 1.0 and 2.0. A value near **1.0** indicates a smooth, trending series (low complexity), while a value near **2.0** indicates a highly complex, space-filling series. A value near **1.5** corresponds to a random walk -- the theoretical fractal dimension of Brownian motion.

The Higuchi algorithm works by constructing sub-series at different scale intervals `k = 1, 2, ..., k_max`. For each scale, it measures the "length" of the sub-series (sum of absolute differences, normalized). The fractal dimension is then estimated as the slope of `log(L(k))` versus `log(1/k)` via linear regression. This approach is computationally efficient and produces robust estimates even for relatively short windows.

This plugin computes the Higuchi FD over multiple rolling windows and derives several features from it. The raw FD values indicate the current complexity regime. The **change** in FD detects transitions between regimes (e.g., from trending to random). The **complexity ratio** measures how far the current FD deviates from 1.5 (random walk), and the **regime** feature provides a discrete classification. ML models can use these features to adapt their behavior -- for example, applying momentum strategies in trending regimes (FD < 1.4) and mean-reversion strategies in complex regimes (FD > 1.6).

## Features

For each window size `W` in the `windows` list (default: 50, 100, 200), the following features are produced:

| Feature | Description |
|---------|-------------|
| `fd_higuchi_{W}` | Raw Higuchi Fractal Dimension computed over a rolling window of `W` bars. Range approximately 1.0 to 2.0. Values near 1.0 = trending, near 1.5 = random walk, near 2.0 = highly complex/mean-reverting. |
| `fd_higuchi_change_{W}` | Change in the Higuchi FD compared to `W` bars ago (`FD[i] - FD[i-W]`). Positive values indicate increasing complexity (transitioning away from trending), negative values indicate decreasing complexity (emerging trend). |
| `fd_complexity_ratio_{W}` | Distance from the random walk value, scaled: `abs(FD - 1.5) * 2.0`. Values near 0.0 = random walk behavior, values near 1.0 = highly structured (either strongly trending or strongly mean-reverting). |
| `fd_regime_{W}` | Discrete regime classification: `-1.0` = trending (FD < 1.4), `0.0` = random walk (1.4 <= FD <= 1.6), `1.0` = mean-reverting (FD > 1.6). |

With default windows `[50, 100, 200]`, this produces 12 features total.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windows` | list[int] | `[50, 100, 200]` | Rolling window sizes for Higuchi Fractal Dimension computation. Shorter windows capture local regime transitions, longer windows provide a more stable complexity measure. Range per element: 20-1000. |
| `k_max` | int | `10` | Maximum interval parameter for the Higuchi algorithm. Controls the number of sub-series scales used to estimate the fractal dimension. Higher values improve accuracy but increase computation time quadratically. Must be less than the smallest window size. Range: 2-50. |

## Usage Notes

- **Warmup period**: Each window size `W` requires `W - 1` bars before the first valid FD value is produced. With the default `windows = [50, 100, 200]`, the first 199 bars will have `NaN` for the 200-bar features.
- **Computation cost**: The Higuchi algorithm has complexity O(W * k_max) per bar, applied over a rolling window. With large windows and datasets, this can be slow. Consider reducing `k_max` or using fewer/smaller windows if performance is a concern.
- **Regime thresholds**: The regime classification uses fixed boundaries at 1.4 and 1.6. These are reasonable defaults for financial time series but are not optimized parameters. The raw `fd_higuchi_{W}` feature provides continuous values for models that can learn their own thresholds.
- **Complementary to Hurst exponent**: The fractal dimension is related to the Hurst exponent by `D = 2 - H` for fractional Brownian motion. A Hurst exponent of 0.5 (random walk) corresponds to FD = 1.5. However, for real financial data this relationship is approximate.
- **Stationarity**: This plugin does not benefit from stationary input data (`benefits_from_stationary = False`). The fractal dimension is computed on raw close prices and is inherently scale-independent.
- **Works on all timeframes**: The fractal dimension can be computed on any timeframe. The window sizes should be adjusted relative to the timeframe -- e.g., on daily data, a 50-bar window covers ~2.5 months; on M15 data, it covers ~12.5 hours.
