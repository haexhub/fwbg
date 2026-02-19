# Topological Features

Extracts topological invariants from price dynamics using Persistent Homology applied to Takens time-delay embeddings, capturing the "shape" of market behavior that traditional statistical features miss.

## Concept

Topological Data Analysis (TDA) studies the shape of data. Rather than measuring statistical moments (mean, variance, skewness), TDA identifies structural properties like connected components, loops, and voids in the data's geometry. These topological features are invariant to continuous deformations -- they capture qualitative patterns that persist across different scales, making them robust to noise and well-suited for detecting regime changes in financial markets.

The plugin first applies Takens time-delay embedding to convert a 1D series of log returns into a point cloud in higher-dimensional space. According to Takens' theorem, this embedding preserves the topological properties of the underlying dynamical system. The point cloud is then analyzed using Persistent Homology (via the Ripser library), which tracks how topological features (connected components and loops) appear and disappear as a scale parameter increases. Features that persist across many scales are considered "real" structure, while short-lived features are noise.

The resulting features give ML models a fundamentally different view of market dynamics. H0 features (connected components) capture fragmentation -- how "broken up" the price structure is. H1 features (loops/cycles) capture cyclical patterns and mean-reversion signals. A market with strong loops suggests recurring price patterns, while a market with only connected components and no loops is trending or random. The persistence entropy and Wasserstein amplitude provide summary statistics of the overall topological complexity, enabling models to distinguish orderly trending regimes from chaotic, structurally complex ones.

## Features

Features are generated for each configured window size `w`. With default parameters (`windows=[50, 100]`), each window produces 10 features.

| Feature | Description |
|---------|-------------|
| `tda_h0_count_{w}` | Number of finite-persistence H0 features (connected components) in the rolling window. Higher counts indicate a more fragmented price structure. |
| `tda_h1_count_{w}` | Number of finite-persistence H1 features (loops/cycles) in the rolling window. Higher counts indicate more cyclical/mean-reverting behavior. |
| `tda_h0_max_pers_{w}` | Maximum persistence of H0 features. High values indicate a dominant structural component that persists across many scales. |
| `tda_h1_max_pers_{w}` | Maximum persistence of H1 features (loops). High values indicate a strong, persistent cyclical pattern in price dynamics. |
| `tda_h0_mean_pers_{w}` | Mean persistence of H0 features. Measures the average structural significance of connected components. |
| `tda_h1_mean_pers_{w}` | Mean persistence of H1 features. Measures the average strength of cyclical patterns. |
| `tda_persistence_entropy_{w}` | Shannon entropy of all persistence values (H0 + H1 combined): `-sum(p_i * log(p_i))`. Higher entropy means topological features are more evenly distributed; lower entropy means a few features dominate. |
| `tda_wasserstein_amp_{w}` | Wasserstein amplitude: `sqrt(sum(persistence^2))` across all finite features. A single scalar summarizing the total topological "energy" of the point cloud. |
| `tda_h1_ratio_{w}` | Ratio of H1 count to H0 count. High values indicate loop-dominated (cyclical) structure; low values indicate component-dominated (trending/fragmented) structure. |
| `tda_max_loop_persistence_{w}` | Ratio of H1 max persistence to H0 max persistence. Measures the relative strength of the strongest loop versus the strongest component. |

**Default feature count:** 10 features per window x 2 windows = **20 features**.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windows` | list[int] | `[50, 100]` | Rolling window sizes for computing TDA features. Each window defines the segment of log returns used for Takens embedding and persistence computation. Larger windows capture longer-range topological structure but are more computationally expensive. |
| `embedding_dim` | int | `3` | Dimension of the Takens time-delay embedding space. Higher dimensions can capture more complex dynamics but require more data points per window. The Takens theorem suggests `embedding_dim >= 2d + 1` where `d` is the dimension of the underlying attractor. |
| `time_delay` | int | `1` | Time delay (lag) between coordinates in the Takens embedding. A delay of 1 uses consecutive returns; larger delays can decorrelate the embedding coordinates and better resolve the attractor structure. |
| `maxdim` | int | `1` | Maximum homology dimension to compute. `1` computes H0 (connected components) and H1 (loops). Setting to `2` would also compute H2 (voids) but is significantly more expensive and rarely needed for 1D price data. |

## Usage Notes

- **Input column:** Uses the `C` (close) column. Log returns are computed internally.
- **Stationarity:** Does not require pre-stationarized input (`benefits_from_stationary: false`).
- **Computational cost:** Persistent homology via Ripser has super-linear complexity in the number of points. Large windows (>200) can be slow. The step-based rolling computation helps, but this plugin is inherently more expensive than statistical indicators.
- **Minimum data requirements:** Each window needs at least `window` data points, and the Takens embedding requires `embedding_dim * time_delay` additional points within each window. A minimum of 3 valid embedded points is enforced per window position.
- **NaN warmup period:** The first `window - 1` bars will have NaN values for each window size, as there is insufficient data for a full rolling window.
- **Feature shifting:** All features are shifted forward by one bar via `shift_features` to prevent look-ahead bias.
- **Error handling:** If Ripser fails on a particular window (e.g., degenerate point cloud), the features for that position are left as NaN rather than causing the entire computation to fail.
- **Interpreting H1 features:** Strong H1 (loop) features often correspond to mean-reverting or oscillating price behavior. The absence of H1 features combined with many H0 components may indicate trending or structurally simple (random walk-like) price action.
