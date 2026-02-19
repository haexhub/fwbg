# Autoencoder Features

Compresses all numeric indicator features into a low-dimensional latent representation using Principal Component Analysis (PCA), producing latent components and a reconstruction error anomaly signal.

## Concept

As indicator plugins generate dozens or hundreds of features, many of these features are correlated or redundant. PCA (equivalent to a linear autoencoder's bottleneck layer) identifies the principal axes of variation in the feature space and projects the data onto a compact set of orthogonal components. This dimensionality reduction helps ML models by removing noise, decorrelating inputs, and focusing on the dominant patterns in the data.

Each latent component (`ae_latent_i`) captures an independent mode of variation across all input indicators. The first component explains the largest share of variance, the second explains the next largest orthogonal share, and so on. Together, they form a compressed "fingerprint" of the current market state as seen through the lens of all upstream indicators.

The reconstruction error feature is particularly valuable for trading models. It measures how well the current observation can be explained by the learned principal components. A high reconstruction error means the current market state is unusual -- it does not fit the patterns seen in the rest of the dataset. This acts as an anomaly detector that can signal regime changes, structural breaks, or data quality issues. Models can learn to reduce position sizing or avoid trading when the reconstruction error spikes.

## Features

With default parameters (`n_components=8`), the plugin generates the following features:

| Feature | Description |
|---------|-------------|
| `ae_latent_0` | First principal component -- captures the largest mode of variation across all input indicators. |
| `ae_latent_1` | Second principal component -- captures the second-largest orthogonal mode of variation. |
| `ae_latent_2` | Third principal component. |
| `ae_latent_3` | Fourth principal component. |
| `ae_latent_4` | Fifth principal component. |
| `ae_latent_5` | Sixth principal component. |
| `ae_latent_6` | Seventh principal component. |
| `ae_latent_7` | Eighth principal component. |
| `ae_reconstruction_error` | Per-row squared reconstruction error: `\|\|x - x_reconstructed\|\|^2`. High values indicate that the current observation is poorly explained by the principal components, signaling an anomalous or out-of-distribution market state. |
| `ae_explained_variance` | Cumulative explained variance ratio across all retained components. A scalar broadcast to all rows. Values near 1.0 mean nearly all variance is captured; lower values indicate significant information loss. |

**Default feature count:** 8 latent components + reconstruction error + explained variance = **10 features**.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | `8` | Number of PCA components (latent dimensions) to extract (range: 1-500). Each component captures an orthogonal mode of variation. More components preserve more information but increase dimensionality. The reconstruction error feature acts as an anomaly detector regardless of this setting. |
| `exclude_prefixes` | list[string] | `["ae_"]` | Column name prefixes to exclude from PCA input. By default excludes the autoencoder's own output columns (`ae_*`) to prevent circular dependencies. Add other prefixes to exclude specific indicator groups from the latent representation. |

## Usage Notes

- **Input columns:** Automatically selects all numeric columns except OHLCV (`O`, `H`, `L`, `C`, `V`) and columns matching `exclude_prefixes`. This means it should run **after** other indicator plugins have added their features.
- **Plugin ordering:** This plugin is a meta-indicator that operates on other indicators' outputs. Ensure it runs last (or near-last) in the indicator pipeline so it has features to compress.
- **Stationarity:** Does not require pre-stationarized input (`benefits_from_stationary: false`). Standardization (zero mean, unit variance) is applied internally via `StandardScaler`.
- **Handling of non-finite values:** Infinite values are replaced with NaN, then all NaN values are imputed with column medians before PCA fitting. This provides robustness against upstream indicators that occasionally produce Inf or NaN.
- **Adaptive component count:** If fewer features are available than `n_components`, the effective number of components is automatically reduced to `min(n_components, n_features - 1, n_samples - 1)`.
- **Feature shifting:** All features are shifted forward by one bar via `shift_features` to prevent look-ahead bias.
- **Interpretation:** The latent components are not directly interpretable in terms of specific indicators. They represent abstract combinations of all input features. The reconstruction error is the most directly actionable feature for risk management.
