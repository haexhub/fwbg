# Regime Cluster

Computes a composite regime score from orthogonal market structure inputs and assigns quantile-based cluster labels (0/1/2) for use in regime-filtered trading strategies.

## Concept

Markets alternate between distinct phases -- trending, choppy, volatile, calm -- and a strategy that works well in one regime may fail catastrophically in another. The regime cluster plugin synthesizes multiple independent market structure indicators into a single composite score that quantifies how favorable current conditions are for directional trading. A high score indicates a trending, persistent, low-entropy environment; a low score indicates a choppy, random, or mean-reverting environment.

The composite score is built from orthogonal inputs: the Hurst exponent (persistence vs. mean-reversion), Shannon entropy (predictability), variance ratio (momentum vs. random walk), volatility rank, and Hurst divergence (regime-shift signal). Each input is z-scored over a rolling window to normalize its scale, then the inputs are equally weighted and averaged. This approach avoids double-counting correlated signals while remaining robust to the absence of individual inputs.

The continuous score is then discretized into cluster labels via rolling quantiles. By default, three regimes are created: unfavorable (cluster 0, lower third), neutral (cluster 1, middle third), and favorable (cluster 2, upper third). These labels can be used directly in the framework's bitmask-based `regime_filter_grid` to selectively enable or disable trading during specific market phases. The score change feature (`regime_cluster_score_chg`) captures regime momentum -- whether conditions are improving or deteriorating over the last 24 bars.

## Features

| Feature | Description |
|---------|-------------|
| `regime_cluster_score` | Composite regime score: equally-weighted average of z-scored inputs. Higher = more favorable for directional trading |
| `regime_cluster_label` | Quantile-based cluster label: 0 = unfavorable, 1 = neutral, 2 = favorable (with default 3 regimes) |
| `regime_cluster_score_chg` | Score change over 24 bars. Positive = conditions improving, negative = deteriorating |
| `regime_cluster_n_inputs` | Number of input signals that were available for score computation. Useful for confidence assessment |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zscore_window` | int | 200 | Rolling window for z-scoring each input signal before averaging |
| `quantile_window` | int | 500 | Rolling window for computing quantile thresholds that define cluster boundaries |
| `n_regimes` | int | 3 | Number of regime clusters (quantile bins). 3 produces labels 0/1/2 |

## Usage Notes

- This plugin depends on the `regime` and `volatility` plugins. Their features must be computed first (columns like `regime_hurst_200`, `regime_entropy_100`, `regime_vr_200_5`, `vol_atr_pct_14_rank`, `regime_hurst_divergence` must be present in the DataFrame).
- The optional input `regime_risk_composite` is used when available but not required.
- All features are shifted by 1 bar to prevent lookahead bias.
- The `manifest.json` sets `benefits_from_stationary: false` since the plugin operates on already-normalized inputs.
- The `quantile_window` should be large enough to capture a representative distribution of scores. A window of 500 bars (~21 trading days of H1 data) provides stable quantile estimates.
- If no input columns are found in the DataFrame, all output features are filled with NaN and `regime_cluster_n_inputs` is set to 0.
- Core inputs and their sign conventions:
  - `regime_hurst_200` (+1): Higher Hurst = more persistent/trending
  - `regime_entropy_100` (-1): Lower entropy = more predictable (sign flipped)
  - `regime_vr_200_5` (+1): Centered at 0 (VR - 1.0), positive = momentum
  - `vol_atr_pct_14_rank` (+1): Higher volatility rank = more opportunity
  - `regime_hurst_divergence` (+1): Short-term vs. long-term Hurst divergence signals regime shifts
