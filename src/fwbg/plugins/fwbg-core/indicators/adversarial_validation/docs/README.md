# Adversarial Validation

Detects distribution shift between recent and older market data by training a classifier to distinguish "old" from "new" observations within a sliding window, producing drift scores and stability signals.

## Concept

Market regimes change over time -- volatility shifts, correlations break down, and the statistical relationships that indicator features capture can drift or break entirely. Adversarial validation quantifies this distribution shift by asking a simple question: "Can a classifier tell apart the recent data from the older data?" If it can (high AUC), the distributions have shifted and the market regime has changed. If it cannot (AUC near 0.5), the data distribution is stable.

The plugin splits each rolling window into an "old" half and a "new" half, then trains a Logistic Regression classifier to distinguish between them using all available numeric indicator features. The classifier's AUC score directly measures how distinguishable the two halves are. The model coefficients reveal which features have shifted most, captured by the maximum feature importance metric.

This gives ML models a powerful meta-signal for adaptive risk management. When the drift score is high, the model knows that the patterns it learned from historical data may no longer apply -- it can reduce position sizing, widen stops, or abstain from trading. The drift acceleration feature captures the speed of regime change, enabling early detection of structural breaks before they fully manifest. The stability score provides the complementary signal, directly usable as a confidence multiplier for position sizing.

## Features

Features are generated for each configured window size `w`. With default parameters (`windows=[100, 200]`), each window produces 5 features.

| Feature | Description |
|---------|-------------|
| `adv_auc_{w}` | AUC score from the adversarial classifier over window `w`. Values near 0.5 indicate no detectable distribution shift (stable regime). Values approaching 1.0 indicate the old and new halves are easily distinguishable (strong regime change). |
| `adv_drift_score_{w}` | Normalized drift score: `clip(2 * (AUC - 0.5), 0, 1)`. Maps the AUC to a 0-1 scale where 0 = no drift and 1 = maximum drift. Easier to interpret and use directly as a risk signal. |
| `adv_stability_{w}` | Stability score: `1 - drift_score`. Inverse of the drift score. Values near 1.0 indicate a stable, predictable regime; values near 0.0 indicate an unstable, shifting regime. Can be used directly as a confidence multiplier. |
| `adv_max_feature_importance_{w}` | Maximum absolute coefficient from the Logistic Regression model. Indicates how strongly the single most-shifted feature contributes to distinguishing old from new data. High values suggest a concentrated shift in one feature dimension. |
| `adv_drift_acceleration_{w}` | First difference (change) of the drift score over consecutive computation steps. Positive values indicate increasing drift (regime is changing faster); negative values indicate stabilization. Useful for early warning of regime transitions. |

**Default feature count:** 5 features per window x 2 windows = **10 features**.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windows` | list[int] | `[100, 200]` | Rolling window sizes for the adversarial validation test. Each window is split into equal old/new halves. Larger windows provide more training data for the classifier (more stable AUC estimates) but detect slower regime changes. Smaller windows are more sensitive to rapid shifts but noisier. |
| `step` | int | `10` | Step size between consecutive classifier evaluations. The classifier is trained every `step` bars and results are forward-filled between evaluations. Larger steps improve performance at the cost of temporal resolution. |
| `max_features` | int | `30` | Maximum number of input features to use per classifier evaluation. When more features are available, a random subset of `max_features` columns is sampled (with a fixed random seed for reproducibility). This controls computational cost and prevents overfitting the adversarial classifier. |
| `exclude_prefixes` | list[string] | `["adv_"]` | Column name prefixes to exclude from adversarial validation input. By default excludes the plugin's own output columns (`adv_*`) to prevent circular dependencies. |

## Usage Notes

- **Input columns:** Automatically selects all numeric columns except OHLCV (`O`, `H`, `L`, `C`, `V`) and columns matching `exclude_prefixes`. Requires at least 3 numeric feature columns to run; returns the input DataFrame unchanged if fewer are available.
- **Plugin ordering:** This is a meta-indicator that operates on other indicators' outputs. It should run **after** the indicators whose drift it needs to detect. Typically one of the last plugins in the indicator pipeline.
- **Computational cost:** Trains a Logistic Regression classifier every `step` bars for each window size. With default parameters (`step=10`), this is manageable but still heavier than pure statistical indicators. Increase `step` for faster execution on large datasets.
- **Stationarity:** Does not require pre-stationarized input (`benefits_from_stationary: false`).
- **NaN warmup period:** The first `window` bars for each window size will have NaN values, as a full window is needed before the first classifier can be trained.
- **Forward-filling:** Between computation steps, AUC and importance values are forward-filled. This means features change in discrete jumps every `step` bars rather than continuously.
- **Feature shifting:** All features are shifted forward by one bar via `shift_features` to prevent look-ahead bias.
- **Handling of non-finite values:** Infinite values are replaced with NaN, rows with any NaN are removed before classifier training, and remaining NaN values are imputed with column medians. This provides robustness against upstream indicators that occasionally produce non-finite values.
- **Random seed:** Feature subsampling uses a fixed random seed (`42`) for reproducibility across runs.
- **Minimum sample size:** The adversarial classifier requires at least 10 valid observations and both classes present. If these conditions are not met for a window position, AUC defaults to 0.5 (no drift detected).
