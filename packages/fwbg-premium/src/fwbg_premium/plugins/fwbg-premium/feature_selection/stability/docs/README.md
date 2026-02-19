# Stability Feature Selection

Meta-selector that runs an inner feature selector multiple times with bootstrap resampling, keeping only features that are consistently selected across runs.

## Concept

A single feature selection run on a fixed dataset can be sensitive to the specific composition of that dataset. Features selected on one sample may not be selected on a slightly different sample, indicating that their apparent importance is fragile and unlikely to generalize. Stability Selection, introduced by Meinshausen & Buhlmann (2010), addresses this by running the selection procedure multiple times on bootstrap resamples and retaining only features that are selected in a sufficient fraction of runs.

The intuition is simple: if a feature is genuinely important, it will be selected regardless of which subset of data is used for evaluation. Features that appear in 80% or 90% of bootstrap runs are almost certainly capturing a real signal. Features that appear in only 20% of runs are likely artifacts of the specific data sample and should be discarded.

This plugin wraps any inner feature selector (defaulting to Boruta) and provides this stability envelope. The result is a highly robust feature set that is resistant to data perturbation and significantly reduces the risk of overfitting in the downstream model.

## Selection Algorithm

1. For each of `n_bootstrap` iterations:
   a. Draw a bootstrap sample of `bootstrap_ratio` fraction of the data (with replacement)
   b. Instantiate the configured inner selector (e.g., Boruta)
   c. Run the inner selector on the bootstrap sample with the provided `inner_params`
   d. Record which features were selected (vote count)
2. After all iterations, compute each feature's selection frequency
3. Keep features selected in >= `threshold` fraction of runs
4. Sort kept features by vote count (most stable first)
5. Apply `max_features` cap if configured

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inner_selector` | `string` | `"boruta"` | Name of the inner feature selector to run on each bootstrap sample |
| `inner_params` | `dict` | `{"n_iter": 5, "n_estimators": 30, "min_z_score": 0.5}` | Parameter dict passed to the inner selector on each bootstrap run |
| `n_bootstrap` | `int` | `10` | Number of bootstrap resampling iterations (min: 1, max: 1000) |
| `threshold` | `float` | `0.6` | Minimum fraction of bootstrap runs a feature must be selected in to be kept (min: 0.0, max: 1.0, step: 0.05) |
| `bootstrap_ratio` | `float` | `0.8` | Fraction of samples drawn (with replacement) per bootstrap iteration (min: 0.1, max: 1.0, step: 0.05) |
| `max_features` | `int` | `None` | Optional hard cap on the number of selected features |

## Usage Notes

- The inner selector is resolved dynamically via `get_feature_selector()`, so any registered feature selector plugin can be used as the inner selector.
- Default inner params are tuned for speed (`n_iter: 5`, `n_estimators: 30`) since Boruta runs `n_bootstrap` times. Total XGBoost fits = `n_bootstrap * n_iter` = 50 by default.
- Higher `threshold` values (e.g., `0.8` or `0.9`) produce smaller, more robust feature sets. Lower values (e.g., `0.5`) are more permissive.
- Bootstrap sampling is done with replacement, meaning some samples may appear multiple times in a single bootstrap iteration while others are left out.
- Metadata includes the full `feature_votes` dictionary (feature name to vote count), which is useful for analyzing how close borderline features were to the threshold.
- A common pipeline is `stability(boruta) -> correlation_filter` to first find robust features, then remove redundant ones.
