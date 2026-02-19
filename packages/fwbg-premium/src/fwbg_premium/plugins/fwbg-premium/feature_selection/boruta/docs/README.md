# Boruta Feature Selection

All-relevant feature selection algorithm that identifies statistically significant features by comparing their importance against randomized shadow features.

## Concept

Traditional feature selection methods based on raw importance scores suffer from a fundamental problem: they cannot distinguish between genuinely informative features and features that appear important purely by chance. Boruta addresses this by introducing a statistical significance test. It creates "shadow features" -- randomly permuted copies of each original feature that, by construction, have no real relationship with the target -- and uses them as a benchmark.

The algorithm trains an XGBoost classifier on the combined set of original and shadow features. Features whose importance consistently exceeds the best shadow feature across multiple iterations are confirmed as genuinely relevant. This approach is rooted in the Boruta method by Kursa & Rudnicki (2010), adapted here with XGBoost as the base learner instead of Random Forest.

The z-score metric quantifies how much each feature's importance exceeds the shadow maximum, normalized by an estimate of the shadow variance. Features with an average z-score above the configurable threshold are selected. Running multiple iterations and averaging z-scores produces stable results that are robust to random variation in any single run.

## Selection Algorithm

For each of `n_iter` iterations:

1. Create shadow features by independently permuting each original feature column
2. Train an XGBoost classifier on the combined original + shadow feature set
3. Extract feature importances from the trained model
4. Record the maximum importance among all shadow features (`shadow_max`)
5. Compute z-scores for each original feature: `(importance - shadow_max) / (shadow_max * 0.1)`
6. Accumulate z-scores across iterations

After all iterations:

1. Compute average z-score per feature (`z_scores_sum / n_iter`)
2. Select features with average z-score >= `min_z_score`
3. Sort selected features by z-score (highest first)
4. Apply `max_features` cap if configured

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | `int` | `10` | Number of Boruta iterations; more iterations produce more stable results (min: 1, max: 1000) |
| `n_estimators` | `int` | `50` | Number of XGBoost trees per iteration for importance estimation (min: 1, max: 10000) |
| `max_depth` | `int` | `4` | Maximum tree depth per XGBoost iteration (min: 1, max: 50) |
| `min_z_score` | `float` | `0.5` | Minimum average z-score for a feature to be accepted as relevant (min: -10.0, max: 100.0) |
| `n_jobs` | `int` | `1` | Number of parallel threads for XGBoost training (min: 1, max: 128) |
| `max_features` | `int` | `None` | Optional hard cap on the number of selected features |

## Usage Notes

- Input data is automatically cleaned: `inf`/`-inf` values are replaced with `NaN`, and `NaN` is filled with `0`.
- Each XGBoost iteration uses a different random state for diversity.
- The shadow standard deviation estimate uses `shadow_max * 0.1` (clamped to a minimum of `1e-10`) as a proxy, since a single max value has no variance of its own.
- Returns metadata including per-feature z-scores, original feature count, and selected count.
- Empty input DataFrames are handled gracefully, returning an empty selection.
- This selector is commonly used as the inner selector for the Stability Selection plugin.
