# Plugin Spec — boruta

**Kind**: feature_selection  •  **Version**: 0.1.0

## Capability

Selects all-relevant features by comparing XGBoost importances against permuted shadow features and keeping those whose average z-score across iterations meets min_z_score.

## Summary

Boruta-style feature selector that, for `n_iter` iterations, concatenates permuted "shadow" copies of X, fits an XGBClassifier on the combined matrix, and computes a per-feature z-score of `(importance - max_shadow_importance) / max(0.1 * max_shadow_importance, 1e-10)`. Averages z-scores across iterations, returns features with `avg_z_score >= min_z_score` sorted descending, optionally truncated to `max_features`. Handles inf/NaN in X (inf→NaN→0), short-circuits on empty column set or single-class y, and uses a seeded numpy Generator when `seed` is provided.

## Inputs

- X: pd.DataFrame of candidate features
- y: np.ndarray of binary target labels
- max_features: optional int cap on the number of selected features
- seed: optional int for reproducible permutations and XGBoost random_state

## Parameters

- `n_iter` (int, default=10): Number of Boruta iterations (shadow-feature comparisons); more iterations = more stable z-score averages.
- `n_estimators` (int, default=50): Number of XGBoost trees per iteration for importance estimation.
- `max_depth` (int, default=4): Maximum XGBoost tree depth per iteration.
- `min_z_score` (float, default=0.5): Minimum average z-score versus shadow features required to accept a feature as relevant.
- `n_jobs` (int, default=1): Number of parallel threads passed to XGBoost.

## Outputs

- selected: List[str] of feature names with avg z-score >= min_z_score, sorted by z-score descending, optionally truncated to max_features
- metadata: dict with keys 'z_scores' (feature -> avg z-score, all originals), 'n_original' (int), 'n_selected' (int)

## Acceptance Criteria

- AC-001: select_features(X, y, ...) returns a (List[str], dict) tuple.
- AC-002: For each of n_iter iterations, shadow features are created by column-wise permutation of X and prefixed with 'shadow_' before an XGBClassifier is fit on the combined matrix.
- AC-003: Per-feature z-score is computed as (importance - shadow_max) / max(shadow_max * 0.1, 1e-10) and averaged across n_iter iterations.
- AC-004: The returned selected list contains exactly the original features whose average z-score is >= min_z_score, sorted by z-score in descending order.
- AC-005: When max_features is a positive integer and len(selected) exceeds it, the list is truncated to the top max_features by z-score.
- AC-006: The returned metadata dict contains 'z_scores' (mapping every original feature to its avg z-score in descending z-score order), 'n_original' equal to len(X.columns), and 'n_selected' equal to len(selected).
- AC-007: Inf values in X are replaced with NaN and all NaN values are filled with 0 before fitting.
- AC-008: When a seed is provided, permutations and XGBoost random_state are derived from numpy.random.default_rng(seed), making runs deterministic.

## Edge Cases

- X has zero columns: returns ([], {}) immediately without fitting any model.
- y contains fewer than 2 unique values: returns ([], {}) immediately.
- max_features is None or <= 0: no cap is applied and all features meeting min_z_score are returned.
- No feature meets min_z_score: selected is [] but metadata still includes z_scores for every original feature plus n_original and n_selected=0.
- shadow_max is 0 (or very small): the denominator is clamped to 1e-10 via max(shadow_max * 0.1, 1e-10), avoiding divide-by-zero.
- X contains inf or NaN values: inf is replaced with NaN and NaN is filled with 0 prior to shadow-feature construction and fitting.

## Assumptions

- y is a binary/multiclass label array suitable for XGBClassifier (the class only guards against len(np.unique(y)) < 2).
- X and y are index-aligned and have matching length.
- The xgboost package is importable in the runtime environment.
