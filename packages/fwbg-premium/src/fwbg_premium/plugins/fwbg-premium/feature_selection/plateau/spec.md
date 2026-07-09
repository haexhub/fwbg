# Plugin Spec — plateau

**Kind**: feature_selection  •  **Version**: 0.1.0

## Capability

Selects features by ranking XGBoost importances with a plateau-stability bonus derived from same-family parameter-neighbor features (e.g. rsi_14 vs rsi_12/rsi_16).

## Summary

Trains an XGBoost classifier on X/y to obtain feature importances, filters out features below min_importance, then computes a plateau score per feature that boosts importance by neighbor stability (low CV across parameter-neighbor features found via numeric-suffix regex like _14, _24h, _5d, _20_) and a plateau factor penalizing outlier-vs-neighbor importance. Features are returned sorted by plateau_score (or raw importance if prefer_plateau=False), optionally capped at max_features. Falls back to top-N-by-importance if no feature clears min_importance.

## Inputs

- X: pandas DataFrame of feature columns (numeric; inf and NaN are replaced with 0 internally)
- y: numpy array of binary targets (0/1) for XGBClassifier fitting
- max_features: optional int cap on returned feature count (None = no cap)

## Parameters

- `n_estimators` (int, default=100): Number of XGBoost trees for computing feature importances
- `max_depth` (int, default=5): Maximum tree depth for the XGBoost importance model
- `min_importance` (float, default=0.01): Minimum feature importance threshold; features below this are excluded before plateau scoring
- `min_neighbors` (int, default=1): Minimum number of parameter-neighbor features required for plateau bonus (otherwise the feature gets a fixed 0.8x penalty)
- `prefer_plateau` (bool, default=True): Sort by plateau score instead of raw importance (recommended for robustness)
- `n_jobs` (int, default=1): Number of parallel threads for XGBoost training

## Outputs

- selected: list of feature column names sorted by plateau_score (or importance if prefer_plateau=False)
- metadata dict with keys: importances (all XGBoost importances), plateau_scores (per filtered feature), plateau_features (features flagged is_plateau=True), n_original, n_with_neighbors, n_selected, method ('plateau' | 'importance' | 'importance_fallback')

## Acceptance Criteria

- AC-001: Returns ([], {}) when X has no columns
- AC-002: Replaces +/-inf with NaN and fills NaN with 0 in X before fitting XGBClassifier
- AC-003: Fits XGBClassifier(n_estimators, max_depth, n_jobs, random_state=42, verbosity=0) on X, y and extracts feature_importances_
- AC-004: Excludes features whose XGBoost importance is below min_importance from plateau scoring
- AC-005: When no feature clears min_importance, falls back to top max_features (or top 10 if max_features is None) by raw importance and sets metadata.method='importance_fallback'
- AC-006: For each filtered feature, finds parameter-neighbors via find_feature_neighbors using numeric-suffix regexes (_(digits)h, _(digits)d, _(digits)_, _(digits)$), matching only the first pattern hit
- AC-007: When a feature has at least min_neighbors neighbor importances, plateau_score = importance * (0.6 + 0.25*stability + 0.15*plateau_factor) where stability=1/(1+cv of neighbors) and plateau_factor=1/(1+0.5*|imp-neighbor_mean|/neighbor_mean)
- AC-008: When a feature has fewer than min_neighbors neighbors, plateau_score = importance * 0.8 and is_plateau=False
- AC-009: Flags is_plateau=True only when stability>0.5 and plateau_factor>0.6
- AC-010: Sorts filtered features by plateau_score descending when prefer_plateau=True, otherwise by raw importance descending
- AC-011: Truncates the sorted list to max_features when max_features is a positive int
- AC-012: Returns metadata containing importances, plateau_scores, plateau_features, n_original, n_with_neighbors, n_selected, and method ('plateau' | 'importance' | 'importance_fallback')

## Edge Cases

- Empty X (no columns) returns ([], {}) without fitting the model
- X containing +/-inf or NaN values is sanitized to 0 before fitting
- No feature meets min_importance -> importance_fallback path returns top-N by raw importance
- Feature name has no numeric suffix matching any pattern -> find_feature_neighbors returns [] and the feature gets the no-neighbor 0.8x penalty
- Neighbor names generated from deltas are only kept when they exist in all_features and are not the feature itself; negative or zero suffix values are skipped
- Only the first matching regex pattern is applied per feature name (order: _Nh, _Nd, _N_, _N$)
- max_features=None or 0 -> no truncation of the sorted list
- Neighbor mean of zero is guarded by a 1e-10 epsilon in stability and plateau_factor divisions

## Assumptions

- y is a binary (0/1) array compatible with XGBClassifier
- X columns are numeric; non-numeric columns are not handled explicitly
- Feature naming follows the numeric-suffix conventions the regex patterns target (e.g. rsi_14, chg_24h, sma_20_slope); features outside these conventions get no neighbors
