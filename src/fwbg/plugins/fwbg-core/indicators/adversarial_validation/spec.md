# Plugin Spec — adversarial_validation

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Detects distribution shift by training a LogisticRegression to separate the older vs newer half of each sliding window and emitting AUC-derived drift, stability, importance, and acceleration features.

## Summary

Adversarial-validation regime-change indicator: for each configured window, fits a scaled LogisticRegression to distinguish the older half of the window from the newer half, then emits per-window adv_auc, adv_drift_score (2*(auc-0.5) clipped to [0,1]), adv_stability (1-drift), adv_max_feature_importance (max |coef|), and adv_drift_acceleration (diff of drift). Forward-fills between step indices, preserves NaN during warmup, and shifts all features by one bar to avoid lookahead.

## Inputs

- OHLCV DataFrame with additional numeric feature columns (non-OHLCV, not starting with any string in exclude_prefixes) that provide the covariates fed to the adversarial classifier

## Parameters

- `windows` (list[int], default=[100, 200]): Sliding window sizes; each window is split in half into an old and new slice compared by the classifier. One family of five output columns is emitted per window.
- `step` (int, default=10): Row stride between recomputations of the adversarial classifier; intermediate rows are forward-filled from the last computed step.
- `max_features` (int, default=30): Maximum number of feature columns fed to the classifier per window; if more numeric features exist, a deterministic random subset (seed 42) of this size is selected.
- `exclude_prefixes` (list[string], default=['adv_']): Column-name prefixes to exclude from the feature set (in addition to OHLCV), used to prevent the indicator's own outputs from being fed back in.

## Outputs

- adv_auc_{w}: raw ROC AUC of the old-vs-new classifier at each computed step, forward-filled between steps
- adv_drift_score_{w}: clip(2*(auc-0.5), 0, 1) — normalized distribution-shift score in [0,1]
- adv_stability_{w}: 1 - adv_drift_score_{w}
- adv_max_feature_importance_{w}: max absolute LogisticRegression coefficient at each step
- adv_drift_acceleration_{w}: first difference of adv_drift_score_{w}

## Acceptance Criteria

- AC-001: Feature-column selection keeps only numeric dtype columns and excludes the OHLCV set {O,H,L,C,V} as well as any column whose name starts with a string in exclude_prefixes.
- AC-002: If fewer than 3 usable feature columns are found, the input DataFrame is returned unchanged with no adv_* columns added.
- AC-003: For each window w in windows, five columns are produced: adv_auc_{w}, adv_drift_score_{w}, adv_stability_{w}, adv_max_feature_importance_{w}, adv_drift_acceleration_{w}.
- AC-004: Within each window, the classifier is trained on rows [i-w : i-w/2] labeled 0 and [i-w/2 : i] labeled 1, evaluated at row indices i in range(window, n_rows, step).
- AC-005: Rows with non-finite values (Inf/-Inf treated as NaN) are dropped before training; remaining NaNs are imputed with column medians (0.0 if the median itself is non-finite).
- AC-006: Features are standardized via StandardScaler before fitting LogisticRegression(max_iter=100, solver='liblinear', C=1.0); AUC is computed from decision_function scores and max_importance from max(|coef_[0]|).
- AC-007: When there are fewer than 10 valid rows, only one class remains after cleaning, or the classifier fit raises, the step returns auc=0.5 and importance=0.0.
- AC-008: When the number of numeric feature columns exceeds max_features, a NumPy RandomState(42) reproducibly subsamples max_features columns for that window's computations.
- AC-009: adv_drift_score_{w} equals clip(2*(auc-0.5), 0, 1); adv_stability_{w} equals 1 - adv_drift_score_{w}; adv_drift_acceleration_{w} equals the first difference of adv_drift_score_{w}.
- AC-010: AUC and importance values are forward-filled between step points; rows before the first computed step (warmup) are set to NaN across all five columns for that window.
- AC-011: All produced feature columns are passed through shift_features(features, df.index) so values at row t reflect information available strictly before t (no lookahead).
- AC-012: get_feature_columns() returns the columns produced by the last compute() call that successfully added features. If compute() exits early (fewer than 3 usable feature columns found, AC-002), _feature_columns is not reset, so the cached value from the prior successful run is preserved and returned. When compute() has never successfully added features, the default-parameter column list is returned instead.

## Edge Cases

- DataFrame has fewer than 3 non-OHLCV numeric feature columns after applying exclude_prefixes — compute() returns df unchanged and adds no columns.
- n_rows <= min(windows) — no step index i satisfies i in range(window, n_rows, step), so every output column for that window is entirely NaN.
- All-NaN or all-Inf feature slice within a window — after Inf→NaN conversion and NaN-row removal, fewer than 10 valid rows remain and the step yields auc=0.5, importance=0.0.
- Constant/degenerate slice where only one class label survives cleaning — step yields auc=0.5, importance=0.0.
- Column with an all-NaN median — imputation falls back to 0.0 for that column instead of propagating NaN.
- n_cols > max_features — a seed-42 random subset of columns is used per window, giving reproducible but window-agnostic subsampling.
- step > 1 — intermediate rows carry forward-filled values from the last computed step, so consecutive outputs are piecewise-constant except at step boundaries.
- Warmup region before the first computed step index — all five output columns for that window remain NaN and are not forward-filled backward.
- LogisticRegression fit raises (e.g., pathological scaled data) — the try/except returns auc=0.5, importance=0.0 for that step instead of propagating the exception.

## Assumptions

- The registered SDK helper shift_features shifts all supplied feature columns by one bar to enforce the no-lookahead invariant required of indicators.
- scikit-learn (LogisticRegression, StandardScaler, roc_auc_score) is available at runtime; it is imported lazily inside _compute_adversarial_auc.
