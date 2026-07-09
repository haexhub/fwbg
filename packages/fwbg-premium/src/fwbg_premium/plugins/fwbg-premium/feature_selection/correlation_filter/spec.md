# Plugin Spec — correlation_filter

**Kind**: feature_selection  •  **Version**: 0.1.0

## Capability

Greedily drops features whose absolute Pearson correlation with any already-kept (more important) feature meets or exceeds a threshold.

## Summary

Redundancy filter that iterates features in the order they arrive (assumed pre-sorted by importance from an upstream selector) and keeps a feature only if its absolute pairwise correlation with every already-selected feature is strictly below `max_correlation`. Optionally caps output at `max_features`. Returns the kept feature list plus metadata describing input/selected/dropped counts, the dropped list, and per-dropped-feature reasons naming the correlated retained feature and correlation value.

## Inputs

- X: pd.DataFrame of pre-selected candidate features (columns assumed ordered by upstream importance)
- y: np.ndarray target array (accepted for interface compatibility but unused)
- max_features: optional int hard cap on number of selected features
- max_correlation: float absolute correlation threshold (default 0.7)

## Parameters

- `max_correlation` (float, default=0.7): Maximum absolute pairwise correlation allowed between kept features; higher-correlated features are dropped
- `max_features` (int, default=None): Optional hard cap on the number of features returned; when reached, iteration stops early

## Outputs

- selected: List[str] of retained feature column names, preserving input order
- metadata dict with keys: n_input, n_selected, n_dropped, dropped (List[str]), drop_reasons (dict mapping dropped feature -> 'kept_feature (r=<value>)')

## Acceptance Criteria

- AC-001: When X has 0 or 1 columns, returns all columns unchanged with metadata {'n_dropped': 0, 'dropped': []}.
- AC-002: Iterates features in the order given by X.columns and keeps a feature only if its absolute correlation with every already-kept feature is strictly less than max_correlation.
- AC-003: A feature is dropped as soon as any already-kept feature has |corr| >= max_correlation with it, and drop_reasons[feat] records that kept feature and the rounded correlation value.
- AC-004: When max_features is provided and truthy, iteration halts once len(selected) reaches max_features; remaining unseen features are neither selected nor listed in dropped.
- AC-005: Uses the absolute pairwise correlation matrix computed once via X.corr().abs() (Pearson by default).
- AC-006: The target array y is not used in the selection decision.
- AC-007: Returned metadata includes n_input, n_selected, n_dropped, dropped, and drop_reasons that are internally consistent (n_input == n_selected + n_dropped when max_features is not reached).

## Edge Cases

- X has zero columns: early-return path returns [] with n_dropped=0 and empty dropped list.
- X has exactly one column: returned as-is regardless of threshold.
- max_features is 0 or None/falsy: treated as 'no cap' because the guard `if max_features and ...` short-circuits on falsy values, so max_features=0 does not truncate output.
- max_correlation boundary: correlation values exactly equal to max_correlation cause the feature to be dropped (comparison is `>=`).
- Constant columns produce NaN correlations from X.corr(); NaN comparisons against max_correlation are False, so such features are kept rather than dropped.
- All features perfectly correlated: only the first (most important) feature is retained, the rest are dropped with reasons referencing it.
- Input columns already sorted by importance is assumed; if unsorted, less important features may displace more important ones only in the sense that order determines who is kept first.

## Assumptions

- Input columns X.columns are pre-sorted from most to least important by an upstream selector (e.g., Stability Boruta).
- Correlation is measured with pandas' default Pearson method via DataFrame.corr().

## Needs Clarification

- [NEEDS CLARIFICATION: get_param_schema() declares only max_correlation, but select_features also accepts max_features — whether max_features should also appear in the schema (with min/max/step) is not stated in the source.]
