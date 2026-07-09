# Plugin Spec — stability

**Kind**: feature_selection  •  **Version**: 0.1.0

## Capability

Selects features that an inner selector chooses in at least a threshold fraction of bootstrap resampling runs.

## Summary

Stability selection wrapper that runs a configurable inner feature selector on repeated bootstrap resamples of the training data and keeps features whose selection frequency across runs meets or exceeds a threshold, optionally capping the result at max_features and ordering by vote count.

## Inputs

- X: pandas DataFrame of candidate features
- y: numpy array of target labels aligned with X
- max_features: optional cap on the number of features returned
- inner_selector: registered feature selector name used on each bootstrap sample
- inner_params: parameter dict forwarded to the inner selector

## Parameters

- `inner_selector` (string, default='boruta'): Name of the inner feature selector to run on each bootstrap sample
- `inner_params` (string, default='{"n_iter": 5, "n_estimators": 30, "min_z_score": 0.5}'): Parameter dict passed to the inner selector on each bootstrap run (source default: {'n_iter': 5, 'n_estimators': 30, 'min_z_score': 0.5}; declared as type 'string' in get_param_schema)
- `n_bootstrap` (int, default=10): Number of bootstrap resampling iterations
- `threshold` (float, default=0.6): Minimum fraction of bootstrap runs a feature must be selected in to be kept
- `bootstrap_ratio` (float, default=0.8): Fraction of samples drawn (with replacement) per bootstrap iteration
- `seed` (int, default=None): Optional RNG seed for reproducible bootstrap resampling (None = nondeterministic)

## Outputs

- List of stable feature names selected in >= threshold fraction of bootstrap runs, ordered by vote count descending
- Metadata dict with feature_votes, n_bootstrap, threshold, n_selected

## Acceptance Criteria

- AC-001: Runs the inner selector (resolved via get_feature_selector(inner_selector)) n_bootstrap times, each time on a bootstrap sample of size int(len(X) * bootstrap_ratio) drawn with replacement from X and y
- AC-002: Only features whose vote count across bootstrap runs is >= threshold * n_bootstrap are returned
- AC-003: Returned feature list is sorted by descending vote count (most stable first)
- AC-004: If max_features is provided and non-zero and len(stable) exceeds it, the returned list is truncated to the top max_features
- AC-005: Bootstrap iterations where the resampled y contains fewer than 2 unique classes are skipped without voting
- AC-006: When seed is provided, resampling is reproducible: two calls with the same seed yield identical bootstrap draws (and, when the inner selector honors 'seed', identical selections)
- AC-007: When seed is not None and inner_params does not already set 'seed', a per-iteration seed derived from the RNG is injected into the inner selector's params via setdefault
- AC-008: Metadata dict returned alongside the selection contains feature_votes, n_bootstrap, threshold, and n_selected

## Edge Cases

- All bootstrap iterations skipped due to single-class y_boot: feature_votes stays empty and an empty stable list is returned with n_selected=0
- No feature reaches the threshold: returns an empty list with populated feature_votes metadata
- max_features is None or 0 (falsy): no truncation is applied regardless of stable length
- inner_params is None: treated as an empty dict when constructing per-iteration params
- seed is None: bootstrap draws are nondeterministic and no seed is injected into inner_params
- inner selector returns None or an empty list for a given bootstrap: that iteration contributes no votes

## Assumptions

- The inner selector name resolves via fwbg.core.get_feature_selector and its class exposes a select_features(X, y, **params) method returning (selected_features, metadata)
- y supports numpy fancy indexing with an integer index array (numpy array or equivalent)
- X is a pandas DataFrame supporting iloc-based row indexing

## Needs Clarification

- [NEEDS CLARIFICATION: get_param_schema declares inner_params with type 'string' while the actual default is a dict — likely a source bug; preserved verbatim in this spec (default encoded as a JSON string here to satisfy the spec schema)]
