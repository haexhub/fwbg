# Plugin Spec — autoencoder_features

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Extracts low-dimensional PCA latent features, per-row reconstruction error, and cumulative explained variance from all numeric non-OHLCV indicator columns.

## Summary

A PCA-based indicator that compresses all numeric feature columns (excluding OHLCV and configurable prefixes) into a bounded number of orthogonal latent components. It also emits a per-row squared reconstruction error (anomaly signal) and a constant cumulative-explained-variance column. Inputs are standardized after Inf→NaN cleanup and column-median imputation; outputs are shifted by one bar to avoid lookahead.

## Inputs

- All numeric (float32/float64/int32/int64) columns in the input DataFrame that are not in {O, H, L, C, V} and do not start with any of the configured exclude_prefixes (default ['ae_']).

## Parameters

- `n_components` (int, default=8): Number of PCA components (latent dimensions) to extract from all numeric indicator features. Each component captures an orthogonal mode of variation. More components preserve more information but increase dimensionality. The reconstruction error feature acts as an anomaly detector regardless of this setting.
- `exclude_prefixes` (list[string], default=['ae_']): Column name prefixes to exclude from PCA input. By default excludes the autoencoder's own output columns (ae_*) to prevent circular dependencies. Add other prefixes to exclude specific indicator groups from the latent representation.

## Outputs

- ae_latent_{i} for i in 0..effective_components-1 (PCA components)
- ae_reconstruction_error (per-row squared reconstruction error in standardized space)
- ae_explained_variance (cumulative explained variance ratio, constant per row)

## Acceptance Criteria

- AC-001: When at least one numeric non-OHLCV, non-excluded feature column exists and effective_components >= 1, the returned DataFrame contains ae_latent_0..ae_latent_{effective_components-1}, ae_reconstruction_error, and ae_explained_variance columns concatenated onto the input.
- AC-002: effective_components is computed as min(n_components, num_feature_cols - 1, num_rows - 1); the number of ae_latent_i columns emitted equals this effective value.
- AC-003: Non-finite values (Inf/-Inf) in the selected feature matrix are replaced with NaN, then NaN entries are imputed with the column's nanmedian (falling back to 0.0 when the median is not finite) before standardization.
- AC-004: Input features are standardized with sklearn StandardScaler prior to PCA fitting.
- AC-005: ae_reconstruction_error equals the row-wise sum of squared differences between the standardized input and its PCA inverse-transform reconstruction.
- AC-006: ae_explained_variance is a constant column equal to the sum of the fitted PCA's explained_variance_ratio_.
- AC-007: All produced feature columns are passed through shift_features(..., df.index), so values are shifted by one bar to prevent lookahead.
- AC-008: Columns named O, H, L, C, V and any column whose name starts with a prefix in exclude_prefixes are never used as PCA inputs.
- AC-009: Only columns with dtype float64, float32, int64, or int32 are eligible as PCA inputs.
- AC-010: get_default_params returns {'n_components': 8, 'exclude_prefixes': ['ae_']}.
- AC-011: get_feature_columns returns the 8 ae_latent_i names plus ae_reconstruction_error and ae_explained_variance, matching the default n_components=8 configuration.

## Edge Cases

- No eligible numeric feature columns after filtering: the original DataFrame is returned unchanged with no ae_* columns added.
- effective_components < 1 (e.g., only one feature column, or a DataFrame with a single row): the original DataFrame is returned unchanged.
- n_components requested exceeds the number of usable features or rows: it is silently clamped to min(n_components, n_features - 1, n_rows - 1).
- A feature column is entirely NaN/Inf, making its nanmedian non-finite: those entries are imputed with 0.0.
- Input contains Inf or -Inf values: they are converted to NaN before median imputation, so PCA never sees non-finite inputs.
- exclude_prefixes is passed as None: it defaults to ['ae_'] inside compute(), preventing the plugin's own outputs from feeding back into PCA input.

## Assumptions

- The fwbg pipeline invokes shift_features correctly so that returned ae_* columns represent information available strictly before the current bar.
- sklearn's PCA and StandardScaler are deterministic for a given standardized input matrix (no external random_state is set here).

## Needs Clarification

- [NEEDS CLARIFICATION: PCA is fit on the entire provided DataFrame in a single call (no rolling/expanding window). Whether callers are expected to pass only historical/training data or the full series is not enforced by this plugin; downstream lookahead safety relies on shift_features.]
