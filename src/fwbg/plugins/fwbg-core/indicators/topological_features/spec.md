# Plugin Spec — topological_features

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes rolling Topological Data Analysis (TDA) features from persistent homology of Takens time-delay embeddings of log-return series.

## Summary

Applies Takens time-delay embedding to windowed log returns of close prices, then uses ripser persistent homology to extract H0/H1 topological features (counts, max/mean persistence, entropy, Wasserstein amplitude, and ratio features) per configured rolling window.

## Inputs

- df['C'] (close price series)

## Parameters

- `windows` (list[int], default=[50, 100]): List of rolling window sizes (in bars) over which TDA features are computed; one set of feature columns is emitted per window.
- `embedding_dim` (int, default=3): Dimension of the Takens time-delay embedding space used to build the point cloud from log returns.
- `time_delay` (int, default=1): Time delay (in bars) between successive coordinates of the Takens embedding.
- `maxdim` (int, default=1): Maximum homology dimension computed by ripser (0 = connected components, 1 = loops).

## Outputs

- tda_h0_count_{w}
- tda_h1_count_{w}
- tda_h0_max_pers_{w}
- tda_h1_max_pers_{w}
- tda_h0_mean_pers_{w}
- tda_h1_mean_pers_{w}
- tda_persistence_entropy_{w}
- tda_wasserstein_amp_{w}
- tda_h1_ratio_{w}
- tda_max_loop_persistence_{w}

## Acceptance Criteria

- AC-001: Registers under the name 'topological_features' via @register_indicator and exposes class TopologicalFeaturesIndicator with version '1.0.0'.
- AC-002: compute() consumes df['C'] and returns the original DataFrame concatenated with TDA feature columns, one group per window in `windows`.
- AC-003: For each window w, produces exactly these 10 columns: tda_h0_count_{w}, tda_h1_count_{w}, tda_h0_max_pers_{w}, tda_h1_max_pers_{w}, tda_h0_mean_pers_{w}, tda_h1_mean_pers_{w}, tda_persistence_entropy_{w}, tda_wasserstein_amp_{w}, tda_h1_ratio_{w}, tda_max_loop_persistence_{w}.
- AC-004: Internally converts close prices to log returns via np.diff(np.log(close), prepend=np.log(close[0])) before embedding.
- AC-005: Uses Takens time-delay embedding with the configured embedding_dim and time_delay to build point clouds from each rolling window of log returns.
- AC-006: Calls ripser.ripser(cloud, maxdim=maxdim) on each window's point cloud to obtain persistence diagrams.
- AC-007: Extracts only finite persistence values (death - birth where death is finite) for statistics.
- AC-008: h0_count and h1_count are counts of finite-persistence points in the H0 and H1 diagrams respectively (0.0 when a diagram is absent).
- AC-009: h{0,1}_max_pers and h{0,1}_mean_pers are max and mean of finite persistences (0.0 when none exist).
- AC-010: persistence_entropy is -sum(p_i * log(p_i)) over normalized finite persistences from H0 and H1 combined (0.0 when total is non-positive or empty).
- AC-011: wasserstein_amp is sqrt(sum(persistence^2)) over combined finite H0+H1 persistences (0.0 when empty).
- AC-012: h1_ratio = safe_divide(h1_count, h0_count) and max_loop_persistence = safe_divide(h1_max_pers, h0_max_pers), computed after the rolling loop.
- AC-013: All feature columns are shifted by one bar via shift_features(...) before being returned, so no lookahead bias is introduced.
- AC-014: get_feature_columns() returns the cached column list set during compute(); when uncalled, returns the sorted default-parameter column list.
- AC-015: get_default_params() returns {'windows': [50, 100], 'embedding_dim': 3, 'time_delay': 1, 'maxdim': 1}.

## Edge Cases

- Windows with fewer than 3 embedded points (n_points = window - (embedding_dim - 1) * time_delay <= 2) are skipped and leave NaN feature values at that position.
- If ripser.ripser raises for a given window, the failure is caught and NaNs are left for that bar's features.
- First (window - 1) bars per window size have NaN feature values because no full window is available.
- Persistence diagrams with no finite points (all deaths infinite or empty diagram) yield 0.0 for count/max/mean/entropy/wasserstein features.
- h1_ratio and max_loop_persistence rely on safe_divide, so a zero H0 count/persistence does not produce a division-by-zero error.
- _takens_embedding returns None when the input segment is too short for the requested embedding_dim/time_delay, in which case the window is skipped.
- compute() assumes df['C'] contains no non-positive or NaN values (uses np.log on close prices directly without guarding).

## Assumptions

- The DataFrame passed to compute() contains a numeric close column named 'C'.
- fwbg_sdk provides BaseIndicator, shift_features, safe_divide, and register_indicator with the semantics used elsewhere in the corpus (feature shift-by-one for no-lookahead, safe_divide for zero-safe division).
- ripser is available at import time; no lazy import is used.
