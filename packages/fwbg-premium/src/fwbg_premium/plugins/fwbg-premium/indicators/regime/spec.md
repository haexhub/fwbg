# Plugin Spec — regime

**Kind**: indicator  •  **Version**: 3.0.0

## Capability

Computes market-regime features from close prices — rolling Hurst exponent (R/S), Shannon entropy of log-returns, and Lo–MacKinlay variance ratio — plus their changes and cross-scale divergence.

## Summary

Regime-detection indicator producing Hurst-exponent, Shannon-entropy, and variance-ratio features (with change and divergence variants) across multiple rolling windows, all shifted by one bar to prevent lookahead. Uses `_original_close` when present (frac-diff aware), otherwise `C`.

## Inputs

- df['C'] (close price)
- df['_original_close'] (optional, used instead of C when present for frac-diff compatibility)

## Parameters

- `hurst_windows` (list[int], default=[100, 200, 500]): Rolling window sizes for Hurst-exponent computation.
- `step` (int, default=10): Step size for rolling Hurst/entropy/variance-ratio computation; intermediate NaNs are forward-filled.
- `entropy_windows` (list[int], default=[50, 100]): Rolling window sizes for Shannon entropy of log-returns.
- `vr_windows` (list[int], default=[100, 200]): Rolling window sizes for the variance-ratio test.
- `vr_lags` (list[int], default=[5, 10]): Aggregation lags q for the Lo–MacKinlay variance ratio.

## Outputs

- regime_hurst_100
- regime_hurst_200
- regime_hurst_500
- regime_hurst_100_chg
- regime_hurst_200_chg
- regime_hurst_divergence
- regime_entropy_50
- regime_entropy_100
- regime_entropy_100_chg
- regime_vr_100_5
- regime_vr_100_10
- regime_vr_200_5
- regime_vr_200_10
- regime_vr_deviation

## Acceptance Criteria

- AC-001: Returns the input DataFrame concatenated with one column per requested feature, using the same index as df.
- AC-002: Uses df['_original_close'] when present, otherwise df['C'], as the price series for all computations.
- AC-003: For each window in hurst_windows, produces a regime_hurst_{window} column with values clipped to [0, 1] (defaulting to 0.5 when insufficient data).
- AC-004: Produces regime_hurst_100_chg (24-bar diff) when 100 is in hurst_windows and regime_hurst_200_chg (48-bar diff) when 200 is in hurst_windows.
- AC-005: Produces regime_hurst_divergence = regime_hurst_100 - regime_hurst_500 when both 100 and 500 are in hurst_windows.
- AC-006: For each window in entropy_windows, produces regime_entropy_{window} from Shannon entropy of log-returns with 10 bins.
- AC-007: Produces regime_entropy_100_chg (24-bar diff of regime_entropy_100) when 100 is in entropy_windows.
- AC-008: For each (window, q) in vr_windows × vr_lags, produces regime_vr_{window}_{q} using the Lo–MacKinlay variance ratio.
- AC-009: Produces regime_vr_deviation = regime_vr_100_5 - 1.0 when 100 is in vr_windows and 5 is in vr_lags.
- AC-010: All returned feature columns are shifted by 1 bar via shift_features to prevent lookahead bias.
- AC-011: get_feature_columns() returns the canonical 14-column list corresponding to the default parameters.
- AC-012: get_default_params() returns hurst_windows=[100,200,500], step=10, entropy_windows=[50,100], vr_windows=[100,200], vr_lags=[5,10].

## Edge Cases

- Hurst computation returns 0.5 when the input series is shorter than 2×max_lag or when fewer than 3 valid (lag, R/S) points can be fitted.
- Hurst slope is clipped into [0.0, 1.0].
- EPSILON is added inside log() to avoid log(0) when prices contain zeros; divisions guard against denominators smaller than EPSILON.
- Shannon entropy returns NaN when fewer than n_bins non-NaN return observations are available in the window.
- Variance ratio returns NaN when fewer than 2×q returns are available or when 1-period variance is below EPSILON.
- Rolling helpers forward-fill NaNs between steps (and, for entropy/VR, only when step > 1) so gaps introduced by step > 1 are propagated from the previous filled value.
- The final shift_features call means the first row of every feature column is NaN even before per-window warm-up NaNs.
- Feature columns declared in get_feature_columns() are only actually produced when the corresponding windows/lags remain in the (defaulted) parameters; non-default parameter sets may omit some of them.

## Assumptions

- df is indexed such that pandas .shift(24)/.shift(48) corresponds to the intended bar offsets for change features.
- df['C'] (or df['_original_close']) is strictly non-negative so that log(price + EPSILON) is well-defined.
