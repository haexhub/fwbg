# Plugin Spec — distribution

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes rolling skewness, kurtosis, their z-scores and changes, return autocorrelation at multiple lags, and composite tail-risk/stability features from close-price returns.

## Summary

Derives return-distribution features from the close price: rolling skewness and (excess) kurtosis over configurable windows, z-score normalisations against their own history, short/medium change deltas, rolling autocorrelation at multiple lags with a change-vs-20-bars-ago signal, and two composite series (a tail-risk score combining kurtosis and negative skewness, and a skewness-stability rolling std). All feature columns are shifted by one bar to prevent lookahead bias.

## Inputs

- df['C'] (close price series used to derive pct_change returns)

## Parameters

- `windows` (list[int], default=[20, 50, 100]): Rolling window lengths (bars) used to compute skewness and kurtosis (and their z-scores/changes/composites).
- `z_score_lookback` (int, default=200): Lookback (bars) for the rolling mean/std used to z-score-normalise skewness and kurtosis against their own history.
- `compute_changes` (bool, default=True): If True and 50 is in windows, emit skew/kurt change-over-10 and change-over-20 features derived from the 50-bar series.
- `autocorr_lags` (list[int], default=[1, 5, 10, 20]): Lags (bars) at which to compute rolling autocorrelation of returns. If 1 is included, also emits dist_autocorr_1_change (vs. 20 bars ago).
- `autocorr_window` (int, default=100): Rolling window (bars) used when computing the autocorrelation at each lag.

## Outputs

- dist_skew_20
- dist_skew_50
- dist_skew_100
- dist_kurt_20
- dist_kurt_50
- dist_kurt_100
- dist_skew_20_z
- dist_skew_50_z
- dist_skew_100_z
- dist_kurt_20_z
- dist_kurt_50_z
- dist_kurt_100_z
- dist_skew_change_10
- dist_skew_change_20
- dist_kurt_change_10
- dist_kurt_change_20
- dist_tail_risk
- dist_stability
- dist_autocorr_1
- dist_autocorr_5
- dist_autocorr_10
- dist_autocorr_20
- dist_autocorr_1_change

## Acceptance Criteria

- AC-001: Returns the input DataFrame concatenated with distribution feature columns; original columns are preserved.
- AC-002: For each window w in windows, produces dist_skew_{w} = rolling skewness of pct_change(C) and dist_kurt_{w} = rolling excess kurtosis of pct_change(C).
- AC-003: For each window w, produces dist_skew_{w}_z and dist_kurt_{w}_z as (value - rolling_mean) / rolling_std over z_score_lookback bars, using safe_divide.
- AC-004: If compute_changes is True and 50 is in windows, emits dist_skew_change_10, dist_skew_change_20, dist_kurt_change_10, dist_kurt_change_20 computed from the 50-bar skew/kurt series.
- AC-005: For each lag in autocorr_lags, emits dist_autocorr_{lag} as the rolling(autocorr_window) autocorrelation of returns at that lag.
- AC-006: If 1 is in autocorr_lags, emits dist_autocorr_1_change = dist_autocorr_1 - dist_autocorr_1.shift(20).
- AC-007: If 50 is in windows, emits dist_tail_risk = (clip(kurt_50, -3, 10)/10 + clip(-skew_50, 0, 3)/3) / 2, and dist_stability = rolling(50).std() of skew_50.
- AC-008: All emitted feature columns are shifted by one bar via shift_features so no feature at index i depends on data at index > i.
- AC-009: get_feature_columns() lists exactly the feature column names the compute() method produces under default params.
- AC-010: get_default_params() returns windows=[20,50,100], z_score_lookback=200, compute_changes=True, autocorr_lags=[1,5,10,20], autocorr_window=100.

## Edge Cases

- Insufficient history: rolling windows return NaN for the first (window-1) rows of skew/kurt and (z_score_lookback-1) rows of z-scores.
- Zero or near-zero rolling std for skew/kurt over z_score_lookback: safe_divide guards against division-by-zero in the z-score computations.
- Autocorrelation window smaller than lag: the lambda returns np.nan when len(x) <= lag, preventing errors on short windows.
- compute_changes=True but 50 not in windows: change features are silently skipped (dist_skew_change_* / dist_kurt_change_* are not emitted).
- Composite features (dist_tail_risk, dist_stability) require 50 in windows; otherwise they are not emitted.
- dist_autocorr_1_change requires 1 in autocorr_lags; otherwise it is not emitted.
- Constant close prices produce zero returns, yielding NaN skewness/kurtosis (undefined moments) and NaN autocorrelation.

## Assumptions

- Input df contains a 'C' (close) column of numeric prices, indexed compatibly with shift_features(..., df.index).
- pandas rolling().skew() / .kurt() are used, so kurtosis is Fisher/excess kurtosis (0 for a normal distribution).

## Needs Clarification

- [NEEDS CLARIFICATION: get_feature_columns() is hardcoded to the default-parameter output set; whether callers rely on it staying in sync when non-default windows/autocorr_lags are used is unclear from the source.]
