# Plugin Spec — fractal_dimension

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Computes rolling Higuchi Fractal Dimension of the close price with derived change, complexity-deviation, and regime features across multiple window sizes.

## Summary

Rolling Higuchi Fractal Dimension (HFD) indicator that measures price series complexity/roughness. For each configured window size, it emits four features: the raw HFD (1.0=smooth/trending, 2.0=random/complex), the change in HFD over one window length (regime-transition signal), a complexity-deviation ratio equal to 2*|FD-1.5|, and a discrete regime label in {-1 trending, 0 random, 1 mean-reverting} using a dead band around FD=1.5 whose half-width is regime_z standard deviations of the Higuchi estimator on a random walk (std ~ 0.486*k_max^0.27/sqrt(window), calibrated by simulation). All features are shifted by one bar to avoid lookahead.

## Inputs

- df["C"] (close price column)

## Parameters

- `regime_z` (float, default=1.96): Half-width of the random dead band in estimator standard deviations; a pure random walk is labelled 0 in about 95% of windows.
- `windows` (list[int], default=[50, 100, 200]): Rolling window sizes over which the Higuchi Fractal Dimension is computed. Shorter windows capture local regime transitions; longer windows give a more stable complexity estimate.
- `k_max` (int, default=10): Maximum interval parameter for the Higuchi algorithm; controls the number of sub-series scales used to estimate the fractal dimension. Must be less than the smallest window size.

## Outputs

- fd_higuchi_{w} — raw Higuchi Fractal Dimension over rolling window w
- fd_higuchi_change_{w} — difference of fd_higuchi_{w} versus w bars prior
- fd_complexity_ratio_{w} — 2 * |fd_higuchi_{w} - 1.5|, i.e. deviation from the random-walk value
- fd_regime_{w} — discrete regime in {-1 trending, 0 random, 1 mean-reverting} (also exposed as a signal column)

## Acceptance Criteria

- AC-001: For each window w in windows, the output DataFrame contains exactly four new columns: fd_higuchi_{w}, fd_higuchi_change_{w}, fd_complexity_ratio_{w}, and fd_regime_{w}.
- AC-002: All emitted feature columns are shifted by one bar via shift_features so that row i never depends on data from row i+1 onward (no lookahead).
- AC-003: fd_higuchi_{w} equals the Higuchi Fractal Dimension computed on the trailing window of size w of df["C"], with NaN for the first w-1 rows (warmup).
- AC-004: fd_higuchi_change_{w} equals fd_higuchi_{w} minus its value w bars earlier (pandas Series.diff(w)).
- AC-005: fd_complexity_ratio_{w} equals 2.0 * |fd_higuchi_{w} - 1.5|.
- AC-006: with band = regime_z * rw_fd_std(w, k_max), fd_regime_{w} equals -1.0 when fd_higuchi_{w} < 1.5 - band, 0.0 when 1.5 - band <= fd_higuchi_{w} <= 1.5 + band, 1.0 when fd_higuchi_{w} > 1.5 + band, and NaN when fd_higuchi_{w} is NaN. For a Gaussian random walk at w=50, k_max=10 the band is about +-0.25 and fewer than ~10% of windows leave it (a fixed 1.4..1.6 band flagged ~45%).
- AC-007: get_signal_columns() returns the list of fd_regime_{w} columns, one per configured window. get_feature_columns(params) and get_signal_columns(params) honour a `windows` entry in params and fall back to the defaults.
- AC-008: The Higuchi computation uses the algorithm: for k in 1..k_max, build k sub-series indexed by strides of k, compute mean sub-series length L(k), then estimate D as the least-squares slope of log(L(k)) versus log(1/k).

## Edge Cases

- Segment length shorter than k_max + 1: _higuchi_fd returns NaN, so the corresponding fd_higuchi_{w} entry is NaN.
- Warmup period: the first w-1 rows of fd_higuchi_{w} (and thus its derived columns) are NaN.
- fd_higuchi_change_{w} produces NaN for the first w non-NaN observations (pandas diff(w) behaviour) and wherever either endpoint of the diff is NaN.
- fd_regime_{w} is NaN wherever fd_higuchi_{w} is NaN (guarded by an explicit valid = ~isnan(fd) mask).
- Degenerate least-squares fit (fewer than 2 positive L(k) values, or a near-zero denominator in the slope formula) causes _higuchi_fd to return NaN rather than raising.
- k_max is internally clamped to min(k_max, n-1) so at least 2 points per sub-series are available.
- If k_max is set larger than the smallest configured window minus 1, that window will still compute (via the internal clamp) but with reduced fidelity.

## Assumptions

- Input DataFrame contains a numeric close-price column named "C".
- shift_features applies a 1-bar forward shift to every emitted feature column, which is what enforces the no-lookahead invariant.
