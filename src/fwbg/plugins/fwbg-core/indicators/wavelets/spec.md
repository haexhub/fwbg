# Plugin Spec — wavelets

**Kind**: indicator  •  **Version**: 2.1.0

## Capability

Decomposes log returns via a causal rolling Discrete Wavelet Transform into per-band power, end-of-window band-pass means, and full-window energy shares.

## Summary

Applies a causal (rolling-window) Discrete Wavelet Transform to the log-return series derived from close prices and emits, for each configured rolling window, the approximation-band power, per-detail-level power and mean, per-detail-level energy share (detail energy / total energy of the trailing DWT window, which sums to 1 together with the approximation share), and a high-frequency-to-low-frequency power ratio (detail level 1 / detail level `levels`). Band power and shares are computed from the coefficients of the whole trailing window (orthogonal DWT in periodization mode), not from the boundary-affected end sample of a reconstruction; white noise therefore puts ~50% of its energy into detail level 1. The `*_mean_*` features still use the end-of-window reconstruction (a causal band-pass filter output at bar i, boundary affected). All features are shifted by one bar to prevent lookahead.

## Inputs

- df['C'] (close prices)

## Parameters

- `wavelet` (choice, default='db4'): Wavelet family for the DWT (Daubechies db1-db10, Symlets sym2-sym5, Coiflets coif1-coif3). db4 is the default and balances time/frequency localization.
- `levels` (int, default=3): Number of DWT decomposition levels. Each level halves the frequency band; level 1 captures the highest-frequency detail, level N the lowest, and the approximation captures the remaining trend.
- `windows` (list[int], default=[10, 20, 50]): Rolling window sizes used to compute per-band energy (mean of squared signal) and mean statistics, and to compute the high/low-frequency ratio.
- `dwt_window` (int, default=256): Maximum number of past bars used per causal DWT computation. Must be >= 2**levels; larger values capture lower-frequency structure at the cost of runtime.

## Outputs

- wt_approx_energy_{w} for each w in windows
- wt_detail_{lvl}_energy_{w} for each detail level 1..levels and each w in windows
- wt_detail_{lvl}_mean_{w} for each detail level 1..levels and each w in windows
- wt_detail_ratio_{lvl} for each detail level 1..levels (energy share: sum of squared level coefficients / sum of squared coefficients of all levels, over the trailing dwt_window)
- wt_high_freq_ratio_{w} for each w in windows (detail level 1 power / detail level `levels` power)

## Acceptance Criteria

- AC-001: Reads close prices from df['C'] and computes log returns with np.diff(np.log(close), prepend=np.log(close[0])); the artificial first return is never used in any DWT window.
- AC-002: Applies pywt.wavedec causally in a rolling window of at most `dwt_window` bars ending at bar i; only past-and-current data influences bar i's features. Band energies use mode='periodization' on the latest multiple of 2**levels samples (exact Parseval), so band power = sum(c**2)/m and shares are exact fractions of the window energy.
- AC-003: For each rolling window w in `windows`, emits wt_approx_energy_{w} and, for each detail level lvl in 1..levels, wt_detail_{lvl}_energy_{w} and wt_detail_{lvl}_mean_{w} (energy = rolling mean over w bars of the band power sum(c**2)/m; mean = rolling mean of the end-of-window detail reconstruction; both need w valid bars).
- AC-004: Emits wt_detail_ratio_{lvl} for each detail level lvl in 1..levels as sum(cD_lvl**2) / (sum(cA**2) + sum over levels of sum(cD**2)) of the trailing DWT window at bar i. For white noise E[wt_detail_ratio_1] = 0.5, E[wt_detail_ratio_2] = 0.25.
- AC-005: Emits wt_high_freq_ratio_{w} for each w in `windows` as safe_divide(detail_level_1_energy_w, detail_level_`levels`_energy_w).
- AC-006: All feature columns are shifted by one bar via shift_features before being concatenated back onto df, so bar i's feature values are derived from data strictly up to bar i-1.
- AC-007: Feature column names are the sorted union of all names above; get_feature_columns(params) returns the names for the given params when supplied, otherwise the sorted names actually produced by the last compute() call, or the default-parameter names (levels=3, windows=[10, 20, 50]) if compute has not been called.
- AC-008: Default parameters are wavelet='db4', levels=3, windows=[10, 20, 50], dwt_window=256; passing windows=None uses [10, 20, 50].

## Edge Cases

- Warm-up: for any bar i whose trailing segment is too short for all `levels` decomposition levels (pywt.dwt_max_level(seg_len, wavelet) < levels, e.g. fewer than 56 bars for db4 with 3 levels) every band quantity is NaN; rolling features additionally need w valid bars.
- Infeasible configuration: if pywt.dwt_max_level(dwt_window, wavelet) < levels a ValueError is raised instead of silently producing NaN levels.
- Reconstruction length mismatch: the last valid index of each reconstructed level (used for the *_mean_* features) is taken as min(seg_len - 1, len(rec) - 1) to guard against pywt.waverec returning a signal longer than the input segment.
- Division safety: high_freq_ratio uses safe_divide and detail_ratio yields NaN for an all-zero window, so zero denominators do not raise.

## Assumptions

- df is indexed such that df.index can be used as the index for the shifted feature DataFrame returned by shift_features.
- df['C'] contains strictly positive close prices (np.log(close) is applied without guarding against non-positive values).
- pywt (PyWavelets) is available in the runtime environment.
