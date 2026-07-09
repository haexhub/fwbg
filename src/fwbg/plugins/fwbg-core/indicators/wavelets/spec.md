# Plugin Spec — wavelets

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Decomposes log returns via a causal rolling Discrete Wavelet Transform into per-level energy, mean, and cross-level ratio features.

## Summary

Applies a causal (rolling-window) Discrete Wavelet Transform to the log-return series derived from close prices and emits, for each configured rolling window, the approximation-band energy, per-detail-level energy and mean, per-detail-level energy ratio (detail energy / total energy), and a high-frequency-to-low-frequency energy ratio (detail level 1 / detail level `levels`). All features are shifted by one bar to prevent lookahead.

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
- wt_detail_ratio_{lvl} for each detail level 1..levels (detail energy / total energy over ratio_window=min(20,n))
- wt_high_freq_ratio_{w} for each w in windows (detail level 1 energy / detail level `levels` energy)

## Acceptance Criteria

- AC-001: Reads close prices from df['C'] and computes log returns with np.diff(np.log(close), prepend=np.log(close[0])).
- AC-002: Applies pywt.wavedec causally in a rolling window of at most `dwt_window` bars ending at bar i; only past-and-current data influences bar i's features.
- AC-003: For each rolling window w in `windows`, emits wt_approx_energy_{w} and, for each detail level lvl in 1..levels, wt_detail_{lvl}_energy_{w} and wt_detail_{lvl}_mean_{w} (energy = pd.Series.rolling(w, min_periods=1).mean() of squared signal; mean = rolling mean of signal).
- AC-004: Emits wt_detail_ratio_{lvl} for each detail level lvl in 1..levels using ratio_window = min(20, n) and safe_divide(detail_energy, total_energy) where total_energy = approx_energy + sum of detail energies over that window.
- AC-005: Emits wt_high_freq_ratio_{w} for each w in `windows` as safe_divide(detail_level_1_energy_w, detail_level_`levels`_energy_w).
- AC-006: All feature columns are shifted by one bar via shift_features before being concatenated back onto df, so bar i's feature values are derived from data strictly up to bar i-1.
- AC-007: Feature column names are the sorted union of all names above; get_feature_columns() returns the sorted names actually produced by the last compute() call, or the default-parameter names (levels=3, windows=[10, 20, 50]) if compute has not been called.
- AC-008: Default parameters are wavelet='db4', levels=3, windows=[10, 20, 50], dwt_window=256; passing windows=None uses [10, 20, 50].

## Edge Cases

- Warm-up: for any bar i whose rolling segment length is less than 2**levels, the approximation and detail signals remain NaN at bar i (rolling energy/mean over these NaNs propagates NaN until enough valid samples accumulate).
- Short input: if pywt.dwt_max_level(seg_len, wavelet) is less than the requested `levels`, actual_lvl is clamped to that maximum and detail levels above actual_lvl remain NaN at that bar.
- Reconstruction length mismatch: the last valid index of each reconstructed level is taken as min(seg_len - 1, len(rec) - 1) to guard against pywt.waverec returning a signal longer than the input segment.
- Small dataset: ratio_window = min(20, n), so on very short series the detail-ratio features are computed over fewer than 20 bars.
- Division safety: all detail_ratio and high_freq_ratio features use safe_divide, so zero denominators do not raise.

## Assumptions

- df is indexed such that df.index can be used as the index for the shifted feature DataFrame returned by shift_features.
- df['C'] contains strictly positive close prices (np.log(close) is applied without guarding against non-positive values).
- pywt (PyWavelets) is available in the runtime environment.
