# Plugin Spec — structure

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes market-structure features from OHLC data: FFT spectral descriptors, path efficiency / fractal dimension, EMA convexity, bars-since-event timers, and VWAP distance/cross metrics.

## Summary

Aggregates a broad set of price-structure indicators for a downstream model: rolling FFT features (dominant frequency, dominant power, spectral energy, spectral entropy, low-frequency ratio) over multiple windows; path efficiency and derived fractal-dimension proxy plus their multi-bar changes; EMA second-derivative convexity, its smoothed value, cross-period divergence and z-score; bars-since-event timers for rolling-high/low breaks, EMA(8/21) cross, RSI extremes and ATR volatility spikes (with log-scaled variants); and VWAP-style features (distance to rolling typical-price mean, share of time above VWAP, bars since VWAP cross). All resulting feature columns are shifted by one bar via shift_features to preclude lookahead.

## Inputs

- df['H']
- df['L']
- df['C']

## Parameters

- `fft_windows` (list[int], default=[64, 128, 256]): Rolling-window lengths used for FFT spectral feature computation; a window is skipped if len(close) < 2*window.
- `path_windows` (list[int], default=[10, 20, 50, 100]): Lookback lengths for path-efficiency and derived fractal-dimension features. Change-features (*_chg) are only emitted for windows 20 and 50.
- `convexity_periods` (list[int], default=[21, 50]): EMA periods used for convexity (second derivative of EMA, scaled by close*1000). convex_divergence requires both 21 and 50; convex_zscore requires 21.
- `event_periods` (list[int], default=[20, 50]): Rolling-window periods used for new-high / new-low breakout event timers (bars-since-event and log1p variants).
- `vwap_windows` (list[int], default=[20, 50, 100]): Rolling windows for VWAP-like typical-price means; structure_vwap_dist_{w} is emitted per window while time-above and bars-since-cross features are hard-wired to a 50-bar VWAP.

## Outputs

- fft_dom_freq_64
- fft_dom_power_64
- fft_energy_64
- fft_entropy_64
- fft_lowfreq_64
- fft_dom_freq_128
- fft_dom_power_128
- fft_energy_128
- fft_entropy_128
- fft_lowfreq_128
- fft_dom_freq_256
- fft_dom_power_256
- fft_energy_256
- fft_entropy_256
- fft_lowfreq_256
- path_efficiency_10
- path_efficiency_20
- path_efficiency_50
- path_efficiency_100
- fractal_dim_10
- fractal_dim_20
- fractal_dim_50
- fractal_dim_100
- path_efficiency_20_chg
- path_efficiency_50_chg
- convex_ema_21
- convex_ema_21_smooth
- convex_ema_50
- convex_ema_50_smooth
- convex_divergence
- convex_zscore
- event_bars_since_high_20
- event_bars_since_high_20_log
- event_bars_since_low_20
- event_bars_since_low_20_log
- event_bars_since_high_50
- event_bars_since_high_50_log
- event_bars_since_low_50
- event_bars_since_low_50_log
- event_bars_since_ema_cross
- event_bars_since_ema_cross_log
- event_bars_since_rsi_extreme
- event_bars_since_rsi_extreme_log
- event_bars_since_vol_spike
- event_bars_since_vol_spike_log
- structure_vwap_dist_20
- structure_vwap_dist_50
- structure_vwap_dist_100
- structure_vwap_time_above
- structure_bars_since_vwap_cross

## Acceptance Criteria

- AC-001: compute(df, ...) returns the input df concatenated with the structure feature columns; the resulting frame has all columns listed in get_feature_columns() when default params are used.
- AC-002: All emitted feature columns are passed through shift_features(features, df.index) so that row i reflects only information available up to and including bar i-1 (no lookahead).
- AC-003: For each fft window w, FFT features are computed on a Hanning-windowed, detrended close-segment via np.fft.rfft; the DC component is dropped and dom_power / entropy / low_freq_ratio are normalized by the summed non-DC power. Bars before index w remain NaN.
- AC-004: path_efficiency_{w} = |C - C.shift(w)| / rolling-sum(|C.diff()|, w) via safe_divide; fractal_dim_{w} = 1 + (1 - path_efficiency_{w}).
- AC-005: path_efficiency_20_chg = pe_20 - pe_20.shift(10) and path_efficiency_50_chg = pe_50 - pe_50.shift(20) are only produced when 20 (resp. 50) is present in path_windows.
- AC-006: convex_ema_{p} = safe_divide(EMA(C, p).diff().diff(), C) * 1000, and convex_ema_{p}_smooth is its 5-bar rolling mean.
- AC-007: convex_divergence = convex_ema_21 - convex_ema_50 and convex_zscore = (convex_ema_21_smooth - rolling_mean_100) / rolling_std_100 are only emitted when the required convexity_periods are present.
- AC-008: Event breakout signals use rolling_high/low with .shift(1) so the current bar is excluded from its own breakout comparison; event_bars_since_* counts bars since the last event and remains NaN until the first event occurs (via _bars_since_event).
- AC-009: Additional event timers are emitted for EMA(8)/EMA(21) cross, RSI(14) extremes (>70 or <30), and ATR(14) volatility spikes (ATR > 2 * shifted 50-bar mean), each with a log1p companion feature.
- AC-010: VWAP proxy uses typical price tp = (H+L+C)/3; structure_vwap_dist_{w} = (C - tp.rolling(w).mean()) / tp.rolling(w).mean(); structure_vwap_time_above is the 20-bar mean of the indicator 'C > VWAP_50', and structure_bars_since_vwap_cross counts bars since that indicator flipped.

## Edge Cases

- If len(close) < 2*w for a given fft window w, that window's FFT features are silently skipped and their columns are absent from the output frame.
- Inside the FFT loop, if the non-DC power sum is below EPSILON (or the non-DC power vector is empty), all five FFT feature entries for that bar remain NaN.
- For an fft window w, indices 0..w-1 remain NaN (loop starts at i=w) and are additionally shifted by shift_features, so w+1 leading rows are NaN in FFT columns.
- Divisions by potentially-zero path length, close, or std are performed via safe_divide to avoid division-by-zero blow-ups.
- _bars_since_event returns NaN for all bars before the first event fires (event_groups == 0), so bars_since_* features start as NaN and only become finite once the underlying condition has triggered at least once.
- np.log1p is applied directly to bars-since counts, so NaN inputs propagate to NaN in the *_log columns (rather than being clipped to zero).
- New-high/new-low tests use '>=' / '<=' against the shifted rolling extremum, so ties with the prior extremum count as events.
- The final shift_features call means row 0 of every feature column is NaN even for features that would otherwise be defined on bar 0 (e.g. FFT trivially, but also event log features).

## Assumptions

- Input DataFrame uses uppercase OHLC column names ('H', 'L', 'C'); 'O' is not read.
- The DataFrame index is monotonically ordered in time so that .shift(), .rolling(), and .diff() have their intended causal semantics.
- shift_features shifts every feature column by exactly one bar; the plugin relies on this rather than shifting individual features itself.
- The `ta` library's ema_indicator, rsi, and average_true_range are used with their default (non-lookahead) implementations.

## Needs Clarification

- [NEEDS CLARIFICATION: The '__init__.py' docstring is in German while the class docstring lists only 'Structure-bezogene Features'; whether the German comments should be preserved verbatim or translated is not stated in the source.]
- [NEEDS CLARIFICATION: structure_vwap_time_above and structure_bars_since_vwap_cross use hard-coded windows (50-bar VWAP, 20-bar time-above) that are not exposed as parameters — unclear whether this is intentional or an oversight.]
