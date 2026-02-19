# Structure

Analyzes price structure through spectral decomposition (FFT), path efficiency, fractal dimension, EMA convexity, time-since-event features, and VWAP-based features.

## Concept

Market prices exhibit structural properties that go beyond simple trend and volatility measures. The structure plugin extracts features that describe the frequency content, geometric efficiency, curvature, and temporal proximity to significant events in the price series. These features help ML models understand not just where price is going, but how it is getting there.

Fast Fourier Transform (FFT) features decompose price into its frequency components. The dominant frequency reveals the primary cycle length, spectral energy measures overall cyclical strength, and spectral entropy quantifies how uniformly energy is distributed across frequencies. Low spectral entropy indicates a dominant cycle (more predictable), while high spectral entropy indicates noise (harder to trade). The low-frequency ratio captures what fraction of energy comes from long-term trends versus short-term oscillations. Path efficiency measures how directly price moves from point A to point B: a straight-line move has efficiency near 1.0, while a choppy, back-and-forth path has efficiency near 0. The fractal dimension (1 + (1 - path_efficiency)) is the inverse view, where 1.0 = smooth trend and 2.0 = space-filling noise.

Convexity features capture the second derivative of EMA trends -- whether the trend is accelerating, decelerating, or at an inflection point. Event-based features encode temporal distance from significant market events (new highs/lows, EMA crosses, RSI extremes, volatility spikes) using both raw and log-transformed bar counts. VWAP (Volume Weighted Average Price approximation via typical price) features measure where price sits relative to the average traded level, providing a reference for institutional fair value.

## Features

| Feature | Description |
|---------|-------------|
| `fft_dom_freq_64` | Dominant frequency in 64-bar FFT window. Primary cycle frequency |
| `fft_dom_power_64` | Power concentration at dominant frequency (0-1). Higher = cleaner cycle |
| `fft_energy_64` | Log of total spectral energy in 64-bar window. Overall cyclical strength |
| `fft_entropy_64` | Spectral entropy in 64-bar window. Low = clear cycle, high = noise |
| `fft_lowfreq_64` | Fraction of energy in low-frequency components (64-bar). High = trend-dominated |
| `fft_dom_freq_128` | Dominant frequency in 128-bar FFT window |
| `fft_dom_power_128` | Power concentration at dominant frequency (128-bar) |
| `fft_energy_128` | Log of total spectral energy (128-bar) |
| `fft_entropy_128` | Spectral entropy (128-bar) |
| `fft_lowfreq_128` | Low-frequency energy fraction (128-bar) |
| `fft_dom_freq_256` | Dominant frequency in 256-bar FFT window |
| `fft_dom_power_256` | Power concentration at dominant frequency (256-bar) |
| `fft_energy_256` | Log of total spectral energy (256-bar) |
| `fft_entropy_256` | Spectral entropy (256-bar) |
| `fft_lowfreq_256` | Low-frequency energy fraction (256-bar) |
| `path_efficiency_10` | Path efficiency over 10 bars. Net displacement / total path length (0-1) |
| `path_efficiency_20` | Path efficiency over 20 bars |
| `path_efficiency_50` | Path efficiency over 50 bars |
| `path_efficiency_100` | Path efficiency over 100 bars |
| `fractal_dim_10` | Fractal dimension proxy over 10 bars. 1 = smooth trend, 2 = noise |
| `fractal_dim_20` | Fractal dimension proxy over 20 bars |
| `fractal_dim_50` | Fractal dimension proxy over 50 bars |
| `fractal_dim_100` | Fractal dimension proxy over 100 bars |
| `path_efficiency_20_chg` | Change in path efficiency(20) over 10 bars. Rising = trend strengthening |
| `path_efficiency_50_chg` | Change in path efficiency(50) over 20 bars |
| `convex_ema_21` | EMA(21) second derivative normalized by price (* 1000). Positive = accelerating up, negative = accelerating down |
| `convex_ema_21_smooth` | 5-bar smoothed version of EMA(21) convexity |
| `convex_ema_50` | EMA(50) second derivative normalized by price (* 1000) |
| `convex_ema_50_smooth` | 5-bar smoothed version of EMA(50) convexity |
| `convex_divergence` | Convexity(21) - Convexity(50). Divergence between short- and medium-term acceleration |
| `convex_zscore` | Z-score of smoothed convexity(21) over 100-bar window |
| `event_bars_since_high_20` | Bars since last 20-bar new high |
| `event_bars_since_high_20_log` | Log-transformed bars since last 20-bar new high |
| `event_bars_since_low_20` | Bars since last 20-bar new low |
| `event_bars_since_low_20_log` | Log-transformed bars since last 20-bar new low |
| `event_bars_since_high_50` | Bars since last 50-bar new high |
| `event_bars_since_high_50_log` | Log-transformed bars since last 50-bar new high |
| `event_bars_since_low_50` | Bars since last 50-bar new low |
| `event_bars_since_low_50_log` | Log-transformed bars since last 50-bar new low |
| `event_bars_since_ema_cross` | Bars since last EMA(8)/EMA(21) crossover |
| `event_bars_since_ema_cross_log` | Log-transformed bars since last EMA crossover |
| `event_bars_since_rsi_extreme` | Bars since RSI(14) was above 70 or below 30 |
| `event_bars_since_rsi_extreme_log` | Log-transformed bars since RSI extreme |
| `event_bars_since_vol_spike` | Bars since ATR exceeded 2x its 50-bar rolling mean |
| `event_bars_since_vol_spike_log` | Log-transformed bars since volatility spike |
| `structure_vwap_dist_20` | (Close - VWAP_20) / VWAP_20. Distance from 20-bar typical price average |
| `structure_vwap_dist_50` | (Close - VWAP_50) / VWAP_50. Distance from 50-bar typical price average |
| `structure_vwap_dist_100` | (Close - VWAP_100) / VWAP_100. Distance from 100-bar typical price average |
| `structure_vwap_time_above` | Fraction of last 20 bars where close was above VWAP(50). Persistence above fair value |
| `structure_bars_since_vwap_cross` | Bars since last VWAP(50) crossover |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fft_windows` | List[int] | [64, 128, 256] | Window sizes for FFT analysis. Must be powers of 2 for optimal FFT performance |
| `path_windows` | List[int] | [10, 20, 50, 100] | Window sizes for path efficiency and fractal dimension |
| `convexity_periods` | List[int] | [21, 50] | EMA periods for convexity (second derivative) computation |
| `event_periods` | List[int] | [20, 50] | Lookback periods for new-high/new-low event detection |
| `vwap_windows` | List[int] | [20, 50, 100] | Window sizes for VWAP (typical price average) computation |

## Usage Notes

- All features are shifted by 1 bar to prevent lookahead bias.
- FFT features require at least `2 * window` data points to compute. Shorter datasets will skip the corresponding FFT window size.
- FFT uses a Hanning window to reduce spectral leakage, and the DC component is excluded from all spectral metrics.
- The VWAP approximation uses the typical price ((H + L + C) / 3) rather than true volume-weighted average price, since volume-weighted computation requires tick data.
- Event features use `shift(1)` on the rolling high/low before comparison to ensure the current bar's high/low does not contaminate the rolling extremum used for detection.
- Log-transformed event features (`*_log`) compress large bar counts, which helps ML models that are sensitive to feature scale.
- The `manifest.json` sets `benefits_from_stationary: true` since FFT and convexity features can benefit from detrended/stationary price inputs.
