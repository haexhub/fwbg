# Wavelets

Decomposes log returns into frequency bands using the Discrete Wavelet Transform (DWT), producing energy and ratio features that characterize market regime across multiple time scales.

## Concept

Financial time series are inherently non-stationary -- their statistical properties shift over time as market regimes change. Traditional Fourier analysis assumes stationarity and provides only global frequency information. Wavelet decomposition solves this by being localized in both time and frequency, making it ideal for capturing transient events like volatility spikes, trend transitions, and microstructure noise.

The Discrete Wavelet Transform recursively splits the signal into frequency bands. At each decomposition level, the signal is separated into a "detail" component (higher frequencies) and an "approximation" component (lower frequencies). Detail level 1 captures the highest frequency content (noise, microstructure effects), while deeper levels capture progressively lower frequencies. The final approximation represents the underlying trend component.

ML models benefit from wavelet features because they encode how energy (variance) is distributed across frequency bands. A trending market concentrates energy in the approximation and lower detail levels, while a choppy, mean-reverting market has most energy in the higher detail levels. The rolling energy and ratio features allow the model to detect these regime shifts in real time and adapt its predictions accordingly.

## Features

With default parameters (`levels=3`, `windows=[10, 20, 50]`), the plugin generates the following features. Feature names are dynamically constructed from the configured levels and windows.

| Feature | Description |
|---------|-------------|
| `wt_approx_energy_{w}` | Rolling mean squared amplitude of the approximation (trend) component over window `w`. Higher values indicate a stronger underlying trend signal. |
| `wt_detail_{lvl}_energy_{w}` | Rolling mean squared amplitude of detail level `lvl` over window `w`. Level 1 = highest frequency (noise), level N = lowest detail frequency. |
| `wt_detail_{lvl}_mean_{w}` | Rolling mean of the reconstructed detail signal at level `lvl` over window `w`. Captures directional bias within each frequency band. |
| `wt_detail_ratio_{lvl}` | Fraction of total energy attributed to detail level `lvl`, computed over a fixed 20-bar smoothing window. Values sum to less than 1 (remainder is approximation energy). |
| `wt_high_freq_ratio_{w}` | Ratio of detail level 1 energy to detail level N energy over window `w`. High values indicate noise-dominated (choppy) markets; low values indicate trend-dominated regimes. |

**Default feature count:** With `levels=3` and `windows=[10, 20, 50]`, this produces 3 approx energy + 9 detail energy + 9 detail mean + 3 detail ratio + 3 high-freq ratio = **27 features**.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wavelet` | choice | `"db4"` | Wavelet family for the DWT. Daubechies wavelets (`db1`-`db10`) are most common for financial data. `db4` provides a good balance between time and frequency localization. `db1` (Haar) is the simplest; higher-order wavelets are smoother. Also supports Symlets (`sym2`-`sym5`) and Coiflets (`coif1`-`coif3`). |
| `levels` | int | `3` | Number of DWT decomposition levels (range: 1-12). Each level halves the frequency band: level 1 captures the highest frequencies (noise/microstructure), level N captures the lowest detail frequencies. The approximation captures the remaining trend component. More levels separate more frequency bands but require longer input series. |
| `windows` | list[int] | `[10, 20, 50]` | Rolling window sizes for computing energy (mean squared amplitude) and mean statistics of each decomposition level (range: 2-5000). Shorter windows capture recent energy shifts; longer windows provide more stable regime characterization. |

## Usage Notes

- **Input column:** Uses the `C` (close) column. Log returns are computed internally.
- **Stationarity:** Does not require pre-stationarized input (`benefits_from_stationary: false`). The wavelet decomposition itself handles non-stationarity.
- **Minimum data length:** The DWT requires at least `2^levels` data points. With `levels=3`, a minimum of 8 bars is needed, though the rolling windows effectively require `max(windows)` bars for meaningful feature values.
- **Feature shifting:** All features are shifted forward by one bar via `shift_features` to prevent look-ahead bias.
- **Decomposition levels vs. data length:** Higher `levels` values require exponentially more data. For intraday data with short histories, keep levels low (2-3). For daily data with years of history, higher levels (4-6) can capture weekly/monthly cycles.
- **Wavelet choice:** `db4` is a robust default. Smoother wavelets (higher order) are better at capturing smooth trends but have longer filter lengths, introducing more edge effects.
