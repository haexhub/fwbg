# CUSUM Structural Break Detection

Detects structural breaks in the price process using the symmetric CUSUM filter from de Prado's "Advances in Financial Machine Learning" (Chapter 2), producing ML-ready features rather than hard event filters.

## Concept

The Cumulative Sum (CUSUM) filter is a sequential analysis technique originally developed for quality control. In the financial context, it monitors cumulative deviations of log returns from their expected value. When the cumulative sum exceeds a threshold, it signals a structural break -- a statistically significant shift in the price process that suggests the underlying regime has changed.

The symmetric variant tracks both positive and negative cumulative deviations simultaneously. A **positive structural break** fires when upward deviations accumulate beyond the threshold, indicating a bullish shift. A **negative structural break** fires when downward deviations accumulate beyond the threshold, indicating a bearish shift. After each event, the cumulative sum resets to zero, and the process resumes.

Rather than using CUSUM events as a hard filter (only trading at event bars), this plugin converts them into continuous ML features. The model receives the current cumulative deviation values, binary event flags, the intensity of the overshoot when an event fires, and the time elapsed since the last event. This allows the ML model to learn nuanced relationships -- for example, that a CUSUM event combined with certain other indicators produces high-quality signals, while CUSUM events in isolation may not be predictive.

## Features

| Feature | Description |
|---------|-------------|
| `cusum_pos_event` | Binary flag (1.0/0.0) indicating a positive structural break occurred at this bar. The positive cumulative sum exceeded the threshold `h = threshold * rolling_std`. |
| `cusum_neg_event` | Binary flag (1.0/0.0) indicating a negative structural break occurred at this bar. The negative cumulative sum dropped below `-h`. |
| `cusum_pos_value` | Current positive cumulative deviation, normalized by the threshold (range 0.0 to 1.0). Values approaching 1.0 indicate a positive break is imminent. Resets to 0 after each positive event. |
| `cusum_neg_value` | Current negative cumulative deviation, normalized by the threshold (range 0.0 to 1.0, sign flipped). Values approaching 1.0 indicate a negative break is imminent. Resets to 0 after each negative event. |
| `cusum_intensity` | Overshoot ratio at the moment of an event: `cumulative_sum / threshold`. Values above 1.0 indicate how far the cumulative sum exceeded the threshold before being detected. 0.0 on non-event bars. |
| `cusum_bars_since` | Number of bars since the last CUSUM event (positive or negative), normalized by the `lookback` parameter. Provides a measure of how "stale" the last structural break signal is. Long gaps between events suggest a stable regime. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | `1.5` | Multiplier applied to the rolling standard deviation to determine the CUSUM event threshold (`h = threshold * rolling_std`). Higher values require larger cumulative deviations before firing a structural break event, reducing false positives but increasing detection lag. Based on de Prado's AFML Ch. 2. Range: 0.5-10.0, step 0.25. |
| `lookback` | int | `100` | Rolling window for computing the expected return (mean) and volatility (std) of log returns. These statistics form the baseline against which cumulative deviations are measured. Longer lookbacks produce more stable baselines but adapt slower to regime changes. Also used to normalize `cusum_bars_since`. Range: 20-1000, step 10. |

## Usage Notes

- **Log returns**: The CUSUM filter operates on log returns (`log(C[i]) - log(C[i-1])`), not raw prices. This ensures the deviation measurement is scale-independent.
- **Adaptive threshold**: The threshold is `h = threshold * rolling_std`, where `rolling_std` is computed over the `lookback` window. This makes the filter adaptive to the current volatility regime -- it requires proportionally larger moves in high-volatility environments.
- **Warmup handling**: The rolling mean and standard deviation require `min_periods=20` bars to produce values. During the warmup period, global estimates (computed over the entire series) are used as fallback. This means early bars use a less precise baseline.
- **Normalization**: `cusum_pos_value` and `cusum_neg_value` are divided by the threshold for scale-independence (range 0 to ~1). `cusum_bars_since` is divided by the `lookback` parameter.
- **Stationarity**: This plugin does not benefit from stationary input data (`benefits_from_stationary = False`). It operates on log returns internally.
- **Reference**: Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 2: Financial Data Structures, Section 2.5.2: The Symmetric CUSUM Filter.
