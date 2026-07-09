# Plugin Spec — cusum_events

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes symmetric CUSUM structural-break features from close-price log returns: positive/negative event flags, normalized cumulative deviations, overshoot intensity, and bars-since-event.

## Summary

Runs a symmetric CUSUM filter over log returns of the close price, using a rolling mean as the expected return and `threshold * rolling_std` as the per-bar event threshold. On a positive or negative structural break the cumulative sum is reset and an event is recorded. Produces six shifted ML feature columns: two binary event flags, two threshold-normalized cumulative-deviation values, an overshoot-ratio intensity, and a lookback-normalized bars-since-event counter.

## Inputs

- df['C'] (close price series)

## Parameters

- `threshold` (float, default=1.5): Multiplier applied to rolling std to form the CUSUM event threshold h = threshold * rolling_std; higher values require larger cumulative deviations before firing an event.
- `lookback` (int, default=100): Rolling window (bars) used to compute the expected log return and its standard deviation, and also used to normalize the bars-since-event feature.

## Outputs

- cusum_pos_event
- cusum_neg_event
- cusum_pos_value
- cusum_neg_value
- cusum_intensity
- cusum_bars_since

## Acceptance Criteria

- AC-001: Computes log returns from df['C'] with the first bar's log return prepended as log(close[0]) so the returns array matches the input length.
- AC-002: Uses a pandas rolling window of size `lookback` with min_periods=20 to compute the expected return (mean) and rolling std of log returns.
- AC-003: Fills NaN warmup values of the rolling mean/std with the global nanmean/nanstd of log returns from index 1 onward.
- AC-004: Runs the symmetric CUSUM filter with s_pos = max(0, s_pos_prev + (r - expected)) and s_neg = min(0, s_neg_prev + (r - expected)), resetting the respective accumulator to 0 whenever |accumulator| exceeds h (only when h > 0).
- AC-005: Sets cusum_pos_event=1 on positive breaks and cusum_neg_event=1 on negative breaks; cusum_intensity records the overshoot ratio |accumulator| / h at the firing bar.
- AC-006: Emits cusum_pos_value = s_pos / h and cusum_neg_value = s_neg / -h (sign-flipped to a 0..1 scale), with 0 substituted where h <= 0.
- AC-007: Emits cusum_bars_since as (i - last_event_bar) / lookback, remaining 0 until the first event fires.
- AC-008: Returns exactly the six feature columns listed in `_FEATURES` via get_feature_columns(), and reports ['cusum_pos_event', 'cusum_neg_event'] as signal columns.
- AC-009: Applies shift_features to all six features so no output at bar i depends on data from bar i or later, then concatenates them onto the input DataFrame preserving its index.

## Edge Cases

- Warmup bars where the rolling mean/std would be NaN (fewer than 20 valid observations) are backfilled with the global nanmean/nanstd of the log-return series so the CUSUM can run from the start.
- Bars where the threshold h is not strictly positive (e.g. zero-variance windows) suppress event firing and yield 0 for cusum_pos_value / cusum_neg_value via the np.where guard.
- The first bar has a synthetic log return of 0 (from prepend=log(close[0])) and the loop starts at i=1, so no CUSUM update or event can occur at index 0.
- Before the first event fires, cusum_bars_since stays at 0 because last_event_bar is initialized to -1 and the counter is only updated once an event has occurred.
- Division-by-zero and invalid warnings during the s_pos/h and s_neg/-h normalization are suppressed with np.errstate; the np.where masks the result to 0 for those bars.

## Assumptions

- Input DataFrame contains a numeric 'C' (close) column with strictly positive values so that np.log is well-defined.
- shift_features shifts every produced feature by one bar to enforce no-lookahead, matching the SDK convention referenced by the constitution.
- The DataFrame index is monotonic and aligns positionally with the numpy arrays used inside compute().
