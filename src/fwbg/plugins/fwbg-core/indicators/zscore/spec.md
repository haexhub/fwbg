# Plugin Spec — zscore

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Rolling z-score of the (log) close against its rolling mean and standard deviation, with back-to-mean flags that serve as mean-reversion exits.

## Summary

For each lookback L: z = (x - mean_L) / std_L with x the log close (default) or close. Mean-reversion rules enter beyond a threshold (`value_check` on `zscore_L`) and exit at the mean through `signal_exit` with `zscore_L_above_mean` (longs) and `zscore_L_below_mean` (shorts).

## Inputs

- df: OHLC; only C is used

## Parameters

- `lookbacks` (list[int], default [20]): windows in bars, each >= 2
- `use_log` (bool, default true)
- `entry_z` (float > 0, default none): entry threshold; needs exactly one lookback

## Outputs

- `zscore_L`, `zscore_L_above_mean`, `zscore_L_below_mean` per lookback
- with `entry_z`: `zscore_entry_long` (z < -entry_z), `zscore_entry_short` (z > entry_z), `zscore_exit_long` (z >= 0), `zscore_exit_short` (z <= 0); the names do not depend on the lookback, so `optimization.indicator_grid` can vary `lookbacks` and `entry_z` under one set of signal rules

Unshifted (decision at the bar's close, `entry_delay=0`); NaN until L bars exist or when the window has no dispersion.
