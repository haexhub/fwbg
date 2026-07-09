# Plugin Spec — aroon

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes Aroon Up and Aroon Down oscillators measuring the number of bars since the highest high and lowest low within a rolling lookback window.

## Summary

Wraps `ta.trend.AroonIndicator` over the H/L columns with a configurable `period` (default 25) to produce two 0-100 features, `aroon_up` and `aroon_down`, shifted by one bar to prevent lookahead bias.

## Inputs

- df['H'] (high price series)
- df['L'] (low price series)

## Parameters

- `period` (int, default=25): Lookback period for Aroon Up/Down. Measures how many bars since the highest high (Aroon Up) or lowest low (Aroon Down) within the window.

## Outputs

- aroon_up
- aroon_down

## Acceptance Criteria

- AC-001: compute() returns the original DataFrame concatenated with two new columns: aroon_up and aroon_down.
- AC-002: Both aroon_up and aroon_down values lie in the range [0, 100].
- AC-003: Features are shifted by one bar via shift_features(...) so row i never depends on data from row i+1 (no lookahead).
- AC-004: get_feature_columns() returns exactly ['aroon_up', 'aroon_down'] regardless of params.
- AC-005: get_default_params() returns {'period': 25}.
- AC-006: Passing a non-default period parameter changes the resulting aroon_up/aroon_down values compared to the default.
- AC-007: get_signal_columns() and get_overlay_columns() both return empty lists.
- AC-008: get_column_group_labels() returns {'aroon': 'Aroon'}.

## Edge Cases

- DataFrame shorter than `period` rows: leading rows produce NaN values in aroon_up/aroon_down (delegated to ta.trend.AroonIndicator).
- First row after shift is always NaN due to shift_features by one bar.
- Constant H/L series (no new highs/lows within window): behaviour delegated to ta library — values remain within [0, 100].
- Very large period (up to schema max of 200) yields more NaN leading rows but no error.

## Assumptions

- _none_
