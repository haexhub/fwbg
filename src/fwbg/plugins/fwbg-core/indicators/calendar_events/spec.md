# Plugin Spec — calendar_events

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Emits deterministic calendar-anomaly features (turn-of-month, quarter-end, triple witching, OpEx, NFP week, year boundary, days-to-month-end, week-of-month) from the datetime index.

## Summary

Derives up to eight calendar-based feature columns from the DataFrame's DatetimeIndex — binary flags for well-known calendar events (turn-of-month, quarter-end, triple witching, monthly OpEx, NFP week, year boundary) and continuous proximity features (normalized days to month end, normalized week of month). Feature groups are toggleable via include_binary and include_proximity. Every column describes the bar's own date: the calendar is deterministic and known in advance, so the values are NOT shifted by one bar (shifting would make Monday's row describe Friday). Only the first row is NaN, as for every indicator.

## Inputs

- df (pandas.DataFrame with a DatetimeIndex — only the index is consumed; no OHLCV columns are read)

## Parameters

- `include_proximity` (bool, default=True): When true, emit continuous proximity features: cal_days_to_month_end, cal_week_of_month.
- `include_binary` (bool, default=True): When true, emit binary event flags: cal_turn_of_month, cal_quarter_end, cal_triple_witching, cal_monthly_opex, cal_nfp_week, cal_year_boundary.
- `calendar` (choice, default=NYSE): exchange holiday calendar (`holidays.financial_holidays`) used to count trading days for cal_turn_of_month, or `weekdays`.

## Outputs

- cal_turn_of_month (binary: 1.0 on the first 3 or last 2 trading days of the month; a closed day takes the position of the last trading day before it)
- cal_quarter_end (binary: 1.0 in last 5 calendar days of Mar/Jun/Sep/Dec)
- cal_triple_witching (binary: 1.0 within 2 days of 3rd Friday of Mar/Jun/Sep/Dec)
- cal_monthly_opex (binary: 1.0 within 2 days of 3rd Friday of any month)
- cal_nfp_week (binary: 1.0 on Monday-Friday of the week containing the month's first Friday; a rule-based approximation, the BLS schedule occasionally deviates and no exceptions are hard-coded)
- cal_year_boundary (binary: 1.0 in last 5 days of Dec or first 5 days of Jan)
- cal_days_to_month_end (float in [0,1]: (days_in_month - day) / days_in_month)
- cal_week_of_month (float in [0,1]: ((day-1)//7) / 4.0)

## Acceptance Criteria

- AC-001: compute() returns the original DataFrame with the enabled calendar feature columns appended (via pd.concat on axis=1).
- AC-002: Feature columns are unshifted and describe the bar's own date; they depend only on the index timestamp (a deterministic calendar), so they cannot carry future information. Row 0 is NaN.
- AC-003: When include_binary=True, the six binary flag columns are produced with values in {0.0, 1.0} using the documented rules (trading days for turn-of-month, calendar days for the others).
- AC-004: When include_proximity=True, the two continuous columns are produced as float64 with the documented formulas.
- AC-005: Setting include_binary=False suppresses the six binary columns; setting include_proximity=False suppresses the two continuous columns.
- AC-006: get_feature_columns() returns the full eight-column list regardless of parameters.
- AC-007: get_signal_columns() returns exactly the six binary event columns; get_calendar_columns() returns the same six (they follow the date alone, so they stay in the ML feature pool).
- AC-008: 3rd-Friday computation (_third_friday) is used for both cal_triple_witching (restricted to Mar/Jun/Sep/Dec) and cal_monthly_opex (all months), each with a +/-2-day window.

## Edge Cases

- DataFrame index must be a DatetimeIndex — the code accesses idx.day, idx.month, idx.days_in_month, idx.dayofyear directly with no fallback.
- Empty DataFrame (n=0): per-row loops for triple_witching / monthly_opex iterate zero times; vectorized branches produce empty arrays; concat should yield an empty frame with the feature columns present.
- Single-row DataFrame: the sole row is the first row, so all emitted feature columns are NaN.
- Months ending on a weekend or holiday: cal_turn_of_month counts trading days, so the last trading day (for example Friday 2023-04-28 before a Sunday month end) is always flagged.
- Intraday bars: every bar of a date carries that date's values.
- cal_week_of_month for day=29,30,31 yields ((day-1)//7)/4.0 = 1.0 (5th week mapped to the upper bound).
- cal_monthly_opex and cal_triple_witching overlap on Mar/Jun/Sep/Dec — both flags fire together in those quarterly windows.
- Per-row Python loops over the index for triple_witching and monthly_opex are O(n) and may be slow on very large frames.

## Assumptions

- The DataFrame index is a pandas DatetimeIndex exposing .day, .month, .days_in_month, and .dayofyear.
- Calendar columns are known in advance, so they are exempt from the one-bar feature shift; the index timestamps themselves are not altered.
- Turn-of-month uses trading days of the `calendar` parameter; the other rules (quarter-end, year boundary, OpEx windows, days-to-month-end, week-of-month) use calendar days.

## Needs Clarification

- [NEEDS CLARIFICATION: Should cal_monthly_opex be suppressed on quarterly-expiry months to avoid perfect collinearity with cal_triple_witching, or is the redundancy intentional for model feature-selection to resolve?]
