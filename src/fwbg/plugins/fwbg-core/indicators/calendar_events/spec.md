# Plugin Spec — calendar_events

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Emits deterministic calendar-anomaly features (turn-of-month, quarter-end, triple witching, OpEx, NFP week, year boundary, days-to-month-end, FOMC cycle, week-of-month) from the datetime index.

## Summary

Derives up to nine calendar-based feature columns from the DataFrame's DatetimeIndex — binary flags for well-known calendar events (turn-of-month, quarter-end, triple witching, monthly OpEx, NFP week, year boundary) and continuous proximity features (normalized days to month end, sinusoidal FOMC-cycle approximation, normalized week of month). Feature groups are toggleable via include_binary and include_proximity. Outputs are shifted by one bar via shift_features to prevent lookahead.

## Inputs

- df (pandas.DataFrame with a DatetimeIndex — only the index is consumed; no OHLCV columns are read)

## Parameters

- `include_proximity` (bool, default=True): When true, emit continuous proximity features: cal_days_to_month_end, cal_fomc_proximity, cal_week_of_month.
- `include_binary` (bool, default=True): When true, emit binary event flags: cal_turn_of_month, cal_quarter_end, cal_triple_witching, cal_monthly_opex, cal_nfp_week, cal_year_boundary.

## Outputs

- cal_turn_of_month (binary: 1.0 when day<=3 or day>=days_in_month-1, else 0.0)
- cal_quarter_end (binary: 1.0 in last 5 calendar days of Mar/Jun/Sep/Dec)
- cal_triple_witching (binary: 1.0 within 2 days of 3rd Friday of Mar/Jun/Sep/Dec)
- cal_monthly_opex (binary: 1.0 within 2 days of 3rd Friday of any month)
- cal_nfp_week (binary: 1.0 on first 5 calendar days of any month)
- cal_year_boundary (binary: 1.0 in last 5 days of Dec or first 5 days of Jan)
- cal_days_to_month_end (float in [0,1]: (days_in_month - day) / days_in_month)
- cal_fomc_proximity (float in [-1,1]: sin(2*pi*day_of_year/46))
- cal_week_of_month (float in [0,1]: ((day-1)//7) / 4.0)

## Acceptance Criteria

- AC-001: compute() returns the original DataFrame with the enabled calendar feature columns appended (via pd.concat on axis=1).
- AC-002: All emitted feature columns are shifted by one bar via shift_features(...) so no row's value depends on same-bar or future timestamps.
- AC-003: When include_binary=True, the six binary flag columns are produced with values in {0.0, 1.0} using the documented calendar-day rules.
- AC-004: When include_proximity=True, the three continuous columns are produced as float64 with the documented formulas.
- AC-005: Setting include_binary=False suppresses the six binary columns; setting include_proximity=False suppresses the three continuous columns.
- AC-006: get_feature_columns() returns the full nine-column list regardless of parameters.
- AC-007: get_signal_columns() returns exactly the six binary event columns.
- AC-008: 3rd-Friday computation (_third_friday) is used for both cal_triple_witching (restricted to Mar/Jun/Sep/Dec) and cal_monthly_opex (all months), each with a +/-2-day window.

## Edge Cases

- DataFrame index must be a DatetimeIndex — the code accesses idx.day, idx.month, idx.days_in_month, idx.dayofyear directly with no fallback.
- Empty DataFrame (n=0): per-row loops for triple_witching / monthly_opex iterate zero times; vectorized branches produce empty arrays; concat should yield an empty frame with the feature columns present.
- Single-row DataFrame: shift_features will shift the sole row's values out, leaving NaN in the emitted feature columns.
- Months with fewer than 31 days: turn-of-month threshold uses days_in_month-1 per-row, so Feb 27/28, Apr 29/30, etc. are handled correctly.
- Leap years: cal_fomc_proximity uses day_of_year with a fixed 46-day period, so the sinusoid's phase drifts by one day across a leap year (documented as an approximation).
- cal_week_of_month for day=29,30,31 yields ((day-1)//7)/4.0 = 1.0 (5th week mapped to the upper bound).
- cal_monthly_opex and cal_triple_witching overlap on Mar/Jun/Sep/Dec — both flags fire together in those quarterly windows.
- Per-row Python loops over the index for triple_witching and monthly_opex are O(n) and may be slow on very large frames.

## Assumptions

- The DataFrame index is a pandas DatetimeIndex exposing .day, .month, .days_in_month, and .dayofyear.
- shift_features applies a uniform one-bar forward shift on all supplied feature arrays keyed by the given index, enforcing the no-lookahead invariant required for indicators.
- Calendar rules encoded here are calendar-day based (not trading-day based); e.g., NFP-week and turn-of-month use raw day-of-month, not exchange sessions.

## Needs Clarification

- [NEEDS CLARIFICATION: FOMC proximity uses a fixed 46-day sinusoid seeded at day-of-year=0 rather than actual FOMC meeting dates — is this intentional as a permanent approximation, or a placeholder for a future calendar-driven implementation?]
- [NEEDS CLARIFICATION: Should cal_monthly_opex be suppressed on quarterly-expiry months to avoid perfect collinearity with cal_triple_witching, or is the redundancy intentional for model feature-selection to resolve?]
