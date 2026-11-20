# Plugin Spec — time_season

**Kind**: indicator  •  **Version**: 3.1.0

## Capability

Derives time/seasonality features (raw + sin/cos encodings of hour/day/month/quarter/week/dayofmonth, configurable session masks with overlap, calendar flags, year progress) from a DatetimeIndex.

## Summary

Indicator that emits intraday, weekly, monthly, quarterly, and yearly time features from the DataFrame's DatetimeIndex. Optionally includes raw integer features, cyclical sin/cos encodings, configurable trading session binary masks (with automatic pairwise overlap column), a trading-day mask, special calendar flags (month/quarter/week start/end), and a year-progress ratio. Every column describes the bar's own timestamp: the calendar and session clock are fixed in advance, so the values are not shifted by one bar (a shift would make Monday's row describe Friday); only the first row is NaN.

## Inputs

- pandas.DataFrame with a pandas.DatetimeIndex (no OHLCV columns required)

## Parameters

- `include_raw` (bool, default=True): Include raw integer time features: hour of day (0-23), day of week (0-6), month (1-12), quarter (1-4), ISO week (1-53), day of month (1-31).
- `include_encoded` (bool, default=True): Include sin/cos cyclical encoding of time features (hour, day of week, month, quarter, week, day of month).
- `include_sessions` (bool, default=True): Include trading session binary features based on the configured sessions parameter.
- `sessions` (string, default='{"asia": [0, 8], "london": [8, 16], "ny": [13, 21]}'): Mapping of session name to [start_hour, end_hour] (UTC hours; naive timestamps are UTC) or [start_hour, end_hour, 'Area/City'] to evaluate the hours in that time zone's local time, which follows daylight saving time. Produces time_session_{name} per session (sorted by name) and a time_session_overlap column when at least two sessions are configured. Wrap-around ranges (start >= end) are supported (e.g. [22, 6]).
- `include_seasonality` (bool, default=True): Include seasonality features: month, quarter, ISO week, day of month (respects include_raw/include_encoded for which variants are emitted).
- `include_calendar` (bool, default=True): Include special calendar binary features: time_month_start (first `month_edge_days` trading days), time_month_end (last `month_edge_days` trading days), time_quarter_end (time_month_end in Mar/Jun/Sep/Dec), time_week_start (Monday), time_week_end (Friday).
- `include_year_progress` (bool, default=True): Include time_year_progress = dayofyear / (365 + is_leap_year).
- `trading_days` (list[int], default=None): Active trading days as ints 0=Mon..6=Sun. When set, emits time_trading_day binary feature; when None (default), the feature is not produced.
- `calendar` (choice, default='weekdays'): trading-day calendar for the month flags: `weekdays` (every Monday-Friday) or an exchange holiday calendar such as `NYSE`.
- `month_edge_days` (int, default=3): number of trading days flagged at the start and at the end of a month.

## Outputs

- time_hour
- time_day
- time_hour_sin
- time_hour_cos
- time_day_sin
- time_day_cos
- time_session_{name} (one per configured session, sorted by name)
- time_session_overlap (only when >=2 sessions configured)
- time_trading_day (only when trading_days is set)
- season_month
- season_quarter
- season_week
- season_dayofmonth
- season_month_sin
- season_month_cos
- season_quarter_sin
- season_quarter_cos
- season_week_sin
- season_week_cos
- season_dayofmonth_sin
- season_dayofmonth_cos
- time_month_start
- time_month_end
- time_quarter_end
- time_week_start
- time_week_end
- time_year_progress

## Acceptance Criteria

- AC-001: Raises ValueError('DataFrame muss einen DateTimeIndex haben') when df.index is not a pandas.DatetimeIndex.
- AC-002: Returns the original DataFrame concatenated with the generated feature columns (via pd.concat on axis=1).
- AC-003: Feature columns are unshifted and describe the bar's own timestamp (deterministic calendar and session clock, no price input), so they cannot carry future information; row 0 is NaN.
- AC-004: With defaults (include_raw, include_encoded, include_sessions, include_seasonality, include_calendar, include_year_progress all True; trading_days=None), get_feature_columns matches the columns produced by compute().
- AC-005: Session masks: for a session [start, end] with start < end the mask is (hour >= start) & (hour < end); when start >= end it wraps around as (hour >= start) | (hour < end). Each mask is emitted as time_session_{name} cast to int, with names sorted alphabetically.
- AC-006: When include_sessions is True and at least 2 sessions are configured, time_session_overlap is 1 wherever any pair of session masks overlaps and 0 otherwise; with fewer than 2 sessions the overlap column is not emitted.
- AC-007: When trading_days is a list of ints, time_trading_day is 1 for rows whose df.index.dayofweek is in that set and 0 otherwise; when trading_days is None the column is omitted.
- AC-008: include_seasonality gates the season_* columns; within it, include_raw controls the integer season columns and include_encoded controls the sin/cos season columns.
- AC-009: Calendar features: time_month_start = trading day of month in 1..month_edge_days, time_month_end = fewer than month_edge_days trading days left in the month, time_quarter_end = time_month_end & (month in {3,6,9,12}), so every month's last trading day is flagged; time_week_start = (dayofweek == 0), time_week_end = (dayofweek == 4), each cast to int.
- AC-010: time_year_progress = dayofyear / (365 + is_leap_year), yielding values in (0, 1].
- AC-011: get_feature_columns(params) returns the list of columns compute would emit for those params; get_signal_columns(params) returns the subset consisting of session (+ overlap) columns, time_trading_day (if enabled), and calendar columns.

## Edge Cases

- DataFrame without a DatetimeIndex: compute raises ValueError.
- sessions supplied as a non-dict value: _parse_sessions falls back to the default asia/london/ny sessions.
- Single-session configuration (len(sessions) == 1): no time_session_overlap column is produced.
- Wrap-around session such as [22, 6]: mask uses OR logic across midnight (hour >= 22 or hour < 6).
- trading_days=None (default): time_trading_day column is not emitted; get_feature_columns/get_signal_columns also omit it.
- Leap years: time_year_progress divides by 366 instead of 365 via is_leap_year.
- The first row of every emitted feature column is NaN (indicator contract).
- ISO week 53: season_week_sin/cos use the ISO year's week count (52 or 53) as the cycle length, so week 53 of a long year is not mapped onto week 1.
- Session tz: an unknown time zone or a session spec that is not [start, end] or [start, end, tz] raises ValueError.

## Assumptions

- df.index is a pandas.DatetimeIndex; timezone handling is delegated to pandas and not modified by this indicator.
- Session hours without a time zone are given in the same clock as df.index.hour (UTC for naive timestamps); with a third list element they are converted to that zone first.

## Needs Clarification

- [NEEDS CLARIFICATION: The 'sessions' param type in get_param_schema is 'session_ranges' (a custom string), which does not map to the SpecParam type enum; represented here as 'string' with the default serialized as JSON. Confirm the intended SpecParam type for dict-shaped session configs.]
