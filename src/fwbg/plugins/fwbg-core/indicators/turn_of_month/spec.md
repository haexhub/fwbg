# Plugin Spec — turn_of_month

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Counts exchange trading days within each calendar month and flags an entry a set number of trading days before the month ends and an exit on a set trading day of the next month.

## Summary

Trading days are weekdays that are not holidays of an exchange calendar from `holidays.financial_holidays` (e.g. `NYSE`, `ECB`), or every weekday for `calendar="weekdays"`. The counts come from the calendar, not from the bars in the data, so a missing bar does not shift them. Entering at `entry_days_left = 1` and exiting at `exit_day = 3` holds the classic turn-of-month window from trading day -1 to +3 (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008). The flag columns have fixed names, so an indicator grid can vary the parameters while the rules stay the same.

## Inputs

- df: any frame with a DatetimeIndex of daily bars labelled with their trading date (only the index is read)

## Parameters

- `calendar` (choice, default `NYSE`): an exchange calendar of `holidays.list_supported_financial()` or `weekdays`; an unknown name raises `ValueError`
- `entry_days_left` (int, default 1): `tom_entry_long` marks the bar with this many trading days left in the month
- `exit_day` (int, default 3): `tom_exit_long` marks this trading day of the month

## Outputs

- `tom_day`: trading day of the month (1 on the first; a bar dated on a closed day counts the trading days before it, 0 before the month's first)
- `tom_days_left`: trading days of the month after the bar's date (0 on the last)
- `tom_entry_long`: 1 when `tom_days_left == entry_days_left`
- `tom_exit_long`: 1 when `tom_day == exit_day`, an exit column for `signal_exit`

The exchange calendar is published in advance, so all columns are unshifted: each refers to the bar's own trading date and is meant for a decision at that bar's close (`entry_delay=0`). As for every indicator, the first row is NaN. Values depend only on a bar's own date, so a prefix of the data gives the same values.
