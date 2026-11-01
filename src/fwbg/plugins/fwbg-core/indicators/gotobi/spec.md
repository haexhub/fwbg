# Plugin Spec — gotobi

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Marks Japanese settlement ("gotobi") days and the bars that close at a given Tokyo hour on them, for rules around the Tokyo 9:55 fixing.

## Summary

Gotobi days are the 5th, 10th, 15th, 20th and 25th of each month and the month's last day, each moved to the previous Japanese business day when it falls on a weekend, a national holiday (`holidays.country_holidays("JP")`) or a bank closure (31 December to 3 January). Japanese importers buy USD at the Tokyo fixing on these days (Ito & Yamada 2017 on order imbalances at the Tokyo fix). A bar belongs to the Tokyo date and hour of its close (start + `bar_minutes`).

## Inputs

- df: any frame with a DatetimeIndex of bar start times (naive = UTC); only the index is read

## Parameters

- `entry_hour` (int, default 10): `gotobi_entry` marks the bar closing at this hour, Tokyo time
- `exit_hour` (int, default 15): `gotobi_exit` marks the bar closing at this hour, Tokyo time
- `bar_minutes` (int, default 60): bar length; a bar closes at its start plus this

## Outputs

- `gotobi_day`: 1 when the bar closes on a gotobi date (Tokyo)
- `gotobi_entry`: 1 for the bar closing at `entry_hour`:00 Tokyo on a gotobi date
- `gotobi_exit`: 1 for the bar closing at `exit_hour`:00 Tokyo on a gotobi date, an exit column for `signal_exit`

The settlement calendar is known in advance, so the columns are unshifted and meant for a decision at the bar's close (`entry_delay=0`). As for every indicator, the first row is NaN. Values depend only on a bar's own timestamp, so a prefix of the data gives the same values.
