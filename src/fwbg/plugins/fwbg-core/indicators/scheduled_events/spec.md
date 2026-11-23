# Plugin Spec — scheduled_events

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Counts exchange trading days from each bar to the next event of a dated events file and flags a window a set number of trading days before the events, with entry and exit edges of that window.

## Summary

Events come from a CSV file with one dated event per row, for example Treasury auctions, central bank meetings or index rebalancing dates; a filter column and values select a subset. Trading days follow `turn_of_month`'s exchange calendar (`holidays.financial_holidays`, e.g. `NYSE`), so the counts do not depend on the bars in the data and also reach events after the last bar. Overlapping windows of clustered events merge into one, so a position held from `evt_window_start` to `evt_window_end` stays open through a cluster.

## Inputs

- df: any frame with a DatetimeIndex of daily bars labelled with their trading date (only the index is read)
- the events file named by `events_file`

## Parameters

- `events_file` (string, default empty): CSV file with the events; `~` is expanded. Empty, unreadable, or missing a named column: a warning and all values NaN
- `date_column` (string, default `date`): column holding the event date. Each cell is parsed on its own, so formats may be mixed (`2024-01-10`, `2024-01-11 14:00`, `Jan 12, 2024`); a timezone offset is dropped and the date as written is kept; blank cells are skipped; a cell without a four-digit year or any other unparsable cell raises `EventDateError` (a `ValueError`) instead of dropping the row. Ambiguous numeric dates such as `01/02/2024` are read month first, so prefer ISO dates
- `filter_column` (string, default empty) and `filter_values` (list of strings, default empty): when `filter_column` is set, only rows whose value is in `filter_values` are events
- `calendar` (choice, default `NYSE`): an exchange calendar of `holidays.list_supported_financial()` or `weekdays`
- `window_start` (int, default 6) and `window_end` (int, default 2): `evt_window` is 1 when an event lies between `window_end` and `window_start` trading days ahead, inclusive; `window_end > window_start` raises `ValueError`

## Outputs

- `evt_days_ahead`: trading days from the bar's date to the next event on or after it (0 on an event day; NaN when no later event is in the file)
- `evt_window`: 1 inside the window, else 0; NaN where `evt_days_ahead` is NaN, i.e. after the file's last event, because later events are unknown rather than absent
- `evt_window_start`: 1 on the bar where `evt_window` turns on (previous bar off); NaN after the file's last event
- `evt_window_end`: 1 on the bar where `evt_window` turns off (previous bar on), an exit column for `signal_exit`. The last window can end after its event when `window_end=0`. Later exit values are NaN.

An event or a bar dated on a closed day counts from the next trading day. Entering at the close of `evt_window_start` and exiting at the close of `evt_window_end` with `window_start=6`, `window_end=2` holds the returns of the five trading days before each event.

The columns are unshifted and meant for a decision at the bar's close (`entry_delay=0`). They are free of lookahead only if the file holds events that were scheduled in advance; recording events after the fact (e.g. unscheduled meetings) leaks the future. As for every indicator, the first row is NaN. Values depend only on a bar's own date and the events file, so a prefix of the data gives the same values.
