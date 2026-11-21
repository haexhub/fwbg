# Plugin Spec — rebalancing_pressure

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Measures the month-to-date return of stocks minus bonds from daily series added by a `data_loading` step and flags month-end entries against the expected rebalancing flow, plus an exit on the month's last trading day.

## Summary

Fixed-weight funds rebalance their stock/bond mix around month end: after stocks outperformed bonds they sell stocks in the last days of the month, after underperformance they buy (Harvey, Mazzoleni & Melone 2025). The input columns (e.g. `macro_spy`, `macro_tlt`, `macro_ief` from a CSV data source) hold on every bar the last close before the bar's date, so on the month's first trading day they hold the previous month's last close, which serves as the base. Trading days come from an exchange calendar (`turn_of_month`'s `trading_days`).

## Inputs

- df with a DatetimeIndex of daily bars labelled with their trading date and the stock and bond columns

## Parameters

- `stock_columns` (list[string], default `["macro_spy"]`), `bond_columns` (list[string], default `["macro_tlt", "macro_ief"]`): each side's month-to-date return is the mean over its columns
- `threshold` (float, default 0.02): minimum absolute drift for an entry
- `entry_days_left` (int, default 3): entries are flagged on the bar with this many trading days left in the month
- `inputs_lagged` (bool, default true): true when the columns already hold the previous trading day's close (the `data_loading` lag); false when they hold the bar's own close, in which case the indicator lags them by one bar itself (daily bars, one bar per date, else `ValueError`). The flag cannot be verified from the data and must match the source: unlagged inputs read as lagged leak the bar's own close
- `calendar` (string, default `NYSE`): exchange calendar, as in `turn_of_month`

## Outputs

- `rebal_drift`: stock minus bond month-to-date return; NaN in a month whose first trading day has no bar
- `rebal_entry_long`: 1 on the entry bar when `rebal_drift < -threshold`
- `rebal_entry_short`: 1 on the entry bar when `rebal_drift > threshold`
- `rebal_exit`: 1 on the month's last trading day (exit column for `signal_exit`)

Unshifted for a decision at the bar's close (the inputs lag by a day, from the loader or from `inputs_lagged=false`); the first row is NaN. A missing input column logs a warning and leaves all values NaN. Values depend only on the bar and earlier bars of its month, so a prefix of the data gives the same values.
