# Calendar Events

Provides deterministic features derived from the datetime index that capture well-documented calendar anomalies in equity and derivatives markets.

## Concept

Financial markets exhibit systematic patterns tied to the calendar. Institutional fund flows concentrate around month-end and quarter-end (rebalancing), options expiration dates create predictable gamma effects, and macroeconomic data releases (like Non-Farm Payrolls, usually on the first Friday of each month) cause recurring volatility patterns. These effects have been documented extensively in academic literature and are collectively known as calendar anomalies.

This plugin encodes these known calendar events as ML features in two forms. **Binary event flags** mark specific windows (e.g., "we are currently in the turn-of-month window" or "this is triple witching week"). **Continuous proximity features** provide smooth, gradient-friendly signals such as the normalized distance to month-end or the normalized week of the month. Both forms are derived purely from the datetime index -- they require no price data.

ML models benefit from calendar features because many trading signals have different predictive power during specific calendar periods. For example, momentum strategies may behave differently during quarter-end rebalancing windows, and volatility regimes often shift around options expiration. By providing these features, the model can learn context-dependent rules without requiring explicit calendar logic in the strategy.

## Features

### Binary Event Flags

| Feature | Description |
|---------|-------------|
| `cal_turn_of_month` | 1.0 if the bar's date is within the first 3 or last 2 trading days of the month (exchange calendar of the `calendar` parameter, default NYSE; a weekend after the month's last trading day keeps the flag). Captures the well-documented turn-of-month effect driven by institutional salary investments, pension fund flows, and portfolio rebalancing. |
| `cal_quarter_end` | 1.0 if the current day is within the last 5 calendar days of March, June, September, or December. Captures quarter-end window dressing and rebalancing flows. |
| `cal_triple_witching` | 1.0 if the current day is within 2 days of the 3rd Friday of March, June, September, or December. Triple witching is the simultaneous expiration of stock options, index options, and index futures, causing elevated volume and volatility. |
| `cal_monthly_opex` | 1.0 if the current day is within 2 days of the 3rd Friday of any month. Monthly options expiration drives gamma exposure changes and pinning effects on underlying prices. |
| `cal_nfp_week` | 1.0 on Monday-Friday of the week that contains the month's first Friday (the week can start in the previous month). Non-Farm Payrolls (NFP) is typically released on the first Friday. The BLS occasionally deviates and no exceptions are hard-coded, so this is a rule-based approximation. |
| `cal_year_boundary` | 1.0 if the current day is in the last 5 days of December or the first 5 days of January. Captures the January effect, tax-loss selling, and new-year portfolio construction. |

### Continuous Proximity Features

| Feature | Description |
|---------|-------------|
| `cal_days_to_month_end` | Normalized distance to month-end, scaled 0.0 to 1.0 (where 0.0 = last day, 1.0 = first day). Provides a smooth signal for month-cycle positioning. |
| `cal_week_of_month` | Normalized week of the month, scaled 0.0 to 1.0. Computed as `((day - 1) // 7) / 4.0`. First week = 0.0, last week approaches 1.0. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_proximity` | bool | `True` | Include continuous proximity features: `cal_days_to_month_end` and `cal_week_of_month`. These provide smooth, gradient-friendly signals for models to learn calendar-based positioning. |
| `include_binary` | bool | `True` | Include binary event flags: `cal_turn_of_month`, `cal_quarter_end`, `cal_triple_witching`, `cal_monthly_opex`, `cal_nfp_week`, and `cal_year_boundary`. Each captures a known calendar anomaly. |
| `calendar` | choice | `NYSE` | Exchange holiday calendar used to count trading days for `cal_turn_of_month`; `weekdays` treats every Monday-Friday as a trading day. |

## Usage Notes

- **No price data required**: All features are computed purely from the datetime index. The `H`, `L`, `C` columns are not used.
- **DatetimeIndex required**: The DataFrame must have a `DatetimeIndex` for date extraction to work.
- **UTC assumption**: Session hours and dates are taken from the index as-is. If your data uses a non-UTC timezone, the calendar events will align to that timezone.
- **Stationarity**: This plugin does not benefit from stationary input data (`benefits_from_stationary = False`). Features are either binary or bounded continuous values.
- **No FOMC feature**: The former `cal_fomc_proximity` column was a 46-day sine wave unrelated to any real meeting date and was removed. For FOMC timing, use the `scheduled_events` indicator with a dated events file.
- **Timing**: Every column describes the bar's own date. The calendar is deterministic and known in advance, so the values are not shifted by one bar (a shift would make Monday's row describe Friday); only the first row is NaN. Intraday bars repeat their date's values.
- **Total feature count**: With default parameters, the plugin produces 8 features (6 binary + 2 continuous). Setting either `include_binary` or `include_proximity` to `False` reduces the feature set accordingly.
