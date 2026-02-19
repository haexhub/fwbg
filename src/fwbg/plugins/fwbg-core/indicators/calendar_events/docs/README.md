# Calendar Events

Provides deterministic features derived from the datetime index that capture well-documented calendar anomalies in equity and derivatives markets.

## Concept

Financial markets exhibit systematic patterns tied to the calendar. Institutional fund flows concentrate around month-end and quarter-end (rebalancing), options expiration dates create predictable gamma effects, and macroeconomic data releases (like Non-Farm Payrolls on the first Friday of each month) cause recurring volatility patterns. These effects have been documented extensively in academic literature and are collectively known as calendar anomalies.

This plugin encodes these known calendar events as ML features in two forms. **Binary event flags** mark specific windows (e.g., "we are currently in the turn-of-month window" or "this is triple witching week"). **Continuous proximity features** provide smooth, gradient-friendly signals such as the normalized distance to month-end or a sinusoidal approximation of the FOMC meeting cycle. Both forms are derived purely from the datetime index -- they require no price data.

ML models benefit from calendar features because many trading signals have different predictive power during specific calendar periods. For example, momentum strategies may behave differently during quarter-end rebalancing windows, and volatility regimes often shift around options expiration. By providing these features, the model can learn context-dependent rules without requiring explicit calendar logic in the strategy.

## Features

### Binary Event Flags

| Feature | Description |
|---------|-------------|
| `cal_turn_of_month` | 1.0 if the current day is within the first 3 or last 2 calendar days of the month. Captures the well-documented turn-of-month effect driven by institutional salary investments, pension fund flows, and portfolio rebalancing. |
| `cal_quarter_end` | 1.0 if the current day is within the last 5 calendar days of March, June, September, or December. Captures quarter-end window dressing and rebalancing flows. |
| `cal_triple_witching` | 1.0 if the current day is within 2 days of the 3rd Friday of March, June, September, or December. Triple witching is the simultaneous expiration of stock options, index options, and index futures, causing elevated volume and volatility. |
| `cal_monthly_opex` | 1.0 if the current day is within 2 days of the 3rd Friday of any month. Monthly options expiration drives gamma exposure changes and pinning effects on underlying prices. |
| `cal_nfp_week` | 1.0 if the current day is within the first 5 calendar days of any month. Non-Farm Payrolls (NFP) is typically released on the first Friday, with anticipation effects starting earlier in the week. |
| `cal_year_boundary` | 1.0 if the current day is in the last 5 days of December or the first 5 days of January. Captures the January effect, tax-loss selling, and new-year portfolio construction. |

### Continuous Proximity Features

| Feature | Description |
|---------|-------------|
| `cal_days_to_month_end` | Normalized distance to month-end, scaled 0.0 to 1.0 (where 0.0 = last day, 1.0 = first day). Provides a smooth signal for month-cycle positioning. |
| `cal_fomc_proximity` | Sinusoidal approximation of the FOMC meeting cycle, using `sin(2 * pi * day_of_year / 46)`. The ~46-day period approximates the ~8 meetings per year. Not exact but provides a smooth cyclic feature. |
| `cal_week_of_month` | Normalized week of the month, scaled 0.0 to 1.0. Computed as `((day - 1) // 7) / 4.0`. First week = 0.0, last week approaches 1.0. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_proximity` | bool | `True` | Include continuous proximity features: `cal_days_to_month_end`, `cal_fomc_proximity`, and `cal_week_of_month`. These provide smooth, gradient-friendly signals for models to learn calendar-based positioning. |
| `include_binary` | bool | `True` | Include binary event flags: `cal_turn_of_month`, `cal_quarter_end`, `cal_triple_witching`, `cal_monthly_opex`, `cal_nfp_week`, and `cal_year_boundary`. Each captures a known calendar anomaly. |

## Usage Notes

- **No price data required**: All features are computed purely from the datetime index. The `H`, `L`, `C` columns are not used.
- **DatetimeIndex required**: The DataFrame must have a `DatetimeIndex` for date extraction to work.
- **UTC assumption**: Session hours and dates are taken from the index as-is. If your data uses a non-UTC timezone, the calendar events will align to that timezone.
- **Stationarity**: This plugin does not benefit from stationary input data (`benefits_from_stationary = False`). Features are either binary or bounded continuous values.
- **FOMC approximation**: The `cal_fomc_proximity` feature uses a sinusoidal approximation rather than actual FOMC meeting dates. It captures the general cyclicality but does not align precisely with specific meeting dates. For exact FOMC alignment, consider supplementing with an external calendar.
- **Total feature count**: With default parameters, the plugin produces 9 features (6 binary + 3 continuous). Setting either `include_binary` or `include_proximity` to `False` reduces the feature set accordingly.
