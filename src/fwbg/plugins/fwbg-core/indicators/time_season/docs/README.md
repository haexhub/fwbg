# Time & Season Indicators

Temporal and calendar-based features that capture intraday session patterns, weekly/monthly seasonality, and cyclical time encoding for ML models.

## Concept

Financial markets exhibit well-documented time-dependent behavior. Volatility and volume follow predictable intraday patterns tied to trading sessions (Asia, London, New York). Monthly seasonality effects (such as the "January Effect" or end-of-month rebalancing) create recurring patterns. Day-of-week effects influence market behavior as institutional flows cluster around specific weekdays. This plugin captures all of these temporal patterns and encodes them in a format suitable for machine learning models.

The key mathematical challenge with time features is their cyclical nature. Hour 23 and hour 0 are only one hour apart, but represented as integers they appear 23 units apart. Sin/cos encoding solves this by mapping cyclical values onto a unit circle: each time value becomes a (sin, cos) pair where nearby times produce nearby coordinates regardless of where they fall in the integer range. For example, `hour_sin = sin(2 * pi * hour / 24)` and `hour_cos = cos(2 * pi * hour / 24)` together uniquely identify each hour while preserving cyclical proximity. The same encoding is applied to day of week, month, quarter, week of year, and day of month.

For ML models, the raw integer features work well with tree-based models (LightGBM, XGBoost) that can naturally split on ordinal values, while the sin/cos encoded features are essential for neural networks and linear models that cannot learn cyclical patterns from raw integers. The trading session flags (Asia, London, NY, overlap) provide direct binary signals for session-based regime detection. Calendar events like month-end and quarter-end flags capture institutional rebalancing flows that create predictable short-term patterns.

## Features

| Feature | Description |
|---------|-------------|
| `time_hour` | Hour of the day (0-23). Raw integer value for tree-based models. |
| `time_day` | Day of the week (0 = Monday, 6 = Sunday). Raw integer value. |
| `time_hour_sin` | Sine component of hour-of-day cyclical encoding. sin(2*pi*hour/24). |
| `time_hour_cos` | Cosine component of hour-of-day cyclical encoding. cos(2*pi*hour/24). |
| `time_day_sin` | Sine component of day-of-week cyclical encoding. sin(2*pi*dow/7). |
| `time_day_cos` | Cosine component of day-of-week cyclical encoding. cos(2*pi*dow/7). |
| `time_session_asia` | Asian trading session flag: 1 during 00:00-08:00 UTC, 0 otherwise. |
| `time_session_london` | London trading session flag: 1 during 08:00-16:00 UTC, 0 otherwise. |
| `time_session_ny` | New York trading session flag: 1 during 13:00-21:00 UTC, 0 otherwise. |
| `time_session_overlap` | Session overlap flag: 1 during London-Asia overlap (08:00-12:00 UTC) or London-NY overlap (13:00-16:00 UTC). Higher liquidity periods. |
| `season_month` | Month of the year (1-12). Raw integer value for tree-based models. |
| `season_quarter` | Quarter of the year (1-4). Raw integer value. |
| `season_week` | ISO week of the year (1-52). Raw integer value. |
| `season_dayofmonth` | Day of the month (1-31). Raw integer value. |
| `season_month_sin` | Sine component of month cyclical encoding. sin(2*pi*month/12). |
| `season_month_cos` | Cosine component of month cyclical encoding. cos(2*pi*month/12). |
| `season_quarter_sin` | Sine component of quarter cyclical encoding. sin(2*pi*quarter/4). |
| `season_quarter_cos` | Cosine component of quarter cyclical encoding. cos(2*pi*quarter/4). |
| `season_week_sin` | Sine component of ISO week cyclical encoding. sin(2*pi*week/52). |
| `season_week_cos` | Cosine component of ISO week cyclical encoding. cos(2*pi*week/52). |
| `season_dayofmonth_sin` | Sine component of day-of-month cyclical encoding. sin(2*pi*day/31). |
| `season_dayofmonth_cos` | Cosine component of day-of-month cyclical encoding. cos(2*pi*day/31). |
| `time_month_start` | Month start flag: 1 when day of month is 1-3, 0 otherwise. Captures institutional month-start rebalancing flows. |
| `time_month_end` | Month end flag: 1 when day of month is 28 or later, 0 otherwise. Captures institutional month-end rebalancing and window dressing. |
| `time_quarter_end` | Quarter end flag: 1 when month is March/June/September/December and day >= 28, 0 otherwise. Captures strong institutional rebalancing at fiscal quarter boundaries. |
| `time_week_start` | Week start flag: 1 on Mondays, 0 otherwise. Captures weekend gap effects and Monday positioning. |
| `time_week_end` | Week end flag: 1 on Fridays, 0 otherwise. Captures end-of-week position squaring and weekend risk reduction. |
| `time_year_progress` | Linear year progress: day-of-year divided by days-in-year (365 or 366). Ranges from 0.0 (January 1) to 1.0 (December 31). Handles leap years. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_raw` | bool | True | Include raw (integer) time features: hour of day (0-23), day of week (0-6), month (1-12), quarter (1-4), ISO week (1-52), day of month (1-31). Useful for tree-based models that can split on ordinal values directly. |
| `include_encoded` | bool | True | Include sin/cos cyclical encoding of time features. Maps periodic values (hour, day, month, etc.) onto a unit circle so that e.g. hour 23 and hour 0 are close together. Essential for neural networks and linear models that cannot learn cyclical patterns from raw integers. |

## Usage Notes

- This plugin does not benefit from stationarity transformations (`benefits_from_stationary: false`), as time features are inherently deterministic and stationary.
- All features are shifted by 1 bar for consistency with other plugins, even though time features do not use OHLC data and would not cause lookahead bias.
- The input DataFrame must have a `DatetimeIndex`. A `ValueError` is raised if the index is not a `DatetimeIndex`.
- Trading session hours are defined in UTC. If your data uses a different timezone, the session flags will be incorrect. Convert your DataFrame index to UTC before applying this plugin.
- The session overlap periods (08:00-12:00 and 13:00-16:00 UTC) typically have the highest liquidity and tightest spreads, making them important for execution-aware models.
- Setting `include_raw=False` while keeping `include_encoded=True` is recommended for neural network and linear model pipelines. Setting `include_encoded=False` while keeping `include_raw=True` is sufficient for pure tree-based models.
- Year progress is a monotonically increasing feature within each year. It captures broad seasonal trends but is not cyclically encoded (use month/quarter encodings for cyclical seasonality).
- The ISO week number may differ from the calendar week for dates near year boundaries.
