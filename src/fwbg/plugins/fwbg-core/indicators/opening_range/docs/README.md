# Opening Range Breakout (ORB)

Computes features based on the Opening Range Breakout concept, capturing intraday price dynamics relative to hourly and session-specific opening ranges.

## Concept

The Opening Range is one of the oldest and most studied intraday patterns. It defines a price range established during the first bars of a time period (typically an hour or trading session). Once established, the range acts as a reference: breakouts above or below the range often signal directional momentum, while price remaining within the range suggests consolidation.

This plugin computes three distinct feature groups. **Rolling ORB** features measure the current bar's relationship to the most recent full-hour opening range -- its width, price position within it, and whether a breakout has occurred. **Session ORB** features do the same for configurable session hours (e.g., Asia open, London open, New York open), with values forward-filled until the next session occurrence. **Statistical features** aggregate breakout behavior over a rolling window, capturing average range size, how often breakouts occur, and whether breakout direction tends to persist.

ML models can leverage these features to detect regime shifts at key session boundaries, identify breakout-versus-fade setups, and incorporate session-specific price dynamics that differ across global trading hours. The ATR normalization ensures features remain scale-independent across instruments and timeframes.

## Features

### Rolling ORB (relative to last full-hour boundary)

| Feature | Description |
|---------|-------------|
| `orb_range` | Opening range (high - low of first `range_bars` bars in the hour) divided by close price. Measures the relative width of the hourly opening range. |
| `orb_position` | Position of the current close within the opening range, scaled 0 to 1. Values below 0 or above 1 indicate the price has moved outside the range. |
| `orb_breakout_up` | Binary flag (1/0) indicating the close is above the opening range high. |
| `orb_breakout_down` | Binary flag (1/0) indicating the close is below the opening range low. |
| `orb_range_vs_atr` | Opening range divided by ATR. Values below 1.0 suggest an unusually narrow range (potential breakout setup), values above 1.0 suggest an unusually wide range. |

### Session ORB (per configured session hour, forward-filled)

For each session hour `HH` in the `sessions` list (default: 00, 08, 13, 14), the following features are produced:

| Feature | Description |
|---------|-------------|
| `orb_sHH_range` | Session opening range divided by close price. |
| `orb_sHH_position` | Position of close within the session opening range (0-1 scale). |
| `orb_sHH_breakout_up` | Binary flag: close is above the session opening range high. |
| `orb_sHH_breakout_down` | Binary flag: close is below the session opening range low. |
| `orb_sHH_range_vs_atr` | Session opening range divided by ATR. |

With default sessions `[0, 8, 13, 14]`, this produces 20 session features:
`orb_s00_range`, `orb_s00_position`, `orb_s00_breakout_up`, `orb_s00_breakout_down`, `orb_s00_range_vs_atr`,
`orb_s08_range`, `orb_s08_position`, `orb_s08_breakout_up`, `orb_s08_breakout_down`, `orb_s08_range_vs_atr`,
`orb_s13_range`, `orb_s13_position`, `orb_s13_breakout_up`, `orb_s13_breakout_down`, `orb_s13_range_vs_atr`,
`orb_s14_range`, `orb_s14_position`, `orb_s14_breakout_up`, `orb_s14_breakout_down`, `orb_s14_range_vs_atr`

### Statistical features (rolling aggregates)

| Feature | Description |
|---------|-------------|
| `orb_stat_avg_range` | Rolling average of the hourly opening range (as % of close) over the `stat_window`. Tracks whether ranges are expanding or contracting. |
| `orb_stat_breakout_rate` | Rolling proportion of bars that broke out of their hourly opening range. High values indicate a breakout-prone regime. |
| `orb_stat_continuation_rate` | Rolling proportion of bars where the breakout direction persisted into the next hour. High values suggest momentum; low values suggest mean-reversion. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `range_bars` | int | `1` | Number of bars defining the opening range after each hour boundary. At M15: 1 bar = 15min range, 2 bars = 30min range. At M5: 1 bar = 5min range. Controls the trade-off between range stability and early signal. Range: 1-12. |
| `atr_period` | int | `14` | ATR period for normalizing the opening range size. The `orb_range_vs_atr` feature divides the opening range by ATR to identify unusually narrow or wide ranges. Range: 5-100. |
| `sessions` | list[int] | `[0, 8, 13, 14]` | UTC hours for session-specific ORB features. Default: 0 (Asia/Tokyo), 8 (London), 13 (NY pre-market), 14 (NY open). Each session produces 5 features that persist until the next occurrence of that session hour. Range per element: 0-23. |
| `stat_window` | int | `20` | Rolling window (in hours) for statistical features: average range, breakout rate, and continuation rate. Larger windows give more stable statistics but react slower to regime changes. Range: 5-200, step 5. |
| `enable_rolling` | bool | `True` | Enable rolling ORB features computed relative to the last full-hour boundary. These 5 features capture the current hour's price action dynamics. |
| `enable_session` | bool | `True` | Enable session-specific ORB features for each configured session hour. Produces 5 features per session that persist until the next session start. |
| `enable_stats` | bool | `True` | Enable statistical features: rolling average range, breakout rate, and continuation rate over the stat_window. |

## Usage Notes

- **Intraday only**: This indicator is designed for intraday timeframes (M1 through H4). On daily bars, it returns the DataFrame unchanged with no features added.
- **DatetimeIndex required**: The DataFrame must have a `DatetimeIndex`. A `ValueError` is raised otherwise.
- **Feature validity**: Rolling and session ORB features are `NaN` during the opening range bars themselves (before the range is fully established). This prevents look-ahead bias.
- **Session hours are UTC**: Configure the `sessions` parameter according to your data's timezone. The defaults assume UTC-indexed data targeting major global session opens.
- **Stationarity**: This plugin does not benefit from stationary input data (`benefits_from_stationary = False`). The features are inherently relative/normalized.
- **Total feature count**: With default parameters, the plugin produces 28 features (5 rolling + 20 session + 3 statistical). Disabling feature groups via `enable_*` flags reduces dimensionality.
