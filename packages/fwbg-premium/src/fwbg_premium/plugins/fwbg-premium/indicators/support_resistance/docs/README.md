# Support & Resistance

Identifies support and resistance zones on H1 and D1 timeframes using swing point detection and clustering, classifies trend strength using Rayner Teo-style moving average alignment, and produces interaction features for ML-driven trading decisions.

## Concept

Support and resistance levels are among the most fundamental concepts in technical analysis. Prices tend to reverse or consolidate at levels where they have previously reversed, because these levels represent zones of concentrated supply (resistance) or demand (support). This plugin automates the detection of these zones by identifying swing highs and swing lows across multiple lookback periods, then clustering nearby levels into zones using ATR-based distance thresholds.

The plugin operates on two timeframes: H1 (using raw swing periods) and D1 (using scaled swing periods that capture daily-level structure from hourly data). For each timeframe, it computes the ATR-normalized distance to the nearest support and resistance zones, the "strength" of each zone (number of touches/confluences), whether price is currently inside a zone, and whether the nearest zone is a "flip zone" (a level that has acted as both support and resistance). Flip zones are particularly significant because they represent polarity changes where former resistance becomes new support and vice versa.

Trend classification follows the Rayner Teo methodology, based on the alignment of 20, 50, and 200-period moving averages. The trend class ranges from -3 (strong downtrend: MA20 < MA50 < MA200 with price below MA20) to +3 (strong uptrend: MA20 > MA50 > MA200 with price above MA20), with 0 representing sideways (MAs not aligned). The interaction features combine S/R proximity with trend context to produce actionable signals: "at support in uptrend" (pullback entry), "at resistance in downtrend" (rally short), and range-trading signals for sideways markets.

## Features

### H1 S/R Features

| Feature | Description |
|---------|-------------|
| `sr_dist_nearest_support` | ATR-normalized distance to nearest H1 support zone below price |
| `sr_dist_nearest_resistance` | ATR-normalized distance to nearest H1 resistance zone above price |
| `sr_support_strength` | Number of swing low touches/confluences at nearest H1 support zone |
| `sr_resistance_strength` | Number of swing high touches/confluences at nearest H1 resistance zone |
| `sr_in_support_zone` | Binary: 1 if price is within zone_proximity ATR of the nearest H1 support |
| `sr_in_resistance_zone` | Binary: 1 if price is within zone_proximity ATR of the nearest H1 resistance |
| `sr_nearest_is_flip_zone` | Binary: 1 if the nearest H1 S/R zone has acted as both support and resistance |

### D1 S/R Features

| Feature | Description |
|---------|-------------|
| `sr_d1_dist_nearest_support` | ATR-normalized distance to nearest D1 support zone below price |
| `sr_d1_dist_nearest_resistance` | ATR-normalized distance to nearest D1 resistance zone above price |
| `sr_d1_support_strength` | Number of touches at nearest D1 support zone |
| `sr_d1_resistance_strength` | Number of touches at nearest D1 resistance zone |
| `sr_d1_in_support_zone` | Binary: 1 if price is within zone_proximity ATR of D1 support |
| `sr_d1_in_resistance_zone` | Binary: 1 if price is within zone_proximity ATR of D1 resistance |
| `sr_d1_nearest_is_flip_zone` | Binary: 1 if the nearest D1 zone has acted as both support and resistance |

### Trend Features

| Feature | Description |
|---------|-------------|
| `sr_trend_class` | Rayner Teo trend classification: -3 to +3 (strong down to strong up, 0 = sideways) |
| `sr_pullback_depth` | ATR-normalized depth of pullback from recent swing extreme in the trend direction |
| `sr_ma_alignment` | Moving average alignment score: -1 to +1. Based on MA20/MA50/MA200 ordering |
| `sr_price_vs_ma20` | (Close - MA20) / ATR. ATR-normalized distance from 20-period MA |
| `sr_price_vs_ma50` | (Close - MA50) / ATR. ATR-normalized distance from 50-period MA |
| `sr_price_vs_ma200` | (Close - MA200) / ATR. ATR-normalized distance from 200-period MA |
| `sr_trend_break` | Trend structure violation: +1 if price breaks above last swing high in downtrend, -1 if below last swing low in uptrend, 0 otherwise |

### Interaction Features

| Feature | Description |
|---------|-------------|
| `sr_at_support_in_uptrend` | Binary: 1 when near H1 support AND trend > 0 (pullback buy setup) |
| `sr_at_resistance_in_downtrend` | Binary: 1 when near H1 resistance AND trend < 0 (rally short setup) |
| `sr_at_support_in_range` | Binary: 1 when near H1 support AND trend = 0 (range buy setup) |
| `sr_at_resistance_in_range` | Binary: 1 when near H1 resistance AND trend = 0 (range sell setup) |
| `sr_range_width` | Total ATR-normalized width between nearest support and resistance |
| `sr_range_position` | Position within the S/R range: 0 = at support, 1 = at resistance |
| `sr_breakout_up` | Binary: 1 on upward breakout from resistance zone |
| `sr_breakout_down` | Binary: 1 on downward breakdown from support zone |
| `sr_at_flipped_support` | Binary: 1 when price is near a former resistance level that may now act as support (temporal flip) |
| `sr_at_flipped_resistance` | Binary: 1 when price is near a former support level that may now act as resistance (temporal flip) |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swing_periods` | list | [5, 10, 20] | Lookback/lookahead periods for swing point detection. Multiple periods capture swings at different scales |
| `lookback` | int | 200 | Number of bars to look back for collecting swing levels into S/R zones |
| `cluster_threshold` | float | 1.5 | ATR multiplier for clustering nearby levels into a single zone. Levels within this distance are merged |
| `atr_period` | int | 14 | Period for ATR computation used in normalization and zone proximity |
| `ma_periods` | list | [20, 50, 200] | Moving average periods for trend classification (Rayner Teo style) |
| `zone_proximity_atr_mult` | float | 0.5 | ATR multiplier defining "in zone" proximity. Price within this distance is considered at the S/R level |
| `d1_bars` | int | 24 | Number of H1 bars per D1 candle. Used to scale swing periods for D1-level detection |

## Usage Notes

- All features are shifted by 1 bar to prevent lookahead bias.
- Swing detection is lookahead-safe by design: a swing high at index `j` is only confirmed at index `j + period`, when enough subsequent bars have formed to validate it as a true swing.
- The D1 S/R zones are computed by scaling the H1 swing periods by `d1_bars` (default: 24x). This captures daily-level structure without requiring separate daily candle data.
- Zone strength (number of touches) indicates confluence -- zones with more touches across multiple swing periods are more significant.
- The `manifest.json` sets `benefits_from_stationary: false` since the plugin works with raw price levels and ATR normalization.
- The `zone_proximity_atr_mult` parameter controls how close price needs to be to a zone to trigger "in zone" signals. The interaction features use `2 * zone_proximity_atr_mult` as their proximity threshold.
- Trend break detection uses the middle swing period from `swing_periods` to identify structural violations.
