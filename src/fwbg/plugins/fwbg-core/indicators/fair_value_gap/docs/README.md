# Fair Value Gap (FVG)

Detects 3-candle imbalance zones from Smart Money Concepts (SMC) and tracks the current price's relationship to active unfilled gaps.

## Concept

A Fair Value Gap is a price imbalance created when a strong directional move leaves a gap between the wicks of three consecutive candles. A **bullish FVG** occurs when candle 3's low is above candle 1's high, creating a gap-up zone that theoretically acts as support. A **bearish FVG** occurs when candle 3's high is below candle 1's low, creating a gap-down zone that acts as resistance.

The core idea from Smart Money Concepts is that institutional order flow creates these imbalances, and price tends to return to "fill" them before continuing. An unfilled FVG represents an area where buy/sell orders may still be resting. Once price penetrates through the gap zone, it is considered "filled" and loses its significance.

This plugin tracks all active (unfilled) FVGs within a configurable lookback window and computes features describing the nearest bullish and bearish gaps: whether they exist, how far away they are, and how large they are. All distance and size measurements are normalized by ATR for scale-independence. ML models can use these features to identify potential support/resistance zones, detect imbalance-driven reversals, and gauge the structural strength of recent price moves.

## Features

| Feature | Description |
|---------|-------------|
| `fvg_bull_active` | Binary flag (1/0) indicating at least one active (unfilled) bullish FVG exists below the current price. |
| `fvg_bear_active` | Binary flag (1/0) indicating at least one active (unfilled) bearish FVG exists above the current price. |
| `fvg_bull_dist` | ATR-normalized distance from the current close to the midpoint of the nearest active bullish FVG below price. `NaN` when no bullish FVG is active. |
| `fvg_bear_dist` | ATR-normalized distance from the current close to the midpoint of the nearest active bearish FVG above price. `NaN` when no bearish FVG is active. |
| `fvg_bull_size` | ATR-normalized size (top - bottom) of the nearest active bullish FVG. Larger gaps indicate stronger imbalances. `NaN` when no bullish FVG is active. |
| `fvg_bear_size` | ATR-normalized size (top - bottom) of the nearest active bearish FVG. `NaN` when no bearish FVG is active. |
| `fvg_in_gap` | Binary flag (1/0) indicating the current close is inside an active FVG zone (bullish or bearish). Price inside a gap suggests it is being filled. |
| `fvg_count` | Total number of currently active (unfilled) FVGs within the lookback window. High counts may indicate a volatile, imbalance-rich environment. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atr_period` | int | `14` | ATR lookback period used to normalize FVG distances and sizes. FVG distances and gap sizes are expressed in ATR units for scale-independence across different price levels and instruments. Range: 2-500. |
| `lookback` | int | `100` | Maximum number of bars an unfilled FVG remains active. After this many bars, stale gaps are discarded. Longer lookbacks track more historical gaps but may include zones that have lost their significance as support/resistance. Range: 10-1000, step 10. |

## Usage Notes

- **FVG detection logic**: A bullish FVG is detected when `High[i-2] < Low[i]` (gap up between candle 1 and candle 3). A bearish FVG is detected when `Low[i-2] > High[i]` (gap down). The middle candle is the "impulse" candle that created the imbalance.
- **Fill conditions**: A bullish FVG is filled (removed) when the current bar's low penetrates below the gap bottom. A bearish FVG is filled when the current bar's high penetrates above the gap top.
- **Distance sign convention**: `fvg_bull_dist` is positive when price is above the bullish gap (the expected condition). `fvg_bear_dist` is positive when price is below the bearish gap. Only the nearest gap in each direction is reported.
- **Warmup period**: The first 2 bars cannot produce FVGs (requires 3 candles). The ATR computation uses a rolling window with `min_periods=1`, so it is available from the first bar.
- **Stationarity**: This plugin does not benefit from stationary input data (`benefits_from_stationary = False`). All features are either binary or ATR-normalized.
- **Works on all timeframes**: Unlike some indicators, FVGs can be detected on any timeframe from M1 to daily.
