# Price Action Indicators

Raw price structure features derived directly from OHLC candlestick data, including candle anatomy, trend structure analysis, gap patterns, streak detection, and optional volume-based confirmation signals.

## Concept

Price action analysis extracts trading information from the most fundamental market data: the open, high, low, and close prices of each bar. While most technical indicators are derived transformations of price, price action features capture the raw microstructure -- how buyers and sellers interacted within each bar, whether the market is making higher highs, whether gaps are being filled, and whether institutional volume confirms the move.

The candle anatomy features (body ratio, range position, shadow ratios) encode the outcome of the supply/demand battle within a single bar. A bar with a large body ratio and high range position shows strong buying conviction. A bar with long upper shadows shows selling pressure at highs. The Higher Highs / Lower Lows framework extends this to multi-bar trend structure -- a market making consistent HH and HL (higher lows) is in a structural uptrend regardless of what oscillators say. Gap analysis captures overnight sentiment shifts and the tendency of prices to "fill" gaps, which is a well-documented market behavior.

For ML models, price action features are among the most predictive because they are closest to the raw data. They are also among the most stationary -- ratios like body ratio and range position are naturally bounded between 0 and 1. The trend structure score provides a single composite measure of whether the market is structurally trending (positive values) or declining (negative values). Volume features, when available, add confirmation or divergence signals that significantly improve prediction quality.

## Features

| Feature | Description |
|---------|-------------|
| `pa_range_pos` | Range position: where close falls within the high-low range. 0.0 = closed at the low, 1.0 = closed at the high. Measures intrabar buying pressure. |
| `pa_body_ratio` | Body ratio: size of the candle body relative to the full range. 0.0 = doji (open equals close), 1.0 = marubozu (full body, no shadows). |
| `pa_body_dir` | Body direction: +1 for bullish (close > open), -1 for bearish (close < open), 0 for doji (close = open). |
| `pa_upper_shadow` | Upper shadow ratio: size of the upper wick relative to the full bar range. High values indicate selling pressure at highs. |
| `pa_lower_shadow` | Lower shadow ratio: size of the lower wick relative to the full bar range. High values indicate buying pressure at lows. |
| `pa_hh` | Higher Highs count: rolling sum of bars that made a higher high than the previous bar, over the lookback period. |
| `pa_ll` | Lower Lows count: rolling sum of bars that made a lower low than the previous bar, over the lookback period. |
| `pa_hl` | Higher Lows count: rolling sum of bars where the low was higher than the previous bar's low. Indicates bullish structure. |
| `pa_lh` | Lower Highs count: rolling sum of bars where the high was lower than the previous bar's high. Indicates bearish structure. |
| `pa_trend_structure` | Trend structure score: (HH + HL) - (LL + LH). Positive values indicate bullish market structure, negative values bearish. Composite structural trend measure. |
| `pa_gap` | Gap size: percentage difference between current open and previous close. Positive = gap up, negative = gap down. |
| `pa_gap_abs` | Absolute gap size: magnitude of the gap regardless of direction. |
| `pa_gap_dir` | Gap direction: +1 for gap up (> 0.1%), -1 for gap down (< -0.1%), 0 for no significant gap. Uses a 0.1% threshold. |
| `pa_gap_filled` | Gap fill indicator: 1 if the current bar's range reached back to the previous close (filling the gap), 0 otherwise. |
| `pa_bullish_streak` | Bullish streak length: number of consecutive bullish candles (close > open). Resets to 0 on a bearish or doji candle. |
| `pa_bearish_streak` | Bearish streak length: number of consecutive bearish candles (close < open). Resets to 0 on a bullish or doji candle. |
| `pa_range_expansion` | Range expansion ratio: current bar's range divided by the 20-bar average range. Values > 1.0 indicate an unusually wide bar (volatility expansion). |
| `pa_inside_bar` | Inside bar flag: 1 when the current bar's range is completely within the previous bar's range (H < prev H and L > prev L). Often precedes breakouts. |
| `pa_outside_bar` | Outside bar flag: 1 when the current bar engulfs the previous bar's range (H > prev H and L < prev L). Signals strong momentum or reversal. |
| `vol_obv_change` | On Balance Volume 5-bar change (percentage). Measures OBV momentum. Positive = accumulation, negative = distribution. Only computed when volume data is available. |
| `vol_mfi` | Money Flow Index (14-period). Volume-weighted RSI, range 0-100. Combines price and volume to identify overbought/oversold with volume confirmation. Only computed when volume data is available. |
| `vol_relative` | Relative volume: current volume divided by 20-bar average volume. Values > 1.0 indicate above-average activity. Only computed when volume data is available. |
| `vol_price_trend` | Volume-price trend: body direction multiplied by relative volume. Positive values = bullish move on high volume (strong confirmation). Only computed when volume data is available. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hh_ll_period` | int | 5 | Rolling window for Higher Highs / Lower Lows / Higher Lows / Lower Highs counters and trend structure score. Counts how many bars made new highs or lows within this window. Shorter values capture micro-structure, longer values capture swing structure. Min: 2, Max: 100. |
| `compute_volume` | bool | True | Whether to compute volume-based features (OBV change, MFI, relative volume, volume-price trend). Requires a volume column (`V` or `Volume`) in the DataFrame. Set to False when volume data is unavailable or unreliable (e.g., crypto spot with fragmented liquidity). |

## Usage Notes

- This plugin benefits from stationarity transformations applied beforehand (`benefits_from_stationary: true` in manifest), particularly for the gap and range features.
- All features are shifted by 1 bar to prevent lookahead bias.
- Candle anatomy features (`pa_range_pos`, `pa_body_ratio`, `pa_upper_shadow`, `pa_lower_shadow`) are naturally bounded between 0 and 1.
- Volume features (`vol_obv_change`, `vol_mfi`, `vol_relative`, `vol_price_trend`) are only computed when a volume column (`V` or `Volume`) is present in the input DataFrame and `compute_volume` is True. If volume is absent, these four features will be missing from the output.
- The gap threshold for directional classification is hardcoded at 0.1% -- gaps smaller than this are classified as direction 0 (no gap).
- The `pa_range_expansion` feature uses a fixed 20-bar rolling mean for normalization.
- Streak features reset to zero on direction change. Very long streaks (> 5-7 bars) are rare and often signal trend exhaustion.
- Inside bars and outside bars are mutually exclusive events that carry different market implications: inside bars signal consolidation/indecision, outside bars signal strong directional conviction or key reversals.
