# Ichimoku

Computes the full Ichimoku Kinko Hyo cloud system with all five traditional lines plus derived features for cloud position, thickness, crosses, and composite signals.

## Concept

The Ichimoku Kinko Hyo ("one-glance equilibrium chart") is a comprehensive technical analysis system developed by Goichi Hosoda. Unlike most indicators that measure a single dimension, Ichimoku provides a complete picture of support/resistance, trend direction, momentum, and future price zones in a single framework. The system consists of five lines: Tenkan-sen (conversion), Kijun-sen (base), Senkou Span A and B (forming the "cloud" or "kumo"), and Chikou Span (lagging).

The cloud (kumo) acts as a dynamic support/resistance zone. Price above the cloud indicates a bullish regime; price below indicates bearish; price inside the cloud indicates indecision. The thickness of the cloud reflects the strength of support/resistance. Tenkan-Kijun crosses function as entry signals, analogous to moving average crossovers but adapted to the Ichimoku framework. The Chikou Span confirms trend direction by comparing the current close to a historical value.

This plugin normalizes all Ichimoku components into ML-friendly features: relative distances (as percentages of price), binary cloud-position flags, cross signals, and composite scores. These features allow ML models to learn the rich multi-dimensional Ichimoku state space, including nuances like "price above a bullish cloud with a fresh TK bullish cross" (strong bullish) versus "price above a bearish cloud" (conflicting signals).

## Features

| Feature | Description |
|---------|-------------|
| `ichi_tenkan` | Tenkan-sen (conversion line): midpoint of highest high and lowest low over tenkan_period |
| `ichi_kijun` | Kijun-sen (base line): midpoint of highest high and lowest low over kijun_period |
| `ichi_senkou_a` | Senkou Span A: midpoint of Tenkan and Kijun, projected forward |
| `ichi_senkou_b` | Senkou Span B: midpoint of highest high and lowest low over senkou_b_period, projected forward |
| `ichi_cloud_thick` | Cloud thickness normalized by close price: (cloud_top - cloud_bottom) / close |
| `ichi_cloud_pos` | Price position within the cloud: (close - cloud_bottom) / (cloud_top - cloud_bottom). Values > 1 = above cloud, < 0 = below cloud |
| `ichi_above_cloud` | Binary: price is above the cloud top |
| `ichi_below_cloud` | Binary: price is below the cloud bottom |
| `ichi_in_cloud` | Binary: price is inside the cloud |
| `ichi_tk_cross` | Tenkan-Kijun spread normalized by close: (tenkan - kijun) / close |
| `ichi_tk_bullish_cross` | Binary: Tenkan crosses above Kijun on this bar |
| `ichi_tk_bearish_cross` | Binary: Tenkan crosses below Kijun on this bar |
| `ichi_price_kijun` | Distance from price to Kijun-sen normalized by close: (close - kijun) / close |
| `ichi_kijun_flat` | Binary: Kijun-sen is flat (change < 0.01% of price), indicating consolidation |
| `ichi_bullish_cloud` | Binary: Senkou Span A > Senkou Span B (cloud color is bullish) |
| `ichi_kumo_twist` | Binary: cloud color changes on this bar (Senkou A and B cross) |
| `ichi_chikou_above` | Binary: current close > close from kijun_period bars ago (Chikou Span confirmation) |
| `ichi_strong_bullish` | Binary: price above cloud AND TK spread positive AND cloud is bullish |
| `ichi_strong_bearish` | Binary: price below cloud AND TK spread negative AND cloud is bearish |
| `ichi_neutral` | Binary: price in cloud OR conflicting signals between position and cloud color |
| `ichi_dist_to_cloud` | Signed distance to nearest cloud edge normalized by price. Positive when above cloud, negative when below |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenkan_period` | `int` | `9` | Window for the Tenkan-sen (conversion line) calculation |
| `kijun_period` | `int` | `26` | Window for the Kijun-sen (base line) and Chikou Span offset |
| `senkou_b_period` | `int` | `52` | Window for the Senkou Span B calculation |

## Usage Notes

- The default periods (9, 26, 52) are Hosoda's original values, designed for daily bars. On hourly data, these correspond to shorter time horizons; consider scaling if needed.
- Senkou Span B requires 52 bars of data before producing valid values, making it the binding warmup constraint.
- The raw line values (`ichi_tenkan`, `ichi_kijun`, `ichi_senkou_a`, `ichi_senkou_b`) are absolute price levels and are not stationary. Set `benefits_from_stationary: true` in the manifest to enable automatic stationarity transforms on these features.
- Normalized features (`ichi_cloud_thick`, `ichi_cloud_pos`, `ichi_tk_cross`, `ichi_price_kijun`, `ichi_dist_to_cloud`) are price-relative and more suitable for cross-instrument models.
- All features are shifted by 1 bar to prevent lookahead bias.
