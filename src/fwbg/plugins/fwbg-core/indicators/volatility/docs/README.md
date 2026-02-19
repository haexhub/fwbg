# Volatility Indicators

Comprehensive volatility measurement features combining channel-based indicators, OHLC-efficient volatility estimators, compression detection, and realized-vs-implied volatility analysis.

## Concept

Volatility is the single most important variable in risk management and position sizing. This plugin goes far beyond simple ATR measurement by incorporating three complementary approaches: traditional channel-based indicators (Bollinger Bands, Keltner, Donchian), advanced OHLC volatility estimators (Garman-Klass, Parkinson, Yang-Zhang), and volatility regime detection through percentile ranking and compression flags.

The OHLC volatility estimators deserve special attention. Close-to-close volatility (the standard approach) discards the vast majority of intrabar price information. The Garman-Klass estimator uses all four OHLC prices and is approximately 7.4 times more efficient than close-only volatility. Parkinson uses the high-low range and is about 5 times more efficient. Yang-Zhang combines overnight return variance, close-to-close variance, and Rogers-Satchell variance, making it the most robust estimator -- it handles both drift and opening jumps correctly. Their formulas:

- **Garman-Klass**: sigma_GK = sqrt(0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2)
- **Parkinson**: sigma_P = sqrt(ln(H/L)^2 / (4 * ln(2)))
- **Yang-Zhang**: sigma_YZ = sqrt(sigma_O^2 + k * sigma_C^2 + (1-k) * sigma_RS^2)

For ML models, volatility features serve multiple purposes. They are direct inputs for position sizing and risk normalization. The volatility compression flag identifies low-volatility regimes that historically precede explosive moves. The ratio between Yang-Zhang and ATR captures volatility regime shifts. Percentile rankings provide a stationary view of whether current volatility is historically high or low.

## Features

| Feature | Description |
|---------|-------------|
| `vol_atr` | Raw Average True Range (14-period). The absolute ATR value, used internally for calculations and as a direct feature. |
| `vol_atr_pct_7` | ATR (7-period) as a percentage of close price. Fast-reacting normalized volatility measure. |
| `vol_atr_pct_14` | ATR (14-period) as a percentage of close price. Standard normalized volatility baseline. |
| `vol_atr_pct_21` | ATR (21-period) as a percentage of close price. Smoother normalized volatility. |
| `vol_bb_pband_20` | Bollinger Bands %B (20-period). Price position within the bands: 0 = at lower band, 0.5 = at middle, 1 = at upper band. Values outside 0-1 indicate price beyond the bands. |
| `vol_bb_wband_20` | Bollinger Bands width (20-period). Distance between upper and lower bands normalized by middle band. Measures volatility expansion/contraction. |
| `vol_kc_pband` | Keltner Channel %B. Price position within the ATR-based channel envelope. |
| `vol_kc_wband` | Keltner Channel width. ATR-based channel width, complementary to Bollinger bandwidth for squeeze detection. |
| `vol_dc_pband` | Donchian Channel %B. Price position within the N-period high-low channel. |
| `vol_dc_wband` | Donchian Channel width. Width of the price range channel, capturing breakout potential. |
| `vol_gk_20` | Garman-Klass volatility estimate (20-bar window). OHLC-efficient estimator, ~7.4x more efficient than close-only. |
| `vol_gk_50` | Garman-Klass volatility estimate (50-bar window). Smoother long-term OHLC volatility. |
| `vol_parkinson_20` | Parkinson volatility estimate (20-bar window). High-low range-based estimator, ~5x more efficient than close-only. |
| `vol_parkinson_50` | Parkinson volatility estimate (50-bar window). Smoother long-term range-based volatility. |
| `vol_yz_20` | Yang-Zhang volatility estimate (20-bar window). The most robust OHLC estimator, handles drift and opening jumps. |
| `vol_yz_50` | Yang-Zhang volatility estimate (50-bar window). Long-term robust volatility estimate. |
| `vol_yz_atr_ratio` | Ratio of Yang-Zhang (20) to ATR percent (14). Divergence between these two volatility measures signals a regime shift (e.g., overnight gaps vs. intrabar movement). |
| `vol_atr_pct_7_rank` | Percentile rank of ATR% (7-period) within a rolling window. 0.0 = historically low, 1.0 = historically high. |
| `vol_atr_pct_14_rank` | Percentile rank of ATR% (14-period) within a rolling window. |
| `vol_atr_pct_21_rank` | Percentile rank of ATR% (21-period) within a rolling window. |
| `vol_bb_wband_20_rank` | Percentile rank of Bollinger bandwidth within a rolling window. Low values indicate a volatility squeeze. |
| `vol_compression` | Binary volatility compression flag. 1.0 when both ATR-14 rank and BB width rank are below the 20th percentile simultaneously, indicating extreme low volatility that often precedes a breakout. |
| `vol_rv_20` | Annualized realized volatility (20-period, using 24h bars). Calculated from log returns, annualized as percentage (sqrt(252*24) * std). |
| `vol_rv_iv_spread` | Realized volatility minus implied volatility (VIX). Positive = realized vol exceeds market expectations. Only computed when `macro_vix` column is present. |
| `vol_rv_iv_ratio` | Ratio of realized to implied volatility. Values above 1.0 mean the market is underpricing volatility. Only computed when `macro_vix` column is present. |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atr_periods` | list[int] | [7, 14, 21] | Periods for ATR calculation, expressed as percentage of price. ATR measures market volatility by decomposing the entire range of a bar. Shorter periods capture recent volatility spikes, longer periods give a smoother baseline. Min: 2, Max: 500. |
| `bb_period` | int | 20 | Lookback period for Bollinger Bands. Defines the SMA center and standard deviation envelope width. The classic 20-period setting corresponds roughly to one trading month. Also used for %B (price position) and bandwidth (volatility squeeze) features. Min: 5, Max: 500. |
| `vol_est_windows` | list[int] | [20, 50] | Rolling window sizes for OHLC-based volatility estimators (Garman-Klass, Parkinson, Yang-Zhang). These estimators are more statistically efficient than close-only volatility. Shorter windows react faster to regime changes, longer windows provide more stable estimates. Min: 5, Max: 500. |

## Usage Notes

- This plugin does not benefit from stationarity transformations (`benefits_from_stationary: false`), as most features are already ratios, percentages, or bounded values.
- All features are shifted by 1 bar to prevent lookahead bias.
- The `vol_rv_iv_spread` and `vol_rv_iv_ratio` features are only computed when a `macro_vix` column is present in the input DataFrame. If VIX data is unavailable, these features will be absent.
- The compression detection uses a rolling window of 100 bars (configurable via `compression_lookback` in `**params`) for percentile ranking.
- Realized volatility calculation assumes 24-hour bars (crypto markets). The annualization factor is `sqrt(252 * 24) * 100`. Adjust if using different bar frequencies.
- The Yang-Zhang estimator is the most computationally expensive but also the most robust. It correctly handles overnight gaps, which is important for markets with distinct sessions.
- The `vol_yz_atr_ratio` is only computed when 20 is included in `vol_est_windows`.
- The `_atr` internal feature is computed but not exposed in `get_feature_columns()` for external use; `vol_atr` serves as the public reference ATR.
