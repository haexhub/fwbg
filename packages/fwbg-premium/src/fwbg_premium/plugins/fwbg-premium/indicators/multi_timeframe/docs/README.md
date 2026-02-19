# Multi-Timeframe

Computes higher-timeframe features (H4, D1, W1, Y1) from H1 bar data, including trend alignment scores, EMA distances, volatility ratios, and momentum divergences across timeframes.

## Concept

Multi-timeframe analysis is a core principle of professional trading: higher timeframes reveal the dominant trend ("big picture"), while lower timeframes provide entry timing. When all timeframes align in the same direction, setups are statistically stronger. When timeframes diverge, it often signals potential reversals or consolidation phases.

This plugin synthesizes H1 (hourly) data into four higher timeframes -- H4 (4-hour), D1 (daily), W1 (weekly), and Y1 (yearly) -- by computing rolling windows of appropriate length. For each timeframe it calculates range position (where price sits within the period's high-low range), EMA distance (how far price deviates from its moving average), trend strength, and technical indicators like ADX, RSI, and Bollinger Band position. The yearly features provide macro context through the 200-day EMA distance and 52-week range position.

The trend alignment features are particularly valuable for ML models: they quantify whether adjacent timeframes agree on direction (H1-H4, H4-D1, D1-W1) and compute a consensus score when all four align. The volatility ratio between H1 and H4 reveals whether short-term volatility is expanding or contracting relative to the intermediate trend. RSI divergence between H1 and H4 can flag momentum exhaustion before price turns.

## Features

| Feature | Description |
|---------|-------------|
| `mtf_h4_trend` | H4 candle trend: (Close - Open_4bars_ago) / H4_Range. Direction and strength of 4-hour move |
| `mtf_h4_range_pos` | Position of current close within the H4 high-low range (0 = at low, 1 = at high) |
| `mtf_h4_ema20_dist` | Distance from H4 EMA(20): (Close - EMA) / Close. Normalized deviation from H4 short-term trend |
| `mtf_h4_ema50_dist` | Distance from H4 EMA(50): (Close - EMA) / Close. Normalized deviation from H4 medium-term trend |
| `mtf_h4_adx` | Average Directional Index on H4 data. Measures trend strength (>25 = trending, <20 = ranging) |
| `mtf_h4_rsi` | RSI computed on H4-equivalent period. Overbought/oversold on H4 timeframe |
| `mtf_h4_atr_pct` | H4 ATR as percentage of close. H4 volatility level |
| `mtf_h4_bb_pband` | Bollinger Band %B on H4 timeframe. Position within Bollinger Bands (0-1 = inside, >1 or <0 = outside) |
| `mtf_d1_range_pos` | Position of current close within the D1 high-low range |
| `mtf_d1_ema20_dist` | Distance from D1 EMA(20) normalized by close |
| `mtf_d1_ema50_dist` | Distance from D1 EMA(50) normalized by close |
| `mtf_d1_trend_strength` | Percentage change of D1 slow EMA over one daily period. Daily trend momentum |
| `mtf_w1_range_pos` | Position of current close within the W1 high-low range |
| `mtf_w1_ema20_dist` | Distance from W1 EMA(20) normalized by close |
| `mtf_w1_ema50_dist` | Distance from W1 EMA(50) normalized by close |
| `mtf_w1_trend_strength` | Percentage change of W1 slow EMA over one weekly period. Weekly trend momentum |
| `mtf_y1_ema200d_dist` | Distance from the 200-day EMA normalized by close. Classic bull/bear market indicator |
| `mtf_y1_52w_range_pos` | Position within the 52-week high-low range (0 = at 52w low, 1 = at 52w high) |
| `mtf_y1_52w_high_dist` | Distance from 52-week high as fraction of close. How far below the yearly high |
| `mtf_y1_52w_low_dist` | Distance from 52-week low as fraction of close. How far above the yearly low |
| `mtf_trend_alignment_h1h4` | Binary: 1 if H1 and H4 trends agree in direction, 0 otherwise |
| `mtf_trend_alignment_h4d1` | Binary: 1 if H4 and D1 trends agree in direction, 0 otherwise |
| `mtf_trend_alignment_d1w1` | Binary: 1 if D1 and W1 trends agree in direction, 0 otherwise |
| `mtf_consensus` | Binary: 1 if all four timeframes (H1, H4, D1, W1) agree in direction |
| `mtf_trend_strength` | Sum of alignment scores (0-3). Number of timeframe pairs in agreement |
| `mtf_vol_ratio_h1h4` | H1 ATR% / H4 ATR%. Intraday volatility relative to intermediate-term (>1 = expanding) |
| `mtf_rsi_divergence` | H1 RSI - H4 RSI. Momentum divergence between timeframes |
| `mtf_d1_above_prev_high` | Binary: 1 if close is above the previous day's high |
| `mtf_d1_below_prev_low` | Binary: 1 if close is below the previous day's low |
| `mtf_d1_dist_to_high` | Normalized distance from the previous day's high to current close |
| `mtf_d1_dist_to_low` | Normalized distance from current close to the previous day's low |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h4_bars` | int | 4 | Number of H1 bars per H4 candle |
| `d1_bars` | int | 24 | Number of H1 bars per D1 candle (24 = 24 hours) |
| `w1_bars` | int | 120 | Number of H1 bars per W1 candle (120 = 5 trading days * 24 hours) |
| `ema_periods` | List[int] | [20, 50] | EMA periods applied at each timeframe. Generates features for each period |
| `include_yearly` | bool | True | Whether to compute yearly features (200d EMA, 52-week range). Requires significant history |

## Usage Notes

- Designed for H1 (hourly) input data. The `h4_bars`, `d1_bars`, and `w1_bars` parameters control how many H1 bars constitute one higher-timeframe candle.
- Yearly features (`mtf_y1_*`) require substantial history: the 200-day EMA needs ~4800 H1 bars, and 52-week range needs ~6240 H1 bars. Set `include_yearly: false` for shorter datasets.
- The `manifest.json` sets `benefits_from_stationary: true` since EMA distance features can benefit from stationary price inputs.
- All features are shifted by 1 bar to prevent lookahead bias.
- The alignment features (`mtf_trend_alignment_*`, `mtf_consensus`, `mtf_trend_strength`) are derived from the EMA(20) distance at each timeframe, so they depend on the `ema_periods` parameter containing 20.
