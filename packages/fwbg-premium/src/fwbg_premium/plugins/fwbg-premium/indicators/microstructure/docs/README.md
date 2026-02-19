# Microstructure

Analyzes intra-bar dynamics and market microstructure signals hidden within OHLC candlestick data, extracting information about buying/selling pressure, wick patterns, and volume-weighted money flow.

## Concept

Traditional technical analysis focuses on price levels and trend direction, but each candlestick contains a wealth of structural information that is rarely exploited by ML models. The microstructure plugin deconstructs every bar into its component parts -- body size, upper/lower wicks, and their ratios -- to reveal the intra-bar battle between buyers and sellers. A bar with a large upper wick relative to its lower wick, for example, signals that sellers absorbed buying pressure at the highs.

These features are grounded in order flow theory: the relationship between open, high, low, and close prices reveals where liquidity was absorbed and which side dominated. The pressure score combines direction (sign of close minus open) with body strength (body as fraction of range) to produce a single measure of conviction. Rolling accumulations of these metrics smooth out noise and expose persistent imbalances over multiple bars.

When volume data is available, the plugin computes volume-weighted variants including the Accumulation/Distribution Line (A/D) and Chaikin Money Flow (CMF). The A/D line tracks cumulative volume-weighted price positioning within each bar's range, while CMF normalizes this over rolling windows to detect sustained institutional buying or selling. ML models can use these features to distinguish between genuine breakouts backed by volume and false moves driven by thin liquidity.

## Features

| Feature | Description |
|---------|-------------|
| `micro_wick_imbalance` | (Upper Wick - Lower Wick) / Bar Range. Positive = selling pressure at highs, negative = buying pressure at lows |
| `micro_intrabar_bias` | (Close - Open) / Bar Range. Positive = bullish bar, negative = bearish bar |
| `micro_body_ratio` | Abs(Close - Open) / Bar Range. How much of the range is body vs. wick (0 = doji, 1 = marubozu) |
| `micro_range_over_atr` | Bar Range / ATR. Normalized bar size relative to recent volatility (>1 = expansion, <1 = contraction) |
| `micro_pressure_score` | sign(Close - Open) * body_ratio. Directional conviction per bar |
| `micro_wick_imbalance_sum` | Rolling sum of wick imbalance over the rolling window. Accumulated directional wick pressure |
| `micro_intrabar_bias_sum` | Rolling sum of intrabar bias over the rolling window. Accumulated bullish/bearish bias |
| `micro_pressure_sum` | Rolling sum of pressure score over the rolling window. Accumulated directional conviction |
| `micro_upper_shadow_max` | Rolling max of upper wick ratio over the rolling window. Peak selling absorption in recent bars |
| `micro_lower_shadow_max` | Rolling max of lower wick ratio over the rolling window. Peak buying absorption in recent bars |
| `micro_direction_consistency` | Absolute value of rolling mean of bar direction (sign of C-O). 1 = all bars same direction, 0 = mixed |
| `micro_vwap_pressure` | Volume-weighted pressure score (rolling). Higher conviction when backed by volume |
| `micro_relative_volume` | Current volume / rolling average volume. Detects volume spikes (>1) and dry-ups (<1) |
| `micro_ad_line` | Accumulation/Distribution Line. Cumulative volume-weighted close location value |
| `micro_ad_zscore` | Z-score of A/D line vs. its 50-bar rolling mean/std. Stationary version for ML consumption |
| `micro_cmf_10` | Chaikin Money Flow (10-bar window). Short-term institutional money flow |
| `micro_cmf_20` | Chaikin Money Flow (20-bar window). Medium-term institutional money flow |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atr_period` | int | 14 | Period for ATR calculation used to normalize bar range |
| `rolling_window` | int | 5 | Window size for rolling accumulations (sums, maxima, consistency) |

## Usage Notes

- All features are shifted by 1 bar to prevent lookahead bias. At bar `i`, the model sees features computed from bar `i-1`.
- Volume-dependent features (`micro_vwap_pressure`, `micro_relative_volume`, `micro_ad_line`, `micro_ad_zscore`, `micro_cmf_10`, `micro_cmf_20`) use volume data when available (column `V`). When volume is absent, fallback approximations based on price-only Close Location Value are used.
- The `micro_ad_line` is a cumulative feature and therefore non-stationary by nature. Use `micro_ad_zscore` instead for ML models that expect stationary inputs.
- Requires OHLC columns (`O`, `H`, `L`, `C`). Volume column (`V`) is optional but recommended.
- The `manifest.json` sets `benefits_from_stationary: false` since most features are already ratios or bounded values.
