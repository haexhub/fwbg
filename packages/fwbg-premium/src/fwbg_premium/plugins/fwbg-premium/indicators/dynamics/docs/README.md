# Dynamics

Computes rate-of-change, lag, and acceleration features for core technical indicators, capturing how fast market conditions are evolving.

## Concept

Static indicator values (e.g. RSI = 65) tell you the current state, but not the trajectory. Is RSI rising toward 65 from oversold territory, or falling from overbought? The dynamics plugin answers this by computing first-order changes (velocity) and second-order changes (acceleration) of key indicators across multiple time horizons.

Change features measure how much an indicator has moved over a given lookback period (4h, 8h, 24h). For RSI, both absolute and percentage changes are computed. For volatility metrics (ATR, Bollinger Band width), percentage changes capture expansion and contraction of market range. ADX changes show whether trend strength is building or fading. MACD and Stochastic changes capture momentum shifts.

Lag features provide the model with explicit access to past indicator states (e.g. "what was RSI 24 bars ago?"), enabling the model to learn temporal patterns without having to infer them. Acceleration features (second derivative) detect inflection points -- when the rate of change itself is changing, signaling the beginning or end of a move. Together, these dynamics features give ML models a rich temporal context for each decision point.

## Features

| Feature | Description |
|---------|-------------|
| `dyn_rsi14_chg_4h` | RSI(14) absolute change over 4 bars |
| `dyn_rsi14_chg_8h` | RSI(14) absolute change over 8 bars |
| `dyn_rsi14_chg_24h` | RSI(14) absolute change over 24 bars |
| `dyn_rsi14_pct_4h` | RSI(14) percentage change over 4 bars |
| `dyn_rsi14_pct_8h` | RSI(14) percentage change over 8 bars |
| `dyn_rsi14_pct_24h` | RSI(14) percentage change over 24 bars |
| `dyn_atr_chg_4h` | ATR percentage (ATR/Close) percentage change over 4 bars |
| `dyn_atr_chg_8h` | ATR percentage change over 8 bars |
| `dyn_atr_chg_24h` | ATR percentage change over 24 bars |
| `dyn_bbwidth_chg_4h` | Bollinger Band width percentage change over 4 bars |
| `dyn_bbwidth_chg_8h` | Bollinger Band width percentage change over 8 bars |
| `dyn_bbwidth_chg_24h` | Bollinger Band width percentage change over 24 bars |
| `dyn_adx_chg_4h` | ADX(14) absolute change over 4 bars |
| `dyn_adx_chg_8h` | ADX(14) absolute change over 8 bars |
| `dyn_adx_chg_24h` | ADX(14) absolute change over 24 bars |
| `dyn_macd_chg_4h` | MACD histogram (normalized by price) change over 4 bars |
| `dyn_macd_chg_8h` | MACD histogram change over 8 bars |
| `dyn_stoch_chg_4h` | Stochastic %K change over 4 bars |
| `dyn_stoch_chg_8h` | Stochastic %K change over 8 bars |
| `lag_rsi14_4h` | RSI(14) value 4 bars ago |
| `lag_rsi14_8h` | RSI(14) value 8 bars ago |
| `lag_rsi14_24h` | RSI(14) value 24 bars ago |
| `lag_atr_4h` | ATR percentage value 4 bars ago |
| `lag_atr_8h` | ATR percentage value 8 bars ago |
| `lag_atr_24h` | ATR percentage value 24 bars ago |
| `lag_adx_4h` | ADX value 4 bars ago |
| `lag_adx_8h` | ADX value 8 bars ago |
| `lag_price_chg_4h` | Price percentage change over 4 bars |
| `lag_price_chg_8h` | Price percentage change over 8 bars |
| `lag_price_chg_24h` | Price percentage change over 24 bars |
| `lag_price_chg_48h` | Price percentage change over 48 bars |
| `accel_rsi` | RSI acceleration: 4h RSI change minus its own 4-bar-ago value |
| `accel_atr` | ATR acceleration: 4h ATR change minus its own 4-bar-ago value |
| `accel_adx` | ADX acceleration: 4h ADX change minus its own 4-bar-ago value |
| `accel_price` | Price acceleration: 4h price change minus its own 4-bar-ago value |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookbacks` | `List[int]` | `[4, 8, 24]` | Lookback periods (in bars) for computing indicator changes |
| `lag_periods` | `List[int]` | `[4, 8, 24, 48]` | Lag periods for historical indicator values and price changes |

## Usage Notes

- The plugin attempts to read pre-computed base indicators from the DataFrame (e.g. `mom_rsi_14`, `vol_atr_pct_14`, `trend_adx_14`). If not present, it computes them from raw OHLC data using the `ta` library.
- Lookback values correspond to hourly bars (H1 timeframe). For example, `4h` means 4 hourly bars. Adjust `lookbacks` and `lag_periods` if using a different timeframe.
- MACD and Stochastic changes are only computed for lookbacks of 4 and 8 (not 24) to avoid overly noisy long-horizon changes.
- Lag features for RSI and ATR use only the first 3 lag periods (default: 4, 8, 24), while price change lags use all 4 periods (4, 8, 24, 48).
- ADX lag features are limited to lags 4 and 8 only.
- All features are shifted by 1 bar to prevent lookahead bias.
- `benefits_from_stationary: false` -- change and percentage-change features are inherently stationary.
