# Market Regime

Combines VIX, credit spreads, equity momentum, and treasury flight-to-quality into a composite risk-on/risk-off regime score derived from macro data.

## Concept

Financial markets alternate between risk-on regimes (investors favor risky assets, volatility is low, credit spreads tighten) and risk-off regimes (investors flee to safe havens, volatility spikes, credit spreads widen). Identifying the current regime is critical for position sizing, strategy selection, and understanding why correlations between assets shift.

This plugin constructs a composite regime indicator from four orthogonal macro components. The VIX z-score captures fear/complacency in equity options markets (inverted so that high VIX = negative/risk-off). The credit spread z-score uses the HYG/LQD ratio as a proxy for high-yield vs. investment-grade bond demand (high ratio = risk appetite). Equity momentum measures the 20-day percentage change in the S&P 500 as a direct gauge of risk appetite. Treasury flight uses TLT (long-term treasury ETF) momentum as an inverted safe-haven indicator (rising TLT = flight to quality = risk-off).

The composite score averages the z-scored components, providing a single continuous measure of the risk regime. Binary flags for risk-on (composite > 0.5) and risk-off (composite < -0.5) offer discrete regime labels. ML models can use both the continuous composite and the binary flags -- the continuous score for nuanced positioning, and the binary flags as regime-conditioning features that modulate other signals.

## Features

| Feature | Description |
|---------|-------------|
| `regime_vix_zscore` | Inverted VIX z-score: negative when VIX is above its rolling mean (risk-off), positive when below (risk-on) |
| `regime_credit_zscore` | Credit spread z-score from HYG/LQD ratio: positive = tightening spreads (risk-on), negative = widening (risk-off) |
| `regime_equity_momentum` | S&P 500 20-day percentage change: positive = bullish equity momentum |
| `regime_treasury_flight` | Inverted TLT 10-day momentum: negative when treasuries are rallying (flight to safety) |
| `regime_risk_composite` | Average of all z-scored components: single continuous risk regime measure |
| `regime_risk_on` | Binary: composite > 0.5 (clear risk-on environment) |
| `regime_risk_off` | Binary: composite < -0.5 (clear risk-off environment) |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window` | `int` | `50` | Lookback window in days for z-score rolling mean and standard deviation |
| `bars_per_day` | `int` | `24` | Number of bars per trading day (used to convert day-based window to bar count) |

## Usage Notes

- This plugin requires macro data columns in the DataFrame. Features are conditionally computed based on available columns:
  - `macro_vix` for VIX z-score
  - `macro_hyg` and `macro_lqd` for credit spread z-score
  - `macro_spx` for equity momentum
  - `macro_tlt` for treasury flight
- If none of these columns are present, the plugin returns the DataFrame unchanged with no additional features.
- The composite score is only computed when at least one z-score component is available. Binary risk-on/risk-off flags derive from the composite.
- The rolling window is `window * bars_per_day` bars (default: 50 * 24 = 1200 bars), requiring significant data history for stable z-scores.
- Equity momentum uses a 20-day lookback (`20 * bars_per_day`), and treasury flight uses 10-day (`10 * bars_per_day`).
- All features are shifted by 1 bar to prevent lookahead bias.
- The `bars_per_day` parameter should match your data's timeframe (24 for H1, 1 for D1, etc.).
