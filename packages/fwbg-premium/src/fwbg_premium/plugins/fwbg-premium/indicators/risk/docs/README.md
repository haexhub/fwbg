# Risk

Computes tail-risk metrics including drawdown state, Conditional Value at Risk (CVaR), volatility-of-volatility, crash probability proxies, and cross-asset correlation features.

## Concept

Risk management is the foundation of sustainable trading. While many indicators focus on identifying profitable entries, risk features quantify the probability and magnitude of adverse outcomes. This plugin provides a comprehensive risk dashboard that ML models can use to scale position sizes, avoid high-risk environments, or tighten exit parameters.

The drawdown features measure how far price has fallen from its rolling peak, how deep the current drawdown is relative to recent history, and how long it has persisted. CVaR (Conditional Value at Risk, also known as Expected Shortfall) goes beyond traditional VaR by measuring the average loss in the tail of the distribution -- answering "when things go bad, how bad do they get?" The plugin computes both VaR and CVaR at 5% and 1% confidence levels across multiple time windows, plus a tail ratio (CVaR 1% / CVaR 5%) that captures how fat the tails are.

Volatility-of-volatility (vol-of-vol) measures the stability of the volatility regime itself. When vol-of-vol is high, the market is transitioning between regimes, creating uncertainty about the appropriate risk parameters. The crash probability proxy combines multiple warning signals -- kurtosis spikes, rising vol-of-vol, correlation decoupling, and extreme CVaR -- into a single composite score (0-1). When macro data (SPX, VIX) is available, the plugin additionally computes rolling correlations, rolling betas, lead-lag relationships, and correlation stability features that capture systemic risk dynamics.

## Features

| Feature | Description |
|---------|-------------|
| `risk_dd_pct_50` | Drawdown percentage over 50-bar rolling peak. (Close - RollingMax) / RollingMax |
| `risk_dd_pct_100` | Drawdown percentage over 100-bar rolling peak |
| `risk_dd_pct_200` | Drawdown percentage over 200-bar rolling peak |
| `risk_dd_ratio_50` | Current drawdown / worst drawdown in last 50 bars. Relative severity |
| `risk_dd_ratio_100` | Current drawdown / worst drawdown in last 100 bars |
| `risk_dd_ratio_200` | Current drawdown / worst drawdown in last 200 bars |
| `risk_bars_since_peak` | Number of bars since the 200-bar rolling peak was reached |
| `risk_bars_since_peak_log` | Log-transformed bars since peak. Compresses large values for ML |
| `risk_recovery_ratio` | Recovery from trough toward peak, clipped to [0, 1]. 1 = fully recovered |
| `risk_var_5_50` | Value at Risk (5th percentile) over 50-bar window |
| `risk_var_1_50` | Value at Risk (1st percentile) over 50-bar window |
| `risk_var_5_100` | Value at Risk (5th percentile) over 100-bar window |
| `risk_var_1_100` | Value at Risk (1st percentile) over 100-bar window |
| `risk_cvar_5_50` | CVaR / Expected Shortfall (5%) over 50-bar window. Average loss when below 5th percentile |
| `risk_cvar_1_50` | CVaR / Expected Shortfall (1%) over 50-bar window. Average loss in extreme tail |
| `risk_cvar_5_100` | CVaR / Expected Shortfall (5%) over 100-bar window |
| `risk_cvar_1_100` | CVaR / Expected Shortfall (1%) over 100-bar window |
| `risk_cvar_tail_ratio` | CVaR(1%, 100) / CVaR(5%, 100). Measures tail fatness -- higher = fatter tails |
| `risk_cvar_5_change` | Change in CVaR(5%, 100) over 20 bars. Rising = worsening tail risk |
| `risk_vol_of_vol_20` | Volatility of ATR changes over 20-bar window. Short-term regime instability |
| `risk_vol_of_vol_50` | Volatility of ATR changes over 50-bar window. Medium-term regime instability |
| `risk_vol_of_vol_100` | Volatility of ATR changes over 100-bar window. Long-term regime instability |
| `risk_vol_of_vol_zscore` | Z-score of vol-of-vol(100) vs. its 200-bar rolling mean/std |
| `risk_vol_of_vol_trend` | Change in vol-of-vol(50) over 10 bars. Rising = increasing regime uncertainty |
| `risk_crash_probability` | Composite crash probability proxy (0-1). Combines kurtosis, vol-of-vol, correlation decoupling, and CVaR signals |
| `risk_crash_prob_change` | Change in crash probability over 10 bars |
| `risk_crash_regime` | Binary: 1 if crash probability exceeds 0.6 threshold |
| `corr_spx_20` | Rolling 20-bar correlation with SPX (requires `macro_spx` column) |
| `corr_spx_50` | Rolling 50-bar correlation with SPX |
| `corr_spx_100` | Rolling 100-bar correlation with SPX |
| `corr_spx_stability` | Change in SPX correlation(50) over 10 bars. Sudden drops = decoupling |
| `corr_spx_decoupling` | Absolute value of correlation stability. Magnitude of correlation shift |
| `lead_lag_spx` | 20-bar momentum of asset minus 20-bar momentum of SPX. Relative performance |
| `beta_spx_50` | Rolling 50-bar beta to SPX: Cov(asset, SPX) / Var(SPX) |
| `beta_spx_100` | Rolling 100-bar beta to SPX |
| `corr_vix_20` | Rolling 20-bar correlation with VIX (requires `macro_vix` or `sent_vix` column) |
| `corr_vix_50` | Rolling 50-bar correlation with VIX |
| `lead_lag_vix` | VIX 10-bar change + asset 10-bar change. Divergence = stress signal |
| `vix_lead_signal` | VIX 10-bar change shifted forward by 5 bars. Tests if VIX leads asset moves |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dd_windows` | List[int] | [50, 100, 200] | Rolling window sizes for drawdown computation |
| `cvar_windows` | List[int] | [50, 100] | Rolling window sizes for VaR/CVaR computation |
| `cvar_percentiles` | List[int] | [5, 1] | Percentiles for VaR/CVaR (5 = 5th percentile, 1 = 1st percentile) |
| `vov_windows` | List[int] | [20, 50, 100] | Rolling window sizes for volatility-of-volatility computation |
| `compute_correlations` | bool | True | Whether to compute cross-asset correlation features (SPX, VIX). Requires macro data columns in the DataFrame |

## Usage Notes

- All risk features are shifted by 1 bar to prevent lookahead bias.
- Correlation features (`corr_*`, `lead_lag_*`, `beta_*`, `vix_lead_*`) are only computed when the corresponding macro data columns are present in the DataFrame (`macro_spx` for SPX features, `macro_vix` or `sent_vix` for VIX features). When absent, these features are simply not generated.
- The crash probability proxy is a composite of up to 4 components (kurtosis from `dist_kurt_50`, vol-of-vol z-score, correlation decoupling, CVaR). The score is normalized by the number of available components, so it remains valid even when some inputs are missing.
- The `manifest.json` sets `benefits_from_stationary: false` since the plugin uses returns and ratios internally.
- CVaR computation requires a minimum of 10 data points in the rolling window to produce a valid estimate.
- Drawdown features are always negative or zero (0 = at peak).
