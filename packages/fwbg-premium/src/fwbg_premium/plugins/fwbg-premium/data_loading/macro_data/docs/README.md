# Macro Data Loader

Computes features from macroeconomic and cross-asset base columns, including lookback changes, derived spreads/ratios, and interest rate differentials.

## Concept

Financial markets are driven by macroeconomic forces -- volatility regimes, yield curves, equity sector rotations, and central bank policy. This plugin transforms raw macro data columns into a rich set of momentum, spread, and ratio features that capture the current macro environment and its recent dynamics.

The plugin is a pure computation engine with no I/O. It expects `macro_*` base columns to already be loaded into the DataFrame by the orchestrator via DataSource. From these base columns, it computes percentage changes at multiple hourly and daily horizons, derived features through subtraction (yield curves, rate spreads) and division (risk ratios, sector ratios), and interest rate differentials relevant for FX carry signals.

This multi-layered feature engineering captures both the level and the trend of macro indicators, enabling the model to detect regime shifts, risk-on/risk-off transitions, and yield curve dynamics that influence FX and equity markets.

## Features

### Lookback Changes

For each `macro_*` base column, the plugin computes percentage changes at configurable hourly and daily lookback periods:

| Feature Pattern | Formula | Description |
|-----------------|---------|-------------|
| `macro_{prefix}_chg_{N}h` | `pct_change(N) * 100` | N-hour percentage change |
| `macro_{prefix}_chg_{N}d` | `pct_change(N * 24) * 100` | N-day percentage change (24 bars/day) |

### Derived Features (Default Set)

| Feature | Operation | Components | Description |
|---------|-----------|------------|-------------|
| `macro_yield_curve_10y_3m` | subtract | TNX - IRX | 10-year minus 3-month yield spread |
| `macro_yield_curve_10y_5y` | subtract | TNX - FVX | 10-year minus 5-year yield spread |
| `macro_yield_curve_10y_2y` | subtract | TNX - US2Y | 10-year minus 2-year yield spread |
| `macro_yield_curve_30y_5y` | subtract | US30Y - US5Y | 30-year minus 5-year yield spread |
| `macro_vix_vvix_ratio` | ratio | VIX / VVIX | Volatility of volatility ratio |
| `macro_risk_ratio_spx_tlt` | ratio | SPX / TLT | Equity vs bond risk appetite |
| `macro_credit_spread_proxy` | ratio | HYG / LQD | High-yield vs investment-grade credit |
| `macro_smallcap_ratio` | ratio | Russell / SPX | Small-cap relative strength |
| `macro_tech_defensive_ratio` | ratio | XLK / XLU | Tech vs utilities sector rotation |
| `macro_yield_spread_us_de` | subtract | TNX - DE10Y | US-Germany yield differential |
| `macro_yield_spread_us_jp` | subtract | TNX - JP10Y | US-Japan yield differential |
| `macro_yield_spread_us_gb` | subtract | TNX - GB10Y | US-UK yield differential |
| `macro_yield_spread_us_au` | subtract | TNX - AU10Y | US-Australia yield differential |

Daily lookback changes are also computed for all derived features.

### Interest Rate Differentials

| Feature | Components | Description |
|---------|------------|-------------|
| `macro_rate_diff_usd_eur` | FED rate - ECB rate | USD/EUR policy rate differential (FX carry signal) |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indicators` | `dict` | 35 macro instruments | Mapping of data file stems to column prefixes |
| `lookbacks_hours` | `list[int]` | `[1, 2, 4, 8, 12, 24]` | Hourly lookback periods for pct_change features |
| `lookbacks_days` | `list[int]` | `[2, 5, 10, 20, 60]` | Daily lookback periods (converted to bars via 24h/day) |
| `derived_features` | `list[dict]` | 13 derived features | List of derived feature specs with `subtract` or `ratio` operations |
| `interest_rates` | `list[dict]` | FED, ECB | Interest rate data sources with file names and lookback periods |
| `interest_rate_diffs` | `list[dict]` | USD-EUR spread | Interest rate differential specs for FX carry signals |

### Default Indicators

The plugin ships with 35 default macro instruments spanning volatility indices (VIX, VVIX, SKEW, VXN), US Treasury yields (IRX, FVX, TNX, TYX, US2Y, US5Y, US30Y), international bond yields (DE10Y, JP10Y, GB10Y, AU10Y), equity indices (SPX, NASDAQ, DOW, Russell, Nikkei, Hang Seng, FTSE, DAX), sector ETFs (XLF, XLE, XLK, XLU, XLP), bond ETFs (TLT, HYG, LQD), currencies (DXY), and commodities (Gold, Oil, Silver futures).

## Usage Notes

- All base columns must be pre-loaded as `macro_*` columns before this plugin executes.
- Daily lookback periods are converted to bars assuming 24 bars per day (H1 timeframe).
- Ratio operations add `1e-10` to the denominator to prevent division by zero.
- Derived features also get daily lookback changes computed, creating momentum signals for spreads and ratios.
- Columns already containing `_chg_` in their name are excluded from base lookback computation to avoid double-differencing.
