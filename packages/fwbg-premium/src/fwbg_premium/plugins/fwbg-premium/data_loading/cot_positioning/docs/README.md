# COT Positioning Data Loader

Computes trading features from CFTC Commitments of Traders (COT) non-commercial positioning data for major FX pairs.

## Concept

The Commitments of Traders report, published weekly by the CFTC, discloses the aggregate positioning of non-commercial (speculative) traders in futures markets. Extreme positioning levels often precede trend reversals, as crowded trades unwind when the majority of speculators are on one side of the market. This plugin transforms raw net positioning data into actionable features that capture the sentiment state and momentum of speculative flows.

The plugin computes z-score normalized positioning levels, detects extreme long/short crowding, identifies crowded trade conditions, and tracks positioning momentum across multiple lookback horizons. Z-score normalization over a rolling 52-week window ensures that positioning levels are comparable across different volatility regimes and time periods.

All computed features are shifted by 1 bar to prevent lookahead bias, since COT data is published with a delay and reflects positions as of the prior Tuesday.

## Features

The plugin expects base columns named `macro_cot_{symbol}` to already be present in the DataFrame (loaded via the DataSource layer). For each COT base column, it generates the following features:

| Feature | Formula | Description |
|---------|---------|-------------|
| `{prefix}_zscore` | `(net - rolling_mean) / rolling_std` | 52-week rolling z-score of net positioning |
| `{prefix}_extreme_long` | `zscore > 2.0` | Binary flag for extreme long positioning |
| `{prefix}_extreme_short` | `zscore < -2.0` | Binary flag for extreme short positioning |
| `{prefix}_crowded` | `abs(zscore) > 1.5` | Binary flag for crowded trade conditions |
| `{prefix}_chg_{N}w` | `net.pct_change(N weeks)` | Positioning momentum over N weeks |

Where `{prefix}` is derived from the base column (e.g., `macro_cot_eurusd` becomes `cot_eurusd`).

### Default Supported Symbols

| File Stem | Column Prefix |
|-----------|---------------|
| `COT_EURUSD_DAY` | `cot_eurusd` |
| `COT_USDJPY_DAY` | `cot_usdjpy` |
| `COT_GBPUSD_DAY` | `cot_gbpusd` |
| `COT_USDCAD_DAY` | `cot_usdcad` |
| `COT_AUDUSD_DAY` | `cot_audusd` |
| `COT_USDCHF_DAY` | `cot_usdchf` |
| `COT_NZDUSD_DAY` | `cot_nzdusd` |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indicators` | `dict` | 7 major FX pairs | Mapping of COT data file stems to column prefixes |
| `lookbacks_weeks` | `list[int]` | `[1, 4, 12, 26]` | Week-based lookback periods for positioning momentum |
| `zscore_window_weeks` | `int` | `52` | Rolling window in weeks for z-score normalization (min: 1, max: 520) |

## Usage Notes

- COT base columns must be pre-loaded as `macro_cot_*` columns before this plugin runs.
- The z-score rolling window is converted from weeks to H1 bars internally (`weeks * 5 * 24`), assuming 5 trading days and 24 hourly bars per day.
- Momentum lookbacks are likewise converted from weeks to H1 bars.
- The minimum periods for the rolling z-score window is set to `window // 4` to allow computation to begin before a full window is available.
- The rolling standard deviation is clipped to a minimum of `1e-6` to avoid division by zero.
- All COT-derived feature columns are shifted forward by 1 bar to prevent lookahead bias.
