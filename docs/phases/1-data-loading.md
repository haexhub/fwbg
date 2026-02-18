# Phase 1: Data Loading

## Purpose

The data loading phase loads external data sources (macro indicators, COT positioning, etc.) and computes derived features from them. This phase executes **before** preprocessing and indicators.

**Important:** DataLoader plugins do **no I/O**. Raw data is loaded beforehand by the orchestrator via configured DataSources and placed in the DataFrame. DataLoader plugins only compute derived features from this already available data.

---

## Three-Layer Architecture

```
DataSource.load()        → I/O: read CSV, fetch API, DB query
    ↓
Orchestrator             → Index alignment: Daily→Intraday mapping, forward-fill
    ↓
DataLoader.execute()     → Computation: lookbacks, derived features, ratios
```

1. **DataSource** (I/O layer): Knows where data comes from and how to read it
2. **Orchestrator**: Aligns different frequencies (e.g., daily data to H1 index)
3. **DataLoader** (plugin): Computes derived features from the aligned data

---

## BaseDataLoader

Module: `fwbg_sdk.data_loaders`

```python
class BaseDataLoader(BasePlugin, ABC):
    phase = PluginPhase.DATA_LOADING
    stateful = False

    @abstractmethod
    def execute(self, ctx: PipelineContext, **params):
        """Compute derived features from raw data in ctx.df."""
```

- `stateful = False` — No fit/transform pattern, no state
- `cacheable = True` — Same result for same input data
- Import: `from fwbg_sdk import BaseDataLoader, register_data_loader`
- Registration: `@register_data_loader("name")`

---

## Available Plugins

### macro_data (fwbg-premium)

Computes macro features from preloaded time series:

| Feature Group | Description | Prefix |
|---------------|-------------|--------|
| VIX | CBOE Volatility Index | `macro_vix_` |
| DXY | US Dollar Index | `macro_dxy_` |
| Yields | US2Y, US5Y, US10Y, US30Y, international yields | `macro_yield_` |
| Yield Curves | Slope (10Y-2Y), steepness (30Y-5Y), term spread | `macro_yield_curve_` |
| Yield Spreads | International rate differentials (US-DE, US-JP, etc.) | `macro_yield_spread_` |

Each feature is computed in multiple lookback variants: `_chg_2d`, `_chg_5d`, `_chg_10d`, `_chg_20d`, `_chg_60d`.

**Derived Features:**
- `vol_rv_iv_ratio` / `vol_rv_iv_spread` — Realized vol vs VIX
- `cross_cot_{pair}_vol_interaction` — COT × Vol interaction
- `cross_cot_{pair}_price_divergence` — Price vs positioning divergence

### cot_positioning (fwbg-premium)

Computes CFTC COT positioning features:

| Feature | Description |
|---------|-------------|
| `cot_{pair}_z_score` | Z-score of net positioning |
| `cot_{pair}_extreme_long/short` | Extreme positioning flags |
| `cot_{pair}_crowded_trade` | Crowded trade indicator |
| `cot_{pair}_weekly_momentum` | Weekly position change |

All features are shifted by 1 bar (lookahead prevention).

---

## DataSources (I/O Layer)

DataSources are the I/O layer. Each source has a `load()` method that returns a `LoadResult`:

```python
@dataclass
class LoadResult:
    data: Dict[str, pd.DataFrame]  # Name → DataFrame
    metadata: Dict[str, Any]       # Additional metadata
    source_name: str               # Source name
```

### Available Source Types

| Type | Class | Description |
|------|-------|-------------|
| `csv` | `CSVSourceConfig` | Local CSV files |
| `rest` | `RESTSourceConfig` | REST APIs (Alpha Vantage, Polygon.io) |
| `websocket` | `WebSocketSourceConfig` | WebSocket streams |
| `database` | `DBSourceConfig` | SQL databases via SQLAlchemy |

### Preconfigured Sources

| Name | Type | Description |
|------|------|-------------|
| `forexsb` | csv | Forex Strategy Builder exports |
| `stooq` | csv | Stooq.com historical data |
| `yahoo` | csv | Yahoo Finance data |
| `downloads` | csv | Manually downloaded data |

### Registering a Custom Source

```python
from fwbg.core.data_sources import register_csv_source

register_csv_source(
    name="my_data",
    path="/path/to/csvs",
    file_pattern="{symbol}_DAY.csv",
)
```

---

## Data Updates

External data must be updated before running the optimizer:

```bash
# Macro data: DXY, VIX (yfinance) + international bond yields (FRED)
python scripts/fetch_macro_data.py

# CFTC COT Positioning: Asset Manager Net Positions for 7 FX pairs
python scripts/fetch_cot_data.py

# All sources (Forex Strategy Builder exports)
python scripts/fetch_all_sources.py
```

### Data Source Overview

| Data | Source | Frequency | History |
|------|--------|-----------|---------|
| DXY, VIX (hourly) | yfinance | H1 | ~2 years |
| US2Y, US5Y, US30Y | FRED (daily, ffill) | D1 | 25+ years |
| DE10Y, JP10Y, GB10Y, AU10Y | FRED (monthly, daily ffill) | D1 | 25+ years |
| COT EURUSD, USDJPY, GBPUSD, ... | CFTC TFF Reports | Weekly (daily ffill) | 2006+ |
| VIX, SPX, TNX, DXY, ... (daily) | Forex Strategy Builder | D1 | varies |

---

## Strategy JSON Configuration

```json
"pipeline": {
  "data_loading": [
    {"name": "macro_data", "source": "forexsb"},
    {"name": "cot_positioning", "source": "forexsb"}
  ]
}
```

The `source` parameter determines which registered DataSource provides the raw data.

---

## Creating a Custom Data Loader Plugin

See [Plugin Development Guide](../plugin-development.md) for the full guide.
