# Phase 1: Data Loading

## Zweck

Die Data-Loading-Phase lädt externe Datenquellen (Makro-Indikatoren, COT-Positioning, etc.) und berechnet daraus abgeleitete Features. Diese Phase wird ausgeführt **bevor** Preprocessing und Indikatoren laufen.

**Wichtig:** DataLoader-Plugins machen **kein I/O**. Die Rohdaten werden vorab vom Orchestrator über konfigurierte DataSources geladen und im DataFrame bereitgestellt. DataLoader-Plugins berechnen nur abgeleitete Features aus diesen bereits vorhandenen Daten.

---

## Dreischichtige Architektur

```
DataSource.load()        → I/O: CSV lesen, API fetchen, DB query
    ↓
Orchestrator             → Index-Alignment: Daily→Intraday Mapping, Forward-Fill
    ↓
DataLoader.execute()     → Computation: Lookbacks, Derived Features, Ratios
```

1. **DataSource** (I/O-Schicht): Weiß woher Daten kommen und wie sie gelesen werden
2. **Orchestrator**: Aligned verschiedene Frequenzen (z.B. Daily-Daten auf H1-Index)
3. **DataLoader** (Plugin): Berechnet abgeleitete Features aus den aligned Daten

---

## BaseDataLoader

Basisklasse: `src/fwbg/plugins/data_loader.py`

```python
class BaseDataLoader(BasePlugin, ABC):
    phase = PluginPhase.DATA_LOADING
    stateful = False

    @abstractmethod
    def execute(self, ctx: PipelineContext, **params):
        """Berechnet abgeleitete Features aus Rohdaten in ctx.df."""
```

- `stateful = False` — Kein fit/transform-Pattern, kein Zustand
- `cacheable = True` — Ergebnis gleich bei gleichen Eingabedaten
- Registrierung: `@register_data_loader("name")`

---

## Verfügbare Plugins

### macro_data (fwbg-premium)

Berechnet Makro-Features aus vorgeladenen Zeitreihen:

| Feature-Gruppe | Beschreibung | Prefix |
|----------------|--------------|--------|
| VIX | CBOE Volatilitätsindex | `macro_vix_` |
| DXY | US Dollar Index | `macro_dxy_` |
| Yields | US2Y, US5Y, US10Y, US30Y, int. Yields | `macro_yield_` |
| Yield Curves | Slope (10Y-2Y), Steepness (30Y-5Y), Term Spread | `macro_yield_curve_` |
| Yield Spreads | International Zinsdifferenzen (US-DE, US-JP, etc.) | `macro_yield_spread_` |

Jedes Feature wird in mehreren Lookback-Varianten berechnet: `_chg_2d`, `_chg_5d`, `_chg_10d`, `_chg_20d`, `_chg_60d`.

**Derived Features:**
- `vol_rv_iv_ratio` / `vol_rv_iv_spread` — Realized Vol vs VIX
- `cross_cot_{pair}_vol_interaction` — COT × Vol Interaction
- `cross_cot_{pair}_price_divergence` — Preis vs Positioning Divergenz

### cot_positioning (fwbg-premium)

Berechnet CFTC COT Positioning-Features:

| Feature | Beschreibung |
|---------|--------------|
| `cot_{pair}_z_score` | Z-Score der Netto-Positionierung |
| `cot_{pair}_extreme_long/short` | Extreme Positioning Flags |
| `cot_{pair}_crowded_trade` | Crowded Trade Indikator |
| `cot_{pair}_weekly_momentum` | Wöchentliche Positionsänderung |

Alle Features werden um 1 Bar geshiftet (Lookahead-Prevention).

---

## DataSources (I/O-Schicht)

DataSources sind die I/O-Schicht. Jede Source hat eine `load()`-Methode die ein `LoadResult` zurückgibt:

```python
@dataclass
class LoadResult:
    data: Dict[str, pd.DataFrame]  # Name → DataFrame
    metadata: Dict[str, Any]       # Zusätzliche Metadaten
    source_name: str               # Quellname
```

### Verfügbare Source-Typen

| Typ | Klasse | Beschreibung |
|-----|--------|--------------|
| `csv` | `CSVSourceConfig` | Lokale CSV-Dateien |
| `rest` | `RESTSourceConfig` | REST APIs (Alpha Vantage, Polygon.io) |
| `websocket` | `WebSocketSourceConfig` | WebSocket Streams |
| `database` | `DBSourceConfig` | SQL-Datenbanken via SQLAlchemy |

### Vorkonfigurierte Quellen

| Name | Typ | Beschreibung |
|------|-----|--------------|
| `forexsb` | csv | Forex Strategy Builder Exports |
| `stooq` | csv | Stooq.com historische Daten |
| `yahoo` | csv | Yahoo Finance Daten |
| `downloads` | csv | Manuell heruntergeladene Daten |

### Eigene Quelle registrieren

```python
from fwbg.core.data_sources import register_csv_source

register_csv_source(
    name="my_data",
    path="/path/to/csvs",
    file_pattern="{symbol}_DAY.csv",
)
```

---

## Daten-Updates

Externe Daten müssen vor dem Optimizer-Lauf aktualisiert werden:

```bash
# Makro-Daten: DXY, VIX (yfinance) + internationale Bond Yields (FRED)
python scripts/fetch_macro_data.py

# CFTC COT Positioning: Asset Manager Net Positions für 7 FX-Paare
python scripts/fetch_cot_data.py

# Alle Quellen (Forex Strategy Builder Exports)
python scripts/fetch_all_sources.py
```

### Datenquellen-Übersicht

| Daten | Quelle | Frequenz | Historie |
|-------|--------|----------|----------|
| DXY, VIX (hourly) | yfinance | H1 | ~2 Jahre |
| US2Y, US5Y, US30Y | FRED (daily, ffill) | D1 | 25+ Jahre |
| DE10Y, JP10Y, GB10Y, AU10Y | FRED (monatlich, daily ffill) | D1 | 25+ Jahre |
| COT EURUSD, USDJPY, GBPUSD, ... | CFTC TFF Reports | Wöchentlich (daily ffill) | 2006+ |
| VIX, SPX, TNX, DXY, ... (daily) | Forex Strategy Builder | D1 | variiert |

---

## Strategy-JSON Konfiguration

```json
"pipeline": {
  "data_loading": [
    {"name": "macro_data", "source": "forexsb"},
    {"name": "cot_positioning", "source": "forexsb"}
  ]
}
```

Der `source`-Parameter bestimmt, aus welcher registrierten DataSource die Rohdaten geladen werden.

---

## Eigenes Data-Loader-Plugin erstellen

Siehe [Plugin Development Guide](../plugin-development.md) für die vollständige Anleitung.
