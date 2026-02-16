# FWBG - ML Trading Strategy Optimizer & Bot

Ein Machine-Learning-basiertes Trading-System mit:
- **Optimizer**: Findet optimale Parameter (TP, SL, Confidence) via Walk-Forward Cross-Validation
- **Bot**: Live-Trading über austauschbare Broker-Adapter

## Quick Start

### Installation

```bash
git clone https://github.com/haexhub/fwbg.git
cd fwbg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optimizer

```bash
fwbg --assets EURUSD
fwbg --strategy-file strategies/exploration.json --assets EURUSD
fwbg --assets EURUSD,GBPUSD,USDJPY
fwbg --strategy-file strategies/exploration.json --asset-classes FOREX
OPTIMIZER_LOG=2 fwbg --assets EURUSD
```

### Trading Bot

```bash
python -m bots.ig                  # Streaming-Modus
python -m bots.ig --no-streaming   # Polling-Modus
```

---

## Projektstruktur

```
fwbg/
├── src/fwbg/
│   ├── plugins/                  # Plugin-System
│   │   ├── fwbg-core/            # Core Plugins (kostenlos)
│   │   │   ├── indicators/       # trend, momentum, volatility, price_action, time_season
│   │   │   ├── exit_strategies/  # fixed
│   │   │   └── risk_management/  # kelly, vol_targeted_kelly
│   │   └── *.py                  # Plugin-Basisklassen
│   ├── core/                     # Config, Registry, Context, DataSources
│   ├── pipeline/                 # Plugin Runner & Pipeline System
│   ├── optimization/             # Walk-Forward CV, Grid Search, Targets
│   ├── simulation/               # Numba-basierte Trade-Simulation
│   ├── data/                     # Datenquellen, Loader, Asset-Definitionen
│   ├── results/                  # Ergebnis-Speicherung & Plotting
│   ├── cli/                      # Command-Line Interface
│   └── adapters/                 # Broker & Datenquellen-Adapter
│
├── packages/
│   └── fwbg-premium/             # Premium Plugins (separates Package)
│       └── indicators/           # regime, structure, risk, distribution, dynamics, ...
│       ├── preprocessing/        # fractional_diff
│       ├── feature_selection/    # boruta, plateau, stability
│       ├── exit_strategies/      # atr_based
│       └── data_loading/         # macro_data, cot_positioning
│
├── strategies/                   # Strategy Configurations (JSON)
├── data/                         # Historical Data (CSV)
└── test_results/                 # Optimization Results
```

---

## Plugin-Architektur

FWBG basiert auf einem Plugin-System. Jede Funktionalität — von Indikatoren über Exit-Strategien bis zu Datenquellen — ist als Plugin implementiert und austauschbar.

### Grundkonzept

Alle Plugins erben von `BasePlugin`:

```python
class BasePlugin(ABC):
    name: str                    # Eindeutiger Name (z.B. "trend")
    phase: PluginPhase           # Pipeline-Phase (z.B. PluginPhase.INDICATORS)
    version: str = "0.1.0"      # Semantische Version (optional)
    stateful: bool = False
    cacheable: bool = True

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext: ...
    def validate(self) -> bool: ...

    @classmethod
    def get_default_params(cls) -> dict: ...

    def get_feature_columns(self) -> List[str]: ...
```

**`stateful`** — Bestimmt ob das Plugin Zustand zwischen Aufrufen speichert, der aus Trainingsdaten gelernt wurde.

- `False` (Default): **Zustandslos.** Das Plugin berechnet bei jedem Aufruf das gleiche Ergebnis, unabhängig von vorherigen Aufrufen. Es gibt keinen `fit()`-Schritt. Beispiel: Die meisten Indikatoren — `trend` berechnet ADX/EMA immer gleich, egal welcher Fold.
- `True`: **Zustandsbehaftet.** Das Plugin hat einen `fit()`-Schritt, der Parameter aus Trainingsdaten lernt. Diese gelernten Parameter werden dann in `execute()`/`transform()` wiederverwendet. `fit()` wird pro CV-Fold **nur auf Train-Daten** aufgerufen (Lookahead-Bias-Schutz). Beispiel: `fractional_diff` — lernt den optimalen d-Wert auf Train-Daten, wendet ihn dann auf Train/Test/OOS an.

**`cacheable`** — Bestimmt ob Ergebnisse gecacht werden können, um redundante Berechnungen über Folds hinweg zu vermeiden.

- `True` (Default): **Cachebar.** Der `PipelineRunner` darf das Ergebnis zwischenspeichern und wiederverwenden, wenn sich die Eingabedaten und Parameter nicht geändert haben. Beispiel: `volatility` — ATR auf den Originaldaten ist für alle Folds identisch, muss nur einmal berechnet werden.
- `False`: **Nicht cachebar.** Das Ergebnis hängt von Zustand ab, der sich zwischen Folds ändert (z.B. gefittete Parameter). Jeder Fold muss neu berechnet werden. Typischerweise haben `stateful=True` Plugins auch `cacheable=False`.

### Plugin-Pakete & Naming

Plugins sind in Paketen organisiert. Jedes Paket hat eine `manifest.json`:

```
plugins/
├── fwbg-core/              # Namespace: "fwbg-core"
│   ├── manifest.json
│   └── indicators/
│       └── trend/
│           ├── manifest.json
│           └── __init__.py  # @register_indicator("trend")
└── fwbg-premium/           # Namespace: "fwbg-premium"
    └── ...
```

**Voll qualifizierte Namen:** `"fwbg-core:trend"`, `"fwbg-premium:regime"`

**Kurznamen:** In Strategy-JSONs kann `"trend"` statt `"fwbg-core:trend"` verwendet werden — die Auflösung erfolgt automatisch.

### Plugin-Discovery

Plugins werden automatisch aus drei Quellen entdeckt:

1. **Core-Pakete:** `src/fwbg/plugins/fwbg-core/` (eingebaute Plugins)
2. **Entry-Point-Pakete:** Installierte Packages mit `fwbg.plugin_packages` Entry Point (z.B. `fwbg-premium`)
3. **User-Pakete:** `~/.fwbg/plugins/` (eigene Plugins)

### Plugin-Tests

Jedes Plugin kann eine eigene `tests.py` im Plugin-Verzeichnis haben. Diese wird über `plugin.run_tests()` oder `plugin.has_tests()` ausgeführt.

---

## Pipeline-System

Die Pipeline verarbeitet Daten in definierten Phasen. Der `PipelineRunner` orchestriert die Ausführung in der korrekten Reihenfolge.

### Phasen (Ausführungsreihenfolge)

```
1. DATA_LOADING      → Externe Daten laden, Features berechnen
2. PREPROCESSING     → OHLC-Daten transformieren (Stationarität)
3. INDICATORS        → Technische Indikatoren berechnen
4. FEATURE_SELECTION → Relevante Features auswählen
5. EXIT_STRATEGIES   → TP/SL-Berechnung
6. RISK_MANAGEMENT   → Positionsgröße und Risk-Controls
7. LABELING          → Training-Labels generieren
8. MODEL             → ML-Modell trainieren / vorhersagen
9. VALIDATION        → Strategie-Performance validieren
```

### PipelineContext

Der Context wird durch alle Phasen gereicht:

```python
@dataclass
class PipelineContext:
    df: pd.DataFrame                  # Haupt-DataFrame
    symbol: str                       # Asset-Symbol (z.B. "EURUSD")
    asset_class: str                  # Asset-Klasse (z.B. "FOREX")
    metadata: Dict[str, Any]          # Inter-Plugin-Kommunikation
    fold_info: Optional[Dict] = None  # Walk-Forward Fold-Info
```

### Parameter-Hierarchie

Parameter werden gemerged (höhere Priorität überschreibt):

1. **Plugin Defaults** (`get_default_params()`)
2. **Config Params** (aus Strategy-JSON)
3. **Global Params** (zur Laufzeit übergeben) — **höchste Priorität**

---

## Plugin-Typen im Detail

### Indikatoren (`BaseIndicator`)

Berechnen technische Features aus OHLCV-Daten. Jeder Indikator erzeugt neue Spalten im DataFrame.

**Basisklasse:**

```python
class BaseIndicator(BasePlugin, ABC):
    phase = PluginPhase.INDICATORS
    group: str = "custom"                    # Feature-Gruppe für Kategorisierung
    benefits_from_stationary: bool = False    # Nach Preprocessing berechnen?

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Berechnet Indicator-Spalten."""

    def get_feature_columns(self) -> List[str]:
        """Gibt Feature-Spaltennamen zurück."""
```

**Pflicht-Helfer:**
- `shift_features(features, index)` — Shiftet Features um 1 Bar. **Muss in jeder `compute()`-Methode verwendet werden** um Lookahead Bias zu verhindern.
- `safe_divide(numerator, denominator)` — Division mit NaN statt Division-by-Zero.

**`benefits_from_stationary`:**
- `False` (Default): Indikator wird **einmalig auf Originaldaten** berechnet (vor dem Preprocessing). Ergebnis wird über Folds hinweg wiederverwendet. Beispiel: `volatility`, `price_action`.
- `True`: Indikator wird **pro Fold auf preprocessed Daten** berechnet. Beispiel: Indikatoren die von stationären Eingangsdaten profitieren.

**Registrierung:** `@register_indicator("name")`

**Verfügbare Indikatoren:**

| Plugin | Paket | Beschreibung | Prefixes |
|--------|-------|--------------|----------|
| `trend` | core | ADX, EMA, SMA, MACD, CCI, Aroon, Supertrend, Efficiency Ratio | `trend_` |
| `momentum` | core | RSI, Stochastic, Williams %R, ROC | `mom_` |
| `volatility` | core | Bollinger Bands, ATR, Volatilitätsschätzer, Vol Compression, RV vs IV | `vol_` |
| `price_action` | core | Range Position, Higher Highs/Lower Lows, Body Ratio, Gaps | `pa_` |
| `time_season` | core | Stunde, Wochentag, Monat, Quartal, Saisonalität | `time_`, `season_` |
| `regime` | premium | Hurst Exponent, Entropy, Variance Ratio | `regime_` |
| `structure` | premium | FFT, Path Statistics, Convexity, Event Flow, VWAP | `struct_` |
| `risk` | premium | Drawdown, CVaR, Volatility of Volatility, Correlations | `risk_` |
| `distribution` | premium | Skewness, Kurtosis, Z-Score | `dist_` |
| `dynamics` | premium | Indikator-Änderungen, Lags, Beschleunigung | `dyn_`, `lag_`, `accel_` |
| `multi_timeframe` | premium | H4/D1/W1/Y1 Multi-Timeframe Features, Trend Alignment, Volatility Ratios | `mtf_` |
| `cross_features` | premium | Kombinierte Signale, COT × Vol Interaction, Positioning Divergence | `cross_` |
| `ichimoku` | premium | Ichimoku Cloud Komponenten | `ichi_` |
| `macro_surprise` | premium | Makro-Überraschungen, Gap-Analyse | `macro_surprise_` |
| `microstructure` | premium | Bar-Microstructure, Tick-Proxies | `micro_` |
| `market_regime` | premium | Risk-On/Off Composite aus VIX, Credit, Equity, Treasury | `regime_risk_`, `regime_vix_` |
| `regime_cluster` | premium | Composite Regime Score → K-Means Clustering (trending/mean-reverting/choppy) | `regime_cluster_` |

**Strategy-JSON:**
```json
"pipeline": {
  "indicators": [
    {"name": "trend", "params": {"adx_periods": [7, 14, 21], "ema_periods": [8, 21, 50]}},
    {"name": "momentum", "params": {"rsi_periods": [7, 14]}},
    {"name": "volatility", "params": {"atr_periods": [7, 14, 21], "compression_lookback": 100}},
    {"name": "regime", "params": {}},
    {"name": "market_regime", "params": {"window": 50}}
  ]
}
```

---

### Preprocessing (`BasePreprocessor`)

Transformiert OHLC-Daten vor der Feature-Berechnung. Folgt dem sklearn **fit/transform-Pattern** um Lookahead Bias zu verhindern: `fit()` lernt nur auf Trainingsdaten, `transform()` wendet gelernte Parameter auf beliebige Daten an.

**Basisklasse:**

```python
class BasePreprocessor(BasePlugin, ABC):
    phase = PluginPhase.PREPROCESSING
    name: str = "base"
    order: int = 100       # Ausführungsreihenfolge (niedriger = früher)
    fitted_: bool = False  # Ob fit() bereits aufgerufen wurde

    def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor":
        """Lernt Parameter von Train-Daten. NIEMALS auf Test/OOS-Daten!"""

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Transformiert DataFrame mit gelernten Parametern."""

    def fit_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Kombiniert fit() und transform() für Train-Daten."""

    def inverse_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Optional: Rücktransformation."""
```

**Lifecycle pro CV-Fold:**
1. `fit()` wird auf Train-Daten aufgerufen → lernt Parameter
2. `transform()` wird auf Train-Daten angewendet → transformierte Train-Daten
3. `transform()` wird auf Test/OOS-Daten angewendet → transformierte Test-Daten (mit Train-Parametern)

**Registrierung:** `@register_preprocessor("name")`

**Verfügbar:**

| Plugin | Paket | Beschreibung |
|--------|-------|--------------|
| `fractional_diff` | premium | Fractional Differentiation für Stationarität (nach López de Prado) |

**Strategy-JSON:**
```json
"pipeline": {
  "preprocessing": [
    {"name": "fractional_diff", "params": {"auto_d": false, "default_d": 0.4, "columns": ["O", "H", "L", "C"]}}
  ]
}
```

---

### Feature Selection (`BaseFeatureSelector`)

Wählt die relevantesten Features für das ML-Modell aus. Wird pro Fold aufgerufen, damit nur auf Train-Daten gelernt wird.

**Basisklasse:**

```python
class BaseFeatureSelector(BasePlugin, ABC):
    phase = PluginPhase.FEATURE_SELECTION
    name: str = "base"

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                       max_features: int = None, **params) -> Tuple[List[str], dict]:
        """
        Wählt die wichtigsten Features aus.

        Args:
            X: Feature-DataFrame
            y: Target-Array (0/1 für Loss/Win)
            max_features: Maximale Anzahl Features (None = unbegrenzt)

        Returns:
            (selected_features, metadata) — Feature-Namen + Zusatzinfos
        """
```

**Registrierung:** `@register_feature_selector("name")`

**Verfügbar:**

| Plugin | Paket | Beschreibung |
|--------|-------|--------------|
| `boruta` | premium | Shadow-Feature-Vergleich — findet alle statistisch relevanten Features |
| `stability` | premium | Bootstrap-basierte Stability Selection — wrapped einen Inner Selector (z.B. Boruta) und behält nur Features die in >threshold der Bootstraps selektiert werden |
| `plateau` | premium | Plateau-basierte Selektion — bewertet Parameter-Stabilität |

**Strategy-JSON:**
```json
"pipeline": {
  "feature_selection": [
    {"name": "stability", "params": {
      "inner_selector": "boruta",
      "inner_params": {"n_iter": 5, "n_estimators": 30, "max_depth": 4, "min_z_score": 0.5},
      "n_bootstrap": 7, "threshold": 0.6, "bootstrap_ratio": 0.8, "max_features": 20
    }}
  ]
}
```

---

### Exit Strategies (`BaseExitStrategy`)

Definieren wie TP/SL-Distanzen berechnet werden. Jede Strategie bestimmt die Interpretation der Grid-Werte und wie über Parameter-Kombinationen iteriert wird.

**Basisklasse:**

```python
class BaseExitStrategy(BasePlugin, ABC):
    phase = PluginPhase.EXIT_STRATEGIES
    name: str = "base"

    @abstractmethod
    def compute_targets(self, df: pd.DataFrame, ctx: SimulationContext,
                       **params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Returns:
            (targets_long, targets_short) — Arrays mit 1.0=Win, 0.0=Loss
        """

    @abstractmethod
    def iterate_grid(self, grid_config: dict, ctx: SimulationContext) -> Iterator[dict]:
        """Iteriert über alle Parameter-Kombinationen aus Grid-Config."""

    @abstractmethod
    def get_cache_key(self, params: dict) -> str:
        """Eindeutiger Cache-Key für Target-Caching."""
```

**Registrierung:** `@register_exit_strategy("name")`

**Verfügbar:**

| Plugin | Paket | Grid-Werte | Beschreibung |
|--------|-------|------------|--------------|
| `fixed` | core | `tp: 40` = 40 × Spread | Konstante TP/SL als Spread-Multiplikatoren |
| `atr_based` | premium | `tp: 1.5` = 1.5 × ATR | Dynamische TP/SL basierend auf Volatilität (ATR) |

**Strategy-JSON:**
```json
"exit_strategy": "atr_based",
"exit_params": {"atr_period": 14, "min_tp_pips": 10, "min_sl_pips": 15}
```

---

### Data Loading (`BaseDataLoader`)

Berechnet abgeleitete Features aus extern geladenen Rohdaten. DataLoader-Plugins machen **kein I/O** — die Rohdaten sind bereits im DataFrame (geladen vom Orchestrator via DataSource).

**Dreischichtige Architektur:**

```
DataSource.load()        → I/O: CSV lesen, API fetchen, DB query
    ↓
Orchestrator             → Index-Alignment: Daily→Intraday Mapping, Forward-Fill
    ↓
DataLoader.execute()     → Computation: Lookbacks, Derived Features, Ratios
```

**Basisklasse:**

```python
class BaseDataLoader(BasePlugin, ABC):
    phase = PluginPhase.DATA_LOADING
    stateful = False

    @abstractmethod
    def execute(self, ctx, **params):
        """Compute derived features from raw data in ctx.df."""
```

**Registrierung:** `@register_data_loader("name")`

**Verfügbar:**

| Plugin | Paket | Beschreibung |
|--------|-------|--------------|
| `macro_data` | premium | Makro-Indikatoren (VIX, Yields, DXY, Yield Spreads, etc.) mit Lookbacks und Derived Features |
| `cot_positioning` | premium | CFTC COT Positioning — Z-Scores, Extremes, Crowded Trade Flags |

**Strategy-JSON:**
```json
"pipeline": {
  "data_loading": [
    {"name": "macro_data", "source": "forexsb"},
    {"name": "cot_positioning", "source": "forexsb"}
  ]
}
```

---

### DataSources

DataSources sind die I/O-Schicht — sie wissen woher Daten kommen und wie sie gelesen werden. Jede Source hat eine `load()`-Methode die ein `LoadResult` zurückgibt.

```python
@dataclass
class LoadResult:
    data: Dict[str, pd.DataFrame]  # Name → DataFrame
    metadata: Dict[str, Any]       # Zusätzliche Metadaten
    source_name: str               # Quellname
```

**Verfügbare Source-Typen:**

| Typ | Klasse | Beschreibung | `load()` |
|-----|--------|--------------|----------|
| `csv` | `CSVSourceConfig` | Lokale CSV-Dateien | Liest CSVs, parst Dates |
| `rest` | `RESTSourceConfig` | REST APIs | API-Endpunkte |
| `websocket` | `WebSocketSourceConfig` | WebSocket Streams | Streaming (kein Batch) |
| `database` | `DBSourceConfig` | SQL-Datenbanken | SQL Queries via SQLAlchemy |

**Vorkonfigurierte Quellen:**

| Name | Typ | Beschreibung |
|------|-----|--------------|
| `forexsb` | csv | Forex Strategy Builder Exports |
| `stooq` | csv | Stooq.com historische Daten |
| `yahoo` | csv | Yahoo Finance Daten |
| `downloads` | csv | Manuell heruntergeladene Daten |

**Eigene Quelle registrieren:**
```python
from fwbg.core.data_sources import register_csv_source

register_csv_source(
    name="my_data",
    path="/path/to/csvs",
    file_pattern="{symbol}_DAY.csv",
)
```

---

### Risk Management (`BaseRiskManager`)

Berechnet Positionsgrößen und Risk-Controls basierend auf Trade-Historie und Performance-Metriken.

**Basisklasse:**

```python
class BaseRiskManager(BasePlugin, ABC):
    phase = PluginPhase.RISK_MANAGEMENT
    name: str = "base"

    @abstractmethod
    def compute_risk_params(self, trades: List[float], win_rate: float,
                           rrr: float, **params) -> Dict[str, Any]:
        """
        Berechnet Risk-Parameter.

        Returns:
            Dict mit mindestens:
            - risk_per_trade: float (Positionsgröße als Kapitalanteil)
            - circuit_breaker: dict (pause_after_losses, pause_bars, enabled)
            - risk_adjustment: dict (original_risk, scale_factor, target_dd)
        """
```

**Registrierung:** `@register_risk_manager("name")`

**Verfügbar:**

| Plugin | Paket | Beschreibung |
|--------|-------|--------------|
| `kelly` | core | Kelly Criterion — optimale Positionsgröße basierend auf Win Rate und RRR |
| `vol_targeted_kelly` | core | Kelly Criterion mit Volatility Targeting — skaliert Positionsgröße mit target_vol / realized_vol |

---

### Broker-Adapter

Der Bot kommuniziert über austauschbare Broker-Adapter. Jeder Adapter implementiert ein einheitliches Interface für Order-Management, Streaming und Account-Daten.

Adapter werden als separate Pakete installiert:
```bash
pip install fwbg-broker-ig     # IG Markets
pip install fwbg-broker-xyz    # Weitere Broker
```

**Registrierung:** `@register_broker_adapter("name")`

Siehe [docs/ADAPTERS.md](docs/ADAPTERS.md) für Details zum Adapter-System.

---

## Eigene Plugins erstellen

### 1. Verzeichnisstruktur

```
~/.fwbg/plugins/
└── my-package/
    ├── manifest.json                    # Package Manifest
    └── indicators/                      # Plugin-Typ Verzeichnis
        └── my_indicator/
            ├── manifest.json            # Plugin Manifest
            ├── __init__.py              # Implementierung
            └── tests.py                 # Optional: Plugin-Tests
```

### 2. Package Manifest (`my-package/manifest.json`)

```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "Meine Trading-Indikatoren",
  "plugins": {
    "indicators": ["my_indicator"]
  }
}
```

### 3. Plugin Manifest (`indicators/my_indicator/manifest.json`)

```json
{
  "name": "my_indicator",
  "version": "1.0.0",
  "description": "Custom Momentum-Indikator",
  "benefits_from_stationary": false
}
```

### 4. Implementierung (`indicators/my_indicator/__init__.py`)

```python
import pandas as pd
import numpy as np
from fwbg.plugins.indicator import BaseIndicator, shift_features, safe_divide
from fwbg.pipeline.base import PluginPhase
from fwbg.core.registry import register_indicator


@register_indicator("my_indicator")
class MyIndicator(BaseIndicator):
    name = "my_indicator"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS
    group = "custom"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        lookback = params.get("lookback", 14)

        features = {}
        returns = df["C"].pct_change()
        features["my_momentum"] = returns.rolling(lookback).mean()
        features["my_volatility"] = returns.rolling(lookback).std()
        features["my_ratio"] = safe_divide(
            features["my_momentum"], features["my_volatility"]
        )

        # PFLICHT: shift_features() verhindert Lookahead Bias
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> list:
        return ["my_momentum", "my_volatility", "my_ratio"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"lookback": 14}

    def validate(self) -> bool:
        return True
```

### 5. In Strategy-JSON verwenden

```json
{
  "pipeline": {
    "indicators": [
      {"name": "my-package:my_indicator", "params": {"lookback": 21}},
      {"name": "trend", "params": {}}
    ]
  }
}
```

Das Plugin wird beim Start automatisch aus `~/.fwbg/plugins/` entdeckt und registriert.

### Plugin-Typ Übersicht

| Typ | Basisklasse | Decorator | Verzeichnis |
|-----|-------------|-----------|-------------|
| Indikator | `BaseIndicator` | `@register_indicator` | `indicators/` |
| Preprocessing | `BasePreprocessor` | `@register_preprocessor` | `preprocessing/` |
| Feature Selection | `BaseFeatureSelector` | `@register_feature_selector` | `feature_selection/` |
| Exit Strategy | `BaseExitStrategy` | `@register_exit_strategy` | `exit_strategies/` |
| Data Loader | `BaseDataLoader` | `@register_data_loader` | `data_loading/` |
| Risk Manager | `BaseRiskManager` | `@register_risk_manager` | `risk_management/` |
| Broker Adapter | `BrokerAdapter` | `@register_broker_adapter` | via pip Entry Points |

---

## CLI-Optionen

| Option | Beschreibung | Beispiel |
|--------|--------------|----------|
| `--assets` | Komma-separierte Asset-Liste | `--assets EURUSD,GBPUSD` |
| `--strategy-file` | Pfad zur Strategy-JSON | `--strategy-file strategies/exploration.json` |
| `--asset-classes` | Nur bestimmte Klassen | `--asset-classes FOREX` |
| `--tags` | Runs nach Tags filtern | `--tags baseline` |
| `--list` | Alle vorhandenen Runs anzeigen | `--list` |
| `--compare` | Runs vergleichen | `--compare RUN1 RUN2` |
| `--no-save` | Ergebnisse nicht speichern | `--no-save` |
| `--load` | Details eines Runs anzeigen | `--load RUN_ID` |
| `--reverse-worst` | Schlechteste Strategien umkehren | `--reverse-worst RUN_ID` |
| `--cpu` | Max CPU-Auslastung (0.0-1.0) | `--cpu 0.8` |
| `--ram-reserve` | Min freier RAM-Anteil | `--ram-reserve 0.25` |
| `--ram-per-worker` | RAM pro Worker in GB | `--ram-per-worker 4.0` |
| `--timeframe` | Timeframe überschreiben | `--timeframe H4` |

---

## Strategy Configuration

Strategies werden in JSON-Dateien unter `strategies/` konfiguriert.

```json
{
  "name": "My Strategy",
  "pipeline": {
    "preprocessing": [
      {"name": "fractional_diff", "params": {"default_d": 0.4}}
    ],
    "indicators": [
      {"name": "trend", "params": {"adx_periods": [7, 14, 21]}},
      {"name": "momentum", "params": {}}
    ],
    "feature_selection": [
      {"name": "boruta", "params": {"max_features": 20}}
    ],
    "data_loading": [
      {"name": "macro_data", "source": "forexsb"},
      {"name": "cot_positioning", "source": "forexsb"}
    ]
  },
  "exit_strategy": "fixed",
  "exit_params": {},
  "model": {"architecture": "long_short_separate"},
  "grids": {
    "FOREX": {
      "tp": [10, 20, 30], "sl": [20, 30, 50], "ct": [0.5, 0.55, 0.6],
      "regime_filter_grid": {
        "condition_grids": [
          {"column": "trend_adx_14", "operator": ">=", "values": [null, 25], "directions": 6, "else_directions": 0}
        ]
      }
    }
  },
  "validation": {
    "folds": 8, "oos_size": 4000, "n_inner_folds": 3, "embargo_bars": 100,
    "sample_weights": true, "probability_calibration": true, "calibration_method": "isotonic",
    "early_pruning": {"enabled": true, "keep_ratio": 0.5, "min_survivors": 10}
  },
  "filters": {"min_rrr": 0, "min_trades": 30},
  "resources": {"ram_per_worker_gb": 4.0, "max_cpu_percent": 0.95, "max_concurrent_assets": 2}
}
```

**Vollständige Parameter-Referenz:** [strategies/README.md](strategies/README.md)

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

**Datenquellen:**

| Daten | Quelle | Frequenz | Historie |
|-------|--------|----------|----------|
| DXY, VIX (hourly) | yfinance | H1 | ~2 Jahre |
| US2Y, US5Y, US30Y | FRED (daily, ffill) | D1 | 25+ Jahre |
| DE10Y, JP10Y, GB10Y, AU10Y | FRED (monatlich, daily ffill) | D1 | 25+ Jahre |
| COT EURUSD, USDJPY, GBPUSD, ... | CFTC TFF Reports | Wöchentlich (daily ffill) | 2006+ |
| VIX, SPX, TNX, DXY, ... (daily) | Forex Strategy Builder | D1 | variiert |

**Yield Curve Shape** — automatisch berechnete Derived Features:
- `macro_yield_curve_10y_2y` — Yield Curve Slope (10Y-2Y, Inversions-Signal)
- `macro_yield_curve_30y_5y` — Long-End Steepness (30Y-5Y)
- `macro_yield_curve_10y_3m` — Term Spread (10Y-3M)

**Yield Spreads** (International):
- `macro_yield_spread_us_de` — US-Germany Zinsdifferenz (EURUSD Carry)
- `macro_yield_spread_us_jp` — US-Japan (USDJPY Carry)
- `macro_yield_spread_us_gb` — US-UK (GBPUSD Carry)
- `macro_yield_spread_us_au` — US-Australia (AUDUSD Carry)

Alle Derived Features inkl. Momentum: `_chg_2d`, `_chg_5d`, `_chg_10d`, `_chg_20d`, `_chg_60d`

**Weitere automatische Features:**
- `vol_rv_iv_ratio` / `vol_rv_iv_spread` — Realized Vol vs VIX (Mean-Reversion/Breakout Signal)
- `cross_cot_{pair}_vol_interaction` — COT Positioning × Vol (Explosivitäts-Signal)
- `cross_cot_{pair}_price_divergence` — Preis vs Positioning Divergenz (Distribution-Signal)

---

## Ergebnisse

Optimierungsergebnisse in `test_results/<timestamp>/`:

```
test_results/20260201_103045_abc123/
├── results.json           # Alle Ergebnisse
├── summary.txt            # Zusammenfassung
└── EURUSD/
    └── best_candidate.json
```

| Status | Bedeutung |
|--------|-----------|
| `significant` | Statistisch signifikanter Edge gefunden |
| `not_significant` | Kein Edge (p-value >= 0.05) |
| `no_candidates` | Keine validen Kandidaten |

---

## Statistische Validierung

Der Optimizer prüft gefundene Strategien in drei Stufen auf statistische Robustheit:

### 1. Monte Carlo Permutation Test

Testet ob die beobachtete Win-Rate signifikant besser als Zufall ist:
- **1000 Permutationen** der Trade-Ergebnisse
- **p-value < 0.05** → Edge ist statistisch signifikant
- Zusätzlich: Equity-Simulation (500 Pfade) für Bankruptcy-Rate

### 2. Deflated Sharpe Ratio (DSR)

*Bailey & López de Prado (2014)*

Korrigiert den beobachteten Sharpe Ratio für **Multiple Testing** — je mehr Grid-Kombinationen getestet werden, desto wahrscheinlicher findet man zufällig einen hohen Sharpe.

```
DSR = Φ((SR_obs - E[max(SR)]) / σ(SR))
```

- **E[max(SR)]** — Erwarteter maximaler Sharpe unter Null-Hypothese (alle Strategien = Zufall)
- **σ(SR)** — Standardabweichung des Sharpe-Schätzers (berücksichtigt Skewness/Kurtosis)
- **DSR > 0.95** → Sharpe ist auch nach Korrektur für Multiple Testing signifikant

### 3. Probability of Backtest Overfitting (PBO)

*Bailey, Borwein, López de Prado, Zhu (2017)*

Misst die Wahrscheinlichkeit, dass die beste In-Sample-Strategie Out-of-Sample schlecht abschneidet.

**Methode: Combinatorial Symmetric Cross-Validation (CSCV)**
- Bei 8 Walk-Forward Folds: **C(8,4) = 70** mögliche IS/OOS-Splits
- Für jeden Split: Prüft ob der beste IS-Combo auch OOS gut rankt
- **PBO > 0.50** → Wahrscheinlich Overfitting

**Ergebnis-Werte im JSON:**

```json
"overfitting": {
  "dsr": {
    "dsr": 0.982,
    "observed_sr": 1.85,
    "expected_max_sr": 2.51,
    "n_strategies": 144,
    "is_significant": true
  },
  "pbo": {
    "pbo": 0.12,
    "n_cscv_splits": 70,
    "is_overfit": false,
    "degradation": 0.88,
    "logit_mean": 1.45
  }
}
```

| Metrik | Gut | Schlecht | Bedeutung |
|--------|-----|---------|-----------|
| DSR | > 0.95 | < 0.50 | Sharpe übersteht Multiple-Testing-Korrektur |
| PBO | < 0.20 | > 0.50 | Beste IS-Strategie bleibt auch OOS stark |

### 4. Feature Stability

Analysiert die Konsistenz der Feature-Selektion über alle Walk-Forward Folds hinweg:

```json
"feature_stability": {
  "stable_count": 12,
  "unstable_count": 3,
  "details": {
    "trend_adx_14": {"count": 8, "stability": 1.0},
    "vol_atr_pct_14_rank": {"count": 6, "stability": 0.75},
    "macro_yield_spread_us_de_chg_5d": {"count": 2, "stability": 0.25}
  }
}
```

- **stability >= 0.50** — Feature wird als stabil eingestuft (in mindestens 50% der Folds selektiert)
- Instabile Features deuten auf Noise-Fitting hin

---

## Weitere Dokumentation

- [strategies/README.md](strategies/README.md) — Strategy Configuration Reference
- [docs/FEATURES.md](docs/FEATURES.md) — Alle verfügbaren Indikatoren & Features
- [docs/ADAPTERS.md](docs/ADAPTERS.md) — Adapter-System für Datenquellen & Broker

---

## Anforderungen

- Python 3.10+
- 16GB+ RAM (für Optimizer)

---

## Lizenz

Proprietär - Nur für internen Gebrauch.
