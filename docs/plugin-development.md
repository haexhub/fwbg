# Plugin Development Guide

Anleitung zum Erstellen eigener FWBG-Plugins. Plugins können Indikatoren, Preprocessors, Feature Selectors, Exit Strategies, Risk Manager oder Data Loader sein.

---

## Quick Start: Custom Indicator

### 1. Verzeichnisstruktur

```
~/.fwbg/plugins/
└── my-package/
    ├── manifest.json                    # Package Manifest
    └── indicators/
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
  "phase": "indicators",
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

Das Plugin wird beim Start automatisch aus `~/.fwbg/plugins/` entdeckt und registriert. Falls der Name eindeutig ist, reicht auch der Kurzname: `"name": "my_indicator"`.

---

## Plugin-Typen Referenz

| Typ | Basisklasse | Decorator | Phase | Vom PipelineRunner ausgeführt? | Verzeichnis |
|-----|-------------|-----------|-------|-------------------------------|-------------|
| Indikator | `BaseIndicator` | `@register_indicator` | INDICATORS | Ja | `indicators/` |
| Preprocessor | `BasePreprocessor` | `@register_preprocessor` | PREPROCESSING | Ja | `preprocessing/` |
| Feature Selector | `BaseFeatureSelector` | `@register_feature_selector` | FEATURE_SELECTION | Ja (im Inner CV) | `feature_selection/` |
| Exit Strategy | `BaseExitStrategy` | `@register_exit_strategy` | EXIT_STRATEGIES | Nein (Optimization-Code) | `exit_strategies/` |
| Risk Manager | `BaseRiskManager` | `@register_risk_manager` | RISK_MANAGEMENT | Nein (Optimization-Code) | `risk_management/` |
| Data Loader | `BaseDataLoader` | `@register_data_loader` | DATA_LOADING | Ja | `data_loading/` |

---

## Plugin-Typ: Indicator

**Datei:** `src/fwbg/plugins/indicator.py`

```python
class BaseIndicator(BasePlugin, ABC):
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True
    group: str = "custom"
    benefits_from_stationary: bool = False

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...
    def get_feature_columns(self) -> List[str]: ...
```

**Pflicht:**
- `shift_features()` am Ende von `compute()` — 1-Bar Shift gegen Lookahead-Bias
- `safe_divide()` für alle Divisionen — NaN statt inf bei Division durch ~0

**Wichtige Attribute:**
- `benefits_from_stationary = False` → Einmal auf Raw-OHLC berechnen, gecacht
- `benefits_from_stationary = True` → Pro Fold auf preprocessed Data berechnen

Detaillierte Dokumentation: [Phase 3: Indicators](phases/3-indicators.md)

---

## Plugin-Typ: Preprocessor

**Datei:** `src/fwbg/plugins/preprocessor.py`

```python
class BasePreprocessor(BasePlugin, ABC):
    phase = PluginPhase.PREPROCESSING
    order: int = 100

    def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor": ...

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...

    def fit_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...
    def inverse_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...
```

**Lifecycle pro Fold:** `reset()` → `fit(train)` → `transform(train)` → `transform(test)`

**Pflicht:**
- `fit()` nur auf Train-Daten aufrufen — Lookahead-Bias-Prevention
- `order` setzen für Reihenfolge bei mehreren Preprocessors

**Beispiel:**

```python
@register_preprocessor("my_normalizer")
class MyNormalizer(BasePreprocessor):
    name = "my_normalizer"
    order = 50

    def fit(self, df, **params):
        self.mean_ = df["C"].mean()
        self.std_ = df["C"].std()
        return super().fit(df, **params)

    def transform(self, df, **params):
        super().transform(df, **params)
        result = df.copy()
        result["C"] = (result["C"] - self.mean_) / self.std_
        return result
```

Detaillierte Dokumentation: [Phase 2: Preprocessing](phases/2-preprocessing.md)

---

## Plugin-Typ: Feature Selector

**Datei:** `src/fwbg/plugins/feature_selector.py`

```python
class BaseFeatureSelector(BasePlugin, ABC):
    phase = PluginPhase.FEATURE_SELECTION

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                       max_features: int = None, **params) -> Tuple[List[str], dict]: ...
```

**Return:** `(selected_feature_names, metadata_dict)`

**Beispiel:**

```python
@register_feature_selector("my_selector")
class MySelector(BaseFeatureSelector):
    name = "my_selector"

    def select_features(self, X, y, max_features=None, **params):
        importances = compute_importances(X, y)
        top_features = sorted(importances, key=importances.get, reverse=True)
        if max_features:
            top_features = top_features[:max_features]
        return top_features, {"importances": importances}
```

Detaillierte Dokumentation: [Phase 4: Feature Selection](phases/4-feature-selection.md)

---

## Plugin-Typ: Exit Strategy

**Datei:** `src/fwbg/plugins/exit_strategy.py`

```python
class BaseExitStrategy(BasePlugin, ABC):
    phase = PluginPhase.EXIT_STRATEGIES

    @abstractmethod
    def compute_targets(self, df, ctx, **params) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def iterate_grid(self, grid_config, ctx) -> Iterator[dict]: ...

    @abstractmethod
    def get_cache_key(self, params) -> str: ...
```

**Drei abstrakte Methoden:**
1. `compute_targets()` — Berechnet Win/Loss-Arrays (1.0/0.0) für Long und Short
2. `iterate_grid()` — Generiert Parameter-Kombinationen aus Grid-Config
3. `get_cache_key()` — Eindeutiger Cache-Key pro Parameterkombination

**Beispiel:**

```python
@register_exit_strategy("my_exit")
class MyExitStrategy(BaseExitStrategy):
    name = "my_exit"

    def compute_targets(self, df, ctx, **params):
        tp = params.get("tp", 30)
        sl = params.get("sl", 20)
        # Simulation via Numba...
        return targets_long, targets_short

    def iterate_grid(self, grid_config, ctx):
        for tp in grid_config.get("tp", [30]):
            for sl in grid_config.get("sl", [20]):
                yield {"tp": tp, "sl": sl}

    def get_cache_key(self, params):
        return f"my_tp{params['tp']}_sl{params['sl']}"
```

Detaillierte Dokumentation: [Phase 5: Exit Strategies](phases/5-exit-strategies.md)

---

## Plugin-Typ: Risk Manager

**Datei:** `src/fwbg/plugins/risk_manager.py`

```python
class BaseRiskManager(BasePlugin, ABC):
    phase = PluginPhase.RISK_MANAGEMENT

    @abstractmethod
    def compute_risk_params(self, trades, win_rate, rrr, **params) -> Dict[str, Any]: ...
```

**Return-Dict muss enthalten:**
- `risk_per_trade`: float — Positionsgröße als Kapitalanteil
- `trade_returns`: List[float] — Per-Trade Returns
- `circuit_breaker`: dict — Pause-Logik bei Verlusten
- `risk_adjustment`: dict — Skalierungsfaktoren

Detaillierte Dokumentation: [Phase 6: Risk Management](phases/6-risk-management.md)

---

## Plugin-Typ: Data Loader

**Datei:** `src/fwbg/plugins/data_loader.py`

```python
class BaseDataLoader(BasePlugin, ABC):
    phase = PluginPhase.DATA_LOADING
    stateful = False

    @abstractmethod
    def execute(self, ctx, **params): ...
```

**Wichtig:** DataLoader machen kein I/O. Die Rohdaten sind bereits im DataFrame (geladen vom Orchestrator). DataLoader berechnen nur abgeleitete Features.

Detaillierte Dokumentation: [Phase 1: Data Loading](phases/1-data-loading.md)

---

## Plugin Testing

Jedes Plugin kann eine `tests.py` im Plugin-Verzeichnis haben:

```python
# my_indicator/tests.py
def test_compute_produces_features():
    from . import MyIndicator
    indicator = MyIndicator()
    df = pd.DataFrame({"O": [...], "H": [...], "L": [...], "C": [...], "V": [...]})
    result = indicator.compute(df)
    assert "my_momentum" in result.columns

def test_shift_applied():
    from . import MyIndicator
    indicator = MyIndicator()
    df = pd.DataFrame({"O": [...], "H": [...], "L": [...], "C": [...], "V": [...]})
    result = indicator.compute(df)
    assert pd.isna(result["my_momentum"].iloc[0])  # Erste Zeile NaN durch shift
```

Tests ausführen:
```python
plugin = MyIndicator()
passed, failed, errors = plugin.run_tests()
print(f"{passed} passed, {failed} failed")
```

---

## Entry-Point Registrierung (pip-installierbare Pakete)

Für Plugin-Pakete die via `pip install` installiert werden, muss ein Entry Point in `pyproject.toml` definiert werden:

```toml
[project.entry-points."fwbg.plugin_packages"]
my-package = "my_package:get_plugins_dir"
```

Die Entry-Point-Funktion gibt den Pfad zum Plugin-Verzeichnis zurück:

```python
# my_package/__init__.py
from pathlib import Path

def get_plugins_dir() -> Path:
    return Path(__file__).parent / "plugins" / "my-package"
```

Das Plugin-Verzeichnis hat die gleiche Struktur wie User-Plugins (manifest.json, Unterverzeichnisse pro Plugin-Typ).

---

## Häufige Fehler

### 1. shift_features() vergessen

```python
# FALSCH — Lookahead Bias!
def compute(self, df, **params):
    features = {"my_rsi": compute_rsi(df["C"])}
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

# RICHTIG
def compute(self, df, **params):
    features = {"my_rsi": compute_rsi(df["C"])}
    features_df = shift_features(features, df.index)  # ← PFLICHT
    return pd.concat([df, features_df], axis=1)
```

Ohne shift_features() sieht das Modell bei Bar `i` den Indikator-Wert von Bar `i` — die aktuelle, noch nicht abgeschlossene Bar. Das erzeugt unrealistische Backtesting-Ergebnisse.

### 2. Preprocessor auf allen Daten fitten

```python
# FALSCH — Lookahead Bias!
preprocessor.fit(all_data)
preprocessor.transform(train_data)
preprocessor.transform(test_data)

# RICHTIG
preprocessor.fit(train_data)           # Nur auf Train!
preprocessor.transform(train_data)
preprocessor.transform(test_data)      # Gleiche Parameter wie fit()
```

### 3. safe_divide() vergessen

```python
# FALSCH — kann inf erzeugen
ratio = momentum / volatility

# RICHTIG — gibt NaN bei Nenner ~0
ratio = safe_divide(momentum, volatility)
```

### 4. benefits_from_stationary falsch gesetzt

- `True` bei Indikatoren die von stationären Daten profitieren (Trend, Moving Averages)
- `False` bei Indikatoren die bereits normalisiert sind (RSI, Stochastic) oder skalenunabhängig (ATR)

Falsch gesetzt: Entweder unnötig langsam (False→True) oder falsche Ergebnisse (True→False, wenn der Indikator tatsächlich stationäre Eingangsdaten braucht).

---

## Weiterführende Dokumentation

- [Architektur & Plugin-System](architecture.md) — Lifecycle, Discovery, Naming
- [Phase 1: Data Loading](phases/1-data-loading.md)
- [Phase 2: Preprocessing](phases/2-preprocessing.md)
- [Phase 3: Indicators](phases/3-indicators.md)
- [Phase 4: Feature Selection](phases/4-feature-selection.md)
- [Phase 5: Exit Strategies](phases/5-exit-strategies.md)
- [Phase 6: Risk Management](phases/6-risk-management.md)
