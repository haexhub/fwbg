# Architektur & Plugin-System

FWBG basiert auf einer Plugin-Pipeline-Architektur. Jede Funktionalität — von Indikatoren über Exit-Strategien bis zu Datenquellen — ist als Plugin implementiert. Plugins können hinzugefügt, ausgetauscht oder komplett entfernt werden, ohne den restlichen Code zu ändern.

---

## Pipeline-Phasen

Der `PluginPhase`-Enum definiert die Ausführungsreihenfolge:

```
1. DATA_LOADING      → Externe Daten laden, Features berechnen
2. PREPROCESSING     → OHLC-Daten transformieren (Stationarität)
3. INDICATORS        → Technische Indikatoren berechnen
4. FEATURE_SELECTION → Relevante Features auswählen
5. EXIT_STRATEGIES   → TP/SL-Berechnung
6. RISK_MANAGEMENT   → Positionsgröße und Risk-Controls
7. LABELING          → Training-Labels generieren (intern)
8. MODEL             → ML-Modell trainieren / vorhersagen (intern)
9. VALIDATION        → Strategie-Performance validieren (intern)
```

**Wichtig:** Der `PipelineRunner` führt die Phasen 1–4 automatisch aus. EXIT_STRATEGIES und RISK_MANAGEMENT werden **direkt vom Optimization-Code** aufgerufen (nicht vom Runner). LABELING, MODEL und VALIDATION sind interne Phasen ohne User-konfigurierbare Plugins.

Detaillierte Phase-Dokumentation: [docs/phases/](phases/)

---

## BasePlugin — Das Plugin-Interface

Alle Plugins erben von `BasePlugin` (`src/fwbg/pipeline/base.py`):

```python
class BasePlugin(ABC):
    # Pflichtattribute (müssen von Subklassen definiert werden)
    name: str                    # Eindeutiger Name (z.B. "trend")
    phase: PluginPhase           # Pipeline-Phase (z.B. PluginPhase.INDICATORS)

    # Optionale Attribute mit Defaults
    version: str = "0.1.0"      # Semantische Version
    stateful: bool = False       # Speichert Zustand über Aufrufe?
    cacheable: bool = True       # Kann das Ergebnis gecacht werden?
    depends_on: List[str] = []   # Abhängigkeiten zu anderen Plugins

    # Methoden
    def execute(self, ctx: PipelineContext, **params) -> PipelineContext: ...
    def fit(self, ctx: PipelineContext, **params) -> None: ...
    def reset(self) -> None: ...
    def validate(self) -> bool: ...

    @classmethod
    def get_default_params(cls) -> dict: ...

    def get_feature_columns(self) -> List[str]: ...
```

### Methoden im Detail

| Methode | Beschreibung |
|---------|--------------|
| `execute(ctx, **params)` | Hauptmethode — verarbeitet PipelineContext und gibt ihn zurück |
| `fit(ctx, **params)` | Lernt Parameter aus Trainingsdaten (nur für `stateful=True` Plugins) |
| `reset()` | Setzt gelernten Zustand zurück (wird zwischen CV-Folds aufgerufen) |
| `validate()` | Prüft ob das Plugin korrekt konfiguriert ist |
| `get_default_params()` | Gibt Default-Parameter zurück (classmethod) |
| `get_feature_columns()` | Gibt die erzeugten Feature-Spaltennamen zurück |

---

## Plugin-Lifecycle: stateful / cacheable / benefits_from_stationary

Diese drei Attribute bestimmen, wann und wie oft ein Plugin ausgeführt wird.

### `stateful` (bool, Default: False)

Bestimmt ob das Plugin Zustand zwischen Aufrufen speichert, der aus Trainingsdaten gelernt wurde.

- **`False` (Default): Zustandslos.** Das Plugin berechnet bei jedem Aufruf das gleiche Ergebnis, unabhängig von vorherigen Aufrufen. Es gibt keinen `fit()`-Schritt. Beispiel: Die meisten Indikatoren — `trend` berechnet ADX/EMA immer gleich, egal welcher Fold.

- **`True`: Zustandsbehaftet.** Das Plugin hat einen `fit()`-Schritt, der Parameter aus Trainingsdaten lernt. Diese gelernten Parameter werden dann in `execute()`/`transform()` wiederverwendet. `fit()` wird pro CV-Fold **nur auf Train-Daten** aufgerufen (Lookahead-Bias-Schutz). Zwischen Folds wird `reset()` aufgerufen. Beispiel: `fractional_diff` — lernt den optimalen d-Wert auf Train-Daten, wendet ihn dann auf Train/Test/OOS an.

### `cacheable` (bool, Default: True)

Bestimmt ob Ergebnisse gecacht werden können, um redundante Berechnungen über Folds hinweg zu vermeiden.

- **`True` (Default): Cachebar.** Der `PipelineRunner` darf das Ergebnis zwischenspeichern und wiederverwenden, wenn sich die Eingabedaten und Parameter nicht geändert haben. Beispiel: `volatility` — ATR auf den Originaldaten ist für alle Folds identisch, muss nur einmal berechnet werden.

- **`False`: Nicht cachebar.** Das Ergebnis hängt von Zustand ab, der sich zwischen Folds ändert (z.B. gefittete Parameter). Jeder Fold muss neu berechnet werden.

### `benefits_from_stationary` (bool, nur Indikatoren, Default: False)

Bestimmt ob ein Indikator auf preprocessed (stationären) oder auf rohen OHLC-Daten berechnet wird. Nur relevant wenn Preprocessing konfiguriert ist.

- **`False` (Default):** Indikator wird **einmalig auf Originaldaten** berechnet (vor dem Preprocessing). Ergebnis wird über alle Folds hinweg wiederverwendet. Beispiel: `volatility`, `momentum`, `price_action`.

- **`True`:** Indikator wird **pro Fold auf preprocessed Daten** berechnet. Das Preprocessing (z.B. fractional differentiation) erzeugt pro Fold andere transformierte Daten, weil `fit()` pro Fold nur auf Train-Daten lernt. Beispiel: `trend` — ADX auf differenzierten Daten gibt andere Werte als auf Raw-Daten.

### Kombinationstabelle

| stateful | cacheable | Verhalten | Beispiel |
|----------|-----------|-----------|----------|
| False | True | Einmal berechnen, über Folds gecacht | `momentum`, `volatility` |
| False | False | Jedes Mal neu berechnen | — |
| True | True | Pro Fold fitten, innerhalb Fold gecacht | — |
| True | False | Pro Fold fitten, nie gecacht | `fractional_diff` |

### Entscheidungshilfe für Plugin-Entwickler

> **Frage 1:** Muss mein Plugin Parameter aus Trainingsdaten lernen?
> - Ja → `stateful = True`, `cacheable = False`
> - Nein → `stateful = False`
>
> **Frage 2:** Gibt mein Plugin bei gleicher Eingabe immer das gleiche Ergebnis?
> - Ja → `cacheable = True`
> - Nein → `cacheable = False`
>
> **Frage 3 (nur Indikatoren):** Profitiert mein Indikator von stationären Eingangsdaten?
> - Ja → `benefits_from_stationary = True`
> - Nein → `benefits_from_stationary = False`

---

## Lifecycle-Diagramm (pro Outer Fold)

```
┌─ Fold Start ──────────────────────────────────────────────────────────┐
│                                                                       │
│  Preprocessors:  reset() → fit(train) → transform(train)             │
│                                         → transform(test)             │
│                                                                       │
│  Stationary Indicators:  compute(preprocessed_data)  [pro Fold]       │
│  Raw Indicators:         (bereits einmalig vorab berechnet + gecacht)  │
│                                                                       │
│  Feature Selection:  select_features(X_train, y_train)                │
│                                                                       │
│  ── Inner CV Loop ──                                                  │
│  │  Exit Strategy:   compute_targets(df, ctx)  [gecacht pro Params]   │
│  │  Model:           train(X_train, y_train) → predict(X_test)        │
│  │  Validation:      evaluate fold results                            │
│  └──────────────────                                                  │
│                                                                       │
│  Risk Management:  compute_risk_params(trades, win_rate, rrr)         │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## PipelineContext

Der Context wird durch alle Phasen gereicht (`src/fwbg/pipeline/context.py`):

```python
@dataclass
class PipelineContext:
    df: pd.DataFrame                  # Haupt-DataFrame mit OHLCV + Features
    symbol: str                       # Asset-Symbol (z.B. "EURUSD")
    asset_class: str                  # Asset-Klasse (z.B. "FOREX", "CRYPTO")
    metadata: Dict[str, Any]          # Inter-Plugin-Kommunikation
    fold_info: Optional[Dict] = None  # Walk-Forward Fold-Info
```

- `df` wird von jedem Plugin erweitert (neue Spalten für Features)
- `metadata` erlaubt Inter-Plugin-Kommunikation (z.B. ein Data Loader speichert hier geladene Zusatzdaten)
- `clone()` erstellt eine Kopie (für parallele Verarbeitung)

---

## PipelineRunner

Der `PipelineRunner` (`src/fwbg/pipeline/runner.py`) orchestriert die Plugin-Ausführung:

### Initialisierung

```python
runner = PipelineRunner(registry=registry, config=pipeline_config)
```

Der Runner:
1. Erstellt Plugin-Instanzen aus der Config
2. Sortiert Plugins innerhalb jeder Phase topologisch nach `depends_on`
3. Validiert alle Abhängigkeiten

### Ausführung

```python
# Stateful Plugins fitten (pro Fold)
runner.fit(ctx)

# Pipeline ausführen
ctx = runner.run(ctx)

# Stateful Plugins zurücksetzen (zwischen Folds)
runner.reset()
```

### Topologische Sortierung (depends_on)

Plugins können Abhängigkeiten innerhalb derselben Phase deklarieren:

```python
class MyPlugin(BaseIndicator):
    name = "my_indicator"
    depends_on = ["trend", "momentum"]  # Kurznamen genügen
```

Der Runner:
- Löst Kurznamen zu FQNs auf ("trend" → "fwbg-core:trend")
- Validiert dass alle Abhängigkeiten existieren und in derselben Phase liegen
- Sortiert mit Kahn's Algorithmus (Abhängigkeiten werden zuerst ausgeführt)
- Erkennt zirkuläre Abhängigkeiten → ValueError

### Parameter-Hierarchie

Parameter werden gemerged (höhere Priorität überschreibt):

```
1. Plugin.get_default_params()    → niedrigste Priorität
2. Strategy-JSON Config Params    → mittlere Priorität
3. Global Runtime Params (CLI)    → höchste Priorität
```

---

## Plugin Discovery

Plugins werden automatisch aus drei Quellen entdeckt — in dieser Reihenfolge:

### 1. Core-Pakete (Builtin)

Verzeichnis: `src/fwbg/plugins/fwbg-core/`

Die mitgelieferten Plugins (trend, momentum, volatility, fixed exit strategy, kelly, etc.).

### 2. Entry-Point-Pakete (pip-installiert)

Installierte Python-Packages mit `fwbg.plugin_packages` Entry Point:

```toml
# In pyproject.toml des Plugin-Pakets:
[project.entry-points."fwbg.plugin_packages"]
fwbg-premium = "fwbg_premium:get_plugins_dir"
```

Die Entry-Point-Funktion gibt den Pfad zum Plugin-Verzeichnis zurück. Beispiel: `fwbg-premium` (`packages/fwbg-premium/`).

### 3. User-Pakete

Verzeichnis: `~/.fwbg/plugins/`

Eigene Plugins im gleichen Verzeichnisformat wie Core-Pakete. Werden zuletzt entdeckt.

### Paketstruktur

Jedes Plugin-Paket hat eine standardisierte Verzeichnisstruktur:

```
my-package/
├── manifest.json              # Package Manifest
├── indicators/
│   └── my_indicator/
│       ├── manifest.json      # Plugin Manifest
│       ├── __init__.py        # Plugin-Klasse
│       └── tests.py           # Optional: Plugin-Tests
├── preprocessing/
│   └── ...
├── exit_strategies/
│   └── ...
├── feature_selection/
│   └── ...
├── risk_management/
│   └── ...
└── data_loading/
    └── ...
```

**Package Manifest** (`manifest.json` im Root):
```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "Mein Plugin-Paket",
  "plugins": {
    "indicators": ["my_indicator"],
    "preprocessing": []
  }
}
```

**Plugin Manifest** (`manifest.json` im Plugin-Verzeichnis):
```json
{
  "name": "my_indicator",
  "version": "1.0.0",
  "description": "Mein Custom Indicator",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

Manifest-Attribute wie `benefits_from_stationary` werden beim Discovery automatisch auf die Plugin-Klasse propagiert.

---

## Name Resolution: FQN vs Kurznamen

### Voll qualifizierte Namen (FQN)

Format: `"paket:plugin"` — z.B. `"fwbg-core:trend"`, `"fwbg-premium:regime"`

FQNs sind immer eindeutig und werden intern von der Registry verwendet.

### Kurznamen

Format: einfach `"plugin"` — z.B. `"trend"`, `"regime"`, `"fractional_diff"`

Kurznamen werden automatisch über `PluginRegistry.resolve_name()` aufgelöst. Die Methode sucht in **allen registrierten Paketen** (nicht nur fwbg-core!), ob ein Plugin mit diesem Namen existiert.

**Wichtig:** Kurznamen funktionieren für **alle Pakete** — Core, Premium und User-Pakete gleichermaßen. Es ist kein Privileg von fwbg-core.

### Mehrdeutigkeit

Wenn zwei Pakete ein Plugin mit demselben Namen registrieren, wirft `resolve_name()` einen `ValueError`:

```
ValueError: Ambiguous plugin name 'trend' — found in: fwbg-core, my-package.
Use fully qualified name: 'fwbg-core:trend' or 'my-package:trend'
```

### Empfehlung

- **Kurznamen** in Strategy-JSONs für Lesbarkeit: `{"name": "trend", "params": {}}`
- **FQN** nur bei Namenskonflikten: `{"name": "fwbg-premium:regime", "params": {}}`

---

## Registration Decorators

Plugins werden bei der Entdeckung automatisch registriert. Zusätzlich gibt es Decorators für explizite Registrierung (`src/fwbg/core/registry.py`):

| Decorator | Plugin-Typ |
|-----------|-----------|
| `@register_indicator("name")` | Indikator |
| `@register_preprocessor("name")` | Preprocessor |
| `@register_feature_selector("name")` | Feature Selector |
| `@register_exit_strategy("name")` | Exit Strategy |
| `@register_risk_manager("name")` | Risk Manager |
| `@register_data_loader("name")` | Data Loader |
| `@register_broker_adapter("name")` | Broker Adapter |

Die Decorators setzen `plugin.name` und registrieren die Klasse im globalen Registry.

---

## Plugin-Tests

Jedes Plugin kann eine eigene `tests.py` im Plugin-Verzeichnis haben:

```python
# my_indicator/tests.py
def test_compute_basic():
    indicator = MyIndicator()
    df = create_sample_ohlcv()
    result = indicator.compute(df)
    assert "my_feature" in result.columns

def test_shift_applied():
    indicator = MyIndicator()
    df = create_sample_ohlcv()
    result = indicator.compute(df)
    assert result["my_feature"].iloc[0] != result["my_feature"].iloc[0]  # NaN from shift
```

Tests werden ausgeführt via:
- `plugin.run_tests()` → (passed, failed, errors)
- `plugin.has_tests()` → True/False

---

## Weiterführende Dokumentation

- [Plugin Development Guide](plugin-development.md) — Eigene Plugins erstellen
- [Phase 1: Data Loading](phases/1-data-loading.md)
- [Phase 2: Preprocessing](phases/2-preprocessing.md)
- [Phase 3: Indicators](phases/3-indicators.md)
- [Phase 4: Feature Selection](phases/4-feature-selection.md)
- [Phase 5: Exit Strategies](phases/5-exit-strategies.md)
- [Phase 6: Risk Management](phases/6-risk-management.md)
- [Phase 7: Validation](phases/7-validation.md)
