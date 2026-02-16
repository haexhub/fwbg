# FWBG — ML Trading Strategy Optimizer

FWBG ist ein Machine-Learning-basiertes Framework zur Walk-Forward-Optimierung von Trading-Strategien. Es findet optimale Parameter (Take-Profit, Stop-Loss, Confidence-Threshold) via Nested Cross-Validation und prüft die statistische Robustheit der Ergebnisse durch multiple Overfitting-Tests.

Das System basiert auf einer **modularen Plugin-Architektur**: Jede Phase der Pipeline — von Indikatoren über Exit-Strategien bis zum Risk Management — ist als austauschbares Plugin implementiert. Eigene Plugins können ohne Änderung am Framework-Code hinzugefügt werden.

---

## Warum FWBG?

- **Plugin-Architektur** — Indikatoren, Preprocessors, Feature Selectors, Exit Strategies und Risk Manager sind Plugins. Jede Phase kann durch eigene Implementierungen erweitert oder komplett ersetzt werden.

- **Walk-Forward Validation** — Nested Cross-Validation mit expandierenden Fenstern, Time-Series Purging (Embargo) und Sample Weights. Kein Lookahead-Bias by Construction.

- **Overfitting-Schutz** — Drei statistische Tests prüfen jede gefundene Strategie: Deflated Sharpe Ratio (Multiple-Testing-Korrektur), Probability of Backtest Overfitting (CSCV), Monte Carlo Permutation Tests.

- **Numba-beschleunigte Simulation** — JIT-kompilierte Trade-Simulation mit paralleler Verarbeitung für schnelle Grid-Search-Durchläufe.

- **Core + Premium Pakete** — Open-Source Core-Indikatoren (Trend, Momentum, Volatility). Premium-Paket mit Regime Detection, Makro-Daten, COT-Positioning, ATR-basierte Exits, Feature Selection.

- **Live-Trading Ready** — Broker-Adapter-System für Live-Ausführung (IG Markets etc.).

---

## Quick Start

### Installation

```bash
git clone https://github.com/haexhub/fwbg.git
cd fwbg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional: Premium Plugins
pip install -e packages/fwbg-premium
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

## Architektur

FWBG verarbeitet Daten in einer Plugin-Pipeline mit definierten Phasen:

```
DATA_LOADING → PREPROCESSING → INDICATORS → FEATURE_SELECTION
    → EXIT_STRATEGIES → RISK_MANAGEMENT → MODEL → VALIDATION
```

Jede Phase kann beliebig viele Plugins enthalten. Der `PipelineRunner` orchestriert die Ausführung in der korrekten Reihenfolge, löst Abhängigkeiten auf und mergt Parameter.

**Detaillierte Architektur-Dokumentation:** [docs/architecture.md](docs/architecture.md)

---

## Pipeline-Phasen

| # | Phase | Zweck | Dokumentation |
|---|-------|-------|---------------|
| 1 | Data Loading | Externe Daten laden (Makro, COT) | [docs/phases/1-data-loading.md](docs/phases/1-data-loading.md) |
| 2 | Preprocessing | Stationaritätstransformationen | [docs/phases/2-preprocessing.md](docs/phases/2-preprocessing.md) |
| 3 | Indicators | Technische Features berechnen | [docs/phases/3-indicators.md](docs/phases/3-indicators.md) |
| 4 | Feature Selection | Relevante Features auswählen | [docs/phases/4-feature-selection.md](docs/phases/4-feature-selection.md) |
| 5 | Exit Strategies | TP/SL-Berechnung (fixed, ATR) | [docs/phases/5-exit-strategies.md](docs/phases/5-exit-strategies.md) |
| 6 | Risk Management | Positionsgröße (Kelly, Vol-Targeted) | [docs/phases/6-risk-management.md](docs/phases/6-risk-management.md) |
| 7 | Validation | Walk-Forward CV, Overfitting-Tests | [docs/phases/7-validation.md](docs/phases/7-validation.md) |

---

## CLI-Referenz

| Option | Beschreibung | Beispiel |
|--------|--------------|----------|
| `--assets` | Komma-separierte Asset-Liste | `--assets EURUSD,GBPUSD` |
| `--strategy-file` | Pfad zur Strategy-JSON | `--strategy-file strategies/exploration.json` |
| `--asset-classes` | Nur bestimmte Klassen | `--asset-classes FOREX` |
| `--timeframe` | Timeframe überschreiben | `--timeframe H4` |
| `--tags` | Runs nach Tags filtern | `--tags baseline` |
| `--list` | Alle vorhandenen Runs anzeigen | `--list` |
| `--compare` | Runs vergleichen | `--compare RUN1 RUN2` |
| `--load` | Details eines Runs anzeigen | `--load RUN_ID` |
| `--reverse-worst` | Schlechteste Strategien umkehren | `--reverse-worst RUN_ID` |
| `--no-save` | Ergebnisse nicht speichern | `--no-save` |
| `--cpu` | Max CPU-Auslastung (0.0-1.0) | `--cpu 0.8` |
| `--ram-reserve` | Min freier RAM-Anteil | `--ram-reserve 0.25` |
| `--ram-per-worker` | RAM pro Worker in GB | `--ram-per-worker 4.0` |

---

## Strategy Configuration

Strategies werden in JSON-Dateien unter `strategies/` konfiguriert:

```json
{
  "name": "My Strategy",
  "pipeline": {
    "indicators": [
      {"name": "trend", "params": {"adx_periods": [7, 14, 21]}},
      {"name": "momentum", "params": {}}
    ],
    "feature_selection": [
      {"name": "stability", "params": {"max_features": 20}}
    ]
  },
  "exit_strategy": "fixed",
  "grids": {
    "FOREX": {"tp": [10, 20, 30], "sl": [20, 30], "ct": [0.5, 0.55, 0.6]}
  },
  "validation": {"folds": 8, "n_inner_folds": 3, "embargo_bars": 100}
}
```

**Vollständige Parameter-Referenz:** [strategies/README.md](strategies/README.md)

---

## Ergebnisse

Optimierungsergebnisse werden in `test_results/<timestamp>/` gespeichert:

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

Details zu den statistischen Tests: [docs/phases/7-validation.md](docs/phases/7-validation.md)

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
│   └── fwbg-premium/             # Premium Plugins (separates pip Package)
│       ├── indicators/           # regime, structure, risk, distribution, dynamics, ...
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

## Dokumentation

### Architektur & Plugin-System
- [Architektur & Plugin-System](docs/architecture.md) — Plugin-Lifecycle, Discovery, Naming, PipelineRunner
- [Plugin Development Guide](docs/plugin-development.md) — Eigene Plugins erstellen

### Pipeline-Phasen
- [Phase 1: Data Loading](docs/phases/1-data-loading.md) — Externe Daten, DataSources
- [Phase 2: Preprocessing](docs/phases/2-preprocessing.md) — Stationaritätstransformationen
- [Phase 3: Indicators](docs/phases/3-indicators.md) — Technische Features, shift_features, safe_divide
- [Phase 4: Feature Selection](docs/phases/4-feature-selection.md) — Boruta, Stability Selection
- [Phase 5: Exit Strategies](docs/phases/5-exit-strategies.md) — Fixed, ATR-based
- [Phase 6: Risk Management](docs/phases/6-risk-management.md) — Kelly, Vol-Targeted Kelly
- [Phase 7: Validation](docs/phases/7-validation.md) — Walk-Forward CV, DSR, PBO, Monte Carlo

### Referenzen
- [Strategy Configuration](strategies/README.md) — Vollständige JSON-Referenz
- [Feature Catalog](docs/FEATURES.md) — Alle verfügbaren Indikatoren & Features
- [Adapter System](docs/ADAPTERS.md) — Broker & Datenquellen-Adapter
- [Robust Validation Guide](docs/ROBUST_VALIDATION_GUIDE.md) — Sample-Bias Detection
- [Live Bias Detection](docs/LIVE_BIAS_DETECTION.md) — Echtzeit-Bias-Checks

---

## Anforderungen

- Python 3.10+
- 16GB+ RAM (für Optimizer)

---

## Lizenz

Proprietär - Nur für internen Gebrauch.
