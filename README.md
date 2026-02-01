# FWBG - ML Trading Strategy Optimizer

Ein Machine-Learning-basierter Optimizer für systematische Trading-Strategien. Der Optimizer findet optimale Parameter-Kombinationen (TP, SL, Confidence Threshold) für verschiedene Assets und Feature-Gruppen mittels Walk-Forward Cross-Validation.

## Inhaltsverzeichnis

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Optimizer Ausführen](#optimizer-ausführen)
4. [Strategy-Konfiguration](#strategy-konfiguration)
5. [Exit-Strategien](#exit-strategien)
6. [Feature-Gruppen](#feature-gruppen)
7. [Feature Selection](#feature-selection)
8. [Ergebnisse](#ergebnisse)
9. [Architektur](#architektur)

---

## Quick Start

```bash
# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# Optimizer mit Standard-Strategie starten
python -m optimizer --assets EURUSD

# Optimizer mit spezifischer Strategie
python -m optimizer --strategy-file strategies/exploration.json --assets EURUSD

# Mehrere Assets parallel
python -m optimizer --strategy-file strategies/exploration.json --assets EURUSD,GBPUSD,USDJPY
```

---

## Installation

### Voraussetzungen

- Python 3.10+
- 16GB+ RAM empfohlen
- Multi-Core CPU (Optimizer nutzt alle verfügbaren Kerne)

### Setup

```bash
# Repository klonen
git clone <repo-url>
cd fwbg

# Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Daten

Historische Preisdaten werden im `data/` Verzeichnis erwartet (CSV-Format):

```
data/
├── forexsb/                    # ForexSB Daten
│   ├── EURUSD_HOUR.csv         # Stündliche OHLC-Daten
│   ├── EURUSD_MINUTE_15.csv    # 15-Minuten Daten
│   ├── GBPUSD_HOUR.csv
│   └── ...
├── stooq/                      # Stooq Daten
│   └── ...
└── downloads/                  # Manuell heruntergeladene Daten
```

CSV-Format: `T,O,H,L,C,V` (Timestamp, Open, High, Low, Close, Volume)

---

## Optimizer Ausführen

### Basis-Kommandos

```bash
# Standard-Run (alle Assets in assets.json)
python -m optimizer

# Spezifische Assets
python -m optimizer --assets EURUSD,GBPUSD

# Mit Strategy-Datei
python -m optimizer --strategy-file strategies/exploration.json

# Nur bestimmte Asset-Klassen
python -m optimizer --asset-class FOREX
```

### Wichtige CLI-Optionen

| Option | Beschreibung | Beispiel |
|--------|--------------|----------|
| `--assets` | Komma-separierte Asset-Liste | `--assets EURUSD,GBPUSD` |
| `--strategy-file` | Pfad zur Strategy-JSON | `--strategy-file strategies/exploration.json` |
| `--asset-class` | Nur bestimmte Klasse | `--asset-class FOREX` |
| `--tags` | Assets mit bestimmten Tags | `--tags major,liquid` |

### Logging

Der Optimizer unterstützt verschiedene Log-Level via Umgebungsvariable:

```bash
# Standard (nur wichtige Infos)
python -m optimizer --assets EURUSD

# Debug-Logging
OPTIMIZER_LOG=2 python -m optimizer --assets EURUSD

# Verbose (alle Details)
OPTIMIZER_LOG=3 python -m optimizer --assets EURUSD
```

---

## Strategy-Konfiguration

Strategien werden als JSON-Dateien im `strategies/` Verzeichnis definiert.

### Minimale Konfiguration

```json
{
  "name": "Meine Strategie",
  "description": "Beschreibung"
}
```

### Vollständige Konfiguration

```json
{
  "name": "Exploration",
  "description": "Breite Parameter-Suche mit ATR-basierten Exits",
  "tags": ["exploration", "atr_based"],

  "model": {
    "architecture": "long_short_separate"
  },

  "features": {
    "preferred_groups": ["trend", "momentum", "volatility"],
    "feature_selection": "boruta",
    "max_features": 30
  },

  "exit_strategy": {
    "mode": "atr_based",
    "atr_based": {
      "atr_period": 14,
      "min_tp_pips": 10,
      "min_sl_pips": 15
    }
  },

  "grids": {
    "FOREX": {
      "tp": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
      "sl": [0.5, 1.0, 1.5, 2.0, 2.5],
      "ct": [0.5, 0.52, 0.55, 0.6, 0.65],
      "timeout_bars": [null, 24, 48, 96]
    }
  },

  "filters": {
    "min_rrr": 0,
    "min_trades": 30
  },

  "resources": {
    "max_cpu_percent": 0.8,
    "xgboost_n_jobs": 0
  }
}
```

### Wichtige Parameter

| Parameter | Beschreibung | Default |
|-----------|--------------|---------|
| `model.architecture` | `"unified"` oder `"long_short_separate"` | `"unified"` |
| `features.max_features` | Max Features pro Modell (0 = kein Limit) | `0` |
| `exit_strategy.mode` | `"fixed"` oder `"atr_based"` | `"fixed"` |
| `filters.min_trades` | Minimum Trades für Validität | `50` |

---

## Exit-Strategien

Der Optimizer unterstützt zwei Exit-Strategie-Modi für TP/SL-Berechnung.

### Fixed Exit (Default)

TP und SL werden als **Spread-Multiplikatoren** definiert. Die tatsächlichen Distanzen sind konstant pro Asset.

```json
{
  "exit_strategy": {
    "mode": "fixed"
  },
  "grids": {
    "FOREX": {
      "tp": [20, 30, 40, 50, 60, 80],
      "sl": [20, 30, 40, 50, 60, 80],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

**Grid-Werte:** `tp: 40, sl: 30` bedeutet:
- TP = 40 × Spread (z.B. bei 1.0 Pip Spread: 40 Pips)
- SL = 30 × Spread (30 Pips)

### ATR-Based Exit

TP und SL werden als **ATR-Multiplikatoren** definiert. Die tatsächlichen Distanzen sind **dynamisch** und passen sich der aktuellen Volatilität an.

```json
{
  "exit_strategy": {
    "mode": "atr_based",
    "atr_based": {
      "atr_period": 14,
      "min_tp_pips": 10,
      "min_sl_pips": 15
    }
  },
  "grids": {
    "FOREX": {
      "tp": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
      "sl": [0.5, 1.0, 1.5, 2.0, 2.5],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

**Grid-Werte:** `tp: 1.5, sl: 1.0` bedeutet:
- TP = 1.5 × ATR (dynamisch pro Bar)
- SL = 1.0 × ATR (dynamisch pro Bar)

**Minimum-Werte:** `min_tp_pips` und `min_sl_pips` sind Spread-basierte Mindestdistanzen, um zu kleine Exits bei niedriger Volatilität zu verhindern.

### Vergleich

| Aspekt | Fixed | ATR-Based |
|--------|-------|-----------|
| TP/SL-Berechnung | Konstant (Spread-Multiples) | Dynamisch (ATR-Multiples) |
| Anpassung an Volatilität | Nein | Ja |
| Grid-Werte | Große Zahlen (10-100) | Kleine Zahlen (0.5-5.0) |
| Spread-Schutz | Implizit (Spread-Basis) | Explizit (min_tp/sl_pips) |
| Use Case | Stabile Märkte | Variable Volatilität |

**Wichtig:** Spread und Slippage werden bei beiden Modi berücksichtigt!

---

## Feature-Gruppen

Features sind in Gruppen organisiert. Jede Gruppe wird separat getestet.

### Verfügbare Gruppen

| Gruppe | Prefixes | Beschreibung |
|--------|----------|--------------|
| `trend` | `trend_`, `ichi_` | ADX, EMA, SMA, MACD, Ichimoku |
| `momentum` | `mom_` | RSI, Stochastic, Williams %R, ROC |
| `volatility` | `vol_` | Bollinger, Keltner, ATR |
| `price_action` | `pa_` | Range, Higher Highs, Gaps |
| `time` | `time_`, `season_` | Zeit- und Saisonalitäts-Features |
| `macro` | `macro_` | VIX, Yields, DXY, Indices |
| `dynamics` | `dyn_`, `lag_`, `accel_` | Änderungen und Verzögerungen |
| `mtf` | `mtf_` | Multi-Timeframe (H4, D1) |
| `distribution` | `dist_` | Skewness, Kurtosis |
| `fft` | `fft_` | Fourier-Zykluserkennung |
| `regime` | `regime_` | Hurst-basierte Marktregime |

### Kombinierte Gruppen

| Gruppe | Enthält |
|--------|---------|
| `trend_momentum` | trend + momentum |
| `macro_vol` | macro + volatility |
| `full_technical` | trend + momentum + volatility + price_action |

### Konfiguration

```json
{
  "features": {
    "preferred_groups": ["trend", "momentum", "volatility"]
  }
}
```

Ohne `preferred_groups` werden alle Gruppen getestet.

---

## Feature Selection

Der Optimizer unterstützt verschiedene Methoden zur automatischen Feature-Auswahl.

### Boruta (Default)

Boruta ist ein "All-Relevant" Algorithmus, der alle statistisch relevanten Features findet.

```json
{
  "features": {
    "feature_selection": "boruta",
    "max_features": 30
  }
}
```

**max_features:** Begrenzt die Anzahl auf die Top-N nach Importance. Empfohlen: 20-30 um Overfitting zu vermeiden.

### Boruta + Plateau

Kombiniert Boruta mit Plateau-Validierung für zusätzliche Stabilität.

```json
{
  "features": {
    "feature_selection": "boruta_plateau"
  }
}
```

### Vergleich

| Methode | Feature-Limit | Stabilität | Speed |
|---------|---------------|------------|-------|
| `boruta` | Optional (max_features) | Mittel | Mittel |
| `boruta_plateau` | Optional | Hoch | Langsamer |
| `importance_based` | Fest (Top-5) | Hoch | Schnell |

---

## Ergebnisse

Ergebnisse werden im `test_results/` Verzeichnis gespeichert.

### Verzeichnisstruktur

```
test_results/
└── 20260201_103045_abc123/
    ├── results.json           # Alle Ergebnisse als JSON
    ├── summary.txt            # Menschenlesbare Zusammenfassung
    └── EURUSD/
        ├── config.json        # Verwendete Konfiguration
        └── best_candidate.json # Beste gefundene Parameter
```

### Status-Codes

| Status | Bedeutung |
|--------|-----------|
| `significant` | Strategie hat statistisch signifikanten Edge |
| `not_significant` | Kein signifikanter Edge gefunden (p-value >= 0.05) |
| `no_candidates` | Keine validen Kandidaten (zu wenig Trades, etc.) |
| `error` | Fehler während der Verarbeitung |

### Ergebnis-Interpretation

```json
{
  "symbol": "EURUSD",
  "status": "significant",
  "best_candidate": {
    "feature_group": "trend_momentum",
    "tp": 1.5,
    "sl": 1.0,
    "ct": 0.55,
    "oos_pnl": 12.5,
    "oos_trades": 127,
    "win_rate": 0.58,
    "p_value": 0.023
  }
}
```

---

## Architektur

### Kern-Komponenten

```
optimizer/
├── __main__.py          # CLI Entry Point
├── cli.py               # Argument Parsing
├── process.py           # Haupt-Optimierungslogik
├── nested_cv.py         # Walk-Forward Cross-Validation
├── simulation.py        # Trade-Simulation (Numba)
├── boruta.py            # Boruta Feature Selection
│
├── exit_strategies/     # Exit-Strategie Module
│   ├── base.py          # Abstrakte Basisklasse
│   ├── fixed/           # Spread-basierte Exits
│   └── atr_based/       # ATR-basierte Exits
│
├── strategy_config.py   # Strategy-JSON Parsing
├── simulation_context.py # Parameter-Container
└── asset_config.py      # Asset-Konfiguration
```

### Optimierungs-Pipeline

```
1. Strategy-JSON laden
   ↓
2. Asset-Daten laden (H1 OHLC + Makro)
   ↓
3. Features berechnen (alle Gruppen)
   ↓
4. Für jede Feature-Gruppe:
   │
   ├─ 5. Grid-Search (TP × SL × Timeout)
   │     │
   │     └─ 6. Walk-Forward CV (N Folds)
   │          │
   │          ├─ 7a. Feature Selection (Boruta)
   │          ├─ 7b. Model Training (XGBoost)
   │          └─ 7c. OOS Validation
   │
   └─ 8. Beste Kandidaten sammeln
   ↓
9. Monte-Carlo Significance Test
   ↓
10. Ergebnisse speichern
```

### Performance-Optimierungen

- **Numba JIT:** Trade-Simulation ist kompiliert für maximale Speed
- **Target-Caching:** Targets werden einmal pro TP/SL berechnet und wiederverwendet
- **Early Termination:** Hoffnungslose Kandidaten werden nach wenigen Folds abgebrochen
- **Parallel Processing:** Feature-Gruppen werden parallel verarbeitet

---

## Weitere Dokumentation

- [docs/FEATURES.md](docs/FEATURES.md) - Detaillierte Feature-Dokumentation
- [strategies/README.md](strategies/README.md) - Strategy-Konfigurations-Referenz

---

## Lizenz

Proprietär - Nur für internen Gebrauch.
