# FWBG - ML Trading Strategy Optimizer & Bot

Ein Machine-Learning-basiertes Trading-System mit:
- **Optimizer**: Findet optimale Parameter (TP, SL, Confidence) via Walk-Forward Cross-Validation
- **Bot**: Live-Trading über IG Markets mit Streaming-Support

## Quick Start

### Installation

```bash
# Repository klonen
git clone https://github.com/haexhub/fwbg.git
cd fwbg

# Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -e ".[dev,ig]"
```

### Optimizer

```bash
# Standard-Run
python -m fwbg.optimizer --assets EURUSD

# Mit Strategy-Datei
python -m fwbg.optimizer --strategy-file strategies/exploration.json --assets EURUSD

# Mehrere Assets
python -m fwbg.optimizer --assets EURUSD,GBPUSD,USDJPY

# Debug-Logging
OPTIMIZER_LOG=2 python -m fwbg.optimizer --assets EURUSD
```

### Trading Bot (IG Markets)

```bash
# Bot im Streaming-Modus starten
python -m bots.ig

# Bot im Polling-Modus (falls Streaming nicht verfügbar)
python -m bots.ig --no-streaming
```

**Account-Setup:**

Erstelle `accounts/<name>/` mit:
- `account_info.json` - IG Credentials
- `assets.json` - Assets und deren Konfiguration

```json
// account_info.json
{
  "credentials": {
    "api_key": "...",
    "username": "...",
    "password": "...",
    "env": "DEMO"  // oder "LIVE"
  },
  "money_management": {
    "max_margin_usage": 0.5,
    "min_lot_size": 0.1
  },
  "metadata": {
    "currency": "EUR"
  }
}
```

---

## Dokumentation

- [docs/FEATURES.md](docs/FEATURES.md) - Alle verfügbaren Indikatoren & Features (~220-300+)
- [docs/ADAPTERS.md](docs/ADAPTERS.md) - Adapter-System für Datenquellen & Broker

---

## Projektstruktur

```
fwbg/
├── src/fwbg/              # Core Library
│   ├── builtins/          # Built-in Plugins
│   │   ├── indicators/    # 15 Indicator Plugins
│   │   ├── exit_strategies/
│   │   └── feature_selection/
│   ├── optimizer/         # Optimization Engine
│   └── core/              # Plugin Registry & Config
│
├── bots/                  # Trading Bots
│   └── ig/                # IG Markets Bot
│       ├── bot.py         # EliteBot (ML-based Trading)
│       └── streaming.py   # Lightstreamer Integration
│
├── strategies/            # Strategy Configurations
├── data/                  # Historical Data (CSV)
└── test_results/          # Optimization Results
```

---

## CLI-Optionen

### Optimizer

| Option | Beschreibung | Beispiel |
|--------|--------------|----------|
| `--assets` | Komma-separierte Asset-Liste | `--assets EURUSD,GBPUSD` |
| `--strategy-file` | Pfad zur Strategy-JSON | `--strategy-file strategies/exploration.json` |
| `--asset-class` | Nur bestimmte Klasse | `--asset-class FOREX` |
| `--tags` | Assets mit bestimmten Tags | `--tags major,liquid` |

### Bot

| Option | Beschreibung |
|--------|--------------|
| `--no-streaming` | Polling-Modus statt Streaming |

---

## Exit-Strategien

### ATR-Based (Default)

TP und SL als ATR-Multiplikatoren (dynamisch, passt sich Volatilität an):

```json
{
  "exit_strategy": "atr_based",
  "grids": {
    "FOREX": {
      "tp": [0.5, 1.0, 1.5, 2.0],
      "sl": [0.5, 1.0, 1.5],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

### Fixed

TP und SL als Spread-Multiplikatoren (konstant):

```json
{
  "exit_strategy": "fixed",
  "grids": {
    "FOREX": {
      "tp": [20, 40, 60],
      "sl": [20, 30, 40],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

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

## Anforderungen

- Python 3.10+
- 16GB+ RAM (für Optimizer)
- IG Markets Account (für Bot)

---

## Lizenz

Proprietär - Nur für internen Gebrauch.
