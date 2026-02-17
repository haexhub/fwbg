# FWBG — ML Trading Strategy Optimizer

FWBG is a machine-learning-based framework for walk-forward optimization of trading strategies. It finds optimal parameters (take-profit, stop-loss, confidence threshold) via nested cross-validation and verifies the statistical robustness of results through multiple overfitting tests.

The system is built on a **modular plugin architecture**: every pipeline phase — from indicators to exit strategies to risk management — is implemented as a swappable plugin. Custom plugins can be added without modifying the framework code.

---

## Why FWBG?

- **Plugin Architecture** — Indicators, preprocessors, feature selectors, exit strategies, and risk managers are all plugins. Every phase can be extended or completely replaced with custom implementations.

- **Walk-Forward Validation** — Nested cross-validation with expanding windows, time-series purging (embargo), and sample weights. No lookahead bias by construction.

- **Overfitting Protection** — Three statistical tests verify every discovered strategy: Deflated Sharpe Ratio (multiple-testing correction), Probability of Backtest Overfitting (CSCV), Monte Carlo permutation tests.

- **Numba-Accelerated Simulation** — JIT-compiled trade simulation with parallel processing for fast grid-search runs.

- **Core + Premium Packages** — Open-source core indicators (trend, momentum, volatility). Premium package with regime detection, macro data, COT positioning, ATR-based exits, feature selection.

- **Live-Trading Ready** — Broker adapter system for live execution (IG Markets etc.).

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
python -m bots.ig                  # Streaming mode
python -m bots.ig --no-streaming   # Polling mode
```

---

## Architecture

FWBG processes data through a plugin pipeline with defined phases:

```
DATA_LOADING → PREPROCESSING → INDICATORS → FEATURE_SELECTION
    → EXIT_STRATEGIES → RISK_MANAGEMENT → MODEL → VALIDATION
```

Each phase can contain any number of plugins. The `PipelineRunner` orchestrates execution in the correct order, resolves dependencies, and merges parameters.

**Detailed architecture documentation:** [docs/architecture.md](docs/architecture.md)

---

## Pipeline Phases

| # | Phase | Purpose | Documentation |
|---|-------|---------|---------------|
| 1 | Data Loading | Load external data (macro, COT) | [docs/phases/1-data-loading.md](docs/phases/1-data-loading.md) |
| 2 | Preprocessing | Stationarity transformations | [docs/phases/2-preprocessing.md](docs/phases/2-preprocessing.md) |
| 3 | Indicators | Compute technical features | [docs/phases/3-indicators.md](docs/phases/3-indicators.md) |
| 4 | Feature Selection | Select relevant features | [docs/phases/4-feature-selection.md](docs/phases/4-feature-selection.md) |
| 5 | Exit Strategies | TP/SL computation (fixed, ATR) | [docs/phases/5-exit-strategies.md](docs/phases/5-exit-strategies.md) |
| 6 | Risk Management | Position sizing (Kelly, vol-targeted) | [docs/phases/6-risk-management.md](docs/phases/6-risk-management.md) |
| 7 | Validation | Walk-forward CV, overfitting tests | [docs/phases/7-validation.md](docs/phases/7-validation.md) |

---

## CLI Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--assets` | Comma-separated asset list | `--assets EURUSD,GBPUSD` |
| `--strategy-file` | Path to strategy JSON | `--strategy-file strategies/exploration.json` |
| `--asset-classes` | Filter by asset class | `--asset-classes FOREX` |
| `--timeframe` | Override timeframe | `--timeframe H4` |
| `--tags` | Filter runs by tags | `--tags baseline` |
| `--list` | Show all existing runs | `--list` |
| `--compare` | Compare runs | `--compare RUN1 RUN2` |
| `--load` | Show run details | `--load RUN_ID` |
| `--reverse-worst` | Reverse worst strategies | `--reverse-worst RUN_ID` |
| `--no-save` | Don't save results | `--no-save` |
| `--cpu` | Max CPU utilization (0.0-1.0) | `--cpu 0.8` |
| `--ram-reserve` | Min free RAM fraction | `--ram-reserve 0.25` |
| `--ram-per-worker` | RAM per worker in GB | `--ram-per-worker 4.0` |

---

## Strategy Configuration

Strategies are configured in JSON files under `strategies/`:

```json
{
  "name": "My Strategy",
  "pipeline": {
    "indicators": [
      {"name": "trend", "params": {"adx_periods": [7, 14, 21]}},
      {"name": "momentum", "params": {}}
    ],
    "feature_selection": [
      {"name": "stability", "params": {"n_bootstrap": 7, "threshold": 0.6}},
      {"name": "correlation_filter", "params": {"max_correlation": 0.7, "max_features": 20}}
    ]
  },
  "exit_strategy": "fixed",
  "grids": {
    "FOREX": {"tp": [10, 20, 30], "sl": [20, 30], "ct": [0.5, 0.55, 0.6]}
  },
  "validation": {"folds": 8, "n_inner_folds": 3, "embargo_bars": 100}
}
```

**Full parameter reference:** [strategies/README.md](strategies/README.md)

---

## Results

Optimization results are stored in `test_results/<timestamp>/`:

```
test_results/20260201_103045_abc123/
├── results.json           # All results
├── summary.txt            # Summary
└── EURUSD/
    └── best_candidate.json
```

| Status | Meaning |
|--------|---------|
| `significant` | Statistically significant edge found |
| `not_significant` | No edge (p-value >= 0.05) |
| `no_candidates` | No valid candidates |

Details on statistical tests: [docs/phases/7-validation.md](docs/phases/7-validation.md)

---

## Project Structure

```
fwbg/
├── src/fwbg/
│   ├── plugins/                  # Plugin system
│   │   ├── fwbg-core/            # Core plugins (free)
│   │   │   ├── indicators/       # trend, momentum, volatility, price_action, time_season, fair_value_gap, cusum_events
│   │   │   ├── exit_strategies/  # fixed
│   │   │   └── risk_management/  # kelly, vol_targeted_kelly
│   │   └── *.py                  # Plugin base classes
│   ├── core/                     # Config, registry, context, data sources
│   ├── pipeline/                 # Plugin runner & pipeline system
│   ├── optimization/             # Walk-forward CV, grid search, targets
│   ├── simulation/               # Numba-based trade simulation
│   ├── data/                     # Data sources, loader, asset definitions
│   ├── results/                  # Result storage & plotting
│   ├── cli/                      # Command-line interface
│   └── adapters/                 # Broker & data source adapters
│
├── packages/
│   └── fwbg-premium/             # Premium plugins (separate pip package)
│       ├── indicators/           # regime, structure, risk, distribution, dynamics, support_resistance, ...
│       ├── preprocessing/        # fractional_diff
│       ├── feature_selection/    # boruta, plateau, stability, correlation_filter
│       ├── exit_strategies/      # atr_based
│       └── data_loading/         # macro_data, cot_positioning
│
├── strategies/                   # Strategy configurations (JSON)
├── data/                         # Historical data (CSV)
└── test_results/                 # Optimization results
```

---

## Documentation

### Architecture & Plugin System
- [Architecture & Plugin System](docs/architecture.md) — Plugin lifecycle, discovery, naming, PipelineRunner
- [Plugin Development Guide](docs/plugin-development.md) — Creating custom plugins

### Pipeline Phases
- [Phase 1: Data Loading](docs/phases/1-data-loading.md) — External data, data sources
- [Phase 2: Preprocessing](docs/phases/2-preprocessing.md) — Stationarity transformations
- [Phase 3: Indicators](docs/phases/3-indicators.md) — Technical features, shift_features, safe_divide
- [Phase 4: Feature Selection](docs/phases/4-feature-selection.md) — Boruta, stability selection, correlation filter
- [Phase 5: Exit Strategies](docs/phases/5-exit-strategies.md) — Fixed, ATR-based
- [Phase 6: Risk Management](docs/phases/6-risk-management.md) — Kelly, vol-targeted Kelly
- [Phase 7: Validation](docs/phases/7-validation.md) — Walk-forward CV, DSR, PBO, Monte Carlo

### References
- [Strategy Configuration](strategies/README.md) — Full JSON reference
- [Feature Catalog](docs/FEATURES.md) — All available indicators & features
- [Adapter System](docs/ADAPTERS.md) — Broker & data source adapters
- [Robust Validation Guide](docs/ROBUST_VALIDATION_GUIDE.md) — Sample bias detection
- [Live Bias Detection](docs/LIVE_BIAS_DETECTION.md) — Real-time bias checks

---

## Requirements

- Python 3.10+
- 16GB+ RAM (for optimizer)

---

## License

Proprietary - Internal use only.
