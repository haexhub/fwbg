# Phase 5: Exit Strategies

## Zweck

Exit Strategies definieren, wie Take-Profit (TP) und Stop-Loss (SL) Distanzen berechnet werden. Jede Strategie bestimmt die Interpretation der Grid-Werte und wie Win/Loss-Targets für die Simulation erzeugt werden.

---

## Wichtig: Nicht vom PipelineRunner ausgeführt

Exit Strategies werden **nicht** vom PipelineRunner orchestriert. Stattdessen werden sie **direkt vom Optimization-Code** (Grid Search) aufgerufen:

1. `iterate_grid()` generiert alle Parameter-Kombinationen
2. `compute_targets()` berechnet Win/Loss-Arrays pro Kombination
3. `get_cache_key()` ermöglicht Caching der Ergebnisse

---

## BaseExitStrategy

Basisklasse: `src/fwbg/plugins/exit_strategy.py`

```python
class BaseExitStrategy(BasePlugin, ABC):
    phase = PluginPhase.EXIT_STRATEGIES

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

- Registrierung: `@register_exit_strategy("name")`

---

## Verfügbare Plugins

### fixed (fwbg-core)

Konstante TP/SL-Distanzen als **Spread-Multiplikatoren**:

- Grid-Wert `tp: 40` bedeutet: TP-Distanz = 40 × Spread des Assets
- Grid-Wert `sl: 20` bedeutet: SL-Distanz = 20 × Spread des Assets

```json
"exit_strategy": "fixed",
"exit_params": {},
"grids": {
  "FOREX": {
    "tp": [10, 20, 30, 40],
    "sl": [15, 20, 30, 50]
  }
}
```

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `timeout_bars` | `None` | Maximale Trade-Dauer in Bars (None = unbegrenzt) |

### atr_based (fwbg-premium)

Dynamische TP/SL-Distanzen basierend auf **Average True Range (ATR)** — die Distanzen passen sich der aktuellen Volatilität an:

- Grid-Wert `tp: 1.5` bedeutet: TP-Distanz = 1.5 × ATR
- Grid-Wert `sl: 1.0` bedeutet: SL-Distanz = 1.0 × ATR

```json
"exit_strategy": "atr_based",
"exit_params": {"atr_period": 14, "min_tp_pips": 10, "min_sl_pips": 15},
"grids": {
  "FOREX": {
    "tp": [1.0, 1.5, 2.0, 2.5],
    "sl": [0.5, 1.0, 1.5, 2.0]
  }
}
```

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `atr_period` | `14` | ATR-Berechnungsperiode |
| `min_tp_pips` | `10` | Mindest-TP in Pips (Minimum-Floor) |
| `min_sl_pips` | `15` | Mindest-SL in Pips (Minimum-Floor) |
| `timeout_bars` | `None` | Maximale Trade-Dauer in Bars |

---

## Grid-Search Integration

Der Grid Search iteriert über alle TP/SL/CT-Kombinationen:

```
iterate_grid(grid_config) → {"tp": 1.5, "sl": 1.0, "timeout_bars": None}
                           → {"tp": 1.5, "sl": 1.5, "timeout_bars": None}
                           → {"tp": 2.0, "sl": 1.0, "timeout_bars": None}
                           → ...
```

Für jede Kombination wird `compute_targets()` aufgerufen, was Win/Loss-Arrays zurückgibt:
- `targets_long[i] = 1.0` → Bar `i` wäre ein gewinnender Long-Trade gewesen
- `targets_long[i] = 0.0` → Bar `i` wäre ein verlierender Long-Trade gewesen

---

## Target-Caching

Targets werden pro Exit-Strategy-Parameterkombination gecacht. Der Cache-Key wird über `get_cache_key()` generiert:

```python
# fixed: "fixed_tp30_sl20_tonone"
# atr_based: "atr_tp1.5_sl1.0_atr14_tonone"
```

### Standard-Caching (2-Tuple)
```python
(targets_long, targets_short)
```

### Sample-Weights-Caching (4-Tuple)
Wenn `sample_weights: true` in der Validation-Config:
```python
(targets_long, targets_short, durations_long, durations_short)
```

Die Durations werden für trade-duration-basierte Sample-Weights im CV benötigt.

---

## Strategy-JSON Konfiguration

### Fixed Exit Strategy

```json
{
  "exit_strategy": "fixed",
  "exit_params": {},
  "grids": {
    "FOREX": {
      "tp": [10, 20, 30, 40],
      "sl": [15, 20, 30, 50],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

### ATR-Based Exit Strategy

```json
{
  "exit_strategy": "atr_based",
  "exit_params": {
    "atr_period": 14,
    "min_tp_pips": 10,
    "min_sl_pips": 15
  },
  "grids": {
    "FOREX": {
      "tp": [1.0, 1.5, 2.0, 2.5],
      "sl": [0.5, 1.0, 1.5, 2.0],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

---

## Eigene Exit-Strategy erstellen

Siehe [Plugin Development Guide](../plugin-development.md) für die vollständige Anleitung.
