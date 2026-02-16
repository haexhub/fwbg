# Phase 5: Exit Strategies

## Purpose

Exit strategies define how take-profit (TP) and stop-loss (SL) distances are calculated. Each strategy determines the interpretation of grid values and how win/loss targets are generated for the simulation.

---

## Important: Not Executed by PipelineRunner

Exit strategies are **not** orchestrated by the PipelineRunner. Instead, they are called **directly by the optimization code** (grid search):

1. `iterate_grid()` generates all parameter combinations
2. `compute_targets()` computes win/loss arrays per combination
3. `get_cache_key()` enables caching of results

---

## BaseExitStrategy

Base class: `src/fwbg/plugins/exit_strategy.py`

```python
class BaseExitStrategy(BasePlugin, ABC):
    phase = PluginPhase.EXIT_STRATEGIES

    @abstractmethod
    def compute_targets(self, df: pd.DataFrame, ctx: SimulationContext,
                       **params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes win/loss targets for long and short.

        Returns:
            (targets_long, targets_short) — Arrays with 1.0=Win, 0.0=Loss
        """

    @abstractmethod
    def iterate_grid(self, grid_config: dict, ctx: SimulationContext) -> Iterator[dict]:
        """Iterates over all parameter combinations from the grid config."""

    @abstractmethod
    def get_cache_key(self, params: dict) -> str:
        """Unique cache key for target caching."""
```

- Registration: `@register_exit_strategy("name")`

---

## Available Plugins

### fixed (fwbg-core)

Constant TP/SL distances as **spread multipliers**:

- Grid value `tp: 40` means: TP distance = 40 × asset spread
- Grid value `sl: 20` means: SL distance = 20 × asset spread

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout_bars` | `None` | Maximum trade duration in bars (None = unlimited) |

### atr_based (fwbg-premium)

Dynamic TP/SL distances based on **Average True Range (ATR)** — distances adapt to the current volatility:

- Grid value `tp: 1.5` means: TP distance = 1.5 × ATR
- Grid value `sl: 1.0` means: SL distance = 1.0 × ATR

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | `14` | ATR calculation period |
| `min_tp_pips` | `10` | Minimum TP in pips (floor) |
| `min_sl_pips` | `15` | Minimum SL in pips (floor) |
| `timeout_bars` | `None` | Maximum trade duration in bars |

---

## Grid Search Integration

The grid search iterates over all TP/SL/CT combinations:

```
iterate_grid(grid_config) → {"tp": 1.5, "sl": 1.0, "timeout_bars": None}
                           → {"tp": 1.5, "sl": 1.5, "timeout_bars": None}
                           → {"tp": 2.0, "sl": 1.0, "timeout_bars": None}
                           → ...
```

For each combination, `compute_targets()` is called, returning win/loss arrays:
- `targets_long[i] = 1.0` → Bar `i` would have been a winning long trade
- `targets_long[i] = 0.0` → Bar `i` would have been a losing long trade

---

## Target Caching

Targets are cached per exit strategy parameter combination. The cache key is generated via `get_cache_key()`:

```python
# fixed: "fixed_tp30_sl20_tonone"
# atr_based: "atr_tp1.5_sl1.0_atr14_tonone"
```

### Standard Caching (2-Tuple)
```python
(targets_long, targets_short)
```

### Sample Weights Caching (4-Tuple)
When `sample_weights: true` in the validation config:
```python
(targets_long, targets_short, durations_long, durations_short)
```

The durations are needed for trade-duration-based sample weights in the CV.

---

## Strategy JSON Configuration

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

## Creating a Custom Exit Strategy

See [Plugin Development Guide](../plugin-development.md) for the complete guide.
