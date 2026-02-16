# Bitmask Regime Filter Design

## Goal

Replace the boolean `_regime_ok` column with a bitmask-based `_regime` column (0-7) that controls which trade directions are allowed per bar. Regime conditions are configurable per asset class, grid-searchable, and optional.

## Architecture

### Bitmask Encoding

```
Bit 2 (4) = Long allowed
Bit 1 (2) = Short allowed
Bit 0 (1) = Sideways allowed (future use)

7 = all allowed (default, no filter)
6 = Long + Short
4 = Long only
2 = Short only
0 = blocked
```

### Signal Loop Integration

In `_simulate_trades_core()` (targets.py):

```python
REGIME_LONG = 4
REGIME_SHORT = 2

for i in range(len(df) - 1):
    if i < next_allowed_entry:
        continue

    direction = None

    if regime[i] & REGIME_LONG and probs_long is not None and probs_long[i, long_win_idx] >= ct_long:
        direction = 1
    elif regime[i] & REGIME_SHORT and probs_short is not None and probs_short[i, short_win_idx] >= ct_short:
        direction = -1

    if direction:
        trade = simulate_pro_trade(...)
```

### Regime Computation

Each regime condition evaluates a feature column against a threshold and determines which directions to SET or CLEAR in the bitmask. Multiple conditions are combined with AND logic (intersection of allowed directions).

```python
def compute_regime_bitmask(df, regime_config):
    """Compute _regime bitmask from conditions.

    Each condition produces a per-bar bitmask.
    Final result = AND of all condition bitmasks (intersection).
    """
    if not regime_config or not regime_config.conditions:
        return np.full(len(df), 7, dtype=np.int8)  # all allowed

    result = np.full(len(df), 7, dtype=np.int8)  # start: all allowed

    for cond in regime_config.conditions:
        cond_mask = _evaluate_condition(df, cond)  # per-bar bitmask
        result = result & cond_mask  # AND: only keep directions allowed by ALL conditions

    return result
```

### Condition Types

A condition maps a feature column + operator + threshold to a bitmask:

```python
@dataclass
class RegimeCondition:
    column: str           # Feature column name (e.g. "trend_adx_14")
    operator: str         # ">=" | "<=" | ">" | "<"
    value: float          # Threshold
    directions: int       # Bitmask to SET when condition is TRUE (e.g. 6 = Long+Short)
    else_directions: int  # Bitmask to SET when condition is FALSE (default: 7 = all)
```

Example: "ADX >= 20 → allow Long+Short, otherwise block all"
```python
RegimeCondition(column="trend_adx_14", operator=">=", value=20, directions=6, else_directions=0)
```

Example: "D1 EMA diff > 0 → allow Long, otherwise allow Short"
```python
RegimeCondition(column="mtf_d1_ema_50_diff", operator=">", value=0, directions=4, else_directions=2)
```

### Strategy JSON Configuration

```json
{
  "grids": {
    "FOREX": {
      "tp": [10, 20, 30],
      "sl": [20, 30],
      "ct": [0.55, 0.6],
      "regime_filter_grid": {
        "condition_grids": [
          {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [null, 20, 25],
            "directions": 6,
            "else_directions": 0
          },
          {
            "column": "mtf_d1_ema_50_diff",
            "operator": ">",
            "values": [null, 0],
            "directions": 4,
            "else_directions": 2
          }
        ]
      }
    },
    "COMMODITY": {
      "tp": [20, 30, 50],
      "sl": [30, 50],
      "ct": [0.55, 0.6],
      "regime_filter_grid": {
        "condition_grids": [
          {
            "column": "vol_ratio_20_100",
            "operator": ">=",
            "values": [null, 1.3],
            "directions": 6,
            "else_directions": 1
          }
        ]
      }
    }
  }
}
```

- `null` in `values` = condition skipped (all directions allowed)
- `directions` = bitmask when condition TRUE
- `else_directions` = bitmask when condition FALSE (default: 7)
- Multiple conditions: cartesian product for grid search, AND for evaluation

### Grid Search Integration

The existing `regime_filter_grid` already creates a cartesian product of condition combinations. Each combination produces a different `_regime` column per bar. This stays the same — only the internal representation changes from boolean to bitmask.

Grid size example (FOREX above):
- ADX: 3 values (null, 20, 25)
- D1 EMA: 2 values (null, 0)
- Total: 3 x 2 = 6 regime combinations per TP/SL/CT combo

### Default Behavior (No Filter)

No `regime_filter_grid` in strategy JSON → `_regime = 7` for all bars. This is equivalent to the current `_regime_ok = True` behavior. Fully backward compatible.

## Files to Modify

| File | Change |
|------|--------|
| `src/fwbg/core/config.py` | `RegimeCondition` dataclass: add `directions`, `else_directions` |
| `src/fwbg/pipeline/features.py` | `compute_regime_filter` → `compute_regime_bitmask`, return int8 array |
| `src/fwbg/optimization/targets.py` | `_simulate_trades_core`: `regime[i]` boolean → `regime[i] & REGIME_LONG/SHORT` |
| `src/fwbg/optimization/process_fold.py` | `_regime_ok` → `_regime`, use new compute function |
| `src/fwbg/optimization/grid_search.py` | Parse new `directions`/`else_directions` fields from grid config |

## Backward Compatibility

- Strategies without `regime_filter_grid` → `_regime = 7` (all allowed), same as before
- Existing `regime_filter_grid` with only `column`/`operator`/`values` → default `directions=7, else_directions=0` (matches current boolean True/False behavior)
- No breaking changes to strategy JSON format

## Regime Filter Ideas (per Asset Class)

### FOREX
- Volatility: `vol_ratio_20_100 >= 1.3` → expansion, allow Long+Short
- Trend Strength: `trend_adx_14 >= 20` → trending, allow Long+Short
- Macro: `macro_us_de_yield_spread_chg > 0` → USD strong, filter direction
- Risk: `macro_vix <= 25` → risk-on, allow all

### INDEX
- Risk: `macro_vix <= 30 AND macro_spx_above_200dma == 1` → risk-on
- Trend: `trend_adx_14 >= 20` → trending

### COMMODITY
- Volatility: `vol_ratio_20_100 >= 1.3` → breakout regime
- Trend: `trend_adx_14 >= 25` → strong trend

### CRYPTO
- Volatility: `vol_ratio_20_100` → high vol = normal, low vol = compression
- Risk: correlated with risk-on/off sentiment
