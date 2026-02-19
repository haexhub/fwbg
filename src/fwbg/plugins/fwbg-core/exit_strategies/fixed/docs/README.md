# Fixed Exit Strategy

Fixed pip-based stop-loss and take-profit exit strategy using spread multipliers.

## Concept

The Fixed Exit Strategy applies constant take-profit (TP) and stop-loss (SL) distances to every trade, regardless of market conditions. Distances are expressed as multiples of the instrument's spread. For example, with a spread of 0.0001 (1 pip), a TP value of 30 means the take-profit is placed 30 pips (0.003) away from entry. This makes parameter tuning intuitive and instrument-agnostic -- the same multiplier values produce proportionally scaled distances across different instruments.

Under the hood, the strategy delegates to Numba-accelerated functions (`compute_targets_numba` / `compute_targets_with_durations_numba`) that walk through OHLC bar data to determine, for each bar, whether a long or short trade would have hit its TP, SL, or timed out. Slippage is modeled as half the spread and is applied to every simulated entry.

Within the walk-forward pipeline, the strategy's `iterate_grid` method produces all TP x SL x timeout combinations for grid search. A `min_rrr` filter can be applied to skip combinations whose reward-to-risk ratio falls below a given threshold, reducing the search space. Results are cached per parameter set using a deterministic cache key of the form `fixed_tp{tp}_sl{sl}_to{timeout}`.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp` | int | `30` | Take-profit distance as spread multiplier (e.g. 30 = 30 pips at spread 0.0001) |
| `sl` | int | `20` | Stop-loss distance as spread multiplier (e.g. 20 = 20 pips at spread 0.0001) |
| `timeout_bars` | int | `None` | Close trade after N bars if neither TP nor SL is hit (`None` = no timeout) |

### Grid Search Parameters

When used in grid search via `iterate_grid`, the grid config supports these additional keys:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp` | list[int] | `[15, 20, 25, 30, 40, 50]` | List of TP multiplier values to search |
| `sl` | list[int] | `[15, 20, 25, 30, 40, 50]` | List of SL multiplier values to search |
| `timeout_bars` | list[int\|None] | `[None]` | List of timeout values to search |
| `min_rrr` | float | `0` | Minimum reward-to-risk ratio; combinations below this are skipped |

## Usage Notes

- TP and SL are **not** in absolute price units. They are multiplied by `ctx.spread` at runtime to produce actual price distances. This keeps configurations portable across instruments.
- Slippage is automatically modeled as `spread * 0.5` and factored into target computation.
- If `timeout_bars` is `None` (the default), trades remain open until TP or SL is hit, up to the global `max_trade_bars` limit from the SimulationContext.
- The `return_durations=True` flag causes the strategy to return four arrays instead of two, adding trade duration arrays for both long and short sides. This is used by the sample-weight pipeline.
- Setting `min_rrr` in the grid config is an effective way to prune unprofitable parameter combinations early. For example, `min_rrr: 1.0` ensures TP >= SL for every tested combination.
