# ATR-Based Exit Strategy

Dynamic take-profit and stop-loss exit strategy that adapts to market volatility using the Average True Range (ATR) indicator.

## Concept

Fixed-pip TP/SL levels fail to account for changing market conditions: a 50-pip stop may be appropriate in a calm market but far too tight during high-volatility events. The ATR-based exit strategy solves this by computing TP and SL distances as multiples of the current ATR at each trade's entry bar. This means wider stops and targets in volatile markets (where price swings are larger) and tighter levels in quiet markets (where a smaller move is significant).

The strategy supports two timeout modes. In fixed timeout mode, trades are closed after a configurable number of bars if neither TP nor SL is hit. In adaptive timeout mode, the timeout is dynamically computed per trade based on the ratio of the current ATR to its moving average: high-volatility periods get shorter timeouts (prices move faster, so resolution should come sooner), while low-volatility periods get longer timeouts (prices need more time to reach their targets).

All trade simulation is JIT-compiled via Numba for maximum performance, with separate implementations for basic targets, targets with trade durations, and adaptive timeout variants. The strategy integrates with the grid search system, iterating over TP/SL multiplier combinations and optionally filtering by minimum risk-reward ratio.

## Exit Logic

For each bar in the dataset:

1. Read the ATR value at the signal bar
2. Compute TP distance: `max(ATR * tp_mult, min_tp_distance)`
3. Compute SL distance: `max(ATR * sl_mult, min_sl_distance)`
4. Simulate a long and short trade from the next bar using these levels
5. If adaptive timeout is enabled, compute per-trade timeout: `base_timeout * (atr_ma / atr_current)`, clamped to `[min_timeout, max_timeout]` with the vol ratio clamped to `[0.25, 4.0]`

Minimum TP/SL distances (in spread multiples) prevent unrealistically tight levels in extremely low-volatility environments.

## Configuration

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp_mult` | `float` | `2.0` | ATR multiplier for take-profit distance (min: 0.1, max: 20.0) |
| `sl_mult` | `float` | `1.5` | ATR multiplier for stop-loss distance (min: 0.1, max: 20.0) |
| `atr_period` | `int` | `14` | ATR lookback period in bars (fallback if no precomputed ATR column exists) |
| `min_tp_pips` | `int` | `10` | Minimum TP distance in spread multiples (min: 1, max: 500) |
| `min_sl_pips` | `int` | `15` | Minimum SL distance in spread multiples (min: 1, max: 500) |
| `timeout_bars` | `int` | `None` | Fixed timeout: close after N bars (ignored when adaptive timeout is enabled) |

### Adaptive Timeout Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adaptive_timeout` | `bool` | `false` | Enable per-trade adaptive timeout based on volatility ratio |
| `base_timeout` | `int` | `48` | Base timeout bars at average volatility (scales with vol ratio) |
| `min_timeout` | `int` | `12` | Minimum adaptive timeout bars (floor for high-vol periods) |
| `max_timeout` | `int` | `96` | Maximum adaptive timeout bars (cap for low-vol periods) |
| `atr_ma_period` | `int` | `200` | Moving average period over ATR for computing the vol ratio |

### Grid Search Keys

The grid search accepts TP/SL values under multiple key names:

- `tp_mult` / `sl_mult` (preferred)
- `atr_tp_mult` / `atr_sl_mult` (explicit naming)
- `tp` / `sl` (generic)
- `min_rrr`: Minimum risk-reward ratio filter (skips combinations where `tp_mult / sl_mult < min_rrr`)

When adaptive timeout is enabled, `timeout_bars` grid values are ignored (timeout is computed per trade), which significantly reduces grid size.

## Usage Notes

- The plugin looks for a precomputed ATR column in this order: `_atr`, `vol_atr`. If neither exists, it computes ATR on the fly using the `ta` library.
- NaN ATR values (at the start of the series) are replaced with `0.0`.
- Slippage is fixed at `spread * 0.5`.
- The `max_bars` limit (maximum simulation horizon per trade) is taken from `ctx.max_trade_bars` or defaults to the full DataFrame length.
- The `return_durations` parameter enables returning trade duration arrays alongside targets, which is used by the sample weighting system.
- The `resolve_distances` method provides per-bar TP/SL distance arrays for use in downstream simulation and analysis.
- Cache keys follow the format `atr_tp{tp_mult}_sl{sl_mult}_to{timeout}` for efficient target caching.
