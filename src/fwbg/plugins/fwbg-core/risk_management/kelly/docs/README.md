# Kelly Position Sizing

Kelly Criterion-based position sizing with drawdown adjustment and automatic circuit breaker optimization.

## Concept

The Kelly Criterion is a formula from information theory that determines the optimal fraction of capital to wager on a bet with a positive expected value. In its full form, the Kelly fraction is computed as `(win_rate * rrr - (1 - win_rate)) / rrr`, where `rrr` is the reward-to-risk ratio. Betting the full Kelly fraction maximizes the long-term geometric growth rate of capital, but produces substantial drawdowns in practice.

This plugin defaults to **quarter-Kelly** (`kelly_fraction = 0.25`), which significantly reduces variance and drawdown while sacrificing only a modest amount of long-term growth. The computed fraction is further capped by `max_risk` to provide a hard upper bound on per-trade risk regardless of how favorable the Kelly formula output is.

Beyond basic Kelly sizing, the plugin applies two layers of protection. First, **drawdown adjustment**: the trade sequence is simulated with the computed risk, and if the resulting maximum drawdown exceeds `target_max_dd`, the risk is scaled down proportionally. Second, **circuit breaker optimization**: the plugin searches a configurable range of consecutive-loss thresholds and pause durations to find the combination that best improves risk-adjusted performance. When a losing streak hits the trigger threshold, trading pauses for the optimal number of bars before resuming.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kelly_fraction` | float | `0.25` | Fraction of full Kelly to use (0.25 = quarter-Kelly for conservative sizing) |
| `max_risk` | float | `0.05` | Maximum risk per trade as fraction of account (hard cap regardless of Kelly) |
| `target_max_dd` | float | `0.30` | Target maximum drawdown; risk is scaled down if simulated DD exceeds this |
| `circuit_breaker_loss_range` | list[int] | `[3, 8]` | Range [min, max] of consecutive losses to search for optimal circuit breaker trigger |
| `circuit_breaker_pause_range` | list[int] | `[5, 30]` | Range [min, max] of pause bars to search for optimal circuit breaker pause duration |

## Output

The `compute_risk_params` method returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `risk_per_trade` | float | Final risk fraction per trade after all adjustments |
| `is_profitable` | bool | Whether the strategy has positive Kelly expectancy |
| `full_kelly` | float | Raw full-Kelly fraction before scaling |
| `trade_returns` | list[float] | Simulated per-trade returns using the computed risk |
| `circuit_breaker.pause_after_losses` | int | Optimal number of consecutive losses before pausing |
| `circuit_breaker.pause_bars` | int | Optimal number of bars to pause after trigger |
| `circuit_breaker.enabled` | bool | Whether the circuit breaker is active |
| `risk_adjustment.original_risk` | float | Risk before drawdown adjustment |
| `risk_adjustment.scale_factor` | float | Multiplier applied to reduce risk (1.0 = no reduction) |
| `risk_adjustment.target_dd` | float | The target max drawdown used for adjustment |

## Usage Notes

- If the Kelly formula produces a zero or negative value (i.e., the strategy has no edge), `is_profitable` is set to `False` and `risk_per_trade` is 0. No trades will be sized.
- The drawdown adjustment is applied **before** the circuit breaker search, so the circuit breaker optimizes over the already-adjusted risk level.
- `circuit_breaker_loss_range` and `circuit_breaker_pause_range` define search bounds, not fixed values. The plugin tests all combinations within these ranges and selects the one that best protects the equity curve.
- Trade returns in the output assume a flat per-trade risk: winners gain `risk * rrr`, losers lose `risk`. This simplified model is used for circuit breaker and drawdown simulation.
