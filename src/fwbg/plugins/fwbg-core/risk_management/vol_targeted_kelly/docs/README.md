# Volatility-Targeted Kelly

Kelly Criterion position sizing with per-trade volatility targeting that scales position size inversely to realized volatility.

## Concept

This plugin extends the standard Kelly position sizing with **volatility targeting**, a technique widely used in quantitative finance to normalize risk exposure across varying market regimes. The core idea is simple: when realized volatility is low, positions are scaled up; when it is high, positions are scaled down. The scaling factor for each trade is `target_vol / realized_vol`, clamped between configurable bounds.

The base position size is determined identically to the standard Kelly plugin: the full Kelly fraction `(win_rate * rrr - (1 - win_rate)) / rrr` is computed, scaled by `kelly_fraction` (default quarter-Kelly), and capped at `max_risk`. Drawdown adjustment and circuit breaker optimization are applied in the same way. The volatility targeting layer then modulates this base risk on a per-trade basis.

Volatility targeting smooths the equity curve by reducing exposure during turbulent periods and increasing it during calm ones. This typically improves risk-adjusted metrics such as the Sharpe ratio without materially affecting total return. The approach is particularly effective for strategies that maintain a consistent edge across volatility regimes but suffer larger drawdowns when markets become erratic.

## Configuration

### Kelly Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kelly_fraction` | float | `0.25` | Fraction of full Kelly to use (0.25 = quarter-Kelly for conservative sizing) |
| `max_risk` | float | `0.05` | Maximum risk per trade as fraction of account (hard cap regardless of Kelly) |
| `target_max_dd` | float | `0.30` | Target maximum drawdown; risk is scaled down if simulated DD exceeds this |
| `circuit_breaker_loss_range` | list[int] | `[3, 8]` | Range [min, max] of consecutive losses to search for optimal circuit breaker trigger |
| `circuit_breaker_pause_range` | list[int] | `[5, 30]` | Range [min, max] of pause bars to search for optimal circuit breaker pause duration |

### Volatility Targeting Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_vol` | float | `15.0` | Target annualized volatility (%) for position scaling; positions scale as `target_vol / realized_vol` |
| `min_scale` | float | `0.25` | Minimum position scale factor (floor for vol-targeting adjustment) |
| `max_scale` | float | `2.0` | Maximum position scale factor (cap for vol-targeting adjustment) |

### Runtime Parameter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rv_values` | list[float] | `None` | Per-trade realized volatility values; must be the same length as `trades`. When `None`, the plugin falls back to flat Kelly sizing with no volatility adjustment. |

## Output

The `compute_risk_params` method returns the same structure as the standard Kelly plugin, with one additional key when volatility targeting is active:

| Key | Type | Description |
|-----|------|-------------|
| `risk_per_trade` | float | Base risk fraction per trade (before vol scaling) |
| `is_profitable` | bool | Whether the strategy has positive Kelly expectancy |
| `full_kelly` | float | Raw full-Kelly fraction before scaling |
| `trade_returns` | list[float] | Per-trade returns with vol-adjusted position sizes applied |
| `circuit_breaker` | dict | Circuit breaker settings (same as Kelly plugin) |
| `risk_adjustment` | dict | Drawdown adjustment details (same as Kelly plugin) |
| `vol_targeting.target_vol` | float | The target volatility used |
| `vol_targeting.mean_scale` | float | Average scale factor across all trades |
| `vol_targeting.min_scale_used` | float | Smallest scale factor observed |
| `vol_targeting.max_scale_used` | float | Largest scale factor observed |
| `vol_targeting.mean_fk_adjusted` | float | Mean effective risk per trade after vol scaling |

## Usage Notes

- When `rv_values` is `None` or its length does not match the number of trades, the plugin silently falls back to standard flat Kelly sizing. No `vol_targeting` key will appear in the output in this case.
- The `target_vol` parameter is in **annualized percentage** terms (e.g., 15.0 means 15% annualized volatility). The `rv_values` provided at runtime must use the same scale for the ratio to be meaningful.
- Scale factors are clamped to `[min_scale, max_scale]` to prevent extreme position sizes. With defaults of 0.25 and 2.0, positions can be reduced to a quarter or doubled relative to the base Kelly size.
- Realized volatility values below `1e-6` are clipped to avoid division-by-zero, resulting in the scale factor being capped at `max_scale`.
- Trade returns in the output reflect the vol-adjusted sizes: winners gain `risk * scale * rrr`, losers lose `risk * scale`. This gives an accurate picture of the strategy's P&L under volatility targeting.
