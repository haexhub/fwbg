# Plugin Spec — vol_targeted_kelly

**Kind**: risk_management  •  **Version**: 0.1.0

## Capability

Sizes each trade using quarter-Kelly risk with drawdown-targeted shrinkage, optimal circuit-breaker search, and per-trade volatility targeting from realized-vol inputs.

## Summary

Quarter-Kelly risk manager that (1) computes fractional-Kelly risk-per-trade capped by max_risk, (2) shrinks it toward target_max_dd via adjust_risk_for_target_dd, (3) searches for an optimal loss-streak circuit breaker, and (4) when per-trade realized volatility is provided, applies a clipped target_vol/realized_vol scale to each trade's return to produce a volatility-targeted trade-return series.

## Inputs

- trades: List[float] — historical per-trade returns (or R-multiples) used for DD adjustment and circuit-breaker search.
- win_rate: float — historical win probability for the Kelly calculation.
- rrr: float — reward-to-risk ratio for the Kelly calculation and win-side trade_returns scaling.
- kelly_fraction, max_risk, target_max_dd, circuit_breaker_loss_range, circuit_breaker_pause_range, target_vol, min_scale, max_scale: keyword-only tunables.
- rv_values: Optional[List[float]] — per-trade realized volatility aligned 1:1 with trades; enables vol targeting when present and length-matched.

## Parameters

- `kelly_fraction` (float, default=0.25): Fraction of full Kelly to use (0.25 = quarter-Kelly for conservative sizing).
- `max_risk` (float, default=0.05): Hard cap on fractional risk per trade regardless of Kelly output.
- `target_max_dd` (float, default=0.3): Target maximum drawdown; risk is scaled down when the simulated DD exceeds this.
- `circuit_breaker_loss_range` (list[int], default=[3, 8]): Inclusive [min, max] range of consecutive losses searched for the optimal circuit-breaker trigger.
- `circuit_breaker_pause_range` (list[int], default=[5, 30]): Inclusive [min, max] range of pause bars searched for the optimal circuit-breaker pause length.
- `target_vol` (float, default=15): Target annualized volatility (%) used to compute per-trade scale = target_vol / realized_vol.
- `min_scale` (float, default=0.25): Lower clamp on the per-trade volatility-targeting scale factor.
- `max_scale` (float, default=2): Upper clamp on the per-trade volatility-targeting scale factor.

## Outputs

- risk_per_trade: float — final fractional risk per trade (fk) after Kelly, cap, and DD adjustment.
- is_profitable: bool — False when fk <= 0, True otherwise.
- full_kelly: float — raw full-Kelly value before fraction/cap/DD adjustments.
- circuit_breaker: dict — {pause_after_losses, pause_bars, enabled} from find_optimal_circuit_breaker.
- risk_adjustment: dict — {original_risk, scale_factor, target_dd} describing the DD-driven shrink.
- trade_returns: List[float] — per-trade signed returns using fk (and per-trade scale when vol targeting is active).
- vol_targeting: dict — emitted only when rv_values is supplied and length-matched; contains target_vol, mean_scale, min_scale_used, max_scale_used, mean_fk_adjusted.

## Acceptance Criteria

- AC-001: Computes full Kelly as (win_rate * rrr - (1 - win_rate)) / rrr, then applies kelly_fraction and caps at max_risk to derive base risk_per_trade (fk); returns full_kelly in the result.
- AC-002: When rrr <= 0 or the fractional Kelly is non-positive, returns risk_per_trade=0, is_profitable=False, a disabled circuit_breaker block, and a risk_adjustment block with scale_factor=1.0 — without invoking DD adjustment or the circuit-breaker search.
- AC-003: When fk > 0, invokes adjust_risk_for_target_dd(trades, fk, rrr, target_max_dd=target_max_dd) and replaces fk with the adjusted risk whenever the returned scale_factor is < 1.0.
- AC-004: Calls find_optimal_circuit_breaker(trades, fk, rrr, loss_range=circuit_breaker_loss_range, pause_range=circuit_breaker_pause_range) and populates circuit_breaker with pause_after_losses, pause_bars, and enabled=(pause_after_losses > 0).
- AC-005: When rv_values is provided and len(rv_values) == len(trades), computes per-trade scales = clip(target_vol / clip(rv, 1e-6, None), min_scale, max_scale), sets trade_returns to [fk*s*rrr for wins, -fk*s for losses], and emits a vol_targeting block containing target_vol, mean_scale, min_scale_used, max_scale_used, and mean_fk_adjusted.
- AC-006: When rv_values is missing or its length does not match trades, falls back to fixed Kelly sizing: trade_returns = [fk*rrr if t > 0 else -fk for t in trades] and no vol_targeting block is emitted.
- AC-007: risk_adjustment.original_risk is reconstructed as adjusted_risk / scale_factor when scale_factor > 0, otherwise defaults to fk; target_dd echoes the target_max_dd parameter.

## Edge Cases

- rrr <= 0 forces full_kelly = 0, which triggers the unprofitable early-return branch (risk_per_trade=0, is_profitable=False).
- kelly_fraction * full_kelly negative or zero: max(0, ...) floors fk at 0, entering the unprofitable branch.
- kelly_fraction * full_kelly exceeds max_risk: fk is clamped to max_risk via min(max_risk, ...).
- adjust_risk_for_target_dd returns scale_factor == 0: original_risk falls back to fk instead of dividing by zero.
- adjust_risk_for_target_dd returns scale_factor >= 1.0: fk is left unchanged (only shrunk when scale_factor < 1.0).
- rv_values is None: falls back to fixed sizing with no vol_targeting block.
- len(rv_values) != len(trades): falls back to fixed sizing (no partial matching); no error raised.
- Realized-vol entries at or below 0: np.clip(rv, 1e-6, None) prevents divide-by-zero and produces a scale capped at max_scale.
- Extreme rv (very high or very low): resulting scale is clipped to [min_scale, max_scale] before being applied.
- target_vol == 0: all scales collapse to min_scale via the clip floor.

## Assumptions

- _none_
