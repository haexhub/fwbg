# Plugin Spec — kelly

**Kind**: risk_management  •  **Version**: 0.1.0

## Capability

Computes per-trade risk using fractional-Kelly sizing capped by a hard maximum, scaled down to a target maximum drawdown, and paired with an optimized consecutive-loss circuit breaker.

## Summary

Quarter-Kelly (configurable fraction) risk manager that derives risk_per_trade from win_rate and reward-to-risk ratio, hard-caps it at max_risk, further scales it down when the simulated drawdown exceeds target_max_dd via adjust_risk_for_target_dd, and searches for an optimal (pause_after_losses, pause_bars) circuit breaker via find_optimal_circuit_breaker. Returns a rich dict with risk_per_trade, is_profitable, full_kelly, per-trade returns, circuit_breaker settings, and risk_adjustment metadata.

## Inputs

- trades: List[float] — sequence of historical trade outcomes (sign indicates win/loss)
- win_rate: float — empirical win probability in [0,1]
- rrr: float — reward-to-risk ratio (payoff on win divided by loss on loss)

## Parameters

- `kelly_fraction` (float, default=0.25): Fraction of full Kelly to use (0.25 = quarter-Kelly for conservative sizing)
- `max_risk` (float, default=0.05): Maximum risk per trade as fraction of account (hard cap regardless of Kelly)
- `target_max_dd` (float, default=0.3): Target maximum drawdown; risk is scaled down if simulated DD exceeds this
- `circuit_breaker_loss_range` (list[int], default=[3, 8]): Range [min, max] of consecutive losses to search for optimal circuit breaker trigger
- `circuit_breaker_pause_range` (list[int], default=[5, 30]): Range [min, max] of pause bars to search for optimal circuit breaker pause duration

## Outputs

- risk_per_trade: float — final risk fraction to use per trade (0 if unprofitable)
- is_profitable: bool — True when the scaled fractional Kelly is strictly positive
- full_kelly: float — raw (unscaled, uncapped) Kelly value
- trade_returns: List[float] — per-trade simulated returns (fk*rrr for wins, -fk for losses)
- circuit_breaker: dict — {pause_after_losses, pause_bars, enabled}
- risk_adjustment: dict — {original_risk, scale_factor, target_dd}

## Acceptance Criteria

- AC-001: Computes full_kelly as (win_rate * rrr - (1 - win_rate)) / rrr when rrr > 0, else 0.
- AC-002: The scaled fractional Kelly fk = max(0, min(max_risk, full_kelly * kelly_fraction)) — i.e. clamped to [0, max_risk].
- AC-003: When fk <= 0, returns is_profitable=False, risk_per_trade=0, trade_returns filled with 0.0 of len(trades), and a disabled circuit_breaker (all zeros).
- AC-004: When fk > 0, calls adjust_risk_for_target_dd(trades, fk, rrr, target_max_dd=target_max_dd) and replaces fk with adjusted_risk whenever scale_factor < 1.0.
- AC-005: When fk > 0, calls find_optimal_circuit_breaker with the (possibly adjusted) fk, rrr, and the two search ranges, and populates circuit_breaker from its 'optimal_pause_after_losses' / 'optimal_pause_bars' fields; circuit_breaker.enabled iff pause_after_losses > 0.
- AC-006: trade_returns[i] equals fk*rrr for each trades[i] > 0 and -fk otherwise (using the final fk).
- AC-007: risk_adjustment reports original_risk (pre-scale), scale_factor (from adjust_risk_for_target_dd), and echoes target_max_dd as target_dd.
- AC-008: get_default_params() returns exactly {kelly_fraction: 0.25, max_risk: 0.05, target_max_dd: 0.30, circuit_breaker_loss_range: [3, 8], circuit_breaker_pause_range: [5, 30]}.

## Edge Cases

- rrr == 0 (or negative): full_kelly is short-circuited to 0, fk becomes 0, unprofitable branch taken.
- Negative-edge inputs (win_rate*rrr < 1-win_rate): full_kelly < 0, fk clamped to 0, unprofitable branch taken.
- full_kelly * kelly_fraction exceeds max_risk: fk saturates at max_risk.
- Empty trades list: unprofitable branch still returns trade_returns=[] of length 0; profitable branch would call the two simulation helpers with an empty list (behaviour defers to those helpers).
- Zero scale_factor returned by adjust_risk_for_target_dd: risk_adjustment.original_risk falls back to fk to avoid a divide-by-zero.
- Circuit breaker search yields optimal_pause_after_losses == 0: circuit_breaker.enabled is False even though pause_bars may be non-zero.
- trades containing zeros: treated as losses in trade_returns (condition is t > 0).

## Assumptions

- win_rate is a probability in [0,1] and rrr is a positive reward-to-risk ratio when the manager is expected to produce a profitable sizing.
- adjust_risk_for_target_dd returns a dict with keys 'scale_factor' and 'adjusted_risk'.
- find_optimal_circuit_breaker returns a dict with keys 'optimal_pause_after_losses' and 'optimal_pause_bars'.
- The circuit_breaker_*_range params are 2-element [min, max] sequences understood by find_optimal_circuit_breaker.
