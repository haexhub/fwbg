"""Deflated Sharpe ratio (Bailey & López de Prado 2014) for research verdicts.

Sharpe ratios here are per return (per trade), never annualised, so trials
with different trade frequencies share one scale. Kurtosis is the plain
fourth standardised moment (3 for normal returns), not excess kurtosis.
Stdlib only: the register must not import the optimizer package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist, fmean, pstdev

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] of *n_trials* independent trials with true SR 0 whose SR
    estimates have variance *sr_variance* (the paper's SR_0)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def probabilistic_sharpe_ratio(
    sr: float, benchmark: float, n_returns: int, skew: float, kurtosis: float
) -> float | None:
    """P(true SR > *benchmark*) for an SR *sr* observed over *n_returns* returns.

    None when undefined: fewer than two returns, or skew and kurtosis that leave
    the SR estimate no positive variance.
    """
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if n_returns < 2 or denominator <= 0:
        return None
    return _NORMAL.cdf((sr - benchmark) * math.sqrt(n_returns - 1) / math.sqrt(denominator))


def deflated_sharpe_ratio(
    sr: float, n_returns: int, skew: float, kurtosis: float, n_trials: int, sr_variance: float
) -> float | None:
    """PSR of *sr* against the SR the best of *n_trials* would show by luck alone."""
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, benchmark, n_returns, skew, kurtosis)


def return_statistics(returns: Sequence[float]) -> dict[str, float | int] | None:
    """What the verdict's DSR needs from one evaluation's per-trade returns.

    Population moments, like the run's own per-trade Sharpe (process.py). None
    for fewer than two returns or a constant (or non-finite) series.
    """
    values = [float(value) for value in returns]
    # statistics raises on non-finite values instead of returning NaN.
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return None
    mean = fmean(values)
    std = pstdev(values, mu=mean)
    if std == 0:
        return None
    z = [(value - mean) / std for value in values]
    return {
        "sr_per_trade": mean / std,
        "skew": fmean(x**3 for x in z),
        "kurtosis": fmean(x**4 for x in z),
        "n_returns": len(values),
    }
