"""Statistics shared by the research views and the verdict."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from fwbg.research.ledger import Trial, TrialKind

# criterion -> (metric, comparison that passes, sign shown on pass, sign shown on fail)
_CHECKS: dict[str, tuple[str, Callable[[float, float], bool], str, str]] = {
    "min_sharpe": ("sharpe", operator.ge, ">=", "<"),
    "min_trades": ("trades", operator.ge, ">=", "<"),
    "max_drawdown": ("max_drawdown", operator.le, "<=", ">"),
    "min_fold_stability": ("fold_stability", operator.ge, ">=", "<"),
    "max_p_value": ("p_value", operator.le, "<=", ">"),
}


@dataclass(frozen=True)
class BestResult:
    trial_id: str
    symbol: str
    sharpe: float
    metrics: Mapping[str, float | int | None]


def development_trials(trials: Iterable[Trial]) -> list[Trial]:
    """Counted trials that saw only development data (no holdout looks)."""
    return [trial for trial in trials if trial.counted and trial.kind is not TrialKind.HOLDOUT]


def best_result(trials: Iterable[Trial]) -> BestResult | None:
    """The symbol evaluation with the highest Sharpe ratio among development trials."""
    candidates = [
        BestResult(str(trial.trial_id), symbol, float(sharpe), values)
        for trial in development_trials(trials)
        for symbol, values in trial.metrics.items()
        if isinstance(sharpe := values.get("sharpe"), (int, float))
    ]
    return max(candidates, key=lambda candidate: candidate.sharpe, default=None)


def criteria_check(
    criteria: Mapping[str, float], metrics: Mapping[str, float | int | None]
) -> list[str]:
    """One line per criterion: pass, fail or n/a (metric missing)."""
    lines = []
    for name, threshold in criteria.items():
        metric, passes, ok_sign, fail_sign = _CHECKS[name]
        value = metrics.get(metric)
        if not isinstance(value, (int, float)):
            lines.append(f"{name} n/a")
        elif passes(value, threshold):
            lines.append(f"{name} pass ({value:g} {ok_sign} {threshold:g})")
        else:
            lines.append(f"{name} fail ({value:g} {fail_sign} {threshold:g})")
    return lines


def evaluations(trials: Iterable[Trial]) -> int:
    """Counted symbol evaluations: the N a multiple-testing correction uses."""
    return sum(len(trial.metrics) for trial in trials if trial.counted)
