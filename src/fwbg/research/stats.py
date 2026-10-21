"""Statistics shared by the research views and the verdict."""

from __future__ import annotations

import operator
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from fwbg.research.dsr import expected_max_sharpe, probabilistic_sharpe_ratio
from fwbg.research.ledger import Trial, TrialKind

MIN_RETURN_SAMPLES = 2
_RETURN_INPUTS = ("sr_per_trade", "n_returns", "skew", "kurtosis")

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


@dataclass(frozen=True)
class ReturnSample:
    trial_id: str
    symbol: str
    sr: float
    n_returns: int
    skew: float
    kurtosis: float


@dataclass(frozen=True)
class DeflatedBest:
    best: ReturnSample
    n_trials: int  # N: counted symbol evaluations of the card (holdout looks included)
    samples: int  # development evaluations V[SR] was estimated from
    sample_variance: float
    variance_floor: float  # mean of 1/(T-1): the per-trade SR's sampling variance under the null
    expected_max_sr: float
    dsr: float | None  # None: PSR undefined for the best's skew and kurtosis

    @property
    def floored(self) -> bool:
        """Whether the floor, not the sample variance, entered E[max SR]."""
        return self.variance_floor > self.sample_variance

    @property
    def sr_variance(self) -> float:
        """The V[SR] used: the sample variance, floored at the null sampling variance."""
        return max(self.sample_variance, self.variance_floor)


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


def return_samples(trials: Iterable[Trial]) -> list[ReturnSample]:
    """Development evaluations that carry the DSR's return statistics (runs from step 5 on).

    Legacy and no_edge evaluations carry none: they count towards N only.
    """
    samples = []
    for trial in development_trials(trials):
        for symbol, values in trial.metrics.items():
            inputs = []
            for key in _RETURN_INPUTS:
                value = values.get(key)
                if isinstance(value, (int, float)):
                    inputs.append(value)
            if len(inputs) == len(_RETURN_INPUTS) and inputs[1] >= 2:
                sr, n_returns, skew, kurtosis = inputs
                samples.append(
                    ReturnSample(
                        str(trial.trial_id),
                        symbol,
                        float(sr),
                        int(n_returns),
                        float(skew),
                        float(kurtosis),
                    )
                )
    return samples


def deflated_best(trials: Sequence[Trial]) -> DeflatedBest | None:
    """DSR of the best development evaluation by per-trade SR; None below MIN_RETURN_SAMPLES.

    V[SR] is floored at the SR's sampling variance under the null (about 1/(T-1)
    per evaluation), so few or near-identical samples cannot collapse E[max SR].
    """
    samples = return_samples(trials)
    if len(samples) < MIN_RETURN_SAMPLES:
        return None
    best = max(samples, key=lambda sample: sample.sr)
    sample_variance = statistics.variance(sample.sr for sample in samples)
    floor = statistics.fmean(1 / (sample.n_returns - 1) for sample in samples)
    n_trials = evaluations(trials)
    expected = expected_max_sharpe(n_trials, max(sample_variance, floor))
    dsr = probabilistic_sharpe_ratio(best.sr, expected, best.n_returns, best.skew, best.kurtosis)
    return DeflatedBest(best, n_trials, len(samples), sample_variance, floor, expected, dsr)
