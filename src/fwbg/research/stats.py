"""Statistics shared by the research views and the verdict."""

from __future__ import annotations

import json
import logging
import math
import operator
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fwbg.research.dsr import expected_max_sharpe, probabilistic_sharpe_ratio
from fwbg.research.ledger import Trial, TrialKind
from fwbg.research.repo import config_path

MIN_RETURN_SAMPLES = 2
# Hand-curated: the walk_forward.selection_protocol from which signal models and
# indicator-grid variants are chosen on inner windows of each fold's training data.
# Rows below it chose on test folds. Keep it at 2 when protocol 3 (causal unified
# merge) arrives: that must not unrank protocol-2 rows unless decided separately.
INNER_SELECTION_PROTOCOL = 2
# Why an evaluation from before INNER_SELECTION_PROTOCOL is not ranked.
SIGNAL_MODEL = "signal-model"
INDICATOR_GRID = "indicator-grid"
_RETURN_INPUTS = ("sr_per_trade", "n_returns", "skew", "kurtosis")

# criterion -> (metric, comparison that passes, sign shown on pass, sign shown on fail)
_CHECKS: dict[str, tuple[str, Callable[[float, float], bool], str, str]] = {
    "min_sharpe": ("sharpe", operator.ge, ">=", "<"),
    "min_trades": ("trades", operator.ge, ">=", "<"),
    "max_drawdown": ("max_drawdown", operator.le, "<=", ">"),
    "min_fold_stability": ("fold_stability", operator.ge, ">=", "<"),
    "max_p_value": ("p_value", operator.le, "<=", ">"),
}

log = logging.getLogger(__name__)


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


def unrankable_configs(research_dir: Path, trials: Iterable[Trial]) -> dict[str, str]:
    """Development fingerprints whose stored config chose its setting on test folds.

    Maps each to its reason: ``SIGNAL_MODEL`` or, for other models, ``INDICATOR_GRID``
    (a config with both counts as a signal model). Each distinct config is read once.
    A trial without a stored config (studies, legacy runs) stays rankable; so does
    one whose config cannot be read or parsed: it is logged, not raised, so a corrupt
    file never blocks a verdict (``show`` still refuses it in its coverage section).
    """
    found = {}
    for fingerprint in dict.fromkeys(trial.fingerprint for trial in development_trials(trials)):
        path = config_path(research_dir, fingerprint)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            log.warning("configs/%s unreadable, ranked as unknown: %s", path.name, exc)
            continue
        strategy = _mapping(_mapping(config).get("strategy"))
        if _mapping(strategy.get("model")).get("type") == "signal":
            found[fingerprint] = SIGNAL_MODEL
        elif _grid_variants(_mapping(strategy.get("optimization")).get("indicator_grid")) > 1:
            found[fingerprint] = INDICATOR_GRID
    return found


def _grid_variants(grid: Any) -> int:
    """The variants fwbg.optimization.indicator_grid.expand_indicator_grid makes.

    Per indicator the product of its value-list lengths, summed over indicators:
    each variant overrides one indicator. Without variants it runs the base once.
    """
    count = sum(
        math.prod(len(values) for values in params.values())
        for params in _mapping(grid).values()
        if isinstance(params, dict)
    )
    return max(count, 1)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _ranked(
    fingerprint: str, values: Mapping[str, float | int | None], unrankable: Mapping[str, str]
) -> bool:
    """Whether a development evaluation may be ranked.

    Before INNER_SELECTION_PROTOCOL, signal models chose their setting and every
    model its indicator-grid variant on the test folds: such evaluations count
    in N but are never the best result or a return sample.
    """
    protocol = values.get("selection_protocol")
    marked = isinstance(protocol, (int, float)) and protocol >= INNER_SELECTION_PROTOCOL
    return marked or fingerprint not in unrankable


def unranked_evaluations(trials: Iterable[Trial], unrankable: Mapping[str, str]) -> dict[str, int]:
    """Development symbol evaluations left out of the ranking, per reason.

    Only evaluations that would otherwise be ranked (a Sharpe ratio or return
    statistics) are counted; a metric-less one has nothing to leave out.
    *unrankable* comes from :func:`unrankable_configs`.
    """
    counts: dict[str, int] = {}
    for trial in development_trials(trials):
        for symbol, values in trial.metrics.items():
            if _ranked(trial.fingerprint, values, unrankable):
                continue
            if isinstance(values.get("sharpe"), (int, float)) or _return_sample("", symbol, values):
                reason = unrankable[trial.fingerprint]
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def unranked_note(counts: Mapping[str, int]) -> str | None:
    """How the views and the verdict state the unranked evaluations; None if there are none."""
    total = sum(counts.values())
    if not total:
        return None
    reasons = ", ".join(
        f"{counts[reason]} {reason}"
        for reason in (SIGNAL_MODEL, INDICATOR_GRID)
        if reason in counts
    )
    noun = f"pre-protocol-{INNER_SELECTION_PROTOCOL} evaluation"
    return f"{plural(total, noun)} not ranked (counted in N): {reasons}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def best_result(trials: Iterable[Trial], unrankable: Mapping[str, str]) -> BestResult | None:
    """The symbol evaluation with the highest Sharpe ratio among ranked development trials.

    *unrankable* comes from :func:`unrankable_configs`.
    """
    candidates = [
        BestResult(str(trial.trial_id), symbol, float(sharpe), values)
        for trial in development_trials(trials)
        for symbol, values in trial.metrics.items()
        if isinstance(sharpe := values.get("sharpe"), (int, float))
        and _ranked(trial.fingerprint, values, unrankable)
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


def return_samples(trials: Iterable[Trial], unrankable: Mapping[str, str]) -> list[ReturnSample]:
    """Ranked development evaluations that carry the DSR's return statistics (step 5 on).

    Legacy and no_edge evaluations carry none, and unranked evaluations
    (see :func:`best_result`) are left out: they count towards N only.
    """
    samples = []
    for trial in development_trials(trials):
        for symbol, values in trial.metrics.items():
            if not _ranked(trial.fingerprint, values, unrankable):
                continue
            sample = _return_sample(str(trial.trial_id), symbol, values)
            if sample is not None:
                samples.append(sample)
    return samples


def _return_sample(
    trial_id: str, symbol: str, values: Mapping[str, float | int | None]
) -> ReturnSample | None:
    """The evaluation's return statistics; None if one is missing or it has under 2 returns."""
    inputs = []
    for key in _RETURN_INPUTS:
        value = values.get(key)
        if isinstance(value, (int, float)):
            inputs.append(value)
    if len(inputs) < len(_RETURN_INPUTS) or inputs[1] < 2:
        return None
    sr, n_returns, skew, kurtosis = inputs
    return ReturnSample(trial_id, symbol, float(sr), int(n_returns), float(skew), float(kurtosis))


def deflated_best(trials: Sequence[Trial], unrankable: Mapping[str, str]) -> DeflatedBest | None:
    """DSR of the best ranked evaluation by per-trade SR; None below MIN_RETURN_SAMPLES.

    V[SR] is floored at the SR's sampling variance under the null (about 1/(T-1)
    per evaluation), so few or near-identical samples cannot collapse E[max SR].
    N counts every evaluation, unranked ones included.
    """
    samples = return_samples(trials, unrankable)
    if len(samples) < MIN_RETURN_SAMPLES:
        return None
    best = max(samples, key=lambda sample: sample.sr)
    sample_variance = statistics.variance(sample.sr for sample in samples)
    floor = statistics.fmean(1 / (sample.n_returns - 1) for sample in samples)
    n_trials = evaluations(trials)
    expected = expected_max_sharpe(n_trials, max(sample_variance, floor))
    dsr = probabilistic_sharpe_ratio(best.sr, expected, best.n_returns, best.skew, best.kurtosis)
    return DeflatedBest(best, n_trials, len(samples), sample_variance, floor, expected, dsr)
