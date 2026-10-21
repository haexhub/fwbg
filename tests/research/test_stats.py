import statistics

import pytest

from fwbg.research.card import CRITERIA_NAMES
from fwbg.research.dsr import deflated_sharpe_ratio
from fwbg.research.ledger import TrialKind, TrialStatus
from fwbg.research.stats import (
    _CHECKS,
    best_result,
    criteria_check,
    deflated_best,
    development_trials,
    evaluations,
    return_samples,
)
from tests.research.factories import make_trial


def test_best_result_ignores_holdout_failed_and_missing_sharpe():
    trials = [
        make_trial(trial_id="T-000001", metrics={"EURUSD": {"sharpe": 0.4}}),
        make_trial(trial_id="T-000002", metrics={"GBPUSD": {"sharpe": 0.9}, "X": {"sharpe": None}}),
        make_trial(
            trial_id="T-000003", kind=TrialKind.HOLDOUT, metrics={"EURUSD": {"sharpe": 2.0}}
        ),
        make_trial(
            trial_id="T-000004", status=TrialStatus.FAILED, metrics={"EURUSD": {"sharpe": 3.0}}
        ),
    ]
    best = best_result(trials)
    assert best is not None
    assert (best.trial_id, best.symbol, best.sharpe) == ("T-000002", "GBPUSD", 0.9)


def test_best_result_without_any_sharpe_is_none():
    assert best_result([make_trial(trial_id="T-000001", metrics={"X": {"sharpe": None}})]) is None
    assert best_result([]) is None


def test_criteria_check():
    assert criteria_check(
        {"min_sharpe": 1.0, "max_p_value": 0.05, "min_trades": 100.0},
        {"sharpe": 0.8, "p_value": 0.01},
    ) == ["min_sharpe fail (0.8 < 1)", "max_p_value pass (0.01 <= 0.05)", "min_trades n/a"]


def test_development_trials_and_evaluations_count_only_counted_trials():
    trials = [
        make_trial(trial_id="T-000001", metrics={"A": {}, "B": {}}),
        make_trial(trial_id="T-000002", kind=TrialKind.HOLDOUT, metrics={"A": {}}),
        make_trial(trial_id="T-000003", kind=TrialKind.SMOKE, metrics={"A": {}}),
        make_trial(trial_id="T-000004", status=TrialStatus.FAILED, metrics={"A": {}}),
    ]
    assert [trial.trial_id for trial in development_trials(trials)] == ["T-000001"]
    assert evaluations(trials) == 3


def test_every_criterion_has_a_check():
    assert _CHECKS.keys() == CRITERIA_NAMES


INPUTS = {"sr_per_trade": 0.1, "n_returns": 101, "skew": 0.0, "kurtosis": 3.0}


def test_return_samples_need_all_inputs_from_counted_development_trials():
    trials = [
        make_trial(trial_id="T-000001", metrics={"A": INPUTS, "B": {"sharpe": 0.0}}),  # B legacy
        make_trial(trial_id="T-000002", metrics={"A": {**INPUTS, "kurtosis": None}}),
        make_trial(trial_id="T-000003", metrics={"A": {**INPUTS, "n_returns": 1}}),
        make_trial(trial_id="T-000004", kind=TrialKind.HOLDOUT, metrics={"A": INPUTS}),
        make_trial(trial_id="T-000005", status=TrialStatus.FAILED, metrics={"A": INPUTS}),
        make_trial(trial_id="T-000006", kind=TrialKind.SMOKE, metrics={"A": INPUTS}),
    ]
    assert [(s.trial_id, s.symbol) for s in return_samples(trials)] == [("T-000001", "A")]


def test_deflated_best_needs_two_samples():
    assert deflated_best([make_trial(trial_id="T-000001", metrics={"A": INPUTS})]) is None


def test_deflated_best_deflates_the_best_sample_by_all_evaluations():
    trials = [
        make_trial(trial_id="T-000001", metrics={"A": INPUTS, "X": {"sharpe": None}}),  # no_edge
        make_trial(trial_id="T-000002", metrics={"A": {**INPUTS, "sr_per_trade": 0.3}}),
        make_trial(trial_id="T-000003", kind=TrialKind.LEGACY, metrics={"L": {"sharpe": 0}}),
        make_trial(
            trial_id="T-000004",
            kind=TrialKind.HOLDOUT,
            metrics={"A": {**INPUTS, "sr_per_trade": 0.9}},
        ),
    ]
    result = deflated_best(trials)
    assert result is not None
    assert (result.best.trial_id, result.best.sr) == ("T-000002", 0.3)
    assert (result.n_trials, result.samples) == (5, 2)  # N counts every evaluation
    assert result.sr_variance == pytest.approx(statistics.variance([0.1, 0.3]))
    assert result.dsr == pytest.approx(deflated_sharpe_ratio(0.3, 101, 0.0, 3.0, 5, 0.02))


def test_deflated_best_is_undefined_for_impossible_moments():
    bad = {**INPUTS, "sr_per_trade": 1.0, "skew": 3.0, "kurtosis": 1.0}
    result = deflated_best([make_trial(trial_id="T-000001", metrics={"A": INPUTS, "B": bad})])
    assert result is not None
    assert result.best.symbol == "B"
    assert result.dsr is None
