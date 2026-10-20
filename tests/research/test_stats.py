from fwbg.research.card import CRITERIA_NAMES
from fwbg.research.ledger import TrialKind, TrialStatus
from fwbg.research.stats import (
    _CHECKS,
    best_result,
    criteria_check,
    development_trials,
    evaluations,
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
