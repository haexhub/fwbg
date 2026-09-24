import pandas as pd
import pytest

from scripts.jev.calibration import accuracy_vs_baseline, agreement_rate, brier_score


def test_brier_score_perfect_predictions_is_zero():
    predicted = pd.Series([1.0, 0.0, 1.0, 0.0])
    actual = pd.Series([1.0, 0.0, 1.0, 0.0])
    assert brier_score(predicted, actual) == 0.0


def test_brier_score_worst_case_is_one():
    predicted = pd.Series([1.0, 0.0])
    actual = pd.Series([0.0, 1.0])
    assert brier_score(predicted, actual) == 1.0


def test_accuracy_vs_baseline_reports_both():
    predicted = pd.Series([0.9, 0.9, 0.1, 0.1])
    actual = pd.Series([1.0, 0.0, 0.0, 0.0])  # majority class is 0 (3 of 4)
    result = accuracy_vs_baseline(predicted, actual, threshold=0.5)
    assert result["model_accuracy"] == 0.75  # gets bar 2 wrong (predicted win, actual loss)
    assert result["baseline_accuracy"] == 0.75  # always-predict-0 baseline


def test_agreement_rate_between_two_providers():
    provider_a = pd.Series([0.9, 0.1, 0.6])
    provider_b = pd.Series([0.8, 0.2, 0.4])
    # threshold 0.5: A says [win, loss, win], B says [win, loss, loss] -> 2/3 agree
    assert agreement_rate(provider_a, provider_b, threshold=0.5) == pytest.approx(2 / 3)


def test_brier_score_excludes_nan_row():
    predicted = pd.Series([1.0, float("nan"), 0.0])
    actual = pd.Series([1.0, 0.0, 0.0])
    # row 1 has a missing prediction and must be excluded, not counted against the score
    assert brier_score(predicted, actual) == 0.0


def test_accuracy_vs_baseline_excludes_nan_predicted_row():
    predicted = pd.Series([0.9, float("nan"), 0.1])
    actual = pd.Series([1.0, 1.0, 0.0])
    # row 1 has a missing prediction; if it were coerced to "loss" it would count as
    # wrong against actual=1.0 (win). Excluding it leaves two rows that both match.
    result = accuracy_vs_baseline(predicted, actual, threshold=0.5)
    assert result["model_accuracy"] == 1.0
    assert result["baseline_accuracy"] == 0.5


def test_agreement_rate_excludes_rows_with_nan_in_either_provider():
    provider_a = pd.Series([0.9, float("nan"), 0.6])
    provider_b = pd.Series([0.8, float("nan"), 0.3])
    # row 1 is unanswered by both providers; it must not count as agreement.
    # Without exclusion this would read as 2/3 (NaN>=threshold is False on both sides).
    assert agreement_rate(provider_a, provider_b, threshold=0.5) == pytest.approx(0.5)
