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
