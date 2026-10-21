"""Deflated Sharpe ratio, Bailey & López de Prado (2014)."""

from math import sqrt
from statistics import NormalDist

import numpy as np
import pytest

from fwbg.research.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    return_statistics,
)

# The paper's numerical example: N = 100 independent trials, variance of the
# annualised SRs 1/2, best annualised SR 2.5, T = 1250 daily returns (5 years),
# skewness -3, kurtosis 10. The paper reports SR_0 = 0.1132 and DSR = 0.9004.
DAILY_VARIANCE = 0.5 / 250
BEST_SR = 2.5 / sqrt(250)


def test_expected_max_sharpe_matches_the_paper():
    assert expected_max_sharpe(100, DAILY_VARIANCE) == pytest.approx(0.1132, abs=5e-4)


def test_deflated_sharpe_ratio_matches_the_paper():
    dsr = deflated_sharpe_ratio(
        sr=BEST_SR,
        n_returns=1250,
        skew=-3.0,
        kurtosis=10.0,
        n_trials=100,
        sr_variance=DAILY_VARIANCE,
    )
    assert dsr == pytest.approx(0.9004, abs=1e-3)


def test_expected_max_sharpe_approximates_the_maximum_of_standard_normals():
    maxima = np.random.default_rng(0).standard_normal((2000, 1000)).max(axis=1)
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(float(maxima.mean()), abs=0.03)


@pytest.mark.parametrize("n_trials, variance", [(1, 0.3), (0, 0.3), (50, 0.0)])
def test_no_selection_bias_without_trials_or_variance(n_trials, variance):
    assert expected_max_sharpe(n_trials, variance) == 0.0


def test_psr_of_normal_returns():
    expected = NormalDist().cdf(0.1 * sqrt(100) / sqrt(1 + 0.5 * 0.01))
    assert probabilistic_sharpe_ratio(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "sr, n_returns, skew, kurtosis",
    [(0.1, 1, 0.0, 3.0), (1.0, 100, 3.0, 1.0)],  # one return; 1 - 3 + 0 < 0
)
def test_psr_is_undefined(sr, n_returns, skew, kurtosis):
    assert probabilistic_sharpe_ratio(sr, 0.0, n_returns, skew, kurtosis) is None


def test_return_statistics_use_population_moments():
    stats = return_statistics([-1.0, 1.0, -1.0, 1.0])
    assert stats == {"sr_per_trade": 0.0, "skew": 0.0, "kurtosis": 1.0, "n_returns": 4}


def test_return_statistics_of_a_skewed_series():
    values = [0.01, -0.02, 0.03, 0.0, 0.05]
    mean, std = np.mean(values), np.std(values)
    z = (np.array(values) - mean) / std
    stats = return_statistics(values)
    assert stats["sr_per_trade"] == pytest.approx(mean / std)
    assert stats["skew"] == pytest.approx(np.mean(z**3))
    assert stats["kurtosis"] == pytest.approx(np.mean(z**4))


@pytest.mark.parametrize(
    "returns", [[], [0.01], [0.01, 0.01], [float("nan"), 0.1], [float("inf"), 0.1]]
)
def test_return_statistics_need_two_varying_returns(returns):
    assert return_statistics(returns) is None
