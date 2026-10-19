"""Integration with the native FWBG filters, plus their portfolio-specific scope."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.overfitting import compute_overfitting_metrics
from fwbg.optimization.robust_validation import create_walk_forward_folds
from fwbg.portfolio import robustness
from fwbg.portfolio.engine import PortfolioSettings, run_portfolio
from fwbg.portfolio.robustness import _check_series, assess_robustness, monthly_returns
from fwbg.simulation.trade import monte_carlo_permutation_test


@pytest.fixture(scope="module")
def long_history():
    """Provide seeded daily prices and a portfolio run spanning fourteen years."""
    dates = pd.bdate_range("2000-01-03", "2013-12-31")
    rng = np.random.default_rng(812)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (len(dates), 2)), axis=0)),
        index=dates,
        columns=["A", "B"],
    )
    settings = PortfolioSettings(
        weights={"A": 0.5, "B": 0.5}, equity_benchmark="A", start="2001-01-01"
    )
    result = run_portfolio(prices, settings)
    return prices, settings, result


def test_monthly_returns_preserve_initial_cost_and_exclude_partial_month():
    """Monthly compounding retains initial costs and drops partial endpoint months."""
    dates = pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-29", "2021-02-01", "2021-02-15"])
    equity = pd.DataFrame({"trend": [100, 99, 110, 112, 115]}, index=dates)
    monthly = monthly_returns(equity)
    assert len(monthly) == 1
    assert monthly.index[0] == pd.Timestamp("2021-01-31")
    assert monthly.trend.iloc[0] == pytest.approx(0.10)
    # Starting mid-month must not manufacture a full January observation.
    assert monthly_returns(equity.iloc[1:]).empty


def test_shared_native_filters_are_used_on_months_not_order_counts(long_history):
    """Native statistics use monthly observations and fixed-rule chronological folds."""
    prices, settings, result = long_history
    with (
        patch.object(
            robustness, "compute_overfitting_metrics", wraps=compute_overfitting_metrics
        ) as dsr,
        patch.object(
            robustness, "monte_carlo_permutation_test", wraps=monte_carlo_permutation_test
        ) as mc,
        patch.object(
            robustness, "create_walk_forward_folds", wraps=create_walk_forward_folds
        ) as planner,
    ):
        report, monthly, folds = assess_robustness(
            result, prices, settings, permutations=99, equity_simulations=10
        )
    assert dsr.call_count == 8  # three portfolios + active return, full + chronological tests
    assert mc.call_count == 24  # monthly, quarterly, half-year blocks for each series
    assert planner.call_count == 1
    first_args = dsr.call_args_list[0].kwargs
    assert first_args["n_trades"] == len(monthly)
    assert first_args["n_trades"] != len(result.trades)
    assert first_args["n_strategies"] == 1
    assert first_args["grid_results_by_fold"] == {}
    direct = monte_carlo_permutation_test(monthly.trend.tolist(), n_permutations=99, random_seed=42)
    assert report["full_sample"]["trend"]["monte_carlo"] == direct
    assert report["walk_forward"]["all_prefixes_invariant"]
    assert folds.fold.nunique() == 8
    assert len(folds) == 24
    assert folds.prefix_invariant.all()
    assert (folds.history_months >= 24).all()
    assert (
        report["walk_forward_oos"]["trend"]["observations"]
        == 8 * report["walk_forward"]["test_months_each"]
    )
    # Shared PBO cannot establish anything for one fixed rule, and we do not
    # turn references/cost scenarios into fabricated optimization candidates.
    assert report["pbo"]["status"] == "not_applicable"
    assert report["pbo"]["pbo"] is None
    assert report["multiple_testing"]["prior_search_count"] is None
    assert not report["overfitting_validation_complete"]
    assert not report["equity_permutation_diagnostic"]["used_as_filter"]
    terminal = report["equity_permutation_diagnostic"]["trend"]
    assert terminal["p5_equity"] == pytest.approx(terminal["p95_equity"])


def test_profitable_strategy_can_fail_benchmark_outperformance():
    """Positive absolute returns can pass while excess returns fail native checks."""
    dates = pd.date_range("2000-01-31", periods=240, freq="ME")
    benchmark = pd.Series(0.02 + 0.01 * np.sin(np.arange(240)), index=dates)
    strategy = benchmark / 2
    absolute = _check_series(strategy, permutations=399, seed=42)
    active = _check_series(strategy, benchmark=benchmark, permutations=399, seed=42)
    assert absolute["checks_passed"]
    assert not active["checks_passed"]
    assert active["monte_carlo"]["p_value"] > 0.95
    assert active["monthly_mean_return"] < 0


def test_block_test_preserves_compounding_and_pairs_same_benchmark_periods():
    """Block tests compare compounded returns over matching benchmark periods."""
    dates = pd.date_range("2000-01-31", periods=25, freq="ME")
    strategy = pd.Series(np.tile([0.2, -0.1, 0.0, 0.03, -0.04], 5), index=dates)
    benchmark = pd.Series(0.02, index=dates)
    check = _check_series(strategy, benchmark=benchmark, permutations=99, seed=42)
    expected_blocks = []
    for start in range(0, 24, 3):
        expected_blocks.append(
            (1 + strategy.iloc[start : start + 3]).prod()
            - (1 + benchmark.iloc[start : start + 3]).prod()
        )
    direct = monte_carlo_permutation_test(expected_blocks, n_permutations=99, random_seed=42)
    block = check["block_sign_permutations"]["3"]
    assert block["observed_pnl"] == pytest.approx(direct["observed_pnl"])
    assert block["discarded_tail_months"] == 1
    assert block["observations"] == 8
    # Native filter refuses <10 observations: cannot quietly mark short data robust.
    assert not block["is_significant"]


def test_flat_or_short_samples_cannot_pass_robustness(long_history):
    """Short histories and zero-return samples cannot satisfy robustness gates."""
    prices, settings, _ = long_history
    short = run_portfolio(prices.loc[:"2001-06-29"], settings)
    report, _, folds = assess_robustness(
        short, prices, settings, permutations=99, equity_simulations=10
    )
    assert report["walk_forward"]["status"] == "insufficient_data"
    assert not report["profitability_checks_passed"]
    assert folds.empty
    flat = pd.Series(0.0, index=pd.date_range("2000-01-31", periods=240, freq="ME"))
    check = _check_series(flat, permutations=99, seed=42)
    assert not check["checks_passed"]
    assert check["monthly_lag1_autocorrelation"] is None


def test_failed_prefix_check_prevents_a_robustness_pass(long_history):
    """A historical-prefix mismatch prevents a positive robustness summary."""
    prices, settings, result = long_history

    def altered_prefix(*args, **kwargs):
        """Perturb the last prefix valuation to simulate a causality failure."""
        prefix = run_portfolio(*args, **kwargs)
        prefix.equity.iloc[-1, 0] += 1
        return prefix

    with patch.object(robustness, "run_portfolio", side_effect=altered_prefix):
        report, _, _ = assess_robustness(
            result, prices, settings, permutations=99, equity_simulations=10
        )
    assert not report["walk_forward"]["all_prefixes_invariant"]
    assert not report["profitability_checks_passed"]


def test_fixed_seed_reproduces_statistics():
    """Repeated native checks with the same seed produce identical results."""
    sample = pd.Series(np.random.default_rng(1).normal(0.004, 0.03, 240))
    one = _check_series(sample, permutations=199, seed=42)
    two = _check_series(sample, permutations=199, seed=42)
    assert one == two
