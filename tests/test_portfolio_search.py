"""Independent rule examples and regressions for candidate-selection bias."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.overfitting import probability_of_backtest_overfitting
from fwbg.portfolio.engine import PortfolioSettings, run_portfolio
from fwbg.portfolio.rules import STRATEGIES, allocation
from fwbg.portfolio.search import holm_adjust, joint_p, run_search


def test_fixed_rules_on_known_returns_and_cash_slots():
    assets = ["SPY", "EFA", "EEM", "IEF", "GLD"]
    settings = PortfolioSettings(
        dict.fromkeys(assets, 0.2),
        "SPY",
        equity_assets=("SPY", "EFA", "EEM"),
        defensive_asset="IEF",
    )
    # Only gold has positive momentum. The second dual slot must remain cash.
    closes = np.array([[100.0] * 5, [90.0, 70.0, 80.0, 95.0, 120.0]])
    expected = {
        "absolute_momentum": [0, 0, 0, 0, 0.2],
        "relative_momentum": [0, 0, 0, 0.5, 0.5],
        "dual_momentum": [0, 0, 0, 0, 0.5],
        "equity_dual_momentum": [0, 0, 0, 0, 0],
        "reversal": [0, 0.5, 0.5, 0, 0],
    }
    for strategy, target in expected.items():
        actual, _ = allocation(closes, np.ones(5) / 5, assets, replace(settings, strategy=strategy))
        np.testing.assert_allclose(actual, target)
    closes[-1, 3] = 105
    actual, _ = allocation(
        closes, np.ones(5) / 5, assets, replace(settings, strategy="equity_dual_momentum")
    )
    np.testing.assert_allclose(actual, [0, 0, 0, 1, 0])
    closes[-1, 1] = 130
    actual, _ = allocation(
        closes, np.ones(5) / 5, assets, replace(settings, strategy="equity_dual_momentum")
    )
    np.testing.assert_allclose(actual, [0, 1, 0, 0, 0])


def test_inverse_volatility_uses_returns_not_price_levels():
    settings = PortfolioSettings(
        {"A": 0.5, "B": 0.5}, "A", strategy="inverse_volatility", lookback_months=2
    )
    target, _ = allocation(
        np.array([[100, 100], [110, 105], [99, 99.75]]), np.array([0.5, 0.5]), ["A", "B"], settings
    )
    np.testing.assert_allclose(target, [1 / 3, 2 / 3])
    target, _ = allocation(np.ones((3, 2)), np.array([0.5, 0.5]), ["A", "B"], settings)
    np.testing.assert_allclose(target, [0, 0])


def test_twelve_month_signal_requires_thirteen_closes_and_delays_entry():
    dates = pd.bdate_range("2020-01-01", "2021-03-31")
    prices = pd.DataFrame({"A": 100.0}, index=dates)
    prices.loc[:"2020-01-31", "A"] = 50
    prices.loc["2021-02-01":, "A"] = 1000
    settings = PortfolioSettings(
        {"A": 1.0}, "A", strategy="absolute_momentum", lookback_months=12, one_way_cost_bps=0
    )
    result = run_portfolio(prices, settings)
    signal = result.signals.iloc[0]
    assert signal.decision_date == pd.Timestamp("2021-01-29")
    assert signal.execution_date == pd.Timestamp("2021-02-01")
    assert signal.score == 1
    assert signal.target_weight == 1
    assert result.equity.loc["2021-02-01", "trend"] == pytest.approx(10000)


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_all_rules_are_causal_unlevered_and_zero_cash_yield(strategy):
    dates = pd.bdate_range("2019-01-01", "2022-06-30")
    assets = ["SPY", "EFA", "EEM", "IEF", "GLD"]
    random = np.random.default_rng(17)
    prices = pd.DataFrame(
        100 * np.exp(random.normal(0.0002, 0.01, (len(dates), 5)).cumsum(axis=0)),
        index=dates,
        columns=assets,
    )
    settings = PortfolioSettings(
        dict.fromkeys(assets, 0.2),
        "SPY",
        strategy=strategy,
        start="2021-01-01",
        lookback_months=12,
        equity_assets=("SPY", "EFA", "EEM"),
        defensive_asset="IEF",
    )
    original = run_portfolio(prices, settings)
    prices.loc["2022-03-15":] *= 0.01
    changed = run_portfolio(prices, settings)
    pd.testing.assert_frame_equal(
        original.equity.loc[:"2022-03-14"], changed.equity.loc[:"2022-03-14"]
    )
    pd.testing.assert_frame_equal(
        original.signals.loc[original.signals.execution_date < "2022-03-15"],
        changed.signals.loc[changed.signals.execution_date < "2022-03-15"],
    )
    assert original.equity.cash.eq(10000).all()
    assert (original.weights[assets + ["cash"]] >= 0).all().all()
    np.testing.assert_allclose(original.weights[assets + ["cash"]].sum(axis=1), 1)
    assert (original.signals.decision_date < original.signals.execution_date).all()


def test_holm_and_joint_test_do_not_select_only_favorable_pvalues():
    assert holm_adjust({"a": 0.01, "b": 0.02, "c": 0.03}) == {"a": 0.03, "b": 0.04, "c": 0.04}
    check = {
        "status": "evaluated",
        "monte_carlo": {"p_value": 0.001},
        "block_sign_permutations": {"3": {"p_value": 0.02}, "6": {"p_value": 0.3}},
    }
    report = {"full_sample": {"trend": check}, "walk_forward_oos": {"trend": check}}
    assert joint_p(report, "trend") == 0.3
    assert joint_p(report, "absent") == 1
    with pytest.raises(ValueError):
        holm_adjust({"bad": float("nan")})


def test_search_writes_all_trials_and_native_pbo(tmp_path):
    dates = pd.bdate_range("2000-01-03", "2013-12-31")
    random = np.random.default_rng(55)
    prices = pd.DataFrame(
        100 * np.exp(random.normal(0.0003, 0.01, (len(dates), 2)).cumsum(axis=0)),
        index=dates,
        columns=["A", "B"],
    )
    prices.rename_axis("date").to_csv(tmp_path / "prices.csv")
    config = {
        "schema_version": 1,
        "name": "synthetic",
        "base_currency": "EUR",
        "price_basis": "total_return",
        "prices": "prices.csv",
        "weights": {"A": 0.5, "B": 0.5},
        "equity_benchmark": "A",
        "start": "2002-01-01",
        "end": "2013-12-31",
        "split_date": "2008-01-01",
    }
    (tmp_path / "base.json").write_text(json.dumps(config))
    protocol = json.loads(Path("examples/portfolio/search_v1.json").read_text())
    protocol.update(
        base_config="base.json",
        candidates=protocol["candidates"][:2],
        equity_assets=["A"],
        defensive_asset="B",
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    output = tmp_path / "result"
    report = run_search(tmp_path / "protocol.json", output)
    assert len(pd.read_csv(output / "comparison.csv")) == 6
    assert len(report["multiple_testing"]["holm_p"]) == 4
    assert report["pbo"]["1"]["n_cscv_splits"] == 70
    matrix = pd.read_csv(output / "pbo_matrix_cost1.csv", index_col=0).to_numpy()
    assert report["pbo"]["1"]["pbo"] == probability_of_backtest_overfitting(matrix)["pbo"]
    monthly = pd.read_csv(
        output / "candidate_monthly_returns_cost1.csv", index_col=0, parse_dates=True
    )
    expected_winner = monthly.loc[monthly.index < "2008-01-01"].mean().idxmax()
    assert report["chronological_selection"]["selected_candidate"] == expected_winner
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    assert (output / "comparison.png").exists()
    with pytest.raises(FileExistsError):
        run_search(tmp_path / "protocol.json", output)
