"""Accounting, timing and data-contract regressions for monthly portfolios."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fwbg.portfolio.cli import run_experiment
from fwbg.portfolio.data import load_inputs
from fwbg.portfolio.download import (
    _ecb_series,
    convert_to_eur,
    prepare_example,
)
from fwbg.portfolio.engine import PortfolioSettings, run_portfolio, summarize


@pytest.fixture
def market():
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    prices = pd.DataFrame({"A": 100.0 + np.arange(len(dates)), "B": 100.0}, index=dates)
    settings = PortfolioSettings(
        weights={"A": 0.5, "B": 0.5},
        equity_benchmark="A",
        lookback_months=2,
        one_way_cost_bps=0,
    )
    return prices, settings


def test_signal_executes_next_session_and_cannot_earn_execution_day_jump(market):
    prices, settings = market
    prices.loc["2020-03-02":, "A"] *= 10
    result = run_portfolio(prices, settings)
    first = result.signals.iloc[0]
    assert first.decision_date == pd.Timestamp("2020-02-28")
    assert first.execution_date == pd.Timestamp("2020-03-02")
    assert first.target_weight == 0.5
    # The tenfold jump happens before the first purchase for every portfolio.
    assert (result.equity.loc["2020-03-02"] == 10_000).all()
    assert result.equity.loc["2020-03-03", "trend"] > 10_000


def test_future_prices_cannot_change_prior_signals_positions_or_returns(market):
    prices, settings = market
    original = run_portfolio(prices, settings)
    changed = prices.copy()
    changed.loc["2020-05-15":, "A"] *= 0.01
    other = run_portfolio(changed, settings)
    pd.testing.assert_frame_equal(
        original.equity.loc[:"2020-05-14"], other.equity.loc[:"2020-05-14"]
    )
    pd.testing.assert_frame_equal(
        original.signals.loc[original.signals.execution_date < pd.Timestamp("2020-05-15")],
        other.signals.loc[other.signals.execution_date < pd.Timestamp("2020-05-15")],
    )
    pd.testing.assert_frame_equal(
        original.weights.loc[original.weights.date < pd.Timestamp("2020-05-15")],
        other.weights.loc[other.weights.date < pd.Timestamp("2020-05-15")],
    )


def test_truncating_history_does_not_treat_partial_month_as_month_end(market):
    prices, settings = market
    full = run_portfolio(prices, settings)
    short = run_portfolio(prices.loc[:"2020-04-15"], settings)
    pd.testing.assert_frame_equal(full.equity.loc[:"2020-04-15"], short.equity)
    assert short.signals.execution_date.max() == pd.Timestamp("2020-04-01")


def test_trend_exit_carries_execution_day_loss_before_selling(market):
    prices, settings = market
    prices.loc["2020-03-31", "A"] = 20
    prices.loc["2020-04-01":, "A"] = 10
    result = run_portfolio(prices, settings)
    exit_signal = result.signals.loc[
        (result.signals.asset == "A")
        & (result.signals.execution_date == pd.Timestamp("2020-04-01"))
    ]
    assert exit_signal.target_weight.iloc[0] == 0
    assert result.equity.loc["2020-04-01", "trend"] < result.equity.loc["2020-03-31", "trend"]
    weights = result.weights.loc[
        (result.weights.portfolio == "trend") & (result.weights.date == pd.Timestamp("2020-04-01"))
    ]
    assert weights.cash.iloc[0] == pytest.approx(1)
    assert weights.A.iloc[0] == 0


def test_costs_are_self_financed_and_buy_hold_does_not_rebalance(market):
    prices, settings = market
    prices.loc[:, :] = 100
    settings = replace(settings, one_way_cost_bps=100)
    result = run_portfolio(prices, settings)
    expected = 10_000 / 1.01
    assert result.equity.iloc[-1]["static"] == pytest.approx(expected)
    assert result.equity.iloc[-1]["equity_buy_hold"] == pytest.approx(expected)
    paid = result.trades.query("portfolio == 'static'").cost.sum()
    assert paid + result.equity.iloc[-1]["static"] == pytest.approx(10_000)
    assert len(result.trades.query("portfolio == 'equity_buy_hold'")) == 1
    assert (result.weights.cash >= 0).all()
    np.testing.assert_allclose(result.weights[["A", "B", "cash"]].sum(axis=1), 1)


def test_inactive_sleeves_remain_cash_not_redistributed(market):
    prices, settings = market
    result = run_portfolio(prices, settings)
    weights = result.weights.query("portfolio == 'trend'").iloc[0]
    assert weights.A == pytest.approx(0.5)
    assert weights.B == 0  # exactly equal to SMA -> not above it
    assert weights.cash == pytest.approx(0.5)


def test_drawdown_includes_losses_inside_a_month(market):
    prices, settings = market
    prices.loc[:, :] = 100
    prices.loc["2020-03-16", :] = 50
    result = run_portfolio(prices, settings)
    metrics = summarize(result.equity)
    assert metrics["static"]["max_drawdown"] == pytest.approx(-0.5)
    assert metrics["static"]["total_return"] == pytest.approx(0)


def test_cash_stays_constant_and_earns_no_return(market):
    prices, settings = market
    prices.loc[:, :] = 100
    result = run_portfolio(prices, settings)
    assert (result.equity.cash == settings.initial_capital).all()
    assert (result.equity.trend == settings.initial_capital).all()
    assert summarize(result.equity)["cash"]["total_return"] == 0


@pytest.mark.parametrize("kind", ["nan", "zero", "duplicate", "gap", "missing_asset"])
def test_bad_price_data_fail_closed(market, kind):
    prices, settings = market
    if kind == "nan":
        prices.iloc[55, 0] = np.nan
    elif kind == "zero":
        prices.iloc[55, 0] = 0
    elif kind == "duplicate":
        prices = pd.concat([prices.iloc[:20], prices.iloc[19:]])
    elif kind == "gap":
        prices = prices.drop(prices.index[50:60])
    else:
        prices = prices.drop(columns="B")
    with pytest.raises(ValueError):
        run_portfolio(prices, settings)


def test_fx_uses_division_and_never_backfills_from_future():
    dates = pd.to_datetime(["2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"A": [120, 120]}, index=dates)
    fx = pd.Series([1.2, 1.5], index=pd.to_datetime(["2020-01-02", "2020-01-06"]))
    np.testing.assert_allclose(convert_to_eur(prices, fx).A, [100, 80])
    with pytest.raises(ValueError, match="missing or stale"):
        convert_to_eur(prices, fx.iloc[1:])


def test_start_and_end_control_evaluation_without_losing_warmup(market):
    prices, settings = market
    result = run_portfolio(prices, replace(settings, start="2020-04-01", end="2020-05-15"))
    assert result.equity.index[0] == pd.Timestamp("2020-03-31")
    assert result.signals.execution_date.min() == pd.Timestamp("2020-04-01")
    assert result.equity.index[-1] == pd.Timestamp("2020-05-15")
    with pytest.raises(ValueError, match="warm-up"):
        run_portfolio(prices.iloc[:20], settings)


def test_zero_cost_benchmark_matches_analytical_buy_hold(market):
    prices, settings = market
    result = run_portfolio(prices, settings)
    execution = result.equity.index[1]
    expected = 10_000 * prices.A.loc[execution:] / prices.A.loc[execution]
    np.testing.assert_allclose(result.equity.equity_buy_hold.iloc[1:], expected)


def _write_inputs(tmp_path, market):
    prices, _ = market
    prices.rename_axis("date").to_csv(tmp_path / "prices.csv")
    config = {
        "schema_version": 1,
        "name": "test",
        "base_currency": "EUR",
        "price_basis": "total_return",
        "prices": "prices.csv",
        "weights": {"A": 0.5, "B": 0.5},
        "equity_benchmark": "A",
        "lookback_months": 2,
        "one_way_cost_bps": 5,
        "split_date": "2020-05-01",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def test_report_roundtrip_provenance_cost_stress_and_chronological_split(tmp_path, market):
    path, _ = _write_inputs(tmp_path, market)
    output = tmp_path / "result"
    report = run_experiment(path, output, cost_multiplier=2)
    assert report["effective_one_way_cost_bps"] == 10
    metrics = report["metrics"]
    for name in metrics["full"]:
        compounded = (1 + metrics["before_split"][name]["total_return"]) * (
            1 + metrics["after_split"][name]["total_return"]
        ) - 1
        assert compounded == pytest.approx(metrics["full"][name]["total_return"])
    assert len(report["provenance"]["prices"]["sha256"]) == 64
    assert report["cash_return"] == 0
    assert "cash_rates" not in report["provenance"]
    assert json.loads((output / "results.json").read_text())["status"] == "historical_research_only"
    for filename in [
        "equity.csv",
        "weights.csv",
        "trades.csv",
        "signals.csv",
        "annual_returns.csv",
        "monthly_returns.csv",
        "robustness.json",
        "robustness_folds.csv",
        "equity.png",
        "report.md",
    ]:
        assert (output / filename).stat().st_size > 0
    with pytest.raises(FileExistsError):
        run_experiment(path, output)


def test_config_typo_and_unadjusted_prices_fail_closed(tmp_path, market):
    path, config = _write_inputs(tmp_path, market)
    config["lookbak_months"] = 10
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unknown"):
        load_inputs(path)
    del config["lookbak_months"]
    config["price_basis"] = "raw"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="total_return"):
        load_inputs(path)


def test_old_cash_interest_config_is_rejected_instead_of_silently_applied(tmp_path, market):
    path, config = _write_inputs(tmp_path, market)
    config["cash_rates"] = "rates.csv"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unknown.*cash_rates"):
        load_inputs(path)


def test_existing_cli_dispatches_portfolio_help():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "fwbg.cli", "portfolio", "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "download-example" in completed.stdout


def test_ecb_holidays_are_not_prices_but_unmarked_missing_data_fail(tmp_path, monkeypatch):
    from fwbg.portfolio import download

    payload = (
        b"TIME_PERIOD,OBS_VALUE,OBS_STATUS\n2020-04-09,1.1,A\n2020-04-10,,H\n2020-04-14,1.2,A\n"
    )
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: BytesIO(payload))
    series, _ = _ecb_series("test", "2020-01-01", "2021-01-01", tmp_path / "raw.csv")
    assert list(series.index) == [pd.Timestamp("2020-04-09"), pd.Timestamp("2020-04-14")]
    payload = payload.replace(b",,H", b",,M")
    with pytest.raises(ValueError, match="non-finite"):
        _ecb_series("test", "2020-01-01", "2021-01-01", tmp_path / "raw.csv")


def test_failed_download_leaves_no_usable_partial_bundle(tmp_path, monkeypatch):
    from fwbg.portfolio import download

    def fail(output, start, end):
        output.mkdir()
        (output / "partial.csv").write_text("partial")
        raise ValueError("provider failure")

    monkeypatch.setattr(download, "_prepare_example", fail)
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="provider failure"):
        prepare_example(destination, "2004-01-01", "2026-01-01")
    assert not destination.exists()
    assert not list(tmp_path.glob(".portfolio-download-*"))


def test_full_example_preparation_retains_inception_and_data_provenance(tmp_path, monkeypatch):
    from fwbg.portfolio import download

    dates = pd.bdate_range("2019-09-02", "2019-10-31")
    columns = pd.MultiIndex.from_product([["Close"], list(download.ASSETS)])
    sample = pd.DataFrame(120.0, index=dates, columns=columns)
    sample.loc[dates[:3], ("Close", "GLD")] = np.nan
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda *a, **k: sample.copy(), __version__="test"),
    )

    def fake_ecb(key, start, end, raw_path):
        raw_path.write_text("raw source snapshot")
        assert key.startswith("EXR"), "interest rates must not be downloaded"
        result = pd.Series(1.2, index=dates)
        return result, "https://example.test/source"

    monkeypatch.setattr(download, "_ecb_series", fake_ecb)
    output = tmp_path / "bundle"
    metadata = prepare_example(output, "2018-01-01", "2020-01-01")
    assert metadata["common_start"] == dates[3].date().isoformat()
    prices = pd.read_csv(output / "prices_eur.csv", index_col="date")
    np.testing.assert_allclose(prices.to_numpy(), 100)
    assert not (output / "cash_rates.csv").exists()
    assert all((output / entry["path"]).exists() for entry in metadata["raw_files"])
    with pytest.raises(FileExistsError):
        prepare_example(output, "2018-01-01", "2020-01-01")
