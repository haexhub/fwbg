"""The holdout look (fwbg.optimization.holdout): develop before the cutoff, evaluate once after it."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig, ValidationConfig
from fwbg.optimization import holdout, process
from fwbg.optimization.causal_reporting import sized_returns
from fwbg.optimization.holdout import HOLDOUT_STATUSES, process_symbol_holdout, symbol_worker
from fwbg.optimization.process import process_symbol
from fwbg.optimization.signal_fold import prepare_signal_fold_data
from fwbg.research.dsr import return_statistics
from fwbg.simulation.trade import monte_carlo_permutation_test
from fwbg.statuses import EXECUTED_STATUSES, NOT_EXECUTED_STATUSES
from tests.optimization.symbol_harness import (
    CUTOFF,
    HOLDOUT,
    LOSING,
    STRATEGY,
    SYMBOL_FILE,
    WINNING,
    hourly_frame,
    patch_development,
    patch_holdout,
    run_inline,
)

LOOK = replace(STRATEGY, holdout_cutoff=CUTOFF)
NOT_REPORTED = {
    "sharpe",
    "fold_stability",
    "tr_trace",
    "monte_carlo",
    "walk_forward",
    "reporting",
    "return_stats",
}


def test_holdout_cutoff_parses_from_the_metadata():
    assert StrategyConfig.from_dict({"holdout_cutoff": CUTOFF}).holdout_cutoff == CUTOFF
    assert StrategyConfig.from_dict({}).holdout_cutoff is None


def test_worker_is_the_holdout_only_with_a_cutoff():
    assert symbol_worker(LOOK).func is process_symbol_holdout
    assert symbol_worker(STRATEGY).func is process_symbol
    assert symbol_worker(LOOK).keywords == {"strategy": LOOK}


def test_development_sees_only_data_before_the_cutoff(monkeypatch):
    seen = []

    def development(path, strategy):
        seen.append(strategy)
        return {"symbol": "EURUSD", "status": "no_edge"}

    monkeypatch.setattr(holdout, "process_symbol", development)
    process_symbol_holdout(SYMBOL_FILE, replace(LOOK, start_date="2022-02-01"))
    [strategy] = seen
    assert (strategy.start_date, strategy.end_date, strategy.holdout_cutoff) == (
        "2022-02-01",
        CUTOFF,
        None,
    )


def test_development_frame_ends_before_the_cutoff(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    frames = []
    variants = process._run_indicator_variants

    def spy(variant_list, df, wf_folds, asset, sym):
        frames.append(df)
        return variants(variant_list, df, wf_folds, asset, sym)

    monkeypatch.setattr(process, "_run_indicator_variants", spy)
    process_symbol_holdout(SYMBOL_FILE, LOOK)
    [frame] = frames
    assert frame.index[-1] < pd.Timestamp(CUTOFF)
    assert frame.index[-1] == hourly_frame().index[hourly_frame().index < pd.Timestamp(CUTOFF)][-1]


@pytest.mark.parametrize(
    "status", ["no_edge", "not_significant", "no_successful_folds", "no_unified_trades", "error"]
)
def test_failed_development_records_its_outcome_and_never_looks(monkeypatch, status):
    monkeypatch.setattr(
        holdout,
        "process_symbol",
        lambda path, strategy: {
            "symbol": "EURUSD",
            "status": status,
            "sharpe": 0.3,
            "fold_stability": 0.5,
            "error": "boom",
        },
    )
    monkeypatch.setattr(holdout, "load_symbol_frame", pytest.fail)
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == status
    assert result["holdout"] == {
        "cutoff": CUTOFF,
        "evaluated": False,
        "reason": f"development ended in {status}",
    }
    assert result["development"]["status"] == status
    assert result["error"] == "boom"
    assert not NOT_REPORTED & result.keys()


def test_holdout_trains_on_development_minus_embargo_and_tests_after_the_cutoff(monkeypatch):
    look = replace(LOOK, validation=ValidationConfig(folds=4, embargo_bars=5))
    patch_development(monkeypatch, WINNING)
    calls = patch_holdout(monkeypatch)
    development = process_symbol(SYMBOL_FILE, replace(look, end_date=CUTOFF, holdout_cutoff=None))
    result = process_symbol_holdout(SYMBOL_FILE, look)
    [call] = calls
    [fold] = call["folds"]
    frame = hourly_frame()
    n_dev = int(frame.index.searchsorted(pd.Timestamp(CUTOFF)))
    assert fold.train_df.index[0] == frame.index[0]
    assert fold.train_df.index[-1] == frame.index[n_dev - 6]  # 5 embargo bars left out
    assert fold.test_df.index[0] == frame.index[n_dev] >= pd.Timestamp(CUTOFF)
    assert frame.index[n_dev - 1] < pd.Timestamp(CUTOFF)
    assert fold.test_df.index[-1] == frame.index[-1]
    assert (fold.train_start, fold.train_end, fold.test_start, fold.test_end) == (
        0,
        n_dev - 5,
        n_dev,
        len(frame),
    )
    assert call["candidate"] == development["unified_candidate"]
    assert call["ctx"].embargo_bars == 5
    assert call["prepare"] is None  # ML preparation
    assert result["holdout"]["embargo_bars"] == 5
    assert result["holdout"]["bars"] == len(frame) - n_dev


def test_holdout_metrics_use_development_sizing_and_only_holdout_trades(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    development = process_symbol(SYMBOL_FILE, replace(STRATEGY, end_date=CUTOFF))
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    config = development["config"]
    trades = [{"pnl_raw": pnl, "result": res} for pnl, res in HOLDOUT]
    returns, pnl, _ = sized_returns(
        trades, config["risk_per_trade"], config["circuit_breaker"], development["tr_trace"]
    )
    losses = [-p for p in development["tr_trace"] if p < 0]
    assert returns[0] == pytest.approx(config["risk_per_trade"] * 2.0 / (sum(losses) / len(losses)))
    assert result["status"] == "ok"
    assert result["reporting"]["pnl_trace"] == pnl
    assert result["reporting"]["equity"]["equity_points"]
    assert result["tr_trace"] == [p for p, _ in HOLDOUT]
    assert result["win_rate"] == pytest.approx(0.7)
    assert result["return_stats"] == return_statistics(returns)
    assert result["monte_carlo"]["p_value"] == monte_carlo_permutation_test(pnl)["p_value"]
    assert result["config"] == config and result["rrr"] == development["rrr"]
    assert result["test_period_years"] == pytest.approx(result["holdout"]["bars"] / (24 * 250))
    assert len(result["trades_detailed"]) == len(HOLDOUT)
    assert not {"fold_stability", "walk_forward", "grid_results", "overfitting"} & result.keys()
    assert result["development"]["fold_stability"] == development["fold_stability"]
    assert result["development"]["sharpe"] == development["sharpe"]
    assert "fold_details" not in result["development"]["walk_forward"]
    assert result["holdout"]["evaluated"] is True and result["holdout"]["cutoff"] == CUTOFF
    # The development circuit breaker pauses some holdout trades.
    assert config["circuit_breaker"]["enabled"] and len(pnl) < len(HOLDOUT)
    assert (result["holdout"]["trades"], result["holdout"]["sized_trades"]) == (
        len(HOLDOUT),
        len(pnl),
    )


def test_holdout_without_trades_is_no_unified_trades(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(
        monkeypatch,
        evaluations=[{"trades": [], "trades_detailed": [], "pnl": 0, "win_rate": 0, "n_trades": 0}],
    )
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == "no_unified_trades" and result["holdout"]["evaluated"] is True
    assert not NOT_REPORTED & result.keys()


def test_too_short_holdout_window_is_not_evaluated(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch, evaluations=[])
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == "insufficient_data"
    assert result["holdout"]["evaluated"] is False


def test_no_data_after_the_cutoff_is_not_evaluated(monkeypatch):
    patch_development(monkeypatch, WINNING)
    calls = patch_holdout(monkeypatch)
    # The frame ends 2022-11-30 07:00: development covers every bar.
    result = process_symbol_holdout(SYMBOL_FILE, replace(LOOK, holdout_cutoff="2022-12-01"))
    assert result["status"] == "no_data" and result["holdout"]["evaluated"] is False
    assert calls == []


def test_a_crash_in_the_holdout_evaluation_is_this_symbols_error(monkeypatch):
    patch_development(monkeypatch, WINNING)

    def crash(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(holdout, "run_unified_simulation", crash)
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == "error"
    assert result["holdout"]["evaluated"] is False
    assert "model exploded" in result["holdout"]["reason"]
    assert "model exploded" in result["error"]
    assert result["development"]["status"] == "ok"
    assert not NOT_REPORTED & result.keys()


def test_signal_models_prepare_the_holdout_like_their_folds(monkeypatch):
    signal = replace(
        LOOK,
        model=replace(LOOK.model, type="signal"),
        filters=replace(LOOK.filters, allowed_hours=[8, 9]),
    )
    patch_development(monkeypatch, WINNING)
    calls = patch_holdout(monkeypatch)
    process_symbol_holdout(SYMBOL_FILE, signal)
    assert calls[0]["prepare"] is prepare_signal_fold_data


def test_every_holdout_status_is_classified():
    assert HOLDOUT_STATUSES <= EXECUTED_STATUSES | NOT_EXECUTED_STATUSES


@pytest.mark.parametrize("pairs, status", [(WINNING, "ok"), (LOSING, "no_edge")])
def test_a_holdout_run_completes_whether_or_not_it_looked(monkeypatch, tmp_path, pairs, status):
    patch_development(monkeypatch, pairs)
    patch_holdout(monkeypatch)
    run = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    config = json.loads((run / "grid_details" / "EURUSD" / "config.json").read_text("utf-8"))
    assert config["status"] == status
