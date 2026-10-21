import json

import pytest

from fwbg.optimization.process import process_symbol
from fwbg.optimization.signal_selection import SELECTION_PROTOCOL
from fwbg.research.legacy import run_outcome
from tests.optimization.symbol_harness import (
    LOSING,
    STRATEGY,
    SYMBOL_FILE,
    WINNING,
    patch_development,
    run_inline,
)


def test_no_edge_has_no_sharpe_and_no_return_statistics(monkeypatch):
    patch_development(monkeypatch, LOSING)
    result = process_symbol(SYMBOL_FILE, STRATEGY)
    assert result["status"] == "no_edge"
    assert result["sharpe"] is None
    assert "return_stats" not in result


def test_reported_results_carry_the_statistics_of_their_sharpe_returns(monkeypatch):
    patch_development(monkeypatch, WINNING)
    result = process_symbol(SYMBOL_FILE, STRATEGY)
    assert result["status"] == "ok"
    stats = result["return_stats"]
    assert stats["n_returns"] == result["reporting"]["n_trades"]
    # The same series the run's per-trade Sharpe (overfitting.dsr.observed_sr) came from.
    assert stats["sr_per_trade"] == pytest.approx(result["overfitting"]["dsr"]["observed_sr"])


def test_run_writes_return_statistics_and_the_overfitting_block(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    sym_dir = run_inline(monkeypatch, tmp_path) / "grid_details" / "EURUSD"
    unified = json.loads((sym_dir / "unified_metrics.json").read_text())
    assert {"sr_per_trade", "skew", "kurtosis", "n_returns", "test_period_years"} <= unified.keys()
    config = json.loads((sym_dir / "config.json").read_text())
    assert {"dsr", "pbo"} <= config["overfitting"].keys()


def test_run_writes_a_null_sharpe_for_no_edge(monkeypatch, tmp_path):
    patch_development(monkeypatch, LOSING)
    sym_dir = run_inline(monkeypatch, tmp_path) / "grid_details" / "EURUSD"
    assert json.loads((sym_dir / "unified_metrics.json").read_text())["sharpe"] is None
    assert json.loads((sym_dir / "config.json").read_text())["metrics"]["sharpe"] is None
    assert "sr_per_trade" not in json.loads((sym_dir / "unified_metrics.json").read_text())


def test_run_records_the_selection_protocol_per_symbol(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    run_dir = run_inline(monkeypatch, tmp_path)
    fold = json.loads((run_dir / "grid_details" / "EURUSD" / "fold_results.json").read_text())
    assert fold["walk_forward"]["selection_protocol"] == SELECTION_PROTOCOL == 2
    assert run_outcome(run_dir)[1]["EURUSD"]["selection_protocol"] == 2
