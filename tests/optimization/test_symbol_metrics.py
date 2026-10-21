import json

import pytest

from fwbg.optimization import process
from fwbg.optimization.process import process_symbol
from fwbg.optimization.selection_protocol import SELECTION_PROTOCOL
from fwbg.research.legacy import run_outcome
from fwbg.results.storage import save_run_results
from tests.optimization import symbol_harness
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


def _zero_trade_folds(monkeypatch, traded):
    """Folds whose chosen setting traded nothing (kept, D2) unless *traded(fold)*."""
    canned = symbol_harness.fold_result

    def fold_result(fold):
        if traded(fold):
            return canned(fold)
        return {**canned(fold), "test_win_rate": 0.0, "test_pnl": 0.0, "test_trades": 0}

    monkeypatch.setattr(symbol_harness, "fold_result", fold_result)
    patch_development(monkeypatch, WINNING)


def test_zero_trade_folds_count_as_folds_but_have_no_win_rate(monkeypatch, tmp_path):
    _zero_trade_folds(monkeypatch, lambda fold: fold.fold_id != 0)
    run_dir = run_inline(monkeypatch, tmp_path)
    fold = json.loads((run_dir / "grid_details" / "EURUSD" / "fold_results.json").read_text())
    wf = fold["walk_forward"]
    assert wf["n_folds"] > 1
    assert (wf["mean_win_rate"], wf["std_win_rate"]) == (0.6, 0.0)
    assert (wf["min_win_rate"], wf["max_win_rate"]) == (0.6, 0.6)
    assert wf["fold_stability"] == (wf["n_folds"] - 1) / wf["n_folds"]


def test_folds_without_any_trade_leave_the_win_rate_undefined(monkeypatch, tmp_path):
    _zero_trade_folds(monkeypatch, lambda fold: False)
    run_dir = run_inline(monkeypatch, tmp_path)
    fold = json.loads((run_dir / "grid_details" / "EURUSD" / "fold_results.json").read_text())
    wf = fold["walk_forward"]
    assert [wf[k] for k in ("mean_win_rate", "std_win_rate", "min_win_rate", "max_win_rate")] == [
        None
    ] * 4
    assert wf["fold_stability"] == 0.0


def _patch_no_successful_folds(monkeypatch):
    """Development where no fold yields a result; one signal grid entry (ct 0.6)."""
    patch_development(monkeypatch, WINNING)
    entry = {"tp_mult": 2, "sl_mult": 4, "ct": 0.6, "inner_val_pnl": 0.0, "inner_trades": 0}

    def no_folds(variant_list, df, wf_folds, asset, sym):
        label, variant = variant_list[0]
        return {
            "label": label,
            "ctx": None,
            "all_fold_results": [],
            "accumulated_grid_results": [entry],
            "fold_indicators": [],
            "precomputed_raw_df": None,
            "preprocessing_configs": [],
        }

    monkeypatch.setattr(process, "_run_indicator_variants", no_folds)


def test_no_successful_folds_names_a_signal_entrys_ct(monkeypatch, capsys):
    _patch_no_successful_folds(monkeypatch)
    result = process_symbol(SYMBOL_FILE, STRATEGY)
    assert result["status"] == "no_successful_folds"
    assert "CT=0.6," in capsys.readouterr().err


def _symbol_config(run_dir):
    return json.loads((run_dir / "grid_details" / "EURUSD" / "config.json").read_text())


def test_a_run_without_successful_folds_records_the_selection_protocol(monkeypatch, tmp_path):
    _patch_no_successful_folds(monkeypatch)
    config = _symbol_config(run_inline(monkeypatch, tmp_path))
    assert (config["status"], config["selection_protocol"]) == ("no_successful_folds", 2)


def test_a_run_without_unified_trades_records_the_selection_protocol(monkeypatch, tmp_path):
    patch_development(monkeypatch, [])
    run_dir = run_inline(monkeypatch, tmp_path)
    assert (_symbol_config(run_dir)["status"], _symbol_config(run_dir)["selection_protocol"]) == (
        "no_unified_trades",
        2,
    )
    assert run_outcome(run_dir)[1]["EURUSD"]["selection_protocol"] == 2


def test_an_errored_symbol_carries_the_selection_protocol(monkeypatch):
    def broken(path):
        raise RuntimeError("boom")

    patch_development(monkeypatch, WINNING)
    monkeypatch.setattr(process, "load_data_aligned", broken)
    result = process_symbol(SYMBOL_FILE, STRATEGY)
    assert (result["status"], result["selection_protocol"]) == ("error", SELECTION_PROTOCOL)


def test_the_fallback_symbol_config_carries_the_register_blocks(tmp_path):
    # save_run_results writes grid_details only for symbols the live callback missed.
    result = {
        "symbol": "EURUSD",
        "status": "no_successful_folds",
        "grid_results": [{"pnl": 0.0}],
        "selection_protocol": SELECTION_PROTOCOL,
    }
    save_run_results(str(tmp_path), [], [], [], [], [], all_results=[result])
    config = json.loads((tmp_path / "grid_details" / "EURUSD" / "config.json").read_text())
    assert config["selection_protocol"] == SELECTION_PROTOCOL
