"""The run files of a holdout look and what the ledger reads from them."""

import json

from fwbg.optimization import holdout
from fwbg.optimization.signal_selection import SELECTION_PROTOCOL
from fwbg.research.legacy import run_outcome
from fwbg.research.ledger import TrialStatus
from tests.optimization.symbol_harness import (
    CUTOFF,
    LOSING,
    WINNING,
    patch_development,
    patch_holdout,
    run_inline,
    run_inline_result,
)

# Filters every harness result passes, so only the holdout rule can stop an export.
PERMISSIVE = {
    "min_annual_return": -100.0,
    "min_sharpe": -100.0,
    "max_drawdown": 2.0,
    "min_fold_stability": 0.0,
}


def test_holdout_look_records_holdout_metrics_and_keeps_development_apart(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    run_dir = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    sym_dir = run_dir / "grid_details" / "EURUSD"
    config = json.loads((sym_dir / "config.json").read_text())
    unified = json.loads((sym_dir / "unified_metrics.json").read_text())
    development = json.loads((sym_dir / "development.json").read_text())
    assert config["status"] == "ok"
    assert config["holdout"]["cutoff"] == CUTOFF and config["holdout"]["evaluated"] is True
    assert not (sym_dir / "fold_results.json").exists()
    assert development["status"] == "ok" and development["fold_stability"] is not None
    assert unified["trades"] == config["holdout"]["sized_trades"]
    assert (unified["n_long"], unified["n_short"]) == (30, 0)  # the holdout's trade directions
    status, metrics = run_outcome(run_dir)
    assert status is TrialStatus.OK
    symbol = metrics["EURUSD"]
    assert symbol["fold_stability"] is None
    assert symbol["p_value"] == config["monte_carlo"]["p_value"]
    assert (symbol["sharpe"], symbol["sr_per_trade"]) == (
        unified["sharpe"],
        unified["sr_per_trade"],
    )
    assert symbol["sharpe"] != development["sharpe"]  # the holdout's, not the development's
    elite = json.loads((run_dir / "results.json").read_text())["elite_results"]
    assert elite[0]["fold_stability"] is None


def test_holdout_after_a_failed_development_records_no_metrics(monkeypatch, tmp_path, capsys):
    patch_development(monkeypatch, LOSING)
    patch_holdout(monkeypatch)
    result, run_dir = run_inline_result(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    # Like the ledger: no symbol reached its holdout, so the run failed (exit 1).
    assert result is None
    assert "no symbol reached its holdout" in capsys.readouterr().out
    sym_dir = run_dir / "grid_details" / "EURUSD"
    config = json.loads((sym_dir / "config.json").read_text())
    assert config["status"] == "no_edge"
    assert config["holdout"] == {
        "cutoff": CUTOFF,
        "evaluated": False,
        "reason": "development ended in no_edge",
    }
    assert not (sym_dir / "unified_metrics.json").exists()
    assert json.loads((sym_dir / "development.json").read_text())["status"] == "no_edge"
    assert run_outcome(run_dir) == (TrialStatus.FAILED, {})  # nothing was looked at


def test_a_holdout_with_no_trades_was_still_looked_at(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch, pairs=[])
    run_dir = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    config = json.loads((run_dir / "grid_details" / "EURUSD" / "config.json").read_text())
    assert (config["status"], config["holdout"]["evaluated"]) == ("no_unified_trades", True)
    status, metrics = run_outcome(run_dir)
    assert (status, list(metrics)) == (TrialStatus.OK, ["EURUSD"])


def test_a_holdout_that_failed_after_its_trades_counts(monkeypatch, tmp_path):
    def crash(*args):
        raise RuntimeError("boom")

    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    monkeypatch.setattr(holdout, "_holdout_result", crash)
    result, run_dir = run_inline_result(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    assert result is not None  # the look happened: the run completes (exit 0), as recorded
    config = json.loads((run_dir / "grid_details" / "EURUSD" / "config.json").read_text())
    assert (config["status"], config["holdout"]["evaluated"]) == ("error", True)
    status, metrics = run_outcome(run_dir)
    assert (status, list(metrics)) == (TrialStatus.OK, ["EURUSD"])


def test_development_file_carries_the_frozen_unified_setting(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    calls = patch_holdout(monkeypatch)
    run_dir = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    development = json.loads((run_dir / "grid_details" / "EURUSD" / "development.json").read_text())
    # JSON round trip of what the holdout evaluated (its params tuple becomes a list).
    assert development["unified_candidate"] == json.loads(json.dumps(calls[0]["candidate"]))


def test_a_run_without_holdout_writes_no_development_file(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    run_dir = run_inline(monkeypatch, tmp_path)
    assert not (run_dir / "grid_details" / "EURUSD" / "development.json").exists()


def test_holdout_look_never_exports_assets(monkeypatch, tmp_path, capsys):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    run_dir = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF, filters=PERMISSIVE)
    assert (
        json.loads((run_dir / "grid_details" / "EURUSD" / "config.json").read_text())["status"]
        == "ok"
    )
    assert json.loads((run_dir / "assets.json").read_text()) == {}
    [row] = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| EURUSD")]
    assert row.rstrip().endswith("| HOLDOUT  |")  # the elite table's status column


def test_the_same_filters_export_a_development_run(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    run_dir = run_inline(monkeypatch, tmp_path, filters=PERMISSIVE)
    assert list(json.loads((run_dir / "assets.json").read_text())) == ["EURUSD"]


def test_holdout_look_keeps_the_developments_selection_protocol(monkeypatch, tmp_path):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    run_dir = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    development = json.loads((run_dir / "grid_details" / "EURUSD" / "development.json").read_text())
    assert development["walk_forward"]["selection_protocol"] == SELECTION_PROTOCOL == 2
    assert run_outcome(run_dir)[1]["EURUSD"]["selection_protocol"] == 2
