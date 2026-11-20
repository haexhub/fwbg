"""The research gate on ``fwbg.portfolio.cli`` (run, search, swing-study)."""

import json

import numpy as np
import pandas as pd
import pytest

from fwbg.portfolio import cli
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, read_trials
from fwbg.research.runs import GATE_ENV

CUTOFF = "2025-04-30"  # of the gate_register card S-0007


@pytest.fixture
def gated(gate_register, monkeypatch):
    monkeypatch.setenv(GATE_ENV, "on")
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    return gate_register


def write_run_inputs(tmp_path, name="config.json"):
    """Daily prices from 2024 into 2025-12, beyond the card's cutoff, and a config for them."""
    dates = pd.bdate_range("2024-01-01", "2025-12-31")
    prices = pd.DataFrame({"A": 100.0 + np.arange(len(dates)), "B": 100.0}, index=dates)
    prices.rename_axis("date").to_csv(tmp_path / "prices.csv")
    config = {
        "schema_version": 1,
        "name": "gate test",
        "base_currency": "EUR",
        "price_basis": "total_return",
        "prices": "prices.csv",
        "weights": {"A": 0.5, "B": 0.5},
        "equity_benchmark": "A",
        "lookback_months": 2,
        "one_way_cost_bps": 5,
        "split_date": "2024-12-01",
    }
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


def run(path, output, *extra):
    cli.main(["run", "--config", str(path), "--output", str(output), *extra])


def test_run_without_research_id_is_refused_before_it_reads_data(
    tmp_path, gated, capsys
):
    config = write_run_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run(config, tmp_path / "out", "--note", "trend")
    assert exc.value.code == 2
    assert "missing_research_id" in capsys.readouterr().err
    assert read_trials(gated / LEDGER_FILENAME) == []
    assert not (tmp_path / "out").exists()


def test_run_reads_only_data_before_the_cutoff_and_is_recorded(tmp_path, gated):
    config = write_run_inputs(tmp_path)
    run(config, tmp_path / "out", "--research-id", "S-0007", "--note", "trend 2m")

    equity = pd.read_csv(tmp_path / "out" / "equity.csv", index_col=0, parse_dates=True)
    assert equity.index.max() < pd.Timestamp(CUTOFF)
    assert equity.index.max() >= pd.Timestamp("2025-04-01")
    report = json.loads((tmp_path / "out" / "results.json").read_text())
    assert report["data_end_exclusive"] == CUTOFF

    [trial] = read_trials(gated / LEDGER_FILENAME)
    assert trial.strategy == "S-0007" and trial.kind is TrialKind.ITERATION
    assert trial.note == "trend 2m" and trial.metrics == {}
    assert trial.window["end"] == CUTOFF


def test_identical_run_needs_a_reason_and_a_rename_does_not_make_it_new(
    tmp_path, gated, capsys
):
    config = write_run_inputs(tmp_path)
    run(config, tmp_path / "one", "--research-id", "S-0007", "--note", "first")
    renamed = json.loads(config.read_text()) | {"name": "another name"}
    config.write_text(json.dumps(renamed))
    with pytest.raises(SystemExit) as exc:
        run(config, tmp_path / "two", "--research-id", "S-0007", "--note", "again")
    assert exc.value.code == 2
    assert "duplicate_fingerprint" in capsys.readouterr().err
    run(
        config,
        tmp_path / "three",
        "--research-id",
        "S-0007",
        "--note",
        "again",
        "--reason",
        "x",
    )
    assert len(read_trials(gated / LEDGER_FILENAME)) == 2


def test_a_different_cost_multiplier_is_a_different_run(tmp_path, gated):
    config = write_run_inputs(tmp_path)
    args = ("--research-id", "S-0007", "--note", "costs")
    run(config, tmp_path / "one", *args)
    run(config, tmp_path / "two", *args, "--cost-multiplier", "2")
    assert len(read_trials(gated / LEDGER_FILENAME)) == 2


def test_search_is_gated_and_gets_the_cutoff_as_its_data_end(
    tmp_path, gated, monkeypatch, capsys
):
    config = write_run_inputs(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"base_config": config.name, "candidates": []}))
    seen = {}

    def fake_search(path, output, end_date=None):
        seen["end_date"] = end_date

    monkeypatch.setattr("fwbg.portfolio.search.run_search", fake_search)
    argv = ["search", "--protocol", str(protocol), "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2 and "missing_research_id" in capsys.readouterr().err
    assert seen == {}

    cli.main([*argv, "--research-id", "S-0007", "--note", "search"])
    assert seen == {"end_date": CUTOFF}
    [trial] = read_trials(gated / LEDGER_FILENAME)
    assert trial.strategy == "S-0007"


def swing_protocol(tmp_path, evaluation_end):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "manifest.json").write_text("{}")
    protocol = tmp_path / "swing.json"
    protocol.write_text(
        json.dumps(
            {
                "data_directory": "data",
                "warmup_start": "2015-01-01",
                "evaluation_start": "2016-01-01",
                "evaluation_end": evaluation_end,
                "markets": [{"symbol": "DAX"}],
            }
        )
    )
    return protocol


def test_swing_study_past_the_cutoff_is_refused(tmp_path, gated, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(
        "fwbg.portfolio.swing_study.run_swing_study", lambda *a: called.append(a)
    )
    protocol = swing_protocol(tmp_path, "2025-12-31")
    argv = [
        "swing-study",
        "--protocol",
        str(protocol),
        "--output",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--research-id", "S-0007", "--note", "swing"])
    assert exc.value.code == 2
    assert "after_cutoff" in capsys.readouterr().err
    assert called == [] and read_trials(gated / LEDGER_FILENAME) == []


def test_swing_study_ending_before_the_cutoff_is_recorded_and_runs(
    tmp_path, gated, monkeypatch
):
    called = []
    monkeypatch.setattr(
        "fwbg.portfolio.swing_study.run_swing_study", lambda *a: called.append(a)
    )
    protocol = swing_protocol(tmp_path, "2025-04-29")
    cli.main(
        [
            "swing-study",
            "--protocol",
            str(protocol),
            "--output",
            str(tmp_path / "out"),
            "--research-id",
            "S-0007",
            "--note",
            "swing",
        ]
    )
    assert len(called) == 1
    [trial] = read_trials(gated / LEDGER_FILENAME)
    assert trial.window["end"] == "2025-04-30"
