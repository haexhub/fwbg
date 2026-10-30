"""The research gate on ``fwbg analyze`` (MFE/MAE exit analysis)."""

import json

import numpy as np
import pandas as pd
import pytest

from fwbg.cli._analyze import run_analyze
from fwbg.exploration.exit_analyzer import analyze_asset
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, read_trials
from fwbg.research.runs import GATE_ENV

CUTOFF = "2025-04-30"  # of the gate_register card S-0007


@pytest.fixture
def gated(tmp_path, gate_register, monkeypatch):
    monkeypatch.setenv(GATE_ENV, "on")
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    return gate_register


def write_csv(path, start="2025-04-01", n=1200):
    """Hourly bars from *start*; the default runs past the card's cutoff."""
    rng = np.random.default_rng(3)
    dates = pd.date_range(start, periods=n, freq="h")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - abs(rng.standard_normal(n)) * 0.2
    frame = pd.DataFrame({"T": dates, "O": open_, "H": high, "L": low, "C": close, "V": 100})
    frame.to_csv(path, index=False)
    return frame


def analyze(tmp_path, *args):
    csv = tmp_path / "EURUSD_HOUR_1.csv"
    if not csv.exists():
        write_csv(csv)
    run_analyze([str(csv), "--output-dir", str(tmp_path / "out"), *args])
    return json.loads((tmp_path / "out" / "EURUSD_HOUR_1.json").read_text(encoding="utf-8"))


def test_analysis_without_research_id_is_refused(tmp_path, gated, capsys):
    with pytest.raises(SystemExit) as exc:
        analyze(tmp_path, "--note", "ranges")
    assert exc.value.code == 2
    assert "--research-id" in capsys.readouterr().err
    assert read_trials(gated / LEDGER_FILENAME) == []
    assert not (tmp_path / "out").exists()


def test_analysis_reads_only_data_before_the_cutoff_and_is_recorded(tmp_path, gated):
    result = analyze(tmp_path, "--research-id", "S-0007", "--note", "TP/SL ranges")

    frame = write_csv(tmp_path / "full.csv")
    before = frame[frame["T"] < pd.Timestamp(CUTOFF)]
    before.to_csv(tmp_path / "EURUSD_PRE.csv", index=False)
    assert len(before) < len(frame)
    expected = analyze_asset(str(tmp_path / "EURUSD_PRE.csv"))
    assert result["window"] == {"start": None, "end": CUTOFF}
    assert result["bars_analyzed"] == expected["bars_analyzed"]
    assert result["suggested_grid"] == expected["suggested_grid"]

    [trial] = read_trials(gated / LEDGER_FILENAME)
    assert trial.strategy == "S-0007" and trial.kind is TrialKind.ITERATION
    assert trial.window == {"start": None, "end": CUTOFF}
    assert trial.note == "TP/SL ranges"
    # No strategy was evaluated: the analysis adds nothing to the card's trial count.
    assert trial.metrics == {}
    stored = gated / "configs" / f"{trial.fingerprint.removeprefix('sha256:')}.json"
    config = json.loads(stored.read_text(encoding="utf-8"))
    assert config["symbols"] == ["EURUSD"]
    assert config["strategy"]["analysis"] == "mfe_mae"


def test_analysis_window_past_the_cutoff_is_refused(tmp_path, gated, capsys):
    strategy = tmp_path / "s.json"
    strategy.write_text(json.dumps({"end_date": "2025-05-15"}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        analyze(tmp_path, "--strategy-file", str(strategy), "--research-id", "S-0007",
                "--note", "ranges")
    assert exc.value.code == 2
    assert "after_cutoff" in capsys.readouterr().err
    assert read_trials(gated / LEDGER_FILENAME) == []
    assert not (tmp_path / "out").exists()


def test_repeating_an_identical_analysis_needs_a_reason(tmp_path, gated, capsys):
    analyze(tmp_path, "--research-id", "S-0007", "--note", "ranges")
    with pytest.raises(SystemExit) as exc:
        analyze(tmp_path, "--research-id", "S-0007", "--note", "again")
    assert exc.value.code == 2
    assert "duplicate_fingerprint" in capsys.readouterr().err

    analyze(tmp_path, "--research-id", "S-0007", "--note", "again", "--reason", "new data file")
    assert len(read_trials(gated / LEDGER_FILENAME)) == 2


def test_unknown_card_is_refused(tmp_path, gated, capsys):
    with pytest.raises(SystemExit) as exc:
        analyze(tmp_path, "--research-id", "S-0099", "--note", "ranges")
    assert exc.value.code == 2
    assert "card_not_found" in capsys.readouterr().err
