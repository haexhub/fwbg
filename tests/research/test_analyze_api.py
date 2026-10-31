"""The research gate on ``POST /api/exploration/exit-optimization``.

The endpoint records the analysis like ``fwbg analyze`` before it reads data,
reads only data before the card's cutoff and refuses like ``POST /api/runs/start``.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fwbg.core.data_sources as ds_mod
from fwbg.api import create_app
from fwbg.cli._analyze import run_analyze
from fwbg.core.data_sources import CSVSourceConfig
from fwbg.data import config as data_config
from fwbg.exploration.exit_analyzer import analyze_asset
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, read_trials
from fwbg.research.runs import GATE_ENV
from tests.research.test_analyze_cli import CUTOFF, write_csv

URL = "/api/exploration/exit-optimization"


@pytest.fixture
def api(tmp_path, gate_register, monkeypatch):
    monkeypatch.setenv(GATE_ENV, "on")
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    source = CSVSourceConfig(name="t1", path=tmp_path / "data")
    source.path.mkdir()
    write_csv(source.path / "EURUSD_HOUR.csv")
    monkeypatch.setitem(ds_mod._DATA_SOURCES, "t1", source)
    with TestClient(create_app()) as client:
        yield client


def post(api, **fields):
    return api.post(URL, json={"source": "t1", "symbol": "EURUSD", **fields})


def test_analysis_without_research_id_is_refused(api, gate_register, tmp_path):
    response = post(api, note="ranges")
    assert response.status_code == 422
    assert [d["code"] for d in response.json()["detail"]["denials"]] == ["missing_research_id"]
    assert read_trials(gate_register / LEDGER_FILENAME) == []
    assert not (tmp_path / "results" / "exploration").exists()


def test_analysis_reads_only_data_before_the_cutoff_and_is_recorded(api, gate_register, tmp_path):
    response = post(api, research_id="S-0007", note="TP/SL ranges")
    assert response.status_code == 200
    result = response.json()

    frame = pd.read_csv(tmp_path / "data" / "EURUSD_HOUR.csv", parse_dates=["T"])
    frame[frame["T"] < pd.Timestamp(CUTOFF)].to_csv(tmp_path / "pre.csv", index=False)
    assert result["window"] == {"start": None, "end": CUTOFF}
    assert result["bars_analyzed"] == analyze_asset(str(tmp_path / "pre.csv"))["bars_analyzed"]

    [trial] = read_trials(gate_register / LEDGER_FILENAME)
    assert trial.strategy == "S-0007" and trial.kind is TrialKind.ITERATION
    assert trial.note == "TP/SL ranges" and trial.metrics == {}


def test_a_repeat_conflicts_and_runs_with_a_reason(api, gate_register):
    assert post(api, research_id="S-0007", note="ranges").status_code == 200
    response = post(api, research_id="S-0007", note="again")
    assert response.status_code == 409
    assert "duplicate_fingerprint" in [d["code"] for d in response.json()["detail"]["denials"]]

    assert post(api, research_id="S-0007", note="again", reason="new data").status_code == 200
    assert len(read_trials(gate_register / LEDGER_FILENAME)) == 2


def test_the_cli_and_the_api_record_the_same_analysis_alike(
    api, gate_register, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(data_config, "DATA_PATH", data_config.DATA_PATH)  # the CLI sets it
    assert post(api, research_id="S-0007", note="from the dashboard").status_code == 200
    strategy = tmp_path / "s.json"
    strategy.write_text('{"datasource": "t1"}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_analyze(["EURUSD_HOUR.csv", "--strategy-file", str(strategy),
                     "--output-dir", str(tmp_path / "out"),
                     "--research-id", "S-0007", "--note", "from the CLI"])
    # The same analysis is a duplicate whichever path ran it first.
    assert exc.value.code == 2
    assert "duplicate_fingerprint" in capsys.readouterr().err
    assert len(read_trials(gate_register / LEDGER_FILENAME)) == 1
