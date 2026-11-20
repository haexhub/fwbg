"""The research gate on ``POST /api/runs/preview`` and its TP/SL handling."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fwbg.api.runs as runs_mod
import fwbg.core.data_sources as ds_mod
import fwbg.pipeline.features as features_mod
from fwbg.api import create_app
from fwbg.core.data_sources import CSVSourceConfig
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, read_trials
from fwbg.research.runs import GATE_ENV
from tests.research.test_analyze_cli import CUTOFF, write_csv

URL = "/api/runs/preview"
STRATEGY = {
    "name": "preview gate",
    "datasource": "t1",
    "timeframe": "HOUR",
    "pipeline": {"indicators": [{"name": "dummy"}]},
    "signal_rules": {"long": {"conditions": [{"column": "sig"}]}},
}


@pytest.fixture
def api(tmp_path, gate_register, monkeypatch):
    monkeypatch.setenv(GATE_ENV, "on")
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "pg.json").write_text(json.dumps(STRATEGY), encoding="utf-8")
    monkeypatch.setattr(runs_mod, "get_strategies_dir", lambda: strategies)
    source = CSVSourceConfig(name="t1", path=tmp_path / "data")
    source.path.mkdir()
    write_csv(source.path / "EURUSD_HOUR.csv")
    monkeypatch.setitem(ds_mod._DATA_SOURCES, "t1", source)

    seen: dict[str, pd.DataFrame] = {}

    def fake_pool(df, indicators):
        seen["df"] = df
        out = df.copy()
        out["sig"] = 0.0
        return out

    monkeypatch.setattr(features_mod, "compute_indicator_pool", fake_pool)
    with TestClient(create_app()) as client:
        client.seen = seen
        yield client


def post(api, **fields):
    return api.post(URL, json={"strategy_name": "pg", "symbol": "EURUSD", **fields})


def test_preview_without_research_id_is_refused_before_data_is_read(api, gate_register):
    response = post(api, note="look")
    assert response.status_code == 422
    assert [d["code"] for d in response.json()["detail"]["denials"]] == [
        "missing_research_id"
    ]
    assert read_trials(gate_register / LEDGER_FILENAME) == []
    assert "df" not in api.seen


def test_preview_without_note_is_refused(api):
    response = post(api, research_id="S-0007")
    assert response.status_code == 422
    assert "missing_note" in [d["code"] for d in response.json()["detail"]["denials"]]


def test_preview_reads_only_data_before_the_cutoff_and_is_recorded(api, gate_register):
    response = post(api, research_id="S-0007", note="signals look sane")
    assert response.status_code == 200
    df = api.seen["df"]
    assert len(df) > 0
    assert df.index.max() < pd.Timestamp(CUTOFF)
    assert response.json()["total_bars"] == len(df)

    [trial] = read_trials(gate_register / LEDGER_FILENAME)
    assert trial.strategy == "S-0007" and trial.kind is TrialKind.ITERATION
    assert trial.note == "signals look sane"
    assert trial.window["end"] == CUTOFF


def test_preview_cannot_reach_holdout_data_by_asking_for_it(api):
    # The data file runs past the cutoff; the full history is not reachable.
    response = post(api, research_id="S-0007", note="all of it", last_n_bars=10_000_000)
    assert response.status_code == 200
    assert api.seen["df"].index.max() < pd.Timestamp(CUTOFF)


def test_identical_preview_is_a_duplicate_without_a_reason(api, gate_register):
    assert post(api, research_id="S-0007", note="first").status_code == 200
    response = post(api, research_id="S-0007", note="again")
    assert response.status_code == 409
    assert "duplicate_fingerprint" in [
        d["code"] for d in response.json()["detail"]["denials"]
    ]
    assert (
        post(api, research_id="S-0007", note="again", reason="new idea").status_code
        == 200
    )
    assert len(read_trials(gate_register / LEDGER_FILENAME)) == 2


def test_different_tp_is_a_different_preview(api):
    assert post(api, research_id="S-0007", note="a", tp=1.5, sl=1.0).status_code == 200
    assert post(api, research_id="S-0007", note="b", tp=2.5, sl=1.0).status_code == 200


def test_preview_simulation_keeps_fractional_tp_and_sl(monkeypatch):
    import fwbg.core.context as context_mod
    import fwbg.optimization.targets as targets_mod

    captured: dict[str, float] = {}

    def fake_core(df, **kwargs):
        captured.update(tp=kwargs["tp"], sl=kwargs["sl"])
        return {"trades_detailed": []}

    monkeypatch.setattr(targets_mod, "_simulate_trades_core", fake_core)
    monkeypatch.setattr(
        context_mod.SimulationContext, "create", staticmethod(lambda a, s: object())
    )
    df = pd.DataFrame(
        {"sig": [0.0, 1.0]}, index=pd.date_range("2024-01-01", periods=2, freq="h")
    )
    runs_mod._simulate_preview_trades(
        df, {"sig"}, set(), 1.5, 0.5, SimpleNamespace(), SimpleNamespace()
    )
    assert captured == {"tp": 1.5, "sl": 0.5}
