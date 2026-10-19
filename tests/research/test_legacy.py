import json

import pytest

from fwbg.research.ledger import TrialKind, TrialStatus
from fwbg.research.legacy import LegacyRunError, load_run, scan_runs
from tests.research.factories import make_run


@pytest.fixture
def preset_base(tmp_path):
    base = tmp_path / "workspace-strategies"
    (base / "configs").mkdir(parents=True)
    (base / "pipelines").mkdir()
    (base / "pipelines" / "orb_simple_v1.json").write_text(
        json.dumps({"indicators": [{"name": "opening_range"}]}), encoding="utf-8"
    )
    return base


def test_reads_metrics_status_code_and_window(tmp_path, preset_base):
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": {"sharpe": 0.14, "trades": 37, "max_drawdown": 0.19, "noise": 1}},
        elite=[{"symbol": "GOLD", "monte_carlo": {"p_value": 0.032}, "fold_stability": 0.67}],
        strategy_extra={"start_date": "2020-01-01", "end_date": "2025-01-01"},
    )
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.OK
    assert run.kind is TrialKind.LEGACY
    assert run.symbols == ("GOLD",)
    assert run.metrics["GOLD"]["sharpe"] == 0.14
    assert run.metrics["GOLD"]["p_value"] == 0.032
    assert run.metrics["GOLD"]["fold_stability"] == 0.67
    assert "noise" not in run.metrics["GOLD"]
    assert run.window == {"start": "2020-01-01", "end": "2025-01-01"}
    assert run.code == {"commit": "abc123", "branch": "main", "dirty": True}
    assert run.unresolved is None and run.config is not None


def test_every_evaluated_symbol_counts_not_only_the_elite(tmp_path, preset_base):
    run = load_run(
        make_run(tmp_path, "run-1", symbols={"GOLD": {"sharpe": 0.5}, "EURUSD": {"sharpe": -0.2}}),
        preset_base,
    )
    assert set(run.metrics) == {"GOLD", "EURUSD"}
    assert run.metrics["EURUSD"]["p_value"] is None


def test_run_without_symbol_metrics_is_failed(tmp_path, preset_base):
    run = load_run(make_run(tmp_path, "run-1", symbols={"GOLD": None}), preset_base)
    assert run.status is TrialStatus.FAILED
    assert run.metrics == {}
    assert run.symbols == ("GOLD",)


def test_agent_smoke_runs_are_smoke(tmp_path, preset_base):
    run = load_run(make_run(tmp_path, "run-1", name="m3_smoke_20260623_231634"), preset_base)
    assert run.kind is TrialKind.SMOKE


def test_non_finite_metrics_become_null(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1")
    (run_dir / "grid_details" / "GOLD" / "unified_metrics.json").write_text(
        '{"sharpe": NaN, "calmar": Infinity, "trades": true}', encoding="utf-8"
    )
    assert load_run(run_dir, preset_base).metrics["GOLD"] == {
        "sharpe": None,
        "calmar": None,
        "trades": None,
        "p_value": None,
        "fold_stability": None,
    }


def test_timeframe_aliases_share_a_fingerprint(tmp_path, preset_base):
    a = load_run(make_run(tmp_path, "run-a", timeframe="DAY"), preset_base)
    b = load_run(make_run(tmp_path, "run-b", timeframe="DAY_1"), preset_base)
    assert a.fingerprint == b.fingerprint


def test_missing_timeframe_falls_back_to_the_runs_effective_timeframe(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1", timeframe=None)
    assert load_run(run_dir, preset_base).config["strategy"]["timeframe"] == "HOUR_1"


def test_presets_resolve_from_the_workspace_when_the_old_directory_is_gone(tmp_path, preset_base):
    run_dir = make_run(
        tmp_path / "results",
        "run-1",
        strategy_extra={"pipeline": "orb_simple_v1", "_strategy_dir": "/nonexistent/configs"},
    )
    run = load_run(run_dir, preset_base)
    assert run.unresolved is None
    assert run.config["strategy"]["pipeline"] == {"indicators": [{"name": "opening_range"}]}


def test_unresolvable_presets_get_a_unique_stand_in(tmp_path, preset_base):
    extra = {"pipeline": "missing_preset", "_strategy_dir": "/nonexistent/configs"}
    a = load_run(make_run(tmp_path, "run-a", strategy_extra=extra), preset_base)
    b = load_run(make_run(tmp_path, "run-b", strategy_extra=extra), preset_base)
    assert a.config is None
    assert "missing_preset" in a.unresolved
    assert a.fingerprint != b.fingerprint
    assert a.status is TrialStatus.OK


def test_scan_skips_directories_without_strategy(tmp_path, preset_base):
    make_run(tmp_path, "run-1")
    (tmp_path / "portfolio").mkdir()
    assert [run.run_id for run in scan_runs(tmp_path, preset_base)] == ["run-1"]


def test_unreadable_config_is_a_legacy_run_error(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1")
    (run_dir / "config.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(LegacyRunError, match="run-1/config.json"):
        load_run(run_dir, preset_base)
