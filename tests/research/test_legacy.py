import json
from datetime import date

import pytest

from fwbg.research.ledger import TrialKind, TrialStatus
from fwbg.research.legacy import LegacyRunError, load_run, scan_runs
from tests.research.factories import make_run


@pytest.fixture
def preset_base(tmp_path):
    """Create a workspace preset tree containing a minimal pipeline."""
    base = tmp_path / "workspace-strategies"
    (base / "configs").mkdir(parents=True)
    (base / "pipelines").mkdir()
    (base / "pipelines" / "orb_simple_v1.json").write_text(
        json.dumps({"indicators": [{"name": "opening_range"}]}), encoding="utf-8"
    )
    return base


def test_reads_metrics_status_code_and_window(tmp_path, preset_base):
    """Read supported metrics, trial status, Git provenance and date bounds."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={
            "GOLD": {"unified": {"sharpe": 0.14, "trades": 37, "max_drawdown": 0.19, "noise": 1}}
        },
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
    assert run.presets_from_workspace is False


def test_patch_error_is_kept_in_the_code_state(tmp_path, preset_base):
    """Keep a recorded patch capture failure visible in the trial's code state."""
    run_dir = make_run(tmp_path, "run-1")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["git"] = {"commit": "abc123", "dirty": True, "patch": None, "patch_error": "boom"}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    run = load_run(run_dir, preset_base)
    assert run.code["patch_error"] == "boom"


def test_every_evaluated_symbol_counts_not_only_the_elite(tmp_path, preset_base):
    """Record metrics for every evaluated symbol, including those outside the elite list."""
    run = load_run(
        make_run(
            tmp_path,
            "run-1",
            symbols={"GOLD": {"unified": {"sharpe": 0.5}}, "EURUSD": {"unified": {"sharpe": -0.2}}},
        ),
        preset_base,
    )
    assert set(run.metrics) == {"GOLD", "EURUSD"}
    assert run.metrics["EURUSD"]["p_value"] is None


def test_run_without_symbol_results_is_failed(tmp_path, preset_base):
    """Mark a run failed when no symbol has stored evaluation results."""
    run = load_run(make_run(tmp_path, "run-1", symbols={"GOLD": None}), preset_base)
    assert run.status is TrialStatus.FAILED
    assert run.metrics == {}
    assert run.symbols == ("GOLD",)


def test_february_symbol_files_are_read(tmp_path, preset_base):
    """Read February flat symbol files and their historical metric fields."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={},
        legacy_files={
            "GOLD": {
                "symbol": "GOLD",
                "status": "ok",
                "holdout_metrics": {"sharpe": 1.2, "win_rate": 0.6, "pnl": 10.0},
                "monte_carlo": {"p_value": 0.01},
                "walk_forward": {"total_trades": 85, "fold_stability": 0.29},
            }
        },
    )
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.OK
    assert run.symbols == ("GOLD",)
    assert run.metrics["GOLD"] == {
        "sharpe": 1.2,
        "win_rate": 0.6,
        "trades": 85,
        "p_value": 0.01,
        "fold_stability": 0.29,
        "selection_protocol": None,
    }


def test_early_march_symbol_config_and_fold_results_are_read(tmp_path, preset_base):
    """Combine early March symbol config metrics with fold result statistics."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={
            "ASX200": {
                "status": "no_edge",
                "result": {"metrics": {"sharpe": 0, "win_rate": 0.41, "rrr": 1.0, "pnl": 54.3}},
                "fold": {"walk_forward": {"fold_stability": 0.625, "total_trades": 150}},
            }
        },
    )
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.OK
    assert run.metrics["ASX200"] == {
        "sharpe": 0,
        "win_rate": 0.41,
        "rrr": 1.0,
        "trades": 150,
        "p_value": None,
        "fold_stability": 0.625,
        "selection_protocol": None,
    }


def test_later_format_takes_values_from_the_symbol_files(tmp_path, preset_base):
    """Prefer later unified metrics while retaining symbol and fold statistics."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={
            "NAS100": {
                "unified": {"sharpe": 4.4, "trades": 884, "max_drawdown": 0.35},
                "result": {"monte_carlo": {"p_value": 0.05}, "metrics": {"sharpe": 9.9}},
                "fold": {"walk_forward": {"fold_stability": 0.75, "total_trades": 1}},
            }
        },
    )
    assert load_run(run_dir, preset_base).metrics["NAS100"] == {
        "sharpe": 4.4,
        "trades": 884,
        "max_drawdown": 0.35,
        "p_value": 0.05,
        "fold_stability": 0.75,
        "selection_protocol": None,
    }


def test_run_whose_symbols_all_failed_is_failed(tmp_path, preset_base):
    """Mark runs failed when every symbol has a known failure status."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": {"status": "error"}, "DAX": {"status": "no_successful_folds"}},
    )
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.FAILED
    assert run.metrics == {}
    assert run.symbols == ("DAX", "GOLD")


def test_only_evaluated_symbols_get_metrics_but_all_are_fingerprinted(tmp_path, preset_base):
    """Exclude failed symbol metrics while retaining all symbols in the config fingerprint."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": {"unified": {"sharpe": 0.5}}, "DAX": {"status": "error"}},
    )
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.OK
    assert set(run.metrics) == {"GOLD"}
    assert run.config["symbols"] == ["DAX", "GOLD"]


@pytest.mark.parametrize(
    "status",
    ["insufficient_data", "no_data", "no_signal_source"],
)
def test_known_non_evaluated_statuses_are_ignored(tmp_path, preset_base, status):
    """Exclude known unevaluated symbols while retaining an evaluated symbol's metrics."""
    run = load_run(
        make_run(
            tmp_path,
            "run-1",
            symbols={
                "GOLD": {"unified": {"sharpe": 0.5}},
                "DAX": {"status": status},
            },
        ),
        preset_base,
    )
    assert run.status is TrialStatus.OK
    assert set(run.metrics) == {"GOLD"}


def test_no_unified_trades_counts_as_evaluated(tmp_path, preset_base):
    """Count a no-unified-trades symbol as evaluated rather than failed."""
    run_dir = make_run(tmp_path, "run-1", symbols={"DAX": {"status": "no_unified_trades"}})
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.OK
    assert run.metrics == {
        "DAX": {"p_value": None, "fold_stability": None, "selection_protocol": None}
    }


@pytest.mark.parametrize("status", ["mystery", None])
def test_unknown_symbol_status_is_a_legacy_run_error(tmp_path, preset_base, status):
    """Reject unknown symbol statuses instead of guessing whether they count."""
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": {"status": status}})
    with pytest.raises(LegacyRunError, match="run-1.*GOLD"):
        load_run(run_dir, preset_base)


def test_symbol_file_values_win_over_the_elite_list(tmp_path, preset_base):
    """Prefer symbol statistics over conflicting aggregate elite values."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={
            "GOLD": {"unified": {"sharpe": 0.5}, "result": {"monte_carlo": {"p_value": 0.02}}},
            "DAX": {"unified": {"sharpe": 0.1}},
        },
        elite=[{"symbol": "GOLD", "monte_carlo": {"p_value": 0.9}, "fold_stability": 0.5}],
    )
    run = load_run(run_dir, preset_base)
    assert run.metrics["GOLD"]["p_value"] == 0.02
    assert run.metrics["GOLD"]["fold_stability"] == 0.5
    assert run.metrics["DAX"]["p_value"] is None
    assert run.metrics["DAX"]["fold_stability"] is None


def test_null_result_blocks_are_ignored(tmp_path, preset_base):
    """Ignore null result blocks while collecting available metrics."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={
            "GOLD": {
                "unified": {"sharpe": 0.5},
                "result": {"monte_carlo": None, "walk_forward": None},
                "fold": {"walk_forward": None},
            }
        },
        elite=[{"symbol": "GOLD", "monte_carlo": None, "fold_stability": 0.4}],
    )
    metrics = load_run(run_dir, preset_base).metrics["GOLD"]
    assert metrics["p_value"] is None
    assert metrics["fold_stability"] == 0.4


def test_lowercase_symbol_directories_are_recorded_upper_case(tmp_path, preset_base):
    """Normalise lowercase directory names into uppercase recorded symbols."""
    run_dir = make_run(tmp_path, "run-1", symbols={"gold": {"unified": {"sharpe": 1.0}}})
    run = load_run(run_dir, preset_base)
    assert run.symbols == ("GOLD",)
    assert run.metrics["GOLD"]["sharpe"] == 1.0


def test_a_symbol_directory_wins_over_a_symbol_file(tmp_path, preset_base):
    """Prefer a symbol directory when a flat historical file also exists."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": {"unified": {"sharpe": 0.5}}},
        legacy_files={"GOLD": {"status": "ok", "metrics": {"sharpe": 9.0}}},
    )
    run = load_run(run_dir, preset_base)
    assert run.symbols == ("GOLD",)
    assert run.metrics["GOLD"]["sharpe"] == 0.5


def test_symlinked_symbol_entries_are_skipped(tmp_path, preset_base):
    """Exclude symlinked symbol entries from legacy results."""
    run_dir = make_run(tmp_path, "run-1")
    details = run_dir / "grid_details"
    (details / "DAX").symlink_to(details / "GOLD", target_is_directory=True)
    assert load_run(run_dir, preset_base).symbols == ("GOLD",)


def test_agent_smoke_runs_are_smoke(tmp_path, preset_base):
    """Classify recognised agent smoke names as smoke trials."""
    run = load_run(make_run(tmp_path, "run-1", name="m3_smoke_20260623_231634"), preset_base)
    assert run.kind is TrialKind.SMOKE


def test_names_starting_with_m_stay_legacy(tmp_path, preset_base):
    """Keep ordinary names beginning with m classified as legacy trials."""
    run = load_run(make_run(tmp_path, "run-1", name="momentum-test"), preset_base)
    assert run.kind is TrialKind.LEGACY


def test_non_finite_metrics_become_null(tmp_path, preset_base):
    """Normalise nonfinite metric values to ledger-compatible nulls."""
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
        "selection_protocol": None,
    }


def test_timeframe_aliases_share_a_fingerprint(tmp_path, preset_base):
    """Canonicalise timeframe aliases to the same configuration fingerprint."""
    a = load_run(make_run(tmp_path, "run-a", timeframe="DAY"), preset_base)
    b = load_run(make_run(tmp_path, "run-b", timeframe="DAY_1"), preset_base)
    assert a.fingerprint == b.fingerprint


def test_missing_timeframe_falls_back_to_the_runs_effective_timeframe(tmp_path, preset_base):
    """Use the run config timeframe when strategy metadata omits it."""
    run_dir = make_run(tmp_path, "run-1", timeframe=None)
    assert load_run(run_dir, preset_base).config["strategy"]["timeframe"] == "HOUR_1"


def test_strategy_timeframe_wins_over_the_config_timeframe(tmp_path, preset_base):
    """Prefer the strategy timeframe over a conflicting run config value."""
    run_dir = make_run(tmp_path, "run-1", timeframe="HOUR_4", config_timeframe="MINUTE_15")
    assert load_run(run_dir, preset_base).config["strategy"]["timeframe"] == "HOUR_4"


def test_presets_at_the_recorded_directory_take_priority(tmp_path, preset_base):
    """Resolve presets from their recorded location before trying the workspace."""
    old = tmp_path / "old-strategies"
    (old / "configs").mkdir(parents=True)
    (old / "pipelines").mkdir()
    (old / "pipelines" / "orb_simple_v1.json").write_text(
        json.dumps({"indicators": [{"name": "old_range"}]}), encoding="utf-8"
    )
    run_dir = make_run(
        tmp_path / "results",
        "run-1",
        strategy_extra={"pipeline": "orb_simple_v1", "_strategy_dir": str(old / "configs")},
    )
    run = load_run(run_dir, preset_base)
    assert run.config["strategy"]["pipeline"] == {"indicators": [{"name": "old_range"}]}
    assert run.presets_from_workspace is False


def test_presets_resolve_from_the_workspace_when_the_old_directory_is_gone(tmp_path, preset_base):
    """Use workspace presets and flag the fallback when recorded presets are missing."""
    run_dir = make_run(
        tmp_path / "results",
        "run-1",
        strategy_extra={"pipeline": "orb_simple_v1", "_strategy_dir": "/nonexistent/configs"},
    )
    run = load_run(run_dir, preset_base)
    assert run.unresolved is None
    assert run.config["strategy"]["pipeline"] == {"indicators": [{"name": "opening_range"}]}
    assert run.presets_from_workspace is True


def test_presets_without_a_recorded_directory_ignore_the_working_directory(
    tmp_path, preset_base, monkeypatch
):
    """Resolve unlocated presets from the workspace rather than the process directory."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "pipelines").mkdir(parents=True)
    (elsewhere / "pipelines" / "orb_simple_v1.json").write_text(
        json.dumps({"indicators": [{"name": "cwd_range"}]}), encoding="utf-8"
    )
    monkeypatch.chdir(elsewhere)
    run_dir = make_run(tmp_path / "results", "run-1", strategy_extra={"pipeline": "orb_simple_v1"})
    run = load_run(run_dir, preset_base)
    assert run.config["strategy"]["pipeline"] == {"indicators": [{"name": "opening_range"}]}
    assert run.presets_from_workspace is True


def test_unresolvable_presets_get_a_unique_stand_in(tmp_path, preset_base):
    """Give unresolved runs distinct fingerprints without a stored canonical config."""
    extra = {"pipeline": "missing_preset", "_strategy_dir": "/nonexistent/configs"}
    a = load_run(make_run(tmp_path, "run-a", strategy_extra=extra), preset_base)
    b = load_run(make_run(tmp_path, "run-b", strategy_extra=extra), preset_base)
    assert a.config is None
    assert "missing_preset" in a.unresolved
    assert a.fingerprint != b.fingerprint
    assert a.status is TrialStatus.OK


def test_non_finite_strategy_values_get_a_stand_in(tmp_path, preset_base):
    """Use an unresolved fingerprint when strategy values cannot be canonicalised."""
    run_dir = make_run(tmp_path, "run-1", strategy_extra={"cost_multiplier": float("nan")})
    run = load_run(run_dir, preset_base)
    assert run.config is None and run.unresolved is not None
    assert run.status is TrialStatus.OK


def test_run_date_comes_from_the_timestamp(tmp_path, preset_base):
    """Derive the legacy run date from the recorded config timestamp."""
    run = load_run(make_run(tmp_path, "run-1", timestamp="2026-02-13T16:37:45"), preset_base)
    assert run.run_date == date(2026, 2, 13)
    assert run.ts == "2026-02-13T16:37:45"


@pytest.mark.parametrize("timestamp", ["2026-W10-1T00:00", "soon", "2026-13-45T00:00"])
def test_unusable_timestamp_is_a_legacy_run_error(tmp_path, preset_base, timestamp):
    """Reject missing or invalid run timestamps with a legacy error."""
    run_dir = make_run(tmp_path, "run-1", timestamp=timestamp)
    with pytest.raises(LegacyRunError, match="run-1"):
        load_run(run_dir, preset_base)


@pytest.mark.parametrize("content", ["{broken", "[]", '{"timeframe": "HOUR_1"}'])
def test_unusable_config_is_a_legacy_run_error(tmp_path, preset_base, content):
    """Report unreadable or malformed config files as legacy errors."""
    run_dir = make_run(tmp_path, "run-1")
    (run_dir / "config.json").write_text(content, encoding="utf-8")
    with pytest.raises(LegacyRunError, match="run-1/config.json"):
        load_run(run_dir, preset_base)


def test_non_object_symbol_file_is_a_legacy_run_error(tmp_path, preset_base):
    """Reject symbol files whose JSON is not an object."""
    run_dir = make_run(tmp_path, "run-1")
    (run_dir / "grid_details" / "GOLD" / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(LegacyRunError, match="run-1/grid_details/GOLD/config.json"):
        load_run(run_dir, preset_base)


def test_an_ungated_holdout_look_is_never_read_as_development(tmp_path, preset_base):
    """A run with a holdout cutoff in its strategy evaluated holdout data: refuse it."""
    make_run(tmp_path, "run-1", strategy_extra={"holdout_cutoff": "2025-04-30"})
    with pytest.raises(LegacyRunError, match="run-1/strategy.json: .*holdout look"):
        scan_runs(tmp_path, preset_base)


def test_scan_skips_directories_without_strategy(tmp_path, preset_base):
    """Ignore directories without strategy metadata during a legacy scan."""
    make_run(tmp_path, "run-1")
    (tmp_path / "portfolio").mkdir()
    assert [run.run_id for run in scan_runs(tmp_path, preset_base)] == ["run-1"]


def test_scan_skips_symlinked_run_directories(tmp_path, preset_base):
    """Exclude symlinked run directories from scanning."""
    make_run(tmp_path, "run-1")
    (tmp_path / "alias").symlink_to(tmp_path / "run-1", target_is_directory=True)
    assert [run.run_id for run in scan_runs(tmp_path, preset_base)] == ["run-1"]


def test_scanning_a_missing_results_directory_is_a_legacy_run_error(tmp_path, preset_base):
    """Report a missing results root as a legacy error."""
    with pytest.raises(LegacyRunError, match="nope"):
        scan_runs(tmp_path / "nope", preset_base)


def test_scan_skips_gated_runs(tmp_path, preset_base):
    """A run whose config.json has a research block is already in the ledger."""
    make_run(tmp_path, "run-1")
    gated = make_run(tmp_path, "run-2")
    config = json.loads((gated / "config.json").read_text(encoding="utf-8"))
    config["research"] = {"research_id": "S-0007"}
    (gated / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert [run.run_id for run in scan_runs(tmp_path, preset_base)] == ["run-1"]


def test_return_statistics_reach_the_ledger(tmp_path, preset_base):
    """Carry the per-trade return statistics the verdict's DSR needs."""
    unified = {
        "sharpe": 1.2,
        "sr_per_trade": 0.11,
        "skew": -0.4,
        "kurtosis": 4.5,
        "n_returns": 230,
        "test_period_years": 2.5,
    }
    run = load_run(make_run(tmp_path, "run-1", symbols={"GOLD": {"unified": unified}}), preset_base)
    assert {k: run.metrics["GOLD"][k] for k in unified} == unified


def test_a_null_sharpe_stays_null(tmp_path, preset_base):
    """A no_edge symbol of a new run records no Sharpe, not a fake zero."""
    run_dir = make_run(
        tmp_path, "run-1", symbols={"GOLD": {"status": "no_edge", "unified": {"sharpe": None}}}
    )
    assert load_run(run_dir, preset_base).metrics["GOLD"]["sharpe"] is None
