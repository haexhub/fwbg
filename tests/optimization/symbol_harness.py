"""process_symbol / run_optimizer on synthetic hours with canned fold and unified results.

Deterministic and fast: no indicators, no training, no grid search. Folds report
TP 2 / SL 1 (RRR 2): kelly sizes WINNING trades and finds no edge in LOSING ones.
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data import config as data_config
from fwbg.optimization import holdout, process
from fwbg.optimization.resource_manager import SimplePoolManager
from fwbg.results import storage
from fwbg.utils import progress

# ``fwbg.cli`` re-exports ``main()``, which shadows the module of the same name.
cli = importlib.import_module("fwbg.cli.main")

SYMBOL = "EURUSD"
SYMBOL_FILE = f"{SYMBOL}_HOUR_1.csv"
ROWS = 8000  # 2022-01-01 .. 2022-11-29, hourly
CUTOFF = "2022-10-15"
METADATA = {
    "name": "harness",
    "timeframe": "HOUR_1",
    "model": {"type": "xgboost"},
    "validation": {"folds": 4},
    "exit_strategies": [{"name": "atr_based", "params": {"tp_mult": 2.0, "sl_mult": 1.0}}],
    # The synthetic hours run around the clock (weekends included); the harness
    # tests the orchestration, not the data, so the forex-hours check is off.
    "data_checks": "off",
}
STRATEGY = StrategyConfig.from_dict(METADATA)
WINNING = [(2.0, 1.0)] * 12 + [(-1.0, 0.0)] * 8  # per fold, (pnl_raw, result): WR 0.6
LOSING = [(2.0, 1.0)] * 4 + [(-1.0, 0.0)] * 16  # WR 0.2: kelly <= 0 -> no_edge


def hourly_frame(rows: int = ROWS) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=rows, freq="h")
    close = 100 + np.cumsum(np.random.RandomState(1).randn(rows) * 0.1)
    return pd.DataFrame({"O": close, "H": close + 0.1, "L": close - 0.1, "C": close}, index=index)


HOLDOUT = [(2.0, 1.0)] * 21 + [(-1.0, 0.0)] * 9  # the holdout fold's trades: WR 0.7


def unified_fold(fold, pairs) -> dict:
    """What evaluate_on_holdout returns for *pairs*, exits spread over the fold's test bars."""
    step = max(1, len(fold.test_df) // max(1, len(pairs)))
    times = fold.test_df.index[::step][: len(pairs)]
    wins = sum(1 for _, res in pairs if res == 1.0)
    return {
        "trades": [{"pnl_raw": pnl, "result": res} for pnl, res in pairs],
        "trades_detailed": [
            {
                "entry_time": str(t),
                "exit_time": str(t),
                "direction": "LONG",
                "result": res,
                "pnl_raw": pnl,
            }
            for t, (pnl, res) in zip(times, pairs)
        ],
        "pnl": sum(pnl for pnl, _ in pairs),
        "win_rate": wins / len(pairs) if pairs else 0,
        "n_trades": len(pairs),
    }


def fold_result(fold) -> dict:
    return {
        "fold_id": fold.fold_id,
        "test_win_rate": 0.6,
        "test_pnl": 10.0,
        "test_trades": 20,
        "inner_val_pnl": 8.0,
        "test_trades_trace": [],
        "best_config": {
            "tp": 2.0,
            "sl": 1.0,
            "ct": 0.5,
            "rrr": 2.0,
            "timeout_bars": None,
            "model_hyperparameters": {},
            "exit_modifier_params": None,
        },
        "selected_features_long": ["f1"],
        "selected_features_short": [],
    }


def patch_development(monkeypatch, pairs) -> None:
    """Every walk-forward fold succeeds; the unified simulation returns *pairs* per fold."""

    def variants(variant_list, df, wf_folds, asset, sym):
        label, variant = variant_list[0]
        return {
            "score": (len(wf_folds), 10.0),
            "label": label,
            "strategy": variant,
            "ctx": SimulationContext.create(asset, variant),
            "all_fold_results": [fold_result(fold) for fold in wf_folds],
            "accumulated_grid_results": [],
            "fold_indicators": [],
            "precomputed_raw_df": None,
            "preprocessing_configs": [],
        }

    monkeypatch.setattr(process, "load_data_aligned", lambda path: hourly_frame())
    monkeypatch.setattr(process, "_run_indicator_variants", variants)
    monkeypatch.setattr(
        process,
        "run_unified_simulation",
        lambda folds, *args, **kwargs: [unified_fold(fold, pairs) for fold in folds],
    )


def patch_holdout(monkeypatch, pairs=HOLDOUT, evaluations=None) -> list[dict]:
    """The holdout's single unified-simulation call; records what it was given."""
    calls = []

    def simulate(
        folds,
        candidates,
        fold_indicators,
        precomputed_raw_df,
        preprocessing_configs,
        ctx,
        sym,
        prepare_data_fn=None,
    ):
        calls.append(
            {"folds": folds, "candidates": candidates, "ctx": ctx, "prepare": prepare_data_fn}
        )
        return evaluations if evaluations is not None else [unified_fold(folds[0], pairs)]

    monkeypatch.setattr(holdout, "run_unified_simulation", simulate)
    return calls


def _inline(self, func, items, progress_callback=None, result_callback=None):
    results = [func(item) for item in items]
    for result in results:
        result_callback(result)
    return results


def run_inline(monkeypatch, tmp_path: Path, **metadata) -> Path:
    """The real run_optimizer with the worker pool inlined; returns the run dir."""
    return run_inline_result(monkeypatch, tmp_path, **metadata)[1]


def run_inline_result(monkeypatch, tmp_path: Path, **metadata) -> tuple[dict | None, Path]:
    """The real run_optimizer (gate off) with the worker pool inlined; its result and run dir.

    A ``holdout_cutoff`` reaches the run through a stand-in gate (``before_run``),
    the only way a real holdout look gets one.
    """
    for name in ("DATA_PATH", "TIMEFRAME", "tf_cfg", "OOS_SIZE", "WINDOW_SIZE", "RESAMPLE_FROM"):
        monkeypatch.setattr(data_config, name, getattr(data_config, name))
    monkeypatch.setattr(progress, "_progress_queue", progress._progress_queue)
    monkeypatch.setenv("TIMEFRAME", "HOUR_1")
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(tmp_path / "results"))
    monkeypatch.setattr(cli, "create_asset_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(SimplePoolManager, "map_adaptive", _inline)
    data = tmp_path / "data"
    data.mkdir()
    (data / SYMBOL_FILE).write_text("time,O,H,L,C\n", encoding="utf-8")
    cutoff = metadata.pop("holdout_cutoff", None)
    before_run = (lambda meta, symbols: {**meta, "holdout_cutoff": cutoff}) if cutoff else None
    result = cli.run_optimizer(
        strategy_metadata={**METADATA, "_cli_data_path": str(data), **metadata},
        run_id="r1",
        before_run=before_run,
    )
    return result, tmp_path / "results" / "r1"
