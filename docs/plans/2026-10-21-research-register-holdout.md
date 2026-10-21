# Research Register Step 5: Holdout Mode and Verdict Statistics (DSR), Implementation Plan

Execution: task by task. Each task gets a review for spec compliance and code quality.

**Goal:** A gated run with `--holdout` (CLI) or `"holdout": true` (API) does three things. It develops the strategy on the data before the card's cutoff, exactly like an iteration. It retrains the frozen unified setting once on all pre-cutoff data, minus the embargo. It evaluates that setting once on the data after the cutoff. The holdout trial's ledger metrics are the holdout metrics only. From now on, new runs store per-symbol return statistics. The verdict reports the full Bailey & López de Prado (2014) Deflated Sharpe Ratio of the best development evaluation, and the holdout result checked against the card's criteria.

**Architecture:**
- **Holdout worker.** A new worker `fwbg.optimization.holdout.process_symbol_holdout` wraps the existing `process_symbol`:
  1. Development: `process_symbol` with `end_date = cutoff`. Walk-forward, selection, unified simulation, risk calibration and Monte Carlo all stay unchanged.
  2. If development ends `ok`: load the window `[start, end)` again, build one holdout fold (train `[0, n_dev - embargo)`, test `[n_dev, end)`), and run it through `run_unified_simulation([fold], …)`. That function calls `evaluate_on_holdout`, so the context is built exactly as in development.
  3. Size the holdout trades with the development's fk, circuit breaker and loss scale.
  4. Compute every reported metric from those sized holdout trades.
- **Dispatch.** `run_optimizer` picks the worker from `StrategyConfig.holdout_cutoff`. The gate writes that key into the effective metadata of a holdout run, so the key is fingerprinted.
- **Statistics.** The DSR maths lives in a stdlib-only module, `fwbg.research.dsr`. Verdict statistics live in `fwbg.research.stats`, and the verdict text in `fwbg.research.commands`.
- **Process growth.** `process.py` and `cli/main.py` grow by only a few lines. New logic goes to new modules. Shared reporting helpers move out of `process.py` into `causal_reporting.py`.

**Tech Stack:** Python 3.14, numpy/pandas, stdlib `statistics.NormalDist`, FastAPI, pytest.

**Documents:**
- Design: `docs/plans/2026-10-19-research-register-design.md`. Relevant parts: gate rules 4-6, the Verdict section, and the open question "Holdout evaluation mechanics", which this step closes.
- Earlier steps: `docs/plans/2026-10-20-research-register-gate.md` and `docs/plans/2026-10-20-research-register-api.md`.

---

## Findings and risks (read first)

All line references below were checked on `caa1e34`.

1. **The ledger does not store each symbol's status** (`legacy.py:239-275`; metrics are `METRIC_KEYS` plus `p_value` and `fold_stability`). Legacy `no_edge` zeros therefore cannot be told apart from real zeros.
   - **Plan: no heuristic.** This is harmless for the DSR. V[SR] and "best" only use rows that carry the new return statistics, and no legacy row and no new `no_edge` row has them. Legacy and `no_edge` rows still count in N, as decided.
   - **Only side effect:** the existing "Best development result" line (ranked by Sharpe) could pick a legacy zero, but only if every other development Sharpe is negative. This goes into the docs.

2. **Bug that would leak a fake `fold_stability` into holdout trials.** `storage.py:347` writes `"fold_stability": r.get("fold_stability", 0)` into the elite entries of `results.json`. `legacy._symbol_metrics` falls back to that value (`legacy.py:270-274`). A holdout trial would then record `fold_stability: 0`. Task 5 changes the default to `None`.

3. **A `null` Sharpe would crash existing code:**
   - `main.py:500/528` formats `Sharpe={sharpe:.2f}` for `no_edge`, inside `on_result_ready`. A `None` would crash the whole run.
   - `api/runs.py:558` (`/api/runs/compare`) sums Sharpe values; `None` raises a `TypeError`.

   Both are fixed in Task 2. The plotting code (`if sharpe:`) and the dashboard (`?? null`) already handle `null`. The fwbg-agents repo must be checked in step 6.

4. **Some files must not be reformatted with `black`:** `process.py`, `cli/main.py`, `causal_reporting.py`, `core/config.py`, `results/storage.py` and `api/runs.py` (`black --check` would reformat them all). Run black only on new modules and on `src/fwbg/research/`; run `ruff` on everything.

5. **Run-level DSR differs from the verdict DSR.** `overfitting.py` estimates E[max SR] with unit variance and N = number of grid combinations, so its DSR reads near 0 for per-trade SRs. Decision 3 asks to write the `overfitting` block into `config.json`; we do that unchanged. It is advisory and documented as not being the verdict DSR. It is not reworked here (YAGNI).

6. **Interpretation of decision 3** (please confirm):
   - V[SR] and "best" come from development trials only (counted, not holdout). Holdout looks are not part of the selection the DSR deflates.
   - N is `evaluations(own)`: every counted symbol evaluation of the card, holdout looks included. This is the existing function and the conservative reading.
   - "Best" is ranked by `sr_per_trade`, the statistic being deflated, not by annualised Sharpe.
   - The minimum of 2 is counted in symbol evaluations that carry return statistics.

7. **Failed holdout looks still count.** If every symbol's development ends before evaluating (for example `no_successful_folds`), the holdout trial is recorded `failed` and still counts as a look (gate rule 6). The next look needs `--reason`. This is conservative and gets documented.

8. **Extra cost per holdout symbol.** Data is loaded and raw indicators are precomputed twice per symbol (once for development, once for the full window). This is the deliberate shortcut (a `ponytail:` comment).
   - **Causality assumption:** raw indicators on `[start, end)` give the same values on development rows as on `[start, cutoff)`. `tests/test_lookahead_bias.py::TestFutureScramble` pins this.
   - Preprocessors and stateful indicators are still fit on the training frame only (`process_fold.py:116-213`).

9. **Holdout runs normally export no `assets.json` entries.** Holdout `ok` results flow into the elite table of `run_optimizer`, but they carry no `fold_stability`, so they fail the elite filter unless `min_fold_stability` is 0. That is acceptable: holdout runs are evaluations, not deployment exports.

10. **Signal models select on the test fold** (`signal_fold.py:71-148`). This makes development metrics and per-trade SRs optimistic, so the DSR's "best" is biased upward. It is out of scope (decision 5) and only documented.

11. **Cosmetic progress glitch.** The development `process_symbol` reports `done` before the holdout phase; the holdout worker reports its own phase and `done`.

12. **`core/config.py` is already 542 LoC**, over the limit before this step. This step adds 2 lines.

**Feasibility:** every operator decision is feasible as stated.

**Ordering change.** I moved "gate/CLI/API flag" after the pipeline and recording tasks. Exposing `--holdout` before the pipeline honours the cutoff would let a holdout trial evaluate walk-forward over the whole window.

---

## Operator decisions (2026-10-21, binding)

1. **Option A holdout.**
   - Walk-forward, selection, unified simulation and risk calibration run only on `[start, cutoff)`.
   - The frozen unified setting is retrained once on all pre-cutoff data minus the embargo, and evaluated once on `[cutoff, end)` with `evaluate_on_holdout`.
   - The development fk, circuit breaker and loss scale are reused.
   - Metrics come from holdout trades only. The development metrics are stored apart (`development.json`); the holdout result carries no development `fold_stability`.
   - If development ends in `no_edge`, `not_significant`, `no_successful_folds` or similar, the holdout symbol records that status with `holdout.evaluated = false` and is not evaluated.
   - The cutoff reaches the worker through the effective metadata (`holdout_cutoff`, fingerprinted). The gate refuses a holdout with `start >= cutoff`.
   - Raw causal indicators may be precomputed on the full window.
2. **Where holdout can be requested:**
   - `--holdout` on the CLI run command, and `holdout` on the API `RunStartRequest` (preflight argv included).
   - `register_trial` gets no holdout parameter and always records iterations. A study config that carries `holdout_cutoff` is refused by the gate (`INVALID_CONFIG`).
3. **DSR from now on:**
   - New runs store per symbol: `sr_per_trade`, `skew`, `kurtosis`, `n_returns`, `test_period_years` (in `unified_metrics.json` and `METRIC_KEYS`). The `overfitting` block goes into `config.json`.
   - Verdict DSR: N = all counted symbol evaluations of the card. V[SR] comes from per-trade SRs of evaluations with the inputs. Best = the best development evaluation with the inputs. Minimum 2, otherwise "too few trials with return statistics".
   - The formula is the full paper version, validated against the paper's example.
   - The verdict also reports the holdout result.
4. **`no_edge`:**
   - New runs write `sharpe: null`.
   - `no_edge` evaluations are counted in N but excluded from V[SR] (they carry no return statistics).
   - The ledger stays append-only. It has no status, see finding 1.
5. **Signal-model test-fold selection** is a later step; this step only documents it.

## Rules for every task

- Work only in the step-5 worktree (branch `feat/research-holdout`). Never touch the main checkout, never write under `~/fwbg`. No bare `git stash`.
- TDD: failing test first, then the code.
  - Registers live in `tmp_path`, through `gate_register` in `tests/research/conftest.py` or `init_research_repo(tmp_path / "research")`.
  - Run directories go to `tmp_path` by patching `storage.RESULTS_BASE_PATH`.
  - `tests/conftest.py` sets `FWBG_RESEARCH_GATE=off` and points `FWBG_WORKSPACE` at `tests/_fixtures/workspace`. Gate tests switch the gate on with `monkeypatch.setenv(GATE_ENV, "on")`.
- Tests are deterministic: synthetic frames from a fixed seed, and the Monte Carlo helpers are already seeded (`random_seed=42`).
- 500-LoC limit for new and touched modules (spaex); keep growth of `process.py` (1151) and `cli/main.py` (1051) to a few lines. Mark deliberate shortcuts with `ponytail:` comments. Imports go at the top; match the surrounding style.
- Commits: short English messages, no AI/model references or trailers (as in steps 3 and 4). Stage only the touched files.
- Before reporting a task, run:
  ```
  ../../.venv/bin/black <new modules and src/fwbg/research/ files you touched>
  ../../.venv/bin/ruff check src/ packages/
  ../../.venv/bin/mypy --follow-imports=silent src/fwbg/research
  PYTHONPATH=src ../../.venv/bin/python -m pytest -p no:cacheprovider tests/research tests/optimization tests/test_cli.py tests/test_api_run_research.py tests/test_api_result_paths.py tests/test_statuses.py -q
  ```
  plus the task's own test files.

**Fixtures to reuse:**
- `gate_register`, `make_card`, `make_trial`, `make_run`: `tests/research/conftest.py` and `tests/research/factories.py`.
- `gated` and `FakeOptimizer`: `tests/research/test_run_cli.py:26-68`.
- The `optimizer` fixture pattern, which restores `data_config` globals, `progress._progress_queue` and `TIMEFRAME`: `tests/research/test_run_cli.py:271-286`.
- `api`, `_start`, `ALLOWED`, `calls["spawn"]`, `calls["preflight"]`: `tests/test_api_run_research.py`.
- `result_client`: `tests/test_api_result_paths.py`.
- The patching style for `process.load_data_aligned` and signal strategies: `tests/optimization/test_signal_source_guard.py`.

---

### Task 1: DSR maths (`fwbg.research.dsr`)

**Files**
- Create: `src/fwbg/research/dsr.py`
- Test: `tests/research/test_dsr.py` (new)

**Step 1: Failing tests**

```python
"""Deflated Sharpe ratio, Bailey & López de Prado (2014)."""

from math import sqrt
from statistics import NormalDist

import numpy as np
import pytest

from fwbg.research.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    return_statistics,
)

# The paper's numerical example: N = 100 independent trials, variance of the
# annualised SRs 1/2, best annualised SR 2.5, T = 1250 daily returns (5 years),
# skewness -3, kurtosis 10. The paper reports SR_0 = 0.1132 and DSR = 0.9004.
DAILY_VARIANCE = 0.5 / 250
BEST_SR = 2.5 / sqrt(250)


def test_expected_max_sharpe_matches_the_paper():
    assert expected_max_sharpe(100, DAILY_VARIANCE) == pytest.approx(0.1132, abs=5e-4)


def test_deflated_sharpe_ratio_matches_the_paper():
    dsr = deflated_sharpe_ratio(
        sr=BEST_SR, n_returns=1250, skew=-3.0, kurtosis=10.0,
        n_trials=100, sr_variance=DAILY_VARIANCE,
    )
    assert dsr == pytest.approx(0.9004, abs=1e-3)


def test_expected_max_sharpe_approximates_the_maximum_of_standard_normals():
    maxima = np.random.default_rng(0).standard_normal((2000, 1000)).max(axis=1)
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(float(maxima.mean()), abs=0.03)


@pytest.mark.parametrize("n_trials, variance", [(1, 0.3), (0, 0.3), (50, 0.0)])
def test_no_selection_bias_without_trials_or_variance(n_trials, variance):
    assert expected_max_sharpe(n_trials, variance) == 0.0


def test_psr_of_normal_returns():
    expected = NormalDist().cdf(0.1 * sqrt(100) / sqrt(1 + 0.5 * 0.01))
    assert probabilistic_sharpe_ratio(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "sr, n_returns, skew, kurtosis",
    [(0.1, 1, 0.0, 3.0), (1.0, 100, 3.0, 1.0)],  # one return; 1 - 3 + 0 < 0
)
def test_psr_is_undefined(sr, n_returns, skew, kurtosis):
    assert probabilistic_sharpe_ratio(sr, 0.0, n_returns, skew, kurtosis) is None


def test_return_statistics_use_population_moments():
    stats = return_statistics([-1.0, 1.0, -1.0, 1.0])
    assert stats == {"sr_per_trade": 0.0, "skew": 0.0, "kurtosis": 1.0, "n_returns": 4}


def test_return_statistics_of_a_skewed_series():
    values = [0.01, -0.02, 0.03, 0.0, 0.05]
    mean, std = np.mean(values), np.std(values)
    z = (np.array(values) - mean) / std
    stats = return_statistics(values)
    assert stats["sr_per_trade"] == pytest.approx(mean / std)
    assert stats["skew"] == pytest.approx(np.mean(z**3))
    assert stats["kurtosis"] == pytest.approx(np.mean(z**4))


@pytest.mark.parametrize("returns", [[], [0.01], [0.01, 0.01], [float("nan"), 0.1]])
def test_return_statistics_need_two_varying_returns(returns):
    assert return_statistics(returns) is None
```

Run `PYTHONPATH=src ../../.venv/bin/python -m pytest -p no:cacheprovider tests/research/test_dsr.py -q`. Expected: FAIL (module missing). The values were checked with stdlib `NormalDist`: SR_0 = 0.113172, DSR = 0.900397, E[max] for N = 1000 = 3.2551.

**Step 2: Implementation** (`src/fwbg/research/dsr.py`)

```python
"""Deflated Sharpe ratio (Bailey & López de Prado 2014) for research verdicts.

Sharpe ratios here are per return (per trade), never annualised, so trials
with different trade frequencies share one scale. Kurtosis is the plain
fourth standardised moment (3 for normal returns), not excess kurtosis.
Stdlib only: the register must not import the optimizer package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist, fmean, pstdev

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] of *n_trials* independent trials with true SR 0 whose SR
    estimates have variance *sr_variance* (the paper's SR_0)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def probabilistic_sharpe_ratio(
    sr: float, benchmark: float, n_returns: int, skew: float, kurtosis: float
) -> float | None:
    """P(true SR > *benchmark*) for an SR *sr* observed over *n_returns* returns.

    None when undefined: fewer than two returns, or skew and kurtosis that leave
    the SR estimate no positive variance.
    """
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if n_returns < 2 or denominator <= 0:
        return None
    return _NORMAL.cdf((sr - benchmark) * math.sqrt(n_returns - 1) / math.sqrt(denominator))


def deflated_sharpe_ratio(
    sr: float, n_returns: int, skew: float, kurtosis: float, n_trials: int, sr_variance: float
) -> float | None:
    """PSR of *sr* against the SR the best of *n_trials* would show by luck alone."""
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, benchmark, n_returns, skew, kurtosis)


def return_statistics(returns: Sequence[float]) -> dict[str, float | int] | None:
    """What the verdict's DSR needs from one evaluation's per-trade returns.

    Population moments, like the run's own per-trade Sharpe (process.py). None
    for fewer than two returns or a constant (or non-finite) series.
    """
    values = [float(value) for value in returns]
    if len(values) < 2:
        return None
    mean = fmean(values)
    std = pstdev(values, mu=mean)
    if not math.isfinite(std) or std == 0:
        return None
    z = [(value - mean) / std for value in values]
    return {
        "sr_per_trade": mean / std,
        "skew": fmean(x**3 for x in z),
        "kurtosis": fmean(x**4 for x in z),
        "n_returns": len(values),
    }
```

**Step 3:** Run `tests/research/test_dsr.py`; it must pass. Then run black and mypy on the research package.

**Commit:** `feat(research): add the deflated Sharpe ratio`

---

### Task 2: Store return statistics, write a `null` Sharpe for `no_edge`, keep the overfitting block

**Files**
- Modify `src/fwbg/optimization/process.py`:
  - `:827`: compute `return_stats` next to `non_ann_sr`.
  - `:935`: `"sharpe": None`.
  - `:1037-1067` and `:1086-1123`: add `"return_stats": return_stats`.
  - Import `return_statistics` at the top.
- Create `src/fwbg/results/symbol_files.py` (`config_extras`).
- Modify `src/fwbg/cli/main.py`:
  - in `on_result_ready`, add `config_data.update(config_extras(result))` before `config.json` is written;
  - after the `unified_metrics` dict, add `unified_metrics.update(result.get("return_stats") or {})`;
  - `:500/:528`: print the Sharpe as `n/a` when it is `None`.
- Modify `src/fwbg/api/runs.py:558`: `sum((m.get("sharpe") or 0) for m in metrics_list)`.
- Create the test helper `tests/optimization/symbol_harness.py`. `tests/optimization/__init__.py` already exists.
- Tests: `tests/optimization/test_symbol_metrics.py` (new), plus a new case in `tests/test_api_result_paths.py`.

**Harness** (`tests/optimization/symbol_harness.py`; Task 4 adds `patch_holdout`):

```python
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
from fwbg.optimization import process, unified_simulation
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
}
STRATEGY = StrategyConfig.from_dict(METADATA)
WINNING = [(2.0, 1.0)] * 12 + [(-1.0, 0.0)] * 8   # per fold, (pnl_raw, result): WR 0.6
LOSING = [(2.0, 1.0)] * 4 + [(-1.0, 0.0)] * 16    # WR 0.2: kelly <= 0 -> no_edge


def hourly_frame(rows: int = ROWS) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=rows, freq="h")
    close = 100 + np.cumsum(np.random.RandomState(1).randn(rows) * 0.1)
    return pd.DataFrame({"O": close, "H": close + 0.1, "L": close - 0.1, "C": close}, index=index)


def unified_fold(fold, pairs) -> dict:
    """What evaluate_on_holdout returns for *pairs*, exits spread over the fold's test bars."""
    step = max(1, len(fold.test_df) // max(1, len(pairs)))
    times = fold.test_df.index[::step][: len(pairs)]
    wins = sum(1 for _, res in pairs if res == 1.0)
    return {
        "trades": [{"pnl_raw": pnl, "result": res} for pnl, res in pairs],
        "trades_detailed": [
            {"entry_time": str(t), "exit_time": str(t), "direction": "LONG",
             "result": res, "pnl_raw": pnl}
            for t, (pnl, res) in zip(times, pairs)
        ],
        "pnl": sum(pnl for pnl, _ in pairs),
        "win_rate": wins / len(pairs) if pairs else 0,
        "n_trades": len(pairs),
    }


def fold_result(fold) -> dict:
    return {
        "fold_id": fold.fold_id, "test_win_rate": 0.6, "test_pnl": 10.0, "test_trades": 20,
        "inner_val_pnl": 8.0, "test_trades_trace": [],
        "best_config": {"tp": 2.0, "sl": 1.0, "ct": 0.5, "rrr": 2.0, "timeout_bars": None,
                        "model_hyperparameters": {}, "exit_modifier_params": None},
        "selected_features_long": ["f1"], "selected_features_short": [],
    }


def patch_development(monkeypatch, pairs) -> None:
    """Every walk-forward fold succeeds; the unified simulation returns *pairs* per fold."""
    def variants(variant_list, df, wf_folds, asset, sym):
        label, variant = variant_list[0]
        return {
            "score": (len(wf_folds), 10.0), "label": label, "strategy": variant,
            "ctx": SimulationContext.create(asset, variant),
            "all_fold_results": [fold_result(fold) for fold in wf_folds],
            "accumulated_grid_results": [], "fold_indicators": [],
            "precomputed_raw_df": None, "preprocessing_configs": [],
        }

    monkeypatch.setattr(process, "load_data_aligned", lambda path: hourly_frame())
    monkeypatch.setattr(process, "_run_indicator_variants", variants)
    monkeypatch.setattr(
        unified_simulation, "run_unified_simulation",
        lambda folds, *args, **kwargs: [unified_fold(fold, pairs) for fold in folds],
    )


def _inline(self, func, items, progress_callback=None, result_callback=None):
    results = [func(item) for item in items]
    for result in results:
        result_callback(result)
    return results


def run_inline(monkeypatch, tmp_path: Path, **metadata) -> Path:
    """The real run_optimizer (gate off) with the worker pool inlined; returns the run dir."""
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
    cli.run_optimizer(
        strategy_metadata={**METADATA, "_cli_data_path": str(data), **metadata}, run_id="r1"
    )
    return tmp_path / "results" / "r1"
```

Why the harness works without extra patches:
- HOUR_1 with 4 folds: `plan_walk_forward(8000, 4, 17500, 400)` returns 4 folds of 906 bars each (train floor 4375). A development frame cut at `CUTOFF` gives 6864 bars and 622 bars per fold.
- Fold 0 is excluded by causal reporting, which leaves 60 causal trades (at least 10).
- If the planner or kelly behaves differently, adjust `ROWS` or the pairs, not production code.

**Step 1: Failing tests** (`tests/optimization/test_symbol_metrics.py`)

```python
import json

import pytest

from fwbg.optimization.process import process_symbol
from fwbg.research.dsr import return_statistics
from tests.optimization.symbol_harness import (
    LOSING, STRATEGY, SYMBOL_FILE, WINNING, patch_development, run_inline,
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
```

In `tests/test_api_result_paths.py`, add `test_compare_tolerates_a_null_sharpe`:
- `run-1/grid_details/A/unified_metrics.json` contains `{"sharpe": null, "pnl": 1}`.
- `.../B/unified_metrics.json` contains `{"sharpe": 1.0, "pnl": 1}`.
- POST `/api/runs/compare` returns 200 and `aggregate.avg_sharpe == 0.5`.

Expected: all FAIL, with KeyError, `0 is not None`, or a `TypeError` in `compare_runs`.

**Step 2: Implementation**

In `process.py`, after `non_ann_sr` (`:827`):
```python
        # Per-trade return statistics of the reporting series (the verdict's DSR inputs).
        return_stats = return_statistics(reporting_returns)
```
In the `no_edge` branch, use `"sharpe": None,` with a comment: `# no edge, no sizing: no Sharpe (a 0 would enter statistics)`. In the not_significant and ok result dicts, add `"return_stats": return_stats,`.

New module `src/fwbg/results/symbol_files.py`:
```python
"""Per-symbol run files beyond the optimizer's classic set."""

_CONFIG_BLOCKS = ("overfitting",)


def config_extras(result: dict) -> dict:
    """Blocks a symbol's config.json carries for the research register."""
    return {key: result[key] for key in _CONFIG_BLOCKS if result.get(key)}
```

In `main.py`, import `config_extras` at the top (next to the other `fwbg.results` imports). For `:500`, use `sharpe_text = "n/a" if sharpe is None else f"{sharpe:.2f}"` in the `Performance:` line.

**Step 3:** The new tests, `tests/optimization/test_causal_reporting.py`, `tests/test_statuses.py` and `tests/test_api_result_paths.py` must pass.

**Commit:** `feat(optimizer): store per-trade return statistics; no Sharpe without an edge`

---

### Task 3: The ledger records the return statistics

**Files**
- Modify: `src/fwbg/research/legacy.py:21-30` (`METRIC_KEYS`)
- Test: `tests/research/test_legacy.py`, `tests/research/test_outcome.py`

**Step 1: Failing tests**

```python
def test_return_statistics_reach_the_ledger(tmp_path, preset_base):
    unified = {"sharpe": 1.2, "sr_per_trade": 0.11, "skew": -0.4, "kurtosis": 4.5,
               "n_returns": 230, "test_period_years": 2.5}
    run = load_run(make_run(tmp_path, "run-1", symbols={"GOLD": {"unified": unified}}),
                   preset_base)
    assert {k: run.metrics["GOLD"][k] for k in unified} == unified


def test_a_null_sharpe_stays_null(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1",
                       symbols={"GOLD": {"status": "no_edge", "unified": {"sharpe": None}}})
    assert load_run(run_dir, preset_base).metrics["GOLD"]["sharpe"] is None
```

In `test_outcome.py`, a lenient `run_outcome` on such a directory keeps `n_returns` as an `int`.

**Step 2:** Extend `METRIC_KEYS` with `"sr_per_trade", "skew", "kurtosis", "n_returns", "test_period_years"`. Add a comment above it:

```python
# The ledger records no per-symbol status: a legacy no_edge symbol shows sharpe 0,
# a new one sharpe null. Neither carries return statistics, so both count towards
# N but never enter the verdict's SR variance (stats.return_samples).
```

**Step 3:** `tests/research` must pass.

**Commit:** `feat(research): record per-trade return statistics in the ledger`

---

### Task 4: The holdout pipeline (`fwbg.optimization.holdout`)

**Files**
- Modify `src/fwbg/core/config.py`:
  - add `holdout_cutoff: Optional[str] = None` after `end_date`, with the comment `# Set by the research gate for a holdout look; see optimization.holdout`;
  - add `holdout_cutoff=data.get("holdout_cutoff")` in `from_dict`.
- Modify `src/fwbg/optimization/process.py`:
  - extract `load_symbol_frame(csv_path, strategy, sym) -> pd.DataFrame | None` from `:428-473` (load, resample, `_apply_date_window`, flat bars, data loading; `None` = no bars);
  - replace the inline Monte Carlo, equity replay and slim-trades code (`:965-1029`) with the helpers below;
  - add `"unified_candidate": unified_candidate` to the ok result.
- Modify `src/fwbg/optimization/causal_reporting.py`, with new helpers that `compute_causal_reporting` and the holdout share:
  - `sized_returns(trades, fk, circuit_breaker, prior_pnls)`: extracted from `:116-133`;
  - `monte_carlo_summary(pnl, returns)`;
  - `equity_replay(exit_times, returns, max_points=2000)`;
  - `slim_trades(trades_detailed)`.
- Create `src/fwbg/optimization/holdout.py`: `process_symbol_holdout`, `symbol_worker`, `holdout_fold`, `development_summary`, `HOLDOUT_STATUSES`.
- Modify `src/fwbg/cli/main.py:45, :578`: `worker_func = symbol_worker(strategy)`, replacing the `process_symbol` import.
- Tests:
  - `tests/optimization/test_holdout.py` (new);
  - `tests/optimization/test_causal_reporting.py` (helper cases);
  - `tests/optimization/test_date_window.py` (`load_symbol_frame`);
  - `tests/research/test_run_cli.py` (dispatch);
  - `tests/optimization/symbol_harness.py` (`patch_holdout`).

**Harness addition:**

```python
from fwbg.optimization import holdout  # with the other imports

HOLDOUT = [(2.0, 1.0)] * 21 + [(-1.0, 0.0)] * 9


def patch_holdout(monkeypatch, pairs=HOLDOUT, evaluations=None) -> list[dict]:
    """The holdout's single unified-simulation call; records what it was given."""
    calls = []

    def simulate(folds, candidate, fold_indicators, precomputed_raw_df,
                 preprocessing_configs, ctx, sym, prepare_data_fn=None):
        calls.append({"folds": folds, "candidate": candidate, "ctx": ctx,
                      "prepare": prepare_data_fn})
        return evaluations if evaluations is not None else [unified_fold(folds[0], pairs)]

    monkeypatch.setattr(holdout, "run_unified_simulation", simulate)
    return calls
```

**Step 1: Failing tests** (`tests/optimization/test_holdout.py`, key cases)

```python
from dataclasses import replace

import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig, ValidationConfig
from fwbg.optimization import holdout
from fwbg.optimization.causal_reporting import sized_returns
from fwbg.optimization.holdout import (
    HOLDOUT_STATUSES, process_symbol_holdout, symbol_worker,
)
from fwbg.optimization.process import process_symbol
from fwbg.optimization.signal_fold import prepare_signal_fold_data
from fwbg.research.dsr import return_statistics
from fwbg.simulation.trade import monte_carlo_permutation_test
from fwbg.statuses import EXECUTED_STATUSES, NOT_EXECUTED_STATUSES
from tests.optimization.symbol_harness import (
    CUTOFF, HOLDOUT, LOSING, STRATEGY, SYMBOL_FILE, WINNING,
    hourly_frame, patch_development, patch_holdout,
)

LOOK = replace(STRATEGY, holdout_cutoff=CUTOFF)


def test_holdout_cutoff_parses_from_the_metadata():
    assert StrategyConfig.from_dict({"holdout_cutoff": CUTOFF}).holdout_cutoff == CUTOFF


def test_worker_is_the_holdout_only_with_a_cutoff():
    assert symbol_worker(LOOK).func is process_symbol_holdout
    assert symbol_worker(STRATEGY).func is process_symbol
    assert symbol_worker(LOOK).keywords == {"strategy": LOOK}


def test_development_sees_only_data_before_the_cutoff(monkeypatch):
    seen = []
    monkeypatch.setattr(holdout, "process_symbol", lambda path, strategy: seen.append(strategy)
                        or {"symbol": "EURUSD", "status": "no_edge"})
    process_symbol_holdout(SYMBOL_FILE, replace(LOOK, start_date="2022-02-01"))
    [development] = seen
    assert (development.start_date, development.end_date, development.holdout_cutoff) == (
        "2022-02-01", CUTOFF, None)


@pytest.mark.parametrize("status", ["no_edge", "not_significant", "no_successful_folds",
                                    "no_unified_trades", "error"])
def test_failed_development_records_its_outcome_and_never_looks(monkeypatch, status):
    monkeypatch.setattr(holdout, "process_symbol",
                        lambda path, strategy: {"symbol": "EURUSD", "status": status,
                                                "sharpe": 0.3, "fold_stability": 0.5})
    monkeypatch.setattr(holdout, "load_symbol_frame", pytest.fail)
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == status
    assert result["holdout"] == {"cutoff": CUTOFF, "evaluated": False,
                                 "reason": f"development ended in {status}"}
    assert result["development"]["status"] == status
    assert not {"sharpe", "fold_stability", "tr_trace", "monte_carlo", "walk_forward",
                "reporting", "return_stats"} & result.keys()


def test_holdout_trains_on_development_minus_embargo_and_tests_after_the_cutoff(monkeypatch):
    look = replace(LOOK, validation=ValidationConfig(folds=4, embargo_bars=5))
    patch_development(monkeypatch, WINNING)
    calls = patch_holdout(monkeypatch)
    development = process_symbol(SYMBOL_FILE, replace(look, end_date=CUTOFF, holdout_cutoff=None))
    process_symbol_holdout(SYMBOL_FILE, look)
    [call] = calls
    [fold] = call["folds"]
    frame = hourly_frame()
    n_dev = int(frame.index.searchsorted(pd.Timestamp(CUTOFF)))
    assert fold.train_df.index[0] == frame.index[0]
    assert fold.train_df.index[-1] == frame.index[n_dev - 6]       # 5 embargo bars left out
    assert fold.test_df.index[0] == frame.index[n_dev] >= pd.Timestamp(CUTOFF)
    assert fold.test_df.index[-1] == frame.index[-1]
    assert call["candidate"] == development["unified_candidate"]
    assert call["prepare"] is None                                  # ML preparation


def test_holdout_metrics_use_development_sizing_and_only_holdout_trades(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    development = process_symbol(SYMBOL_FILE, replace(STRATEGY, end_date=CUTOFF))
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    config = development["config"]
    trades = [{"pnl_raw": pnl, "result": res} for pnl, res in HOLDOUT]
    returns, pnl, _ = sized_returns(trades, config["risk_per_trade"],
                                    config["circuit_breaker"], development["tr_trace"])
    scale = sum(-p for p in development["tr_trace"] if p < 0) / sum(
        1 for p in development["tr_trace"] if p < 0)
    assert returns[0] == pytest.approx(config["risk_per_trade"] * 2.0 / scale)
    assert result["status"] == "ok"
    assert result["reporting"]["pnl_trace"] == pnl
    assert result["return_stats"] == return_statistics(returns)
    assert result["monte_carlo"]["p_value"] == monte_carlo_permutation_test(pnl)["p_value"]
    assert result["config"] == config and result["rrr"] == development["rrr"]
    assert not {"fold_stability", "walk_forward", "grid_results", "overfitting"} & result.keys()
    assert result["development"]["fold_stability"] == development["fold_stability"]
    assert "fold_details" not in result["development"]["walk_forward"]
    assert result["holdout"]["evaluated"] is True and result["holdout"]["cutoff"] == CUTOFF


def test_holdout_without_trades_is_no_unified_trades(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch, evaluations=[
        {"trades": [], "trades_detailed": [], "pnl": 0, "win_rate": 0, "n_trades": 0}])
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == "no_unified_trades" and result["holdout"]["evaluated"] is True


def test_too_short_holdout_window_is_not_evaluated(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch, evaluations=[])
    result = process_symbol_holdout(SYMBOL_FILE, LOOK)
    assert result["status"] == "insufficient_data"
    assert result["holdout"]["evaluated"] is False


def test_no_data_after_the_cutoff_is_not_evaluated(monkeypatch):
    patch_development(monkeypatch, WINNING)
    patch_holdout(monkeypatch)
    result = process_symbol_holdout(SYMBOL_FILE, replace(LOOK, holdout_cutoff="2022-11-30"))
    # Development (end 2022-11-30) covers every bar; nothing lies after the cutoff.
    assert result["status"] == "no_data" and result["holdout"]["evaluated"] is False


def test_signal_models_prepare_the_holdout_like_their_folds(monkeypatch):
    signal = replace(LOOK, model=replace(LOOK.model, type="signal"),
                     filters=replace(LOOK.filters, allowed_hours=[8, 9]))
    patch_development(monkeypatch, WINNING)
    [call] = (calls := patch_holdout(monkeypatch)) or calls
    ...
```

For the signal case, write it plainly: call `process_symbol_holdout`, then assert `calls[0]["prepare"] is prepare_signal_fold_data`.

```python
def test_every_holdout_status_is_classified():
    assert HOLDOUT_STATUSES <= EXECUTED_STATUSES | NOT_EXECUTED_STATUSES
```

Other tests in this task:
- `test_causal_reporting.py`:
  - `test_sized_returns_apply_the_circuit_breaker_and_prior_loss_scale`: `fk = 0.02`; the circuit breaker pauses 1 trade after 2 losses; prior pnl `[-2, -4, 3]` gives scale 3. Assert the returns, the pnl list and the taken indices.
  - `test_monte_carlo_summary_keys`: the seven keys that `process.py` reported before.
- `test_date_window.py`: `test_load_symbol_frame_applies_the_half_open_window`, which patches `fwbg.optimization.process.load_data_aligned`; and a case where it returns `None` when no bar is left.
- `test_run_cli.py`: `test_run_optimizer_dispatches_a_holdout_look`, modelled on `test_run_optimizer_runs_with_the_gated_metadata`. `before_run` returns `{**metadata, "holdout_cutoff": "2024-01-01"}`; assert `captured[0].func is process_symbol_holdout`.

Expected: FAIL (module and helpers missing).

**Step 2: Implementation**

In `causal_reporting.py`, inside the fold loop of `compute_causal_reporting`, replace the CB mask, scale and append block with:
```python
                returns, pnl, taken = sized_returns(fold_trades, fk, risk_result["circuit_breaker"], prior_pnls)
                causal_returns.extend(returns); causal_pnl.extend(pnl)
                causal_trade_indices.extend(flat_offset + i for i in taken)
                excluded_trades += len(fold_trades) - len(taken)
```

```python
def sized_returns(trades, fk, circuit_breaker, prior_pnls):
    """Size *trades* with risk parameters calibrated beforehand (no hindsight).

    fk and circuit breaker come from earlier data; the loss scale (cf.
    pnl_to_returns) is the mean loss of *prior_pnls*. Returns (returns, pnl,
    indices of the trades the circuit breaker let through).
    """
    mask = circuit_breaker_mask(
        [t["result"] for t in trades],
        circuit_breaker["pause_after_losses"] if circuit_breaker.get("enabled") else 0,
        circuit_breaker.get("pause_bars", 0),
    )
    losses = [abs(p) for p in prior_pnls if p < 0]
    scale = float(np.mean(losses)) if losses else float(np.mean(np.abs(prior_pnls))) or 1.0
    taken = [i for i, keep in enumerate(mask) if keep]
    pnl = [trades[i]["pnl_raw"] for i in taken]
    return [fk * p / scale for p in pnl], pnl, taken
```

`monte_carlo_summary(pnl, returns)` calls `monte_carlo_permutation_test(pnl, n_permutations=1000)` and `monte_carlo_equity_from_returns(returns, n_simulations=500)`. It returns the dict `process.py:976-984` built, using the same keys. `process.py` then reads `mc_summary["is_significant"]`, `["p_value"]`, `["equity_median"]` and `["bankruptcy_rate"]`. `equity_replay` and `slim_trades` are moved unchanged from `process.py:1002-1011` and `:1019-1029`; the alignment check stays in the caller.

`src/fwbg/optimization/holdout.py` (about 220 LoC):

```python
"""Holdout look: develop before the card's cutoff, evaluate once after it.

Option A (research register step 5): walk-forward, selection, unified
simulation and risk calibration see only [start, cutoff) — process_symbol with
end_date = cutoff. The frozen unified setting is then retrained once on every
development bar except the embargo and evaluated once on [cutoff, end), sized
with the development's fk, circuit breaker and loss scale. Reported metrics
come from the holdout trades alone; the development outcome is kept apart.

Causality: raw (stationary-free) indicators are precomputed on the whole
window [start, end) so the first holdout bars have their warm-up. That gives
development rows the same values as an iteration only because registered
indicators are causal (pinned by tests/test_lookahead_bias.py). Preprocessors
and stateful indicators are fit on the training frame only (process_fold).
"""

HOLDOUT_STATUSES = frozenset({"ok", "not_significant", "no_unified_trades",
                              "insufficient_data", "no_data"})
_DEVELOPMENT_KEYS = ("status", "reason", "error", "pnl", "win_rate", "rrr", "sharpe",
                     "calmar", "fold_stability", "test_period_years", "monte_carlo",
                     "return_stats", "overfitting", "config", "best_config",
                     "indicator_variant")


def symbol_worker(strategy):
    """The per-symbol worker of a run: a holdout look when the gate set a cutoff."""
    worker = process_symbol_holdout if strategy.holdout_cutoff else process_symbol
    return partial(worker, strategy=strategy)


def process_symbol_holdout(csv_path, strategy):
    cutoff = strategy.holdout_cutoff
    development = process_symbol(
        csv_path, dataclasses.replace(strategy, end_date=cutoff, holdout_cutoff=None))
    status = development.get("status")
    if status != "ok":
        return _not_evaluated(development, cutoff, f"development ended in {status}")
    sym = development["symbol"]
    report_phase(sym, "Holdout evaluation...")
    try:
        result = _evaluate(csv_path, strategy, development)
    except Exception as exc:  # like process_symbol: a crash is this symbol's error
        tb = traceback.format_exc()
        log(1, f"Holdout evaluation failed: {exc}", sym)
        result = _not_evaluated(development, cutoff, f"holdout evaluation failed: {exc}",
                                status="error")
        result["error"] = f"{exc}\n{tb}"
    report_done(sym, result["status"])
    return result
```

`_evaluate(csv_path, strategy, development)`:
1. `df = load_symbol_frame(csv_path, strategy, sym)`, then `n_dev = int(df.index.searchsorted(pd.Timestamp(cutoff)))`. If `df is None` or `n_dev == len(df)`, return `_not_evaluated(..., "no data after the cutoff", status="no_data")`.
2. `variant = _variant(strategy, development.get("indicator_variant"))`, using `expand_indicator_grid`: the single variant, or the one whose label matches.
3. `ctx = SimulationContext.create(get_asset(sym), variant)`.
4. `fold = holdout_fold(df, n_dev, ctx.embargo_bars)`.
5. `fold_indicators, precomputed_raw_df, _ = precompute_indicators(df, variant, sym)`.
6. `prepare = prepare_signal_fold_data if ctx.model_type == "signal" else None`.
7. Run the single simulation:
   ```python
   # ponytail: one-fold run_unified_simulation (logs say "Unified Simulation") so the
   # holdout builds its context exactly like development; a dedicated function only if it diverges.
   evaluations = run_unified_simulation([fold], development["unified_candidate"],
       fold_indicators, precomputed_raw_df, variant.get_preprocessing(), ctx, sym,
       prepare_data_fn=prepare)
   ```
8. An empty list gives `_not_evaluated(..., "holdout window too short to evaluate", status="insufficient_data")`. Otherwise return `_holdout_result(development, evaluations[0], fold, cutoff, ctx.embargo_bars)`.

Add a module-level comment on `load_symbol_frame` being called twice: `# ponytail: data loaded and raw indicators precomputed a second time for the full window; cache only if holdout runs get frequent.`

`holdout_fold(df, n_dev, embargo_bars)` returns `WalkForwardFold(fold_id=0, train_start=0, train_end=max(0, n_dev - embargo_bars), test_start=n_dev, test_end=len(df), train_df=df.iloc[:train_end].copy(), test_df=df.iloc[n_dev:].copy())`.

`_holdout_result(development, evaluation, fold, cutoff, embargo)`:
1. `config = development["config"]`.
2. `returns, pnl, taken = sized_returns(evaluation["trades"], config["risk_per_trade"], config["circuit_breaker"], development["tr_trace"])`.
3. `years = len(fold.test_df) / (data_config.tf_cfg["bars_per_hour"] * 24 * 250)`.
4. `holdout = {"cutoff", "evaluated": True, "train_start", "train_end", "test_start", "test_end" (ISO of the first and last bars), "bars", "embargo_bars", "trades": len(evaluation["trades"]), "sized_trades": len(pnl)}`.
5. If there are no trades, return `{"symbol", "status": "no_unified_trades", "holdout", "development": development_summary(development)}`.
6. Otherwise:
   - `mc = monte_carlo_summary(pnl, returns)`;
   - `sharpe = calculate_sharpe_ratio(returns, trades_per_year=len(pnl) / years if years > 0 else len(pnl))`;
   - `equity = equity_replay([detailed[i]["exit_time"] for i in taken], returns)` if the detailed trades are aligned (same length and all have `exit_time`), else `None`;
   - status `"ok" if mc["is_significant"] else "not_significant"`.
7. The result dict has: `symbol`, `status`, `pnl` (sum of sized pnl), `win_rate` (over all holdout trades), `rrr` (development), `sharpe`, `calmar` (`calculate_calmar_from_returns(returns)`), `tr_trace` (raw pnl of all holdout trades), `test_period_years` (years), `config` (development), `currencies` (development), `monte_carlo`, `return_stats` (`return_statistics(returns)`), `reporting` (`{"causal": True, "n_trades": len(pnl), "pnl_trace": pnl, "equity": equity}`), `trades_detailed` (`slim_trades(detailed)`), `holdout`, and `development`.

`development_summary(development)` copies `_DEVELOPMENT_KEYS` that are present, plus `walk_forward` without `fold_details`. `_not_evaluated(development, cutoff, reason, status=None)` returns `{"symbol", "status": status or development["status"], "holdout": {"cutoff", "evaluated": False, "reason"}, "development": development_summary(development)}`, with the development `error` copied when `status` is `None`.

**Step 3:** The task's tests and the whole of `tests/optimization` must pass. `test_causal_reporting.py` guards that the extraction keeps behaviour.

**Commit:** `feat(optimizer): evaluate a holdout look once after the cutoff`

---

### Task 5: Write the holdout result and record it in the ledger

**Files**
- Modify `src/fwbg/results/symbol_files.py`: add `"holdout"` to `_CONFIG_BLOCKS`; new `write_development(sym_dir, result, encoder)`, which writes `development.json` when `result["development"]` is present.
- Modify `src/fwbg/cli/main.py` (`on_result_ready`, one call after `config.json`): `write_development(sym_dir, result, _SafeJsonEncoder)`.
- Modify `src/fwbg/results/storage.py:347`: `"fold_stability": r.get("fold_stability"),` (finding 2).
- Test: `tests/optimization/test_holdout_files.py` (new).

**Step 1: Failing tests**

```python
import json

from fwbg.research.legacy import run_outcome
from fwbg.research.ledger import TrialStatus
from tests.optimization.symbol_harness import (
    CUTOFF, LOSING, WINNING, patch_development, patch_holdout, run_inline,
)


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
    status, metrics = run_outcome(run_dir)
    assert status is TrialStatus.OK
    symbol = metrics["EURUSD"]
    assert symbol["fold_stability"] is None
    assert symbol["p_value"] == config["monte_carlo"]["p_value"]
    assert (symbol["sharpe"], symbol["sr_per_trade"]) == (unified["sharpe"], unified["sr_per_trade"])
    elite = json.loads((run_dir / "results.json").read_text())["elite_results"]
    assert elite[0]["fold_stability"] is None


def test_holdout_after_a_failed_development_records_no_metrics(monkeypatch, tmp_path):
    patch_development(monkeypatch, LOSING)
    patch_holdout(monkeypatch)
    run_dir = run_inline(monkeypatch, tmp_path, holdout_cutoff=CUTOFF)
    sym_dir = run_dir / "grid_details" / "EURUSD"
    config = json.loads((sym_dir / "config.json").read_text())
    assert config["status"] == "no_edge"
    assert config["holdout"] == {"cutoff": CUTOFF, "evaluated": False,
                                 "reason": "development ended in no_edge"}
    assert not (sym_dir / "unified_metrics.json").exists()
    assert json.loads((sym_dir / "development.json").read_text())["status"] == "no_edge"
    _, metrics = run_outcome(run_dir)
    assert all(value is None for value in metrics["EURUSD"].values())  # counted, no metrics
```

Expected: FAIL (no `holdout` block, no `development.json`, elite `fold_stability == 0`).

**Step 2:** Implement as listed. If `save_run_results`, `_write_summary` or the elite loop reads a key the holdout result lacks, add that key to the holdout result in `holdout.py`, not as a default in storage.

**Step 3:** The tests, `tests/research` and `tests/optimization` must pass.

**Commit:** `feat(research): write holdout looks apart from their development`

---

### Task 6: `--holdout` through gate, CLI and API

**Files**
- Modify `src/fwbg/research/gate.py`:
  - `_apply_window`: for a holdout, refuse `start >= cutoff` right after parsing (before the open-end return).
  - `check`: refuse a `holdout_cutoff` the caller supplied (`INVALID_CONFIG`); for a holdout, set `metadata["holdout_cutoff"] = card.holdout_cutoff.isoformat()` before fingerprinting.
  - Docstrings of `RunRequest` and `check`.
- Modify `src/fwbg/research/runs.py`:
  - `add_run_arguments`: `--holdout`;
  - `ResearchRun.holdout: bool = False`, passed into `RunRequest(..., holdout=self.holdout)`;
  - `_print_check`: one extra line for holdout looks: `split: development .. {cutoff} | holdout {cutoff} .. {end or 'open'}`.
- Modify `src/fwbg/cli/main.py`: `holdout=args.holdout` in `ResearchRun(...)`. In the gate-off branch: `if args.holdout: parser.error(f"--holdout needs the research gate ({GATE_ENV} is off)")`. The existing `--check` message stays unchanged.
- Modify `src/fwbg/api/runs.py` (`RunStartRequest.holdout: bool = False`) and `src/fwbg/api/run_preflight.py` (`cli_command`: `if body.holdout: cmd.append("--holdout")`).
- Modify the `src/fwbg/research/study.py` docstring: "Studies record iterations only; a holdout look runs through `python -m fwbg.cli --holdout`. A config carrying `holdout_cutoff` is refused."
- Tests:
  - `tests/research/test_gate.py`, `tests/research/test_runs.py`, `tests/research/test_run_cli.py`;
  - `tests/test_api_run_research.py`, `tests/research/test_study.py`.

**Step 1: Failing tests**

`test_gate.py`:
```python
def test_holdout_records_the_card_cutoff_in_its_configuration():
    decision = check(make_card(), request(holdout=True), [])
    assert decision.effective_metadata["holdout_cutoff"] == "2025-04-30"
    assert decision.fingerprint == fingerprint(decision.effective_metadata, SYMBOLS)
    assert decision.fingerprint != check(make_card(), request(), []).fingerprint
    assert "holdout_cutoff" not in METADATA


@pytest.mark.parametrize("end", [None, "2025-09-01"])
@pytest.mark.parametrize("start", ["2025-04-30", "2025-06-01"])
def test_holdout_must_start_before_the_cutoff(start, end):
    metadata = {**METADATA, "start_date": start, **({"end_date": end} if end else {})}
    decision = check(make_card(), request(strategy_metadata=metadata, holdout=True), [])
    assert codes(decision) == {DenialCode.INVALID_WINDOW}


def test_holdout_starting_before_the_cutoff_is_allowed():
    metadata = {**METADATA, "start_date": "2024-01-01"}
    assert check(make_card(), request(strategy_metadata=metadata, holdout=True), []).allowed


@pytest.mark.parametrize("holdout", [False, True])
def test_a_configuration_must_not_set_the_holdout_cutoff(holdout):
    metadata = {**METADATA, "holdout_cutoff": "2025-04-30"}
    decision = check(make_card(), request(strategy_metadata=metadata, holdout=holdout), [])
    assert codes(decision) == {DenialCode.INVALID_CONFIG}
```

`test_runs.py`, `test_holdout_run_is_on_record_as_a_holdout`:
- `ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "look", holdout=True).before_run(METADATA, ["eurusd"])` returns metadata with `holdout_cutoff == "2025-04-30"` and no `end_date`;
- `read_pending(gate_register)[0].trial.kind is TrialKind.HOLDOUT`;
- the marker config's strategy carries `holdout_cutoff`.

`test_run_cli.py`:
- Give `FakeOptimizer` `self.metadata = metadata` after `before_run`.
- `test_holdout_flag_runs_a_holdout_look`: `--holdout` gives a trial with kind `HOLDOUT`, and `gated.fake.metadata["holdout_cutoff"] == "2025-04-30"`.
- `test_check_json_reports_a_holdout`: payload `kind == "holdout"`.
- `test_gate_off_refuses_the_holdout_flag`: exit 2, `"--holdout needs the research gate"` in stderr, no call.

`test_api_run_research.py`:
- `test_holdout_reaches_the_cli_and_its_preflight`: `_start(client, research_id="S-0001", note="n", holdout=True)` puts `"--holdout"` in `calls["spawn"][0]` and `calls["preflight"] == [[*cmd, "--check-json"]]`.
- `test_holdout_is_off_by_default`: `"--holdout"` not in the command.
- Extend `test_research_text_starting_with_a_dash_parses_in_the_cli`, or add a case, so that `parse_known_args` sees `args.holdout is True`.

`test_study.py`:
- `test_a_study_cannot_look_at_the_holdout`: `record(gate_register, config={**CONFIG, "holdout_cutoff": "2025-04-30"})` raises `GateDenied` with `[DenialCode.INVALID_CONFIG]`; the ledger stays empty.

Run `tests/research/test_gate.py tests/research/test_runs.py tests/research/test_run_cli.py tests/research/test_study.py tests/test_api_run_research.py`. Expected: FAIL.

**Step 2: Implementation (gate)**

```python
    if "holdout_cutoff" in request.strategy_metadata:
        denials.append(Denial(
            DenialCode.INVALID_CONFIG,
            "holdout_cutoff is set by the register for holdout runs; remove it from the strategy",
        ))
    window_denial = _apply_window(card, metadata, request.holdout)
    ...
    if request.holdout:
        # The worker develops on [start, cutoff) and looks at [cutoff, end); fingerprinted.
        metadata["holdout_cutoff"] = card.holdout_cutoff.isoformat()
```

In `_apply_window`, right after the parse loop:

```python
    if holdout and start is not None and start >= cutoff:
        return Denial(
            DenialCode.INVALID_WINDOW,
            f"a holdout run develops on data before the cutoff {cutoff}; start_date is {start}",
        )
```

Help text for `--holdout`: "look at the card's holdout once: develop before its cutoff, evaluate after it". `main.py` grows by 3 lines.

**Step 3:** The listed tests, the whole of `tests/research`, and `tests/test_api_run_spawn.py` / `tests/test_api_run_start_validation.py` must pass.

**Commits:** `feat(research): gate holdout looks and hand them the cutoff`, then `feat(api): start holdout looks`

---

### Task 7: Verdict statistics, DSR and holdout result

**Files**
- Modify `src/fwbg/research/stats.py`: `MIN_RETURN_SAMPLES`, `ReturnSample`, `DeflatedBest`, `return_samples`, `deflated_best`.
- Modify `src/fwbg/research/commands.py` (`verdict_entry`; new `_dsr_line`, `_holdout_lines`).
- Tests: `tests/research/test_stats.py`, `tests/research/test_commands.py`.

**Step 1: Failing tests**

`test_stats.py`:
```python
INPUTS = {"sr_per_trade": 0.1, "n_returns": 101, "skew": 0.0, "kurtosis": 3.0}


def test_return_samples_need_all_inputs_from_counted_development_trials():
    trials = [
        make_trial(trial_id="T-000001", metrics={"A": INPUTS, "B": {"sharpe": 0.0}}),  # B legacy
        make_trial(trial_id="T-000002", metrics={"A": {**INPUTS, "kurtosis": None}}),
        make_trial(trial_id="T-000003", metrics={"A": {**INPUTS, "n_returns": 1}}),
        make_trial(trial_id="T-000004", kind=TrialKind.HOLDOUT, metrics={"A": INPUTS}),
        make_trial(trial_id="T-000005", status=TrialStatus.FAILED, metrics={"A": INPUTS}),
        make_trial(trial_id="T-000006", kind=TrialKind.SMOKE, metrics={"A": INPUTS}),
    ]
    assert [(s.trial_id, s.symbol) for s in return_samples(trials)] == [("T-000001", "A")]


def test_deflated_best_needs_two_samples():
    assert deflated_best([make_trial(trial_id="T-000001", metrics={"A": INPUTS})]) is None


def test_deflated_best_deflates_the_best_sample_by_all_evaluations():
    trials = [
        make_trial(trial_id="T-000001", metrics={"A": INPUTS, "X": {"sharpe": None}}),  # no_edge
        make_trial(trial_id="T-000002", metrics={"A": {**INPUTS, "sr_per_trade": 0.3}}),
        make_trial(trial_id="T-000003", kind=TrialKind.LEGACY, metrics={"L": {"sharpe": 0}}),
        make_trial(trial_id="T-000004", kind=TrialKind.HOLDOUT,
                   metrics={"A": {**INPUTS, "sr_per_trade": 0.9}}),
    ]
    result = deflated_best(trials)
    assert (result.best.trial_id, result.best.sr) == ("T-000002", 0.3)
    assert (result.n_trials, result.samples) == (5, 2)            # N counts every evaluation
    assert result.sr_variance == pytest.approx(statistics.variance([0.1, 0.3]))
    assert result.dsr == pytest.approx(deflated_sharpe_ratio(0.3, 101, 0.0, 3.0, 5, 0.02))


def test_deflated_best_is_undefined_for_impossible_moments():
    bad = {**INPUTS, "sr_per_trade": 1.0, "skew": 3.0, "kurtosis": 1.0}
    trials = [make_trial(trial_id="T-000001", metrics={"A": INPUTS, "B": bad})]
    assert deflated_best(trials).dsr is None
```

`test_commands.py`:
- Update `test_verdict_closes_the_card_with_its_statistics`: its last line becomes `"Deflated Sharpe ratio: too few trials with return statistics (0 of 2 needed).\n"`.
- `test_verdict_reports_the_deflated_sharpe_ratio`: two trials carrying the `INPUTS` (SR 0.1 and 0.3). Build the expected line from `deflated_best(read_trials(...))` with the format below and assert it is in the body.
- `test_verdict_reports_holdout_results_against_the_criteria`: a holdout trial with `metrics={"GOLD": {"sharpe": 1.4, "trades": 80, "p_value": 0.01, "fold_stability": None}, "EURUSD": {"p_value": None, "fold_stability": None}}`. The body contains:
  - `"Holdout result T-000002 (window open..open, cutoff 2025-04-30): GOLD min_sharpe pass (1.4 >= 1); min_trades fail (80 < 100) | EURUSD no holdout metrics."`
  - Use the default card criteria (`min_sharpe`, `min_trades`) and a window `{"start": None, "end": None}`.
- `test_verdict_reports_a_failed_holdout_look`: `"Holdout result T-000002 (...): failed, no symbol evaluated."`

**Step 2: Implementation**

`stats.py` (stays under 140 LoC):

```python
MIN_RETURN_SAMPLES = 2
_RETURN_INPUTS = ("sr_per_trade", "n_returns", "skew", "kurtosis")


@dataclass(frozen=True)
class ReturnSample:
    trial_id: str
    symbol: str
    sr: float
    n_returns: int
    skew: float
    kurtosis: float


@dataclass(frozen=True)
class DeflatedBest:
    best: ReturnSample
    n_trials: int        # N: counted symbol evaluations of the card (holdout looks included)
    samples: int         # development evaluations V[SR] was estimated from
    sr_variance: float
    expected_max_sr: float
    dsr: float | None    # None: PSR undefined for the best's skew and kurtosis


def return_samples(trials: Iterable[Trial]) -> list[ReturnSample]:
    """Development evaluations that carry the DSR's return statistics (runs from step 5 on).

    Legacy and no_edge evaluations carry none: they count towards N only.
    """
    samples = []
    for trial in development_trials(trials):
        for symbol, values in trial.metrics.items():
            sr, n, skew, kurtosis = (values.get(key) for key in _RETURN_INPUTS)
            if all(isinstance(v, (int, float)) for v in (sr, n, skew, kurtosis)) and n >= 2:
                samples.append(ReturnSample(str(trial.trial_id), symbol, float(sr), int(n),
                                            float(skew), float(kurtosis)))
    return samples


def deflated_best(trials: Sequence[Trial]) -> DeflatedBest | None:
    """DSR of the best development evaluation by per-trade SR; None below MIN_RETURN_SAMPLES."""
    samples = return_samples(trials)
    if len(samples) < MIN_RETURN_SAMPLES:
        return None
    best = max(samples, key=lambda sample: sample.sr)
    variance = statistics.variance(sample.sr for sample in samples)
    n_trials = evaluations(trials)
    expected = expected_max_sharpe(n_trials, variance)
    dsr = probabilistic_sharpe_ratio(best.sr, expected, best.n_returns, best.skew, best.kurtosis)
    return DeflatedBest(best, n_trials, len(samples), variance, expected, dsr)
```

`commands.py`:
- `verdict_entry` drops the placeholder line.
- After the best-result block, it appends `_dsr_line(own)` and then `*_holdout_lines(card, own)`.
- The existing `Holdout: used by …` line stays.

```python
def _dsr_line(own: Sequence[Trial]) -> str:
    result = deflated_best(own)
    if result is None:
        have = len(return_samples(own))
        return (f"Deflated Sharpe ratio: too few trials with return statistics "
                f"({have} of {MIN_RETURN_SAMPLES} needed).")
    best = result.best
    basis = (f"per-trade SR {best.sr:.4f} over {best.n_returns} returns; "
             f"N={result.n_trials} symbol evaluations; V[SR]={result.sr_variance:.3g} from "
             f"{result.samples} evaluations with return statistics; "
             f"E[max SR]={result.expected_max_sr:.4f}")
    if result.dsr is None:
        return (f"Deflated Sharpe ratio: not defined for {best.trial_id} {best.symbol} "
                f"(its skew and kurtosis leave no positive SR variance; {basis}).")
    return f"Deflated Sharpe ratio: {result.dsr:.3f} for {best.trial_id} {best.symbol} ({basis})."


def _holdout_lines(card: ResearchCard, own: Sequence[Trial]) -> list[str]:
    lines = []
    for trial in own:
        if trial.kind is not TrialKind.HOLDOUT:
            continue
        window = f"{trial.window.get('start') or 'open'}..{trial.window.get('end') or 'open'}"
        head = f"Holdout result {trial.trial_id} (window {window}, cutoff {trial.holdout_cutoff})"
        if trial.status is not TrialStatus.OK:
            lines.append(f"{head}: failed, no symbol evaluated.")
            continue
        parts = [
            f"{symbol} " + "; ".join(criteria_check(card.success_criteria, values))
            if any(isinstance(v, (int, float)) for v in values.values())
            else f"{symbol} no holdout metrics"
            for symbol, values in trial.metrics.items()
        ]
        lines.append(f"{head}: {' | '.join(parts)}.")
    return lines
```

Note: `min_fold_stability` will show `n/a` for holdout trials, by design.

**Step 3:** `tests/research` must pass; run mypy on the research package. `commands.py` ends up around 370 LoC.

**Commit:** `feat(research): report the deflated Sharpe ratio and holdout results in verdicts`

---

### Task 8: Docs

**Files**
- Modify `docs/plans/2026-10-19-research-register-design.md`.
- Modify `CLAUDE.md` and `AGENTS.md` (Research register rule).

**Design doc changes:**
- **Status:** steps 1-5 implemented.
- **Problem paragraph:** the nested-CV `holdout_ratio` is hard-coded to 0.0 in `process_fold.py:418`. The register's holdout replaces it.
- **Fingerprint:** a holdout run's metadata carries `holdout_cutoff`, which the gate writes. A strategy may not set it (`INVALID_CONFIG`).
- **Gate rule 5:** a holdout's start must lie before the cutoff (422).
- **API section:** `RunStartRequest.holdout` (done).
- **`register_trial`:** iterations only.
- **New section "Holdout look (step 5)"**, which replaces the open question:
  - Option A mechanics and the embargo;
  - sizing without hindsight;
  - metrics from holdout trades only;
  - files: `config.json` `holdout` block, `development.json`, `unified_metrics.json`;
  - statuses (`HOLDOUT_STATUSES`; development outcome with `evaluated: false`);
  - the causality assumption (pointer to `tests/test_lookahead_bias.py`);
  - a failed look still counts as a look.
- **Ledger:** the new metric keys; `no_edge` writes `sharpe: null` from step 5; the ledger has no per-symbol status, so legacy zeros count in N only; `no_edge` rows still carry naive `max_drawdown` and `annual_return`.
- **Verdict:**
  - DSR definition: N, V[SR] from development evaluations with return statistics, best by per-trade SR, minimum 2, the paper's formula;
  - holdout result lines;
  - the run-level `overfitting.dsr` in `config.json` is advisory (unit variance, N = grid combinations) and is not the verdict DSR.
- **Known limitation:** signal models select their exit combination on the test fold (`signal_fold.py:71-148`), which inflates development metrics and the DSR's best SR. Fixing it is a later step.
- **Testing:** DSR checked against the paper (done).

**Rule line** (both files, after the `--reason` bullet):
```markdown
- A holdout look (`--holdout`) evaluates the frozen setting once on data after
  the card's cutoff; use it when iterating is done. A further look needs `--reason`.
```

**Commit:** `docs: record step 5 of the research register`

---

## After all tasks

- Final whole-branch review.
- Full suite: `PYTHONPATH=src ../../.venv/bin/python -m pytest -p no:cacheprovider -q`. The 6 known environment failures from step 4 remain.
- Smoke test on a scratch copy of the register (remote removed), with `FWBG_RESEARCH_DIR` and the results directory in the scratch dir, on a cheap real strategy (DAY_1):
  1. Run an iteration; then `--check --holdout` shows the split.
  2. A `--holdout` run writes `config.json` with the `holdout` block, plus `development.json` and holdout `unified_metrics.json`.
  3. The trial has kind `holdout` and `fold_stability` null.
  4. A second `--holdout` without `--reason` is refused.
  5. A `research verdict` on the scratch card shows the DSR line (or "too few") and the holdout result.

### Critical Files for Implementation
- src/fwbg/optimization/holdout.py (new; `process_symbol_holdout`, `symbol_worker`)
- src/fwbg/optimization/process.py (`load_symbol_frame` extraction, `return_stats`, `no_edge` `sharpe: None`, `unified_candidate`)
- src/fwbg/research/gate.py (holdout start check, `holdout_cutoff` injection and refusal)
- src/fwbg/research/stats.py and src/fwbg/research/dsr.py (new; DSR maths and verdict statistics)
- src/fwbg/cli/main.py (`on_result_ready` file writes, worker dispatch, `--holdout` wiring)
