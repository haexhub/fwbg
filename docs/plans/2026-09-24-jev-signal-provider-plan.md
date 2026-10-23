# Jev Signal Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a provider-agnostic batch pipeline that asks Jev-style decision
models (official TypeSafe Jev + itemis's self-hosted SemIf) a per-bar
triple-barrier question for EUR/USD, caches both providers' answers, computes
calibration/accuracy/agreement metrics directly from the cache, and exposes
the result to fwbg's existing `models/signal` plugin via a new `jev_signal`
DataLoader plugin — no changes to fwbg's model/validation framework.

**Architecture:** See `docs/plans/2026-09-24-jev-signal-provider-design.md`
for the full rationale. Summary: `scripts/fetch_jev_predictions.py` calls
both providers per bar with an identical `state`+`questions` payload (two
`noul` questions: `is_long_win`, `is_short_win` — mirroring exactly what
`FixedExitStrategy.compute_targets()` / `compute_targets_numba()` already
compute as `targets_long`/`targets_short`), caches results to CSV (columns
`is_long_win`/`is_short_win`, unprefixed — provider identity lives in the
filename, `{asset}_{provider.name}_tp{tp}_sl{sl}.csv`; this is close to but
not exactly the default `{symbol}_{timeframe}.csv` pattern `register_csv_source`
expects out of the box — registering it needs a custom `file_pattern`, not
zero-config compatibility), and a new
`jev_signal` DataLoader plugin maps the cached probability columns to
`_composed_signal_long`/`_composed_signal_short` (with the mandatory 1-bar
shift). `models/signal` (already exists, unmodified) reads those columns.

**Tech Stack:** Python 3.13, `requests` (already a dependency), `pandas`,
`numpy`, fwbg's existing `fwbg_sdk` (`BaseDataLoader`, `register_data_loader`,
`shift_features`), `FixedExitStrategy`/`compute_targets_numba` (reused, not
reimplemented), `pytest`.

**Verified against the real codebase** (not guessed): `SignalModel`
(`src/fwbg/plugins/fwbg-core/models/signal/__init__.py`), `FixedExitStrategy`
+ `compute_targets_numba` (`src/fwbg/plugins/fwbg-core/exit_strategies/fixed/__init__.py`,
`src/fwbg/simulation/numba_core.py:580`), `register_csv_source`
(`src/fwbg/core/data_sources.py:659`), `get_asset("EURUSD")` → spread
0.00018 / point 0.0001 (`src/fwbg/data/assets.py:151`).

**Known unknown, not resolved by this plan:** the *real* response JSON shape
of both Jev APIs. Nobody on this task has made a live call yet. Task 1 below
isolates that risk into one small, swappable function
(`client.parse_response`) with a documented assumption — everything else is
built and tested against that assumed shape so only one function needs to
change once a real response is seen. **Do not run the batch script against
the real APIs before verifying this against an actual response** (see
"Manual verification" at the end).

---

### Task 1: Provider client

**Files:**
- Create: `scripts/jev/__init__.py` (empty)
- Create: `scripts/jev/client.py`
- Test: `tests/jev/test_client.py`
- Create: `tests/jev/__init__.py` (empty)

**Step 1: Write the failing test**

```python
# tests/jev/test_client.py
from unittest.mock import patch, MagicMock

from scripts.jev.client import JevProvider, ask, parse_response

OFFICIAL = JevProvider(
    name="jev_official",
    base_url="https://api.example-jev.invalid/v1/decide",
    api_key_env="JEV_OFFICIAL_API_KEY",
)


def test_ask_builds_correct_request(monkeypatch):
    monkeypatch.setenv("JEV_OFFICIAL_API_KEY", "test-key")
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "results": {
            "is_long_win": {"probability": 0.62},
            "is_short_win": {"probability": 0.15},
        }
    }
    fake_response.raise_for_status.return_value = None

    with patch("scripts.jev.client.requests.post", return_value=fake_response) as post:
        result = ask(
            OFFICIAL,
            state="EURUSD @ 1.0850, RSI(14)=61, ...",
            questions={
                "is_long_win": {
                    "type": "noul",
                    "instructions": "Will TP be hit before SL/timeout for a long entry here?",
                    "criteria": {"true": "TP hit first", "false": "SL hit first or timeout"},
                },
                "is_short_win": {
                    "type": "noul",
                    "instructions": "Will TP be hit before SL/timeout for a short entry here?",
                    "criteria": {"true": "TP hit first", "false": "SL hit first or timeout"},
                },
            },
        )

    called_url, called_kwargs = post.call_args[0][0], post.call_args[1]
    assert called_url == OFFICIAL.base_url
    assert called_kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert called_kwargs["json"]["state"].startswith("EURUSD")
    assert set(called_kwargs["json"]["questions"]) == {"is_long_win", "is_short_win"}
    assert result == {"is_long_win": 0.62, "is_short_win": 0.15}


def test_parse_response_missing_question_raises():
    import pytest

    with pytest.raises(KeyError):
        parse_response({"results": {"is_long_win": {"probability": 0.5}}}, ["is_long_win", "is_short_win"])
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/jev-signal-provider && uv run pytest tests/jev/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.jev.client'`

**Step 3: Write minimal implementation**

```python
# scripts/jev/client.py
"""Thin HTTP client for Jev-shaped decision APIs (state + questions -> probabilities).

ASSUMPTION (unverified against a real API as of 2026-09-24): the response is
JSON shaped as {"results": {"<question_name>": {"probability": <float>, ...}}}.
If a real call comes back differently, fix parse_response() only — every
other module in scripts/jev/ depends on ask()'s return value
(dict[question_name] -> float), not on the raw response shape.
"""
from dataclasses import dataclass, field
import os

import requests


@dataclass(frozen=True)
class JevProvider:
    name: str
    base_url: str
    api_key_env: str
    extra_headers: dict = field(default_factory=dict)
    timeout_s: float = 10.0


def parse_response(raw: dict, question_names: list[str]) -> dict[str, float]:
    results = raw["results"]
    return {name: float(results[name]["probability"]) for name in question_names}


def ask(provider: JevProvider, state: str, questions: dict) -> dict[str, float]:
    api_key = os.environ[provider.api_key_env]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **provider.extra_headers,
    }
    response = requests.post(
        provider.base_url,
        headers=headers,
        json={"state": state, "questions": questions},
        timeout=provider.timeout_s,
    )
    response.raise_for_status()
    return parse_response(response.json(), list(questions.keys()))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/jev/test_client.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scripts/jev/__init__.py scripts/jev/client.py tests/jev/__init__.py tests/jev/test_client.py
git commit -m "feat: add Jev provider HTTP client"
```

---

### Task 2: State/prompt builder

**Files:**
- Create: `scripts/jev/prompt.py`
- Test: `tests/jev/test_prompt.py`

**Step 1: Write the failing test**

```python
# tests/jev/test_prompt.py
import pandas as pd

from scripts.jev.prompt import build_state_text, build_questions


def test_build_state_text_includes_ohlc_and_indicators():
    row = pd.Series({
        "O": 1.0845, "H": 1.0851, "L": 1.0840, "C": 1.0849,
        "mom_rsi_14": 61.2, "trend_ema_21": 1.0830,
    })
    text = build_state_text(row, ohlc_window=None)
    assert "1.0849" in text
    assert "mom_rsi_14=61.2" in text
    assert "trend_ema_21=1.083" in text


def test_build_state_text_skips_nan_indicators():
    row = pd.Series({"O": 1.0, "H": 1.0, "L": 1.0, "C": 1.0, "mom_rsi_14": float("nan")})
    text = build_state_text(row, ohlc_window=None)
    assert "mom_rsi_14" not in text


def test_build_questions_shape():
    questions = build_questions(tp_pips=20, sl_pips=20, horizon_bars=30)
    assert set(questions) == {"is_long_win", "is_short_win"}
    assert questions["is_long_win"]["type"] == "noul"
    assert "20" in questions["is_long_win"]["instructions"]
    assert "30" in questions["is_long_win"]["instructions"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/jev/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# scripts/jev/prompt.py
import pandas as pd


def build_state_text(row: pd.Series, ohlc_window: pd.DataFrame | None) -> str:
    parts = [f"O={row['O']:.5g} H={row['H']:.5g} L={row['L']:.5g} C={row['C']:.5g}"]
    for name, value in row.items():
        if name in ("O", "H", "L", "C"):
            continue
        if pd.isna(value):
            continue
        parts.append(f"{name}={value:.5g}")
    return " ".join(parts)


def build_questions(tp_pips: float, sl_pips: float, horizon_bars: int) -> dict:
    def _question(direction: str) -> dict:
        return {
            "type": "noul",
            "instructions": (
                f"Opening a {direction} position now with take-profit "
                f"{tp_pips} pips and stop-loss {sl_pips} pips: is take-profit "
                f"reached before stop-loss or before {horizon_bars} bars pass?"
            ),
            "criteria": {
                "true": "Take-profit reached first",
                "false": "Stop-loss reached first, or neither within the horizon",
            },
        }

    return {
        "is_long_win": _question("long"),
        "is_short_win": _question("short"),
    }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/jev/test_prompt.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add scripts/jev/prompt.py tests/jev/test_prompt.py
git commit -m "feat: add Jev state/question prompt builder"
```

---

### Task 3: Label computation (reuse FixedExitStrategy, don't reimplement)

**Files:**
- Create: `scripts/jev/labels.py`
- Test: `tests/jev/test_labels.py`

**Step 1: Write the failing test**

```python
# tests/jev/test_labels.py
import numpy as np
import pandas as pd

from scripts.jev.labels import compute_labels


def test_compute_labels_matches_fixed_exit_strategy_directly():
    # Small synthetic OHLC series: price ramps up steadily, so a long
    # entry near the start should win, a short entry should lose.
    n = 50
    close = 1.0800 + np.linspace(0, 0.0100, n)  # steady uptrend, 100 pips over 50 bars
    df = pd.DataFrame({
        "O": close, "H": close + 0.0002, "L": close - 0.0002, "C": close,
    })

    labels = compute_labels(df, symbol="EURUSD", tp=20, sl=20, timeout_bars=30)

    assert list(labels.columns) == ["targets_long", "targets_short"]
    assert len(labels) == n
    # Early bars in a strong uptrend: long should win, short should not.
    assert labels["targets_long"].iloc[0] == 1.0
    assert labels["targets_short"].iloc[0] == 0.0


def test_compute_labels_uses_asset_spread():
    # Different assets have different spreads; this just checks the function
    # doesn't hardcode EURUSD's spread internally in a way that breaks lookup.
    from fwbg.data.assets import get_asset

    asset = get_asset("EURUSD")
    assert asset.spread == 0.00018
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/jev/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# scripts/jev/labels.py
import pandas as pd

from fwbg.data.assets import get_asset
from fwbg.plugins import load_plugins  # ensures @register_exit_strategy runs
from fwbg_sdk.registry import get_exit_strategy


class _MinimalContext:
    """Just enough of SimulationContext for FixedExitStrategy.compute_targets()."""

    def __init__(self, spread: float, max_trade_bars: int):
        self.spread = spread
        self.max_trade_bars = max_trade_bars
        self.entry_modifier = None


def compute_labels(df: pd.DataFrame, symbol: str, tp: float, sl: float, timeout_bars: int) -> pd.DataFrame:
    load_plugins()
    asset = get_asset(symbol)
    strategy = get_exit_strategy("fixed")()
    ctx = _MinimalContext(spread=asset.spread, max_trade_bars=timeout_bars)

    class _Params:
        tp_value = tp
        sl_value = sl

    targets_long, targets_short = strategy.compute_targets(
        df, ctx, params=_Params(), timeout_bars=timeout_bars
    )
    return pd.DataFrame({"targets_long": targets_long, "targets_short": targets_short}, index=df.index)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/jev/test_labels.py -v`
Expected: PASS (2 tests)

If `load_plugins()` / `get_exit_strategy` import paths don't match (plugin
discovery in fwbg is dynamic — confirm the real function names with
`grep -rn "def load_plugins\|def get_exit_strategy" src/fwbg/`), fix the
imports before proceeding; this is the one place in the plan most likely to
need a small adjustment against real code that wasn't fully traced.

**Step 5: Commit**

```bash
git add scripts/jev/labels.py tests/jev/test_labels.py
git commit -m "feat: compute triple-barrier labels via existing FixedExitStrategy"
```

---

### Task 4: Resumable CSV cache

**Files:**
- Create: `scripts/jev/cache.py`
- Test: `tests/jev/test_cache.py`

**Step 1: Write the failing test**

```python
# tests/jev/test_cache.py
import pandas as pd

from scripts.jev.cache import PredictionCache


def test_cache_round_trip(tmp_path):
    path = tmp_path / "EURUSD_M1.csv"
    cache = PredictionCache(path)
    assert cache.pending_timestamps(all_timestamps=["t0", "t1", "t2"]) == ["t0", "t1", "t2"]

    cache.append("t0", {"is_long_win": 0.6, "is_short_win": 0.1})
    cache.append("t1", {"is_long_win": 0.4, "is_short_win": 0.3})

    assert cache.pending_timestamps(all_timestamps=["t0", "t1", "t2"]) == ["t2"]

    reloaded = PredictionCache(path)
    df = reloaded.as_dataframe()
    assert list(df.index) == ["t0", "t1"]
    assert df.loc["t0", "is_long_win"] == 0.6


def test_cache_survives_reopen_after_partial_run(tmp_path):
    path = tmp_path / "EURUSD_M1.csv"
    PredictionCache(path).append("t0", {"is_long_win": 0.6, "is_short_win": 0.1})

    resumed = PredictionCache(path)
    assert resumed.pending_timestamps(all_timestamps=["t0", "t1"]) == ["t1"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/jev/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# scripts/jev/cache.py
from pathlib import Path

import pandas as pd


class PredictionCache:
    """Append-only CSV cache, keyed by timestamp. Safe to resume after a
    partial/interrupted batch run — already-answered timestamps are skipped.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._df = pd.read_csv(self.path, index_col=0)
        else:
            self._df = pd.DataFrame()

    def pending_timestamps(self, all_timestamps: list) -> list:
        done = set(self._df.index)
        return [t for t in all_timestamps if t not in done]

    def append(self, timestamp, values: dict) -> None:
        self._df.loc[timestamp, list(values.keys())] = list(values.values())
        self._df.to_csv(self.path)

    def as_dataframe(self) -> pd.DataFrame:
        return self._df.copy()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/jev/test_cache.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scripts/jev/cache.py tests/jev/test_cache.py
git commit -m "feat: add resumable CSV cache for Jev batch predictions"
```

---

### Task 5: Calibration & agreement analysis

**Files:**
- Create: `scripts/jev/calibration.py`
- Test: `tests/jev/test_calibration.py`

**Step 1: Write the failing test**

```python
# tests/jev/test_calibration.py
import pandas as pd

from scripts.jev.calibration import brier_score, accuracy_vs_baseline, agreement_rate


def test_brier_score_perfect_predictions_is_zero():
    predicted = pd.Series([1.0, 0.0, 1.0, 0.0])
    actual = pd.Series([1.0, 0.0, 1.0, 0.0])
    assert brier_score(predicted, actual) == 0.0


def test_brier_score_worst_case_is_one():
    predicted = pd.Series([1.0, 0.0])
    actual = pd.Series([0.0, 1.0])
    assert brier_score(predicted, actual) == 1.0


def test_accuracy_vs_baseline_reports_both():
    predicted = pd.Series([0.9, 0.9, 0.1, 0.1])
    actual = pd.Series([1.0, 0.0, 0.0, 0.0])  # majority class is 0 (3 of 4)
    result = accuracy_vs_baseline(predicted, actual, threshold=0.5)
    assert result["model_accuracy"] == 0.75  # gets bar 2 wrong (predicted win, actual loss)
    assert result["baseline_accuracy"] == 0.75  # always-predict-0 baseline


def test_agreement_rate_between_two_providers():
    provider_a = pd.Series([0.9, 0.1, 0.6])
    provider_b = pd.Series([0.8, 0.2, 0.4])
    # threshold 0.5: A says [win, loss, win], B says [win, loss, loss] -> 2/3 agree
    assert agreement_rate(provider_a, provider_b, threshold=0.5) == pytest.approx(2 / 3)
```

Add `import pytest` at the top of the test file (needed for `pytest.approx`).

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/jev/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# scripts/jev/calibration.py
import pandas as pd


def brier_score(predicted: pd.Series, actual: pd.Series) -> float:
    return float(((predicted - actual) ** 2).mean())


def accuracy_vs_baseline(predicted: pd.Series, actual: pd.Series, threshold: float) -> dict:
    predicted_class = (predicted >= threshold).astype(float)
    model_accuracy = float((predicted_class == actual).mean())
    baseline_class = float(actual.mean() >= 0.5)
    baseline_accuracy = float((actual == baseline_class).mean())
    return {"model_accuracy": model_accuracy, "baseline_accuracy": baseline_accuracy}


def agreement_rate(provider_a: pd.Series, provider_b: pd.Series, threshold: float) -> float:
    class_a = provider_a >= threshold
    class_b = provider_b >= threshold
    return float((class_a == class_b).mean())
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/jev/test_calibration.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add scripts/jev/calibration.py tests/jev/test_calibration.py
git commit -m "feat: add Brier score, accuracy-vs-baseline, and provider agreement metrics"
```

---

### Task 6: Batch runner script (ties Tasks 1-4 together)

**Files:**
- Create: `scripts/fetch_jev_predictions.py`
- Test: `tests/jev/test_fetch_jev_predictions.py`

**Step 1: Write the failing test**

Test the orchestration logic with a fake provider (no real network), proving
resume-after-partial-run works and both providers get called with identical
questions.

```python
# tests/jev/test_fetch_jev_predictions.py
import pandas as pd

from scripts.fetch_jev_predictions import run_batch
from scripts.jev.client import JevProvider


def test_run_batch_calls_each_provider_once_per_bar(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"O": [1.08, 1.081], "H": [1.082, 1.083], "L": [1.079, 1.080], "C": [1.081, 1.082]},
        index=["t0", "t1"],
    )
    calls = []

    def fake_ask(provider, state, questions):
        calls.append(provider.name)
        return {"is_long_win": 0.5, "is_short_win": 0.5}

    monkeypatch.setattr("scripts.fetch_jev_predictions.ask", fake_ask)

    provider = JevProvider(name="fake", base_url="http://x", api_key_env="X")
    run_batch(df, provider=provider, tp_pips=20, sl_pips=20, horizon_bars=30, cache_path=tmp_path / "out.csv")

    assert calls == ["fake", "fake"]
    cached = pd.read_csv(tmp_path / "out.csv", index_col=0)
    assert list(cached.index) == ["t0", "t1"]


def test_run_batch_skips_already_cached_bars(tmp_path, monkeypatch):
    df = pd.DataFrame({"O": [1.0], "H": [1.0], "L": [1.0], "C": [1.0]}, index=["t0"])
    (tmp_path / "out.csv").write_text("timestamp,is_long_win,is_short_win\nt0,0.9,0.1\n")

    calls = []
    monkeypatch.setattr("scripts.fetch_jev_predictions.ask", lambda *a, **k: calls.append(1) or {"is_long_win": 0.0, "is_short_win": 0.0})

    provider = JevProvider(name="fake", base_url="http://x", api_key_env="X")
    run_batch(df, provider=provider, tp_pips=20, sl_pips=20, horizon_bars=30, cache_path=tmp_path / "out.csv")

    assert calls == []  # t0 was already cached, no API call made
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/jev/test_fetch_jev_predictions.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# scripts/fetch_jev_predictions.py
"""Batch-call a Jev provider for every bar of an OHLC(+indicator) DataFrame.

Usage:
    uv run python scripts/fetch_jev_predictions.py --provider official \
        --asset EURUSD --tp 20 --sl 20 --horizon-bars 30
"""
import argparse

import pandas as pd

from scripts.jev.cache import PredictionCache
from scripts.jev.client import JevProvider, ask
from scripts.jev.prompt import build_questions, build_state_text

PROVIDERS = {
    "official": JevProvider(
        name="jev_official",
        base_url="https://api.example-jev.invalid/v1/decide",  # TODO: real URL, confirmed manually
        api_key_env="JEV_OFFICIAL_API_KEY",
    ),
    "semif": JevProvider(
        name="jev_semif",
        base_url="https://llms.itemis.cloud/typesafe/v1/systemone",
        api_key_env="LITELLM_API_KEY",
    ),
}


def run_batch(df: pd.DataFrame, provider: JevProvider, tp_pips: float, sl_pips: float,
              horizon_bars: int, cache_path) -> None:
    cache = PredictionCache(cache_path)
    pending = cache.pending_timestamps(list(df.index))
    questions = build_questions(tp_pips=tp_pips, sl_pips=sl_pips, horizon_bars=horizon_bars)

    for timestamp in pending:
        row = df.loc[timestamp]
        state = build_state_text(row, ohlc_window=None)
        result = ask(provider, state=state, questions=questions)
        cache.append(timestamp, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS.keys(), required=True)
    parser.add_argument("--asset", default="EURUSD")
    parser.add_argument("--tp", type=float, required=True, help="Take-profit, pips")
    parser.add_argument("--sl", type=float, required=True, help="Stop-loss, pips")
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument("--features-csv", required=True,
                         help="Pre-computed OHLC+indicator CSV, indexed by timestamp "
                              "(see docs/plans/2026-09-24-jev-signal-provider-plan.md, Task 7)")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, index_col=0)
    provider = PROVIDERS[args.provider]
    cache_path = f"data/jev_cache/{args.asset}_{provider.name}_tp{int(args.tp)}_sl{int(args.sl)}.csv"
    run_batch(df, provider, args.tp, args.sl, args.horizon_bars, cache_path)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/jev/test_fetch_jev_predictions.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scripts/fetch_jev_predictions.py tests/jev/test_fetch_jev_predictions.py
git commit -m "feat: add batch runner script for Jev predictions"
```

---

### Task 7: `jev_signal` DataLoader plugin

**Files:**
- Create: `src/fwbg/plugins/custom/data_loading/jev_signal/__init__.py`
- Create: `src/fwbg/plugins/custom/data_loading/jev_signal/manifest.json`
- Create: `src/fwbg/plugins/custom/data_loading/jev_signal/spec.md`
- Create: `src/fwbg/plugins/custom/data_loading/jev_signal/tests.py` (repo convention: plugin tests live alongside the plugin, collected via `testpaths` in `pyproject.toml`)

**Step 1: Write the failing test**

```python
# src/fwbg/plugins/custom/data_loading/jev_signal/tests.py
import pandas as pd

from fwbg.plugins.custom.data_loading.jev_signal import JevSignalLoader


class _Ctx:
    def __init__(self, df):
        self.df = df


def test_maps_raw_probability_columns_with_shift():
    df = pd.DataFrame({
        "jev_official_is_long_win": [0.9, 0.2, 0.7],
        "jev_official_is_short_win": [0.1, 0.8, 0.3],
    })
    ctx = _Ctx(df)

    JevSignalLoader().execute(ctx, provider="jev_official")

    # Mandatory 1-bar shift: row 0 is NaN, row 1 sees row 0's value, etc.
    assert pd.isna(ctx.df["_composed_signal_long"].iloc[0])
    assert ctx.df["_composed_signal_long"].iloc[1] == 0.9
    assert ctx.df["_composed_signal_short"].iloc[1] == 0.1


def test_missing_columns_produce_no_signal():
    df = pd.DataFrame({"C": [1.0, 1.1]})
    ctx = _Ctx(df)

    JevSignalLoader().execute(ctx, provider="jev_official")

    assert (ctx.df["_composed_signal_long"].fillna(0) == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest src/fwbg/plugins/custom/data_loading/jev_signal/tests.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/fwbg/plugins/custom/data_loading/jev_signal/__init__.py
"""Maps cached Jev provider probability columns to fwbg's composed-signal
columns, so the existing `models/signal` plugin can consume them unmodified.

Expects `jev_{provider}_is_long_win` / `jev_{provider}_is_short_win` base
columns already present in ctx.df (loaded by the orchestrator via a
CSV DataSource registered over scripts/fetch_jev_predictions.py's cache
output — see docs/plans/2026-09-24-jev-signal-provider-plan.md).
"""
import pandas as pd

from fwbg_sdk import BaseDataLoader, register_data_loader, shift_features


@register_data_loader("jev_signal")
class JevSignalLoader(BaseDataLoader):
    name = "jev_signal"
    version = "1.0.0"

    def execute(self, ctx, **params):
        provider = params.get("provider", "jev_official")
        long_col = f"jev_{provider}_is_long_win"
        short_col = f"jev_{provider}_is_short_win"

        long_values = ctx.df[long_col] if long_col in ctx.df.columns else pd.Series(0.0, index=ctx.df.index)
        short_values = ctx.df[short_col] if short_col in ctx.df.columns else pd.Series(0.0, index=ctx.df.index)

        features = {
            "_composed_signal_long": long_values,
            "_composed_signal_short": short_values,
        }
        shifted = shift_features(features, ctx.df.index)
        ctx.df = pd.concat([ctx.df, shifted], axis=1)
```

```json
// src/fwbg/plugins/custom/data_loading/jev_signal/manifest.json
{
  "name": "jev_signal",
  "version": "1.0.0",
  "description": "Maps cached Jev provider probabilities to composed signal columns",
  "phase": "data_loading"
}
```

```markdown
<!-- src/fwbg/plugins/custom/data_loading/jev_signal/spec.md -->
# Plugin Spec — jev_signal

**Kind**: data_loader • **Version**: 1.0.0

## Capability

Maps `jev_{provider}_is_long_win` / `jev_{provider}_is_short_win` base
columns (populated by a registered CSV DataSource reading
`scripts/fetch_jev_predictions.py` output) to `_composed_signal_long` /
`_composed_signal_short`, applying the mandatory 1-bar shift.

## Parameters

- `provider` (string, default="jev_official"): which cached provider's
  columns to read (`jev_official` or `jev_semif`).

## Edge Cases

- Missing base columns → composed signal columns are all 0 (no lookahead
  risk, just no signal).
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest src/fwbg/plugins/custom/data_loading/jev_signal/tests.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/fwbg/plugins/custom/data_loading/jev_signal/
git commit -m "feat: add jev_signal DataLoader plugin"
```

---

## What this plan does NOT cover (deliberately)

- **Real API credentials/response verification.** Task 1's `parse_response`
  assumption must be checked against one real call to each provider before
  `scripts/fetch_jev_predictions.py` is run for real. Do this manually:
  ```bash
  curl -s https://llms.itemis.cloud/typesafe/v1/systemone \
    -H "Authorization: Bearer $LITELLM_API_KEY" -H "Content-Type: application/json" \
    -d '{"state": "...", "questions": {"is_long_win": {...}, "is_short_win": {...}}}'
  ```
  and update `parse_response` if the shape differs.
- **Feature/indicator CSV generation** (the `--features-csv` input to Task 6's
  script) — this plan assumes it already exists. It should be produced by
  running fwbg's existing DATA_LOADING + INDICATORS phases for EUR/USD and
  exporting the resulting DataFrame; the exact call site (CLI vs. the
  existing `/api/chart/indicator` endpoint vs. a small new script) needs a
  quick look at `src/fwbg/pipeline/` before writing — deferred to keep this
  plan's Task 6 testable in isolation first.
- **Registering the CSV DataSource + strategy JSON for a live Stage B run**
  (`fwbg --assets EURUSD`) — **not mechanical, corrected after Task 7's code
  review found a real gap**: the generic orchestrator every real run goes
  through, `run_data_loading()` (`src/fwbg/data/loader.py:171-267`, used by
  both live trading and backtesting), only proceeds if the raw source frame
  has a `Close` column (`loader.py:226`) and hardcodes the output column
  name as `f"macro_{prefix}"` (`loader.py:228`), plus applies daily→intraday
  forward-fill alignment built for once-a-day macro series. Jev's cache
  (`scripts/jev/cache.py`) is an already-bar-aligned, two-column
  (`is_long_win`/`is_short_win`) series with no `Close` column and no
  forward-fill need — this orchestrator cannot ingest that shape as-is.
  Whoever picks this up needs to either extend `run_data_loading()` with a
  second, non-macro ingestion path, or bypass it with a bespoke loader for
  this one data source. Not a config-only step; a small design decision of
  its own, deferred here on purpose (this plan's scope was proving out
  Jev's calibration signal, not building general external-signal ingestion).
  **Second, separate gap found in the same final review**: the cache's raw
  columns (`is_long_win`/`is_short_win`, unprefixed) don't match what
  `jev_signal`'s `execute()` looks up (`{provider}_is_long_win`, e.g.
  `jev_official_is_long_win`) — provider identity lives only in the cache
  filename today. Whoever wires the DataSource needs a rename/prefix step
  on top of solving the `Close`/`macro_` prefix problem above (needed
  regardless, since both providers' predictions will eventually need to
  coexist in the same `ctx.df`, disambiguated by prefix). `JevSignalLoader`'s
  missing-column fallback is silent (returns an all-zero signal, no error),
  so forgetting this step fails quietly, not loudly — worth remembering.
- Any code for `jev_official`'s real base URL — placeholder in Task 6, fill
  in from TypeSafe's actual docs once you have API access.
