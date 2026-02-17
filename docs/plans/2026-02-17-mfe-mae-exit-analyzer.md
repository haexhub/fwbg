# MFE/MAE Exit Analyzer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** CLI command `fwbg analyze` that computes MFE/MAE statistics per asset and outputs JSON with suggested TP/SL grids.

**Architecture:** Numba-accelerated MFE/MAE computation on raw OHLC data, ATR-normalized percentiles, capture-rate matrix via bar-by-bar forward simulation, JSON output per asset.

**Tech Stack:** Numba (@njit, prange), numpy, pandas (data loading only), argparse, tabulate (CLI output), json

---

### Task 1: Numba Core — `compute_mfe_mae`

**Files:**
- Create: `src/fwbg/analysis/__init__.py`
- Create: `src/fwbg/analysis/mfe_mae.py`
- Test: `tests/analysis/test_mfe_mae.py`

**Step 1: Write failing tests for compute_mfe_mae**

```python
# tests/analysis/test_mfe_mae.py
import numpy as np
import pytest

from fwbg.analysis.mfe_mae import compute_mfe_mae


class TestComputeMfeMae:
    def test_basic_uptrend(self):
        """In a steady uptrend, MFE long should be large, MAE long small."""
        n = 100
        open_ = np.arange(100.0, 100.0 + n, dtype=np.float64)
        high = open_ + 0.5
        low = open_ - 0.1
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        # In uptrend, long MFE should be positive and substantial
        assert np.nanmean(mfe_l[:n - 11]) > 1.0
        # Long MAE should be small relative to MFE
        assert np.nanmean(mae_l[:n - 11]) < np.nanmean(mfe_l[:n - 11])

    def test_basic_downtrend(self):
        """In a steady downtrend, MFE short should be large, MAE short small."""
        n = 100
        open_ = np.arange(200.0, 200.0 - n, -1.0, dtype=np.float64)
        high = open_ + 0.1
        low = open_ - 0.5
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        assert np.nanmean(mfe_s[:n - 11]) > 1.0
        assert np.nanmean(mae_s[:n - 11]) < np.nanmean(mfe_s[:n - 11])

    def test_output_shapes(self):
        n = 50
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        assert mfe_l.shape == (n,)
        assert mae_l.shape == (n,)
        assert mfe_s.shape == (n,)
        assert mae_s.shape == (n,)

    def test_last_bars_nan(self):
        """Last max_bars entries should be NaN (not enough forward data)."""
        n = 50
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        max_bars = 10
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=max_bars)
        # Last bar can't look forward at all
        assert np.isnan(mfe_l[-1])

    def test_entry_is_next_bar_open(self):
        """Entry should be next bar's open, not current bar."""
        open_ = np.array([100.0, 105.0, 110.0, 115.0, 120.0], dtype=np.float64)
        high = open_ + 1.0
        low = open_ - 1.0
        mfe_l, mae_l, _, _ = compute_mfe_mae(open_, high, low, max_bars=2)
        # Bar 0: entry = open_[1] = 105. Forward bars 1,2.
        # MFE long = max(high[1], high[2]) - entry = max(106, 111) - 105 = 6.0
        assert abs(mfe_l[0] - 6.0) < 1e-10

    def test_mfe_mae_non_negative(self):
        """MFE and MAE should always be >= 0."""
        rng = np.random.default_rng(42)
        n = 500
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        open_ = close + rng.standard_normal(n) * 0.1
        high = np.maximum(open_, close) + abs(rng.standard_normal(n)) * 0.3
        low = np.minimum(open_, close) - abs(rng.standard_normal(n)) * 0.3
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=20)
        valid = ~np.isnan(mfe_l)
        assert np.all(mfe_l[valid] >= -1e-10)
        assert np.all(mae_l[valid] >= -1e-10)
        assert np.all(mfe_s[valid] >= -1e-10)
        assert np.all(mae_s[valid] >= -1e-10)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/analysis/test_mfe_mae.py -v
```
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement compute_mfe_mae**

```python
# src/fwbg/analysis/__init__.py
# (empty)

# src/fwbg/analysis/mfe_mae.py
"""MFE/MAE (Maximum Favorable/Adverse Excursion) computation via Numba."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def compute_mfe_mae(open_, high, low, max_bars):
    """Compute MFE/MAE for every bar looking forward max_bars.

    Entry = next bar's open (no look-ahead bias).
    Returns 4 arrays of length n: mfe_long, mae_long, mfe_short, mae_short.
    Last bars where forward window is insufficient are NaN.
    """
    n = len(open_)
    mfe_long = np.full(n, np.nan)
    mae_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)
    mae_short = np.full(n, np.nan)

    for i in prange(n - 1):
        entry = open_[i + 1]
        end = min(i + 1 + max_bars, n)
        if end <= i + 1:
            continue
        best_high = high[i + 1]
        worst_low = low[i + 1]
        for j in range(i + 2, end):
            if high[j] > best_high:
                best_high = high[j]
            if low[j] < worst_low:
                worst_low = low[j]
        mfe_long[i] = best_high - entry
        mae_long[i] = entry - worst_low
        mfe_short[i] = entry - worst_low
        mae_short[i] = best_high - entry

    return mfe_long, mae_long, mfe_short, mae_short
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/analysis/test_mfe_mae.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/fwbg/analysis/__init__.py src/fwbg/analysis/mfe_mae.py tests/analysis/test_mfe_mae.py
git commit -m "feat: add MFE/MAE Numba core computation"
```

---

### Task 2: Capture Rate Computation

**Files:**
- Modify: `src/fwbg/analysis/mfe_mae.py`
- Modify: `tests/analysis/test_mfe_mae.py`

**Step 1: Write failing tests for compute_capture_rates**

```python
from fwbg.analysis.mfe_mae import compute_capture_rates


class TestComputeCaptureRates:
    def test_basic_shape(self):
        """Output shape matches tp_values x sl_values."""
        n = 200
        rng = np.random.default_rng(42)
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        high = open_ + abs(rng.standard_normal(n)) * 0.5
        low = open_ - abs(rng.standard_normal(n)) * 0.5
        atr = np.full(n, 1.0)
        tp_vals = np.array([0.5, 1.0, 1.5])
        sl_vals = np.array([0.5, 1.0, 1.5])
        wr_long, wr_short, trade_counts = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=20
        )
        assert wr_long.shape == (3, 3)
        assert wr_short.shape == (3, 3)
        assert trade_counts.shape == (3, 3)

    def test_wider_sl_more_wins(self):
        """Wider SL should yield same or higher win rate."""
        n = 1000
        rng = np.random.default_rng(42)
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
        high = open_ + abs(rng.standard_normal(n)) * 0.4
        low = open_ - abs(rng.standard_normal(n)) * 0.4
        atr = np.full(n, 0.5)
        tp_vals = np.array([1.0])
        sl_vals = np.array([0.5, 1.0, 2.0])
        wr_long, _, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=30
        )
        # Wider SL → same or more wins
        assert wr_long[0, 1] >= wr_long[0, 0] - 0.01
        assert wr_long[0, 2] >= wr_long[0, 1] - 0.01

    def test_tighter_tp_more_wins(self):
        """Tighter TP should yield same or higher win rate."""
        n = 1000
        rng = np.random.default_rng(42)
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
        high = open_ + abs(rng.standard_normal(n)) * 0.4
        low = open_ - abs(rng.standard_normal(n)) * 0.4
        atr = np.full(n, 0.5)
        tp_vals = np.array([0.5, 1.0, 2.0])
        sl_vals = np.array([1.0])
        wr_long, _, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=30
        )
        # Tighter TP → same or more wins
        assert wr_long[0, 0] >= wr_long[1, 0] - 0.01
        assert wr_long[1, 0] >= wr_long[2, 0] - 0.01

    def test_simultaneous_hit_counts_as_loss(self):
        """When TP and SL hit same bar, should count as loss (conservative)."""
        # Price opens at 100, next bar has H=102, L=98
        # With TP=1.0*ATR and SL=1.0*ATR (ATR=1.0), both are hit
        open_ = np.array([100.0, 100.0, 102.0], dtype=np.float64)
        high = np.array([100.5, 102.0, 103.0], dtype=np.float64)
        low = np.array([99.5, 98.0, 101.0], dtype=np.float64)
        atr = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        tp_vals = np.array([1.0])
        sl_vals = np.array([1.0])
        wr_long, _, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=2
        )
        # Bar 0 → entry=100.0, bar 1 H=102>101(TP), L=98<99(SL) → both hit → loss
        assert wr_long[0, 0] < 0.5
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement compute_capture_rates**

```python
@njit(cache=True, parallel=True)
def compute_capture_rates(open_, high, low, atr, tp_values, sl_values, max_bars):
    """Compute win rates for each TP/SL combination (in ATR multiples).

    For each bar, iterates forward to determine if TP or SL is hit first.
    Simultaneous hit = loss (conservative).

    Returns:
        wr_long: (n_tp, n_sl) win rate array for long trades
        wr_short: (n_tp, n_sl) win rate array for short trades
        trade_counts: (n_tp, n_sl) number of resolved trades
    """
    n = len(open_)
    n_tp = len(tp_values)
    n_sl = len(sl_values)

    wr_long = np.zeros((n_tp, n_sl))
    wr_short = np.zeros((n_tp, n_sl))
    trade_counts = np.zeros((n_tp, n_sl))

    for ti in prange(n_tp):
        for si in range(n_sl):
            wins_l = 0
            wins_s = 0
            total = 0
            for i in range(n - 1):
                entry = open_[i + 1]
                if entry <= 0.0 or atr[i] <= 0.0:
                    continue
                tp_dist = atr[i] * tp_values[ti]
                sl_dist = atr[i] * sl_values[si]
                end = min(i + 1 + max_bars, n)
                hit_l = 0  # 0=timeout, 1=tp, -1=sl
                hit_s = 0
                for j in range(i + 1, end):
                    # Long: TP if high >= entry + tp_dist, SL if low <= entry - sl_dist
                    tp_hit_l = high[j] >= entry + tp_dist
                    sl_hit_l = low[j] <= entry - sl_dist
                    if tp_hit_l and sl_hit_l:
                        hit_l = -1  # Conservative: loss
                        break
                    elif tp_hit_l:
                        hit_l = 1
                        break
                    elif sl_hit_l:
                        hit_l = -1
                        break
                    # Short: TP if low <= entry - tp_dist, SL if high >= entry + sl_dist
                    tp_hit_s = low[j] <= entry - tp_dist
                    sl_hit_s = high[j] >= entry + sl_dist
                    if tp_hit_s and sl_hit_s:
                        hit_s = -1
                    elif tp_hit_s:
                        hit_s = 1
                    elif sl_hit_s:
                        hit_s = -1
                # Only count resolved trades (TP or SL hit, not timeout)
                if hit_l != 0:
                    total += 1
                    if hit_l == 1:
                        wins_l += 1
                if hit_s != 0:
                    if hit_s == 1:
                        wins_s += 1

            if total > 0:
                wr_long[ti, si] = wins_l / total
                wr_short[ti, si] = wins_s / total
                trade_counts[ti, si] = total

    return wr_long, wr_short, trade_counts
```

Note: Short trade tracking inside the forward loop needs its own state (not mixed with long break).
The actual implementation should track long and short independently within the inner j-loop.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/fwbg/analysis/mfe_mae.py tests/analysis/test_mfe_mae.py
git commit -m "feat: add capture rate computation for TP/SL grid analysis"
```

---

### Task 3: Exit Analyzer — Orchestration

**Files:**
- Create: `src/fwbg/analysis/exit_analyzer.py`
- Modify: `tests/analysis/test_mfe_mae.py` (add integration tests)

**Step 1: Write failing tests**

```python
from fwbg.analysis.exit_analyzer import analyze_asset


class TestAnalyzeAsset:
    def test_returns_valid_structure(self, tmp_path):
        """analyze_asset returns dict with required keys."""
        # Create a simple CSV
        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=500)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        assert "symbol" in result
        assert "mfe_mae" in result
        assert "capture_matrix" in result
        assert "suggested_grid" in result
        assert "long" in result["mfe_mae"]
        assert "short" in result["mfe_mae"]

    def test_suggested_grid_has_tp_sl(self, tmp_path):
        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=1000)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        grid = result["suggested_grid"]
        assert "tp" in grid
        assert "sl" in grid
        assert len(grid["tp"]) >= 2
        assert len(grid["sl"]) >= 2

    def test_capture_matrix_sorted_by_edge(self, tmp_path):
        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=1000)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        if len(result["capture_matrix"]) > 1:
            edges = [r["edge_long"] for r in result["capture_matrix"]]
            assert edges == sorted(edges, reverse=True)
```

Helper:

```python
def _write_test_csv(path, n=500):
    """Write a realistic OHLCV CSV for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - abs(rng.standard_normal(n)) * 0.2
    vol = rng.integers(100, 10000, size=n)
    df = pd.DataFrame({"T": dates, "O": open_, "H": high, "L": low, "C": close, "V": vol})
    df.to_csv(path, index=False)
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement exit_analyzer.py**

Key functions:
- `analyze_asset(data_file, exit_strategy, exit_params, max_bars)` → dict
- `_compute_atr(high, low, close, period)` → numpy array
- `_compute_percentiles(values, percentile_list)` → dict
- `_suggest_grid(mfe_pct_long, mfe_pct_short, mae_pct_long, mae_pct_short, capture_rates)` → dict
- `_build_capture_matrix(wr_long, wr_short, trade_counts, tp_vals, sl_vals)` → list[dict]
- `format_terminal_output(result)` → str
- `write_json(result, output_path)` → None

ATR computation: simple True Range / EMA, no dependency on ta library needed.
Percentile list: [10, 25, 50, 60, 75, 85, 90, 95]
Capture matrix scanned at fine resolution: np.arange(0.2, 4.1, 0.1) for both TP and SL.
Top entries sorted by edge_long descending.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/fwbg/analysis/exit_analyzer.py tests/analysis/test_mfe_mae.py
git commit -m "feat: add exit analyzer with MFE/MAE percentiles and grid suggestion"
```

---

### Task 4: CLI Integration

**Files:**
- Modify: `src/fwbg/cli/main.py`

**Step 1: Write the analyze subcommand handler**

In `main.py`, after the `api` subcommand block (around line 642), add:

```python
if len(sys.argv) > 1 and sys.argv[1] == "analyze":
    from fwbg.cli._analyze import run_analyze
    run_analyze(sys.argv[2:])
    return
```

**Create:** `src/fwbg/cli/_analyze.py`

```python
"""CLI handler for `fwbg analyze` — MFE/MAE exit analysis."""

import argparse
import json
import os
import glob
from datetime import datetime

from fwbg.analysis.exit_analyzer import analyze_asset, format_terminal_output, write_json
from fwbg.data.config import DATA_PATH, TIMEFRAME


def run_analyze(argv):
    parser = argparse.ArgumentParser(
        description="Analyze optimal TP/SL ranges via MFE/MAE analysis"
    )
    parser.add_argument("asset", nargs="?", help="Asset file (e.g. BRENT_HOUR.csv)")
    parser.add_argument("--asset-class", help="Analyze all assets of a class (FOREX, INDEX, COMMODITY, CRYPTO)")
    parser.add_argument("--strategy-file", help="Strategy JSON for exit strategy config")
    parser.add_argument("--max-bars", type=int, default=48, help="Forward window (default: 48)")
    parser.add_argument("--output-dir", default="test_results/analyze", help="Output directory")
    args = parser.parse_args(argv)

    if not args.asset and not args.asset_class:
        parser.error("Either asset file or --asset-class required")

    # Load exit strategy config from strategy file
    exit_strategy = "atr_based"
    exit_params = {"atr_period": 14}
    if args.strategy_file:
        with open(args.strategy_file) as f:
            strat = json.load(f)
        exit_strategy = strat.get("exit_strategy", "atr_based")
        exit_params = strat.get("exit_params", {})

    # Resolve file list
    files = _resolve_files(args.asset, args.asset_class)
    os.makedirs(args.output_dir, exist_ok=True)

    for data_file in files:
        symbol = os.path.basename(data_file).replace(".csv", "")
        print(f"\n{'='*60}")
        print(f"  Analyzing {symbol}")
        print(f"{'='*60}")

        result = analyze_asset(data_file, exit_strategy, exit_params, args.max_bars)
        print(format_terminal_output(result))

        out_path = os.path.join(args.output_dir, f"{symbol}.json")
        write_json(result, out_path)
        print(f"\n  -> {out_path}")


def _resolve_files(asset, asset_class):
    if asset:
        path = os.path.join(DATA_PATH, asset) if not os.path.isabs(asset) else asset
        if not os.path.exists(path):
            # Try without path prefix
            candidates = glob.glob(f"{DATA_PATH}/{asset}")
            if candidates:
                return candidates
            raise FileNotFoundError(f"Data file not found: {path}")
        return [path]

    from fwbg.data.assets import AssetRegistry
    registry = AssetRegistry()
    symbols = registry.symbols_by_class(asset_class.upper())
    files = []
    for sym in symbols:
        path = os.path.join(DATA_PATH, f"{sym}_{TIMEFRAME}.csv")
        if os.path.exists(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No data files found for asset class {asset_class}")
    return sorted(files)
```

**Step 2: Test manually**

```bash
python -m fwbg analyze BRENT_HOUR.csv --max-bars 48
python -m fwbg analyze --asset-class COMMODITY --strategy-file strategies/exploration_scalping.json
```

**Step 3: Commit**

```bash
git add src/fwbg/cli/_analyze.py src/fwbg/cli/main.py
git commit -m "feat: add fwbg analyze CLI command"
```

---

### Task 5: Full Integration Test & Polish

**Files:**
- Modify: `tests/analysis/test_mfe_mae.py`

**Step 1: Add CLI integration test**

```python
class TestCLIIntegration:
    def test_analyze_single_asset(self, tmp_path):
        """End-to-end: analyze a single asset file."""
        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=1000)
        out_dir = tmp_path / "output"
        from fwbg.cli._analyze import run_analyze
        run_analyze([str(csv_path), "--output-dir", str(out_dir), "--max-bars", "20"])
        json_file = out_dir / "TEST_HOUR.json"
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert data["symbol"] == "TEST"
        assert len(data["capture_matrix"]) > 0
        assert len(data["suggested_grid"]["tp"]) >= 2
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/analysis/ -v
python -m pytest tests/ -x -q  # Full suite to ensure no regressions
```

**Step 3: Run on real data for smoke test**

```bash
python -m fwbg analyze BRENT_HOUR.csv --strategy-file strategies/exploration_scalping.json
```

**Step 4: Commit**

```bash
git add tests/analysis/test_mfe_mae.py
git commit -m "test: add integration tests for exit analyzer"
```

---

### Task 6: FastAPI Endpoint

**Files:**
- Create: `src/fwbg/api/analyze.py`
- Modify: `src/fwbg/api/__init__.py` (register router)

**Step 1: Create analyze router**

```python
# src/fwbg/api/analyze.py
"""API endpoints for MFE/MAE exit analysis."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    asset: str  # e.g. "BRENT_HOUR.csv"
    exit_strategy: str = "atr_based"
    exit_params: dict = {"atr_period": 14}
    max_bars: int = 48


@router.post("")
def run_analysis(req: AnalyzeRequest):
    """Run MFE/MAE analysis for a single asset. Returns full result."""
    import os
    from fwbg.analysis.exit_analyzer import analyze_asset, write_json
    from fwbg.api.deps import get_test_results_dir
    from fwbg.data.config import DATA_PATH

    data_file = os.path.join(DATA_PATH, req.asset)
    if not os.path.exists(data_file):
        raise HTTPException(404, f"Data file not found: {req.asset}")

    result = analyze_asset(data_file, req.exit_strategy, req.exit_params, req.max_bars)

    # Cache result to disk
    out_dir = os.path.join(get_test_results_dir(), "analyze")
    os.makedirs(out_dir, exist_ok=True)
    symbol = req.asset.replace(".csv", "")
    write_json(result, os.path.join(out_dir, f"{symbol}.json"))

    return result


@router.get("")
def list_analyses():
    """List all cached analysis results."""
    import json
    import os
    from fwbg.api.deps import get_test_results_dir

    out_dir = os.path.join(get_test_results_dir(), "analyze")
    if not os.path.isdir(out_dir):
        return []

    results = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(out_dir, f)
        with open(path) as fh:
            data = json.load(fh)
        results.append({
            "symbol": data.get("symbol"),
            "timeframe": data.get("timeframe"),
            "exit_strategy": data.get("exit_strategy"),
            "analyzed_at": data.get("analyzed_at"),
            "bars_analyzed": data.get("bars_analyzed"),
            "suggested_grid": data.get("suggested_grid"),
        })
    return results


@router.get("/{symbol}")
def get_analysis(symbol: str):
    """Get cached analysis result for a symbol."""
    import json
    import os
    from fwbg.api.deps import get_test_results_dir

    out_dir = os.path.join(get_test_results_dir(), "analyze")
    # Try exact match, then with _HOUR suffix
    for candidate in [f"{symbol}.json", f"{symbol}_HOUR.json"]:
        path = os.path.join(out_dir, candidate)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)

    raise HTTPException(404, f"No analysis found for {symbol}")
```

**Step 2: Register router in create_app()**

In `src/fwbg/api/__init__.py`, add:
```python
from fwbg.api.analyze import router as analyze_router
app.include_router(analyze_router)
```
Note: No `/api` prefix needed since the router already has `prefix="/api/analyze"`.

**Step 3: Test manually**

```bash
# Start API server
python -m fwbg api &

# Test endpoints
curl -X POST http://localhost:8420/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"asset": "BRENT_HOUR.csv", "max_bars": 48}'

curl http://localhost:8420/api/analyze
curl http://localhost:8420/api/analyze/BRENT
```

**Step 4: Commit**

```bash
git add src/fwbg/api/analyze.py src/fwbg/api/__init__.py
git commit -m "feat: add /api/analyze endpoints for MFE/MAE analysis"
```
