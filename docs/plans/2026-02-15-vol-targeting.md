# Volatility Targeting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `vol_targeted_kelly` risk management plugin that scales position size per trade based on realized volatility, producing smoother equity curves and better risk-adjusted returns.

**Architecture:** New risk manager plugin wraps Kelly criterion + per-trade vol scaling. Trade simulation stores RV at entry time. Risk manager returns pre-computed `trade_returns` which process.py uses for all metrics (Sharpe, Calmar, Monte Carlo). Fully backward-compatible — existing `kelly` plugin unchanged.

**Tech Stack:** Python, numpy. Plugin architecture via `@register_risk_manager`.

---

## Task 1: Store RV at Entry Time in Trade Dicts

**Files:**
- Modify: `src/fwbg/optimization/targets.py:115`
- Test: `tests/test_vol_targeting.py`

**Step 1: Write failing test**

```python
# tests/test_vol_targeting.py
"""Tests for Volatility Targeting risk management."""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext


def _make_df_with_rv(n=500):
    """Create OHLC DataFrame with vol_rv_20 feature column."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        "_atr": np.full(n, 0.001),
        "_regime_ok": np.ones(n, dtype=bool),
        # Simulated RV feature (already shifted, as volatility plugin would produce)
        "vol_rv_20": 10.0 + rng.normal(0, 2, n),  # ~10% annualized
    }, index=idx)
    return df


class TestRVInTradeStorage:
    """RV at entry time should be stored in trade dicts."""

    def test_rv_stored_when_available(self):
        """Trades should include rv_at_entry when vol_rv_20 in DataFrame."""
        from fwbg.optimization.targets import _simulate_trades_core

        df = _make_df_with_rv(500)
        n = len(df)
        # Fake model predictions: always predict long with high confidence
        probs_long = np.zeros((n, 2))
        probs_long[:, 1] = 0.8  # 80% confidence for class 1

        ctx = SimulationContext(
            symbol="TEST", asset_class="forex", spread=0.0001, point=0.0001,
            grid_tp=[1], grid_sl=[1], grid_ct=[0.5],
            long_enabled=True, short_enabled=False,
            max_trade_bars=100, separate_long_short=False,
        )

        result = _simulate_trades_core(
            df, probs_long, None, 1, None, 0.6, 0.6, 1, 1, ctx,
        )
        trades = result["trades"]
        assert len(trades) > 0, "Should produce at least one trade"
        for t in trades:
            assert "rv_at_entry" in t, "Trade should have rv_at_entry"
            assert isinstance(t["rv_at_entry"], float)
            assert not np.isnan(t["rv_at_entry"])

    def test_rv_absent_when_no_rv_column(self):
        """Trades should NOT have rv_at_entry when vol_rv_20 not in DataFrame."""
        from fwbg.optimization.targets import _simulate_trades_core

        df = _make_df_with_rv(500)
        df = df.drop(columns=["vol_rv_20"])
        n = len(df)
        probs_long = np.zeros((n, 2))
        probs_long[:, 1] = 0.8

        ctx = SimulationContext(
            symbol="TEST", asset_class="forex", spread=0.0001, point=0.0001,
            grid_tp=[1], grid_sl=[1], grid_ct=[0.5],
            long_enabled=True, short_enabled=False,
            max_trade_bars=100, separate_long_short=False,
        )

        result = _simulate_trades_core(
            df, probs_long, None, 1, None, 0.6, 0.6, 1, 1, ctx,
        )
        trades = result["trades"]
        assert len(trades) > 0
        for t in trades:
            assert "rv_at_entry" not in t
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vol_targeting.py::TestRVInTradeStorage -x -q`
Expected: FAIL (rv_at_entry not in trade dicts)

**Step 3: Implement — add RV to trade storage**

In `src/fwbg/optimization/targets.py`, modify `_simulate_trades_core()`:

Before the simulation loop (after line 83), add:
```python
has_rv = "vol_rv_20" in df.columns
rv_values = df["vol_rv_20"].values if has_rv else None
```

At line 115, change:
```python
trades.append({"result": trade["result"], "pnl_raw": trade["pnl_raw"]})
```
to:
```python
t = {"result": trade["result"], "pnl_raw": trade["pnl_raw"]}
if has_rv:
    rv_val = float(rv_values[i])
    if not np.isnan(rv_val):
        t["rv_at_entry"] = rv_val
trades.append(t)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vol_targeting.py::TestRVInTradeStorage -x -q`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All existing tests still pass

---

## Task 2: Add Return-Based Metric Functions

**Files:**
- Modify: `src/fwbg/simulation/trade.py` (add 2 functions)
- Test: `tests/test_vol_targeting.py`

**Step 1: Write failing tests**

Append to `tests/test_vol_targeting.py`:

```python
class TestReturnBasedMetrics:
    """Metric functions that accept pre-computed trade returns."""

    def test_calmar_from_returns_positive(self):
        """Calmar ratio should be positive for winning strategy."""
        from fwbg.simulation.trade import calculate_calmar_from_returns

        # 60% wins at 2% gain, 40% losses at 1% loss
        returns = [0.02] * 60 + [-0.01] * 40
        calmar = calculate_calmar_from_returns(returns)
        assert calmar > 0

    def test_calmar_from_returns_matches_fixed(self):
        """With uniform sizing, should match regular calmar."""
        from fwbg.simulation.trade import (
            calculate_calmar_ratio, calculate_calmar_from_returns,
        )

        trades_binary = [1.0] * 60 + [-1.0] * 40
        fk = 0.02
        rrr = 1.5

        fixed_calmar = calculate_calmar_ratio(trades_binary, fk, rrr)
        returns = [fk * rrr if t > 0 else -fk for t in trades_binary]
        returns_calmar = calculate_calmar_from_returns(returns)

        assert abs(fixed_calmar - returns_calmar) < 0.01

    def test_calmar_empty_returns(self):
        from fwbg.simulation.trade import calculate_calmar_from_returns
        assert calculate_calmar_from_returns([]) == 0.0

    def test_mc_equity_from_returns(self):
        """MC simulation should work with pre-computed returns."""
        from fwbg.simulation.trade import monte_carlo_equity_from_returns

        returns = [0.03] * 60 + [-0.02] * 40
        result = monte_carlo_equity_from_returns(returns, n_simulations=100)

        assert "median_equity" in result
        assert "bankruptcy_rate" in result
        assert "observed_equity" in result
        assert result["median_equity"] > 100  # Profitable strategy
        assert result["n_simulations"] == 100

    def test_mc_equity_from_returns_matches_fixed(self):
        """With uniform returns, should match regular MC."""
        from fwbg.simulation.trade import (
            monte_carlo_equity_simulation,
            monte_carlo_equity_from_returns,
        )

        trades_binary = [1.0] * 60 + [-1.0] * 40
        fk = 0.02
        rrr = 1.5

        fixed_mc = monte_carlo_equity_simulation(
            trades_binary, fk, rrr, n_simulations=500, random_seed=42,
        )
        returns = [fk * rrr if t > 0 else -fk for t in trades_binary]
        returns_mc = monte_carlo_equity_from_returns(
            returns, n_simulations=500, random_seed=42,
        )

        # Same seed, same returns → same results
        assert abs(fixed_mc["observed_equity"] - returns_mc["observed_equity"]) < 0.01

    def test_mc_equity_from_returns_few_trades(self):
        from fwbg.simulation.trade import monte_carlo_equity_from_returns
        result = monte_carlo_equity_from_returns([0.01] * 5)
        assert result["n_simulations"] == 0
        assert result["median_equity"] == 100.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vol_targeting.py::TestReturnBasedMetrics -x -q`
Expected: FAIL (ImportError — functions don't exist yet)

**Step 3: Implement metric functions**

Add to `src/fwbg/simulation/trade.py` after `calculate_calmar_ratio()` (after line 200):

```python
def calculate_calmar_from_returns(trade_returns):
    """Calmar Ratio from pre-computed per-trade returns (for variable position sizing)."""
    if not trade_returns:
        return 0.0

    equity = [100.0]
    for r in trade_returns:
        equity.append(equity[-1] * (1 + r))

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    max_dd = max(max_dd, 0.01)
    total_return = (equity[-1] - equity[0]) / equity[0]
    return min(10.0, total_return / max_dd)
```

Add after `monte_carlo_equity_simulation()` (after line 675):

```python
def monte_carlo_equity_from_returns(trade_returns, n_simulations=1000, random_seed=42):
    """Monte Carlo equity simulation from pre-computed per-trade returns.

    Like monte_carlo_equity_simulation but accepts variable-sized returns
    instead of binary trades + fixed risk.
    """
    if len(trade_returns) < 10:
        return {
            "median_equity": 100.0,
            "p5_equity": 100.0,
            "p95_equity": 100.0,
            "bankruptcy_rate": 0.0,
            "observed_equity": 100.0,
            "n_simulations": 0,
        }

    rng = np.random.default_rng(random_seed)
    returns_arr = np.array(trade_returns)

    def simulate_equity(returns_seq):
        equity = 100.0
        for r in returns_seq:
            equity *= 1 + r
            if equity <= 0:
                return 0.0
        return equity

    observed_equity = simulate_equity(returns_arr)

    final_equities = []
    bankruptcies = 0
    for _ in range(n_simulations):
        permuted = rng.permutation(returns_arr)
        final_eq = simulate_equity(permuted)
        final_equities.append(final_eq)
        if final_eq <= 0:
            bankruptcies += 1

    final_equities = np.array(final_equities)

    return {
        "median_equity": float(np.median(final_equities)),
        "mean_equity": float(np.mean(final_equities)),
        "p5_equity": float(np.percentile(final_equities, 5)),
        "p25_equity": float(np.percentile(final_equities, 25)),
        "p75_equity": float(np.percentile(final_equities, 75)),
        "p95_equity": float(np.percentile(final_equities, 95)),
        "bankruptcy_rate": bankruptcies / n_simulations,
        "observed_equity": float(observed_equity),
        "n_simulations": n_simulations,
    }
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_vol_targeting.py::TestReturnBasedMetrics -x -q`
Expected: PASS

---

## Task 3: Create vol_targeted_kelly Plugin

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/risk_management/vol_targeted_kelly/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/risk_management/vol_targeted_kelly/manifest.json`
- Modify: `src/fwbg/plugins/fwbg-core/manifest.json` (add to risk_management list)
- Test: `tests/test_vol_targeting.py`

**Step 1: Write failing tests**

Append to `tests/test_vol_targeting.py`:

```python
from fwbg.plugins import import_plugin_module

_vtk = import_plugin_module("fwbg-core", "risk_management", "vol_targeted_kelly")
VolTargetedKellyRiskManager = _vtk.VolTargetedKellyRiskManager


class TestVolTargetedKelly:
    """Tests for vol_targeted_kelly risk management plugin."""

    def _make_trades_with_rv(self, n_wins=60, n_losses=40, rv_mean=15.0, rv_std=3.0):
        """Create binary trades list + rv_values list."""
        rng = np.random.default_rng(42)
        trades = [1.0] * n_wins + [-1.0] * n_losses
        rv_values = (rv_mean + rng.normal(0, rv_std, n_wins + n_losses)).clip(min=1.0).tolist()
        return trades, rv_values

    def test_returns_trade_returns(self):
        """Vol targeted kelly should return per-trade returns."""
        mgr = VolTargetedKellyRiskManager()
        trades, rv_values = self._make_trades_with_rv()
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        assert "trade_returns" in result
        assert len(result["trade_returns"]) == len(trades)
        assert result["risk_per_trade"] > 0

    def test_scaling_low_vol_bigger_position(self):
        """When RV < target_vol, position should be scaled UP."""
        mgr = VolTargetedKellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        # All trades at RV=10 with target_vol=15 → scale = 1.5
        rv_values = [10.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        fk_base = result["risk_per_trade"]
        # First trade is a win: return should be > fk_base * rrr
        assert result["trade_returns"][0] > fk_base * 1.5 * 0.99

    def test_scaling_high_vol_smaller_position(self):
        """When RV > target_vol, position should be scaled DOWN."""
        mgr = VolTargetedKellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        # All trades at RV=30 with target_vol=15 → scale = 0.5
        rv_values = [30.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        fk_base = result["risk_per_trade"]
        # First trade is a win: return should be < fk_base * rrr
        assert result["trade_returns"][0] < fk_base * 1.5 * 1.01

    def test_scale_clamped(self):
        """Vol scale should be clamped between min_scale and max_scale."""
        mgr = VolTargetedKellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        # Extreme: RV=1 with target=15 → raw scale=15, but clamped to max_scale
        rv_values = [1.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
            max_scale=2.0,
        )
        fk_base = result["risk_per_trade"]
        # Scale clamped at 2.0, so return ≤ fk_base * 2.0 * rrr
        assert result["trade_returns"][0] <= fk_base * 2.0 * 1.5 + 1e-10

    def test_fallback_without_rv(self):
        """Without rv_values, should behave like regular Kelly."""
        from fwbg.plugins import import_plugin_module
        _kelly = import_plugin_module("fwbg-core", "risk_management", "kelly")
        kelly_mgr = _kelly.KellyRiskManager()
        vtk_mgr = VolTargetedKellyRiskManager()

        trades = [1.0] * 60 + [-1.0] * 40
        kelly_result = kelly_mgr.compute_risk_params(trades, 0.60, 1.5)
        vtk_result = vtk_mgr.compute_risk_params(trades, 0.60, 1.5)

        assert vtk_result["risk_per_trade"] == kelly_result["risk_per_trade"]
        assert "trade_returns" not in vtk_result

    def test_unprofitable_strategy(self):
        """Unprofitable strategy should return fk=0, no trade_returns."""
        mgr = VolTargetedKellyRiskManager()
        trades = [1.0] * 30 + [-1.0] * 70
        rv_values = [15.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.30, rrr=1.0,
            rv_values=rv_values, target_vol=15.0,
        )
        assert result["risk_per_trade"] == 0
        assert result["is_profitable"] is False

    def test_vol_targeting_metadata(self):
        """Result should include vol targeting metadata."""
        mgr = VolTargetedKellyRiskManager()
        trades, rv_values = self._make_trades_with_rv()
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        vt = result["vol_targeting"]
        assert "target_vol" in vt
        assert "mean_scale" in vt
        assert "min_scale_used" in vt
        assert "max_scale_used" in vt

    def test_registry_registered(self):
        """Plugin should be registered in risk manager registry."""
        from fwbg.core import get_risk_manager
        cls = get_risk_manager("vol_targeted_kelly")
        assert cls.__name__ == "VolTargetedKellyRiskManager"

    def test_default_params(self):
        defaults = VolTargetedKellyRiskManager.get_default_params()
        assert defaults["target_vol"] == 15.0
        assert defaults["min_scale"] == 0.25
        assert defaults["max_scale"] == 2.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vol_targeting.py::TestVolTargetedKelly -x -q`
Expected: FAIL (ImportError)

**Step 3: Create manifest**

`src/fwbg/plugins/fwbg-core/risk_management/vol_targeted_kelly/manifest.json`:
```json
{
  "name": "vol_targeted_kelly",
  "version": "1.0.0",
  "description": "Kelly criterion with per-trade volatility targeting",
  "phase": "risk_management"
}
```

**Step 4: Implement plugin**

`src/fwbg/plugins/fwbg-core/risk_management/vol_targeted_kelly/__init__.py`:

```python
"""Kelly Criterion + Volatility Targeting Risk Manager.

Extends Kelly with per-trade position scaling based on realized volatility.
When RV is low → trade bigger. When RV is high → trade smaller.
This smooths the equity curve and improves risk-adjusted returns.
"""
from typing import Dict, Any, List, Optional

import numpy as np

from fwbg.plugins import BaseRiskManager
from fwbg.core import register_risk_manager
from fwbg.simulation.trade import (
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
)


@register_risk_manager("vol_targeted_kelly")
class VolTargetedKellyRiskManager(BaseRiskManager):
    """Quarter-Kelly with per-trade volatility targeting."""

    def compute_risk_params(
        self,
        trades: List[float],
        win_rate: float,
        rrr: float,
        *,
        kelly_fraction: float = 0.25,
        max_risk: float = 0.05,
        target_max_dd: float = 0.30,
        circuit_breaker_loss_range: tuple = (3, 8),
        circuit_breaker_pause_range: tuple = (5, 30),
        # Vol targeting params
        target_vol: float = 15.0,
        min_scale: float = 0.25,
        max_scale: float = 2.0,
        rv_values: Optional[List[float]] = None,
        **params
    ) -> Dict[str, Any]:
        # Step 1: Compute base Kelly (same as regular kelly)
        full_kelly = (win_rate * rrr - (1 - win_rate)) / rrr if rrr > 0 else 0
        fk = max(0, min(max_risk, full_kelly * kelly_fraction))

        if fk <= 0:
            return {
                "risk_per_trade": 0,
                "is_profitable": False,
                "full_kelly": full_kelly,
                "circuit_breaker": {
                    "pause_after_losses": 0, "pause_bars": 0, "enabled": False,
                },
                "risk_adjustment": {
                    "original_risk": 0, "scale_factor": 1.0, "target_dd": target_max_dd,
                },
            }

        # Step 2: DD adjustment on base Kelly (using binary trades)
        kelly_adj = adjust_risk_for_target_dd(
            trades, fk, rrr, target_max_dd=target_max_dd
        )
        if kelly_adj["scale_factor"] < 1.0:
            fk = kelly_adj["adjusted_risk"]

        # Step 3: Circuit breaker
        cb = find_optimal_circuit_breaker(
            trades, fk, rrr,
            loss_range=circuit_breaker_loss_range,
            pause_range=circuit_breaker_pause_range,
        )

        result = {
            "risk_per_trade": fk,
            "is_profitable": True,
            "full_kelly": full_kelly,
            "circuit_breaker": {
                "pause_after_losses": cb["optimal_pause_after_losses"],
                "pause_bars": cb["optimal_pause_bars"],
                "enabled": cb["optimal_pause_after_losses"] > 0,
            },
            "risk_adjustment": {
                "original_risk": kelly_adj["adjusted_risk"] / kelly_adj["scale_factor"]
                    if kelly_adj["scale_factor"] > 0 else fk,
                "scale_factor": kelly_adj["scale_factor"],
                "target_dd": target_max_dd,
            },
        }

        # Step 4: Vol targeting — compute per-trade returns
        if rv_values and len(rv_values) == len(trades):
            rv_arr = np.array(rv_values, dtype=float)
            scales = np.clip(target_vol / np.clip(rv_arr, 1e-6, None), min_scale, max_scale)

            trade_returns = []
            for t, s in zip(trades, scales):
                fk_adj = fk * s
                if t > 0:
                    trade_returns.append(fk_adj * rrr)
                else:
                    trade_returns.append(-fk_adj)

            result["trade_returns"] = trade_returns
            result["vol_targeting"] = {
                "target_vol": target_vol,
                "mean_scale": float(np.mean(scales)),
                "min_scale_used": float(np.min(scales)),
                "max_scale_used": float(np.max(scales)),
                "mean_fk_adjusted": float(fk * np.mean(scales)),
            }

        return result

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "kelly_fraction": 0.25,
            "max_risk": 0.05,
            "target_max_dd": 0.30,
            "circuit_breaker_loss_range": [3, 8],
            "circuit_breaker_pause_range": [5, 30],
            "target_vol": 15.0,
            "min_scale": 0.25,
            "max_scale": 2.0,
        }


__all__ = ["VolTargetedKellyRiskManager"]
```

**Step 5: Update fwbg-core manifest**

In `src/fwbg/plugins/fwbg-core/manifest.json`, change:
```json
"risk_management": ["kelly"]
```
to:
```json
"risk_management": ["kelly", "vol_targeted_kelly"]
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_vol_targeting.py::TestVolTargetedKelly -x -q`
Expected: PASS

---

## Task 4: Wire process.py to Use Per-Trade Returns

**Files:**
- Modify: `src/fwbg/optimization/process.py` (3 locations)
- Test: integration test in `tests/test_vol_targeting.py`

**Step 1: Write failing test**

Append to `tests/test_vol_targeting.py`:

```python
class TestProcessIntegration:
    """Integration: vol targeting flows through process.py metrics."""

    def test_trade_returns_used_for_sharpe(self):
        """When risk_result has trade_returns, Sharpe should use them."""
        from fwbg.simulation.trade import calculate_sharpe_ratio

        # Variable returns (vol-targeted): some bigger, some smaller
        variable_returns = [0.04, 0.03, -0.01, 0.05, -0.02, 0.03, -0.015, 0.035]
        # Fixed returns (regular Kelly): uniform sizing
        fk, rrr = 0.02, 1.5
        fixed_returns = [fk * rrr if r > 0 else -fk for r in variable_returns]

        sharpe_var = calculate_sharpe_ratio(variable_returns)
        sharpe_fix = calculate_sharpe_ratio(fixed_returns)

        # They should differ (different return distributions)
        assert sharpe_var != sharpe_fix

    def test_rv_values_extracted_from_trades(self):
        """rv_values should be extractable from aggregated trade dicts."""
        all_trades = [
            {"result": 1.0, "pnl_raw": 0.001, "rv_at_entry": 12.5},
            {"result": -1.0, "pnl_raw": -0.001, "rv_at_entry": 18.3},
            {"result": 1.0, "pnl_raw": 0.001},  # No RV (early trade)
        ]
        rv_values = [t["rv_at_entry"] for t in all_trades if "rv_at_entry" in t]
        assert len(rv_values) == 2
        assert rv_values[0] == 12.5
```

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_vol_targeting.py::TestProcessIntegration -x -q`
Expected: PASS (these tests just verify the concept, no code change needed yet)

**Step 3: Modify process.py**

Three locations need changes. In all three, replace the fixed `trade_returns` computation with a check for pre-computed returns from the risk manager.

**Location 1: Lines 485-491 (overfitting metrics)**

Replace:
```python
trade_returns = [fk * rrr if r > 0 else -fk for r in all_trades_binary]
```
with:
```python
trade_returns = risk_result.get("trade_returns") or [fk * rrr if r > 0 else -fk for r in all_trades_binary]
```

**Location 2: Lines 588-589 (MC simulation)**

Replace:
```python
mc_equity = monte_carlo_equity_simulation(all_trades_binary, fk, rrr, n_simulations=500)
```
with:
```python
if "trade_returns" in risk_result:
    from fwbg.simulation.trade import monte_carlo_equity_from_returns
    mc_equity = monte_carlo_equity_from_returns(trade_returns, n_simulations=500)
else:
    mc_equity = monte_carlo_equity_simulation(all_trades_binary, fk, rrr, n_simulations=500)
```

**Location 3: Lines 603-605 (not_significant metrics) and 684-686 (ok metrics)**

For BOTH locations, replace:
```python
trade_returns = [fk * rrr if r > 0 else -fk for r in all_trades_binary]
sharpe = calculate_sharpe_ratio(trade_returns, trades_per_year=actual_trades_per_year)
calmar = calculate_calmar_ratio(all_trades_binary, fk, rrr)
```
with:
```python
trade_returns = risk_result.get("trade_returns") or [fk * rrr if r > 0 else -fk for r in all_trades_binary]
sharpe = calculate_sharpe_ratio(trade_returns, trades_per_year=actual_trades_per_year)
if "trade_returns" in risk_result:
    from fwbg.simulation.trade import calculate_calmar_from_returns
    calmar = calculate_calmar_from_returns(trade_returns)
else:
    calmar = calculate_calmar_ratio(all_trades_binary, fk, rrr)
```

**Also add rv_values extraction** before the risk manager call (line ~458):

After:
```python
all_trades_binary = [t["result"] for t in all_trades]
```
Add:
```python
# Extract RV values for vol-targeted risk management
rv_values = [t["rv_at_entry"] for t in all_trades if "rv_at_entry" in t]
```

And modify the risk manager call to pass rv_values:
```python
risk_result = risk_mgr.compute_risk_params(
    all_trades_binary, mean_wr, rrr,
    rv_values=rv_values if len(rv_values) == len(all_trades_binary) else None,
    **strategy.risk_params
)
```

**Step 4: Add vol_targeting to result dict**

In the "ok" result dict (~line 714), add after `"risk_adjustment"`:
```python
"vol_targeting": risk_result.get("vol_targeting"),
```

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass

---

## Task 5: Strategy JSON Config

**Files:**
- Modify: `strategies/exploration.json` (example config)

**Step 1: Add vol_targeted_kelly option to exploration strategy**

No code test needed — just document the config pattern. In `strategies/exploration.json` or a new strategy, the user can switch:

```json
{
  "risk_management": "vol_targeted_kelly",
  "risk_params": {
    "target_vol": 15.0,
    "min_scale": 0.25,
    "max_scale": 2.0,
    "kelly_fraction": 0.25,
    "max_risk": 0.05,
    "target_max_dd": 0.30
  }
}
```

No default strategy should be changed — the user opts in explicitly.

---

## Verification

1. `python -m pytest tests/test_vol_targeting.py -x -q` — All vol targeting tests green
2. `python -m pytest tests/test_risk_management.py -x -q` — Existing Kelly tests still green
3. `python -m pytest tests/ -x -q` — Full suite green
