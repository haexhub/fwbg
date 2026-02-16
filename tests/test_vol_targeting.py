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
        "_regime": np.full(n, 7, dtype=np.int8),
        "vol_rv_20": 10.0 + rng.normal(0, 2, n),
    }, index=idx)
    return df


def _make_ctx():
    return SimulationContext(
        symbol="TEST", asset_class="forex", spread=0.0001, point=0.0001,
        grid_tp=[1], grid_sl=[1], grid_ct=[0.5],
        long_enabled=True, short_enabled=False,
        max_trade_bars=100, separate_long_short=False,
    )


# =============================================================================
# TASK 1: RV in Trade Storage
# =============================================================================


class TestRVInTradeStorage:

    def test_rv_stored_when_available(self):
        from fwbg.optimization.targets import _simulate_trades_core

        df = _make_df_with_rv(500)
        n = len(df)
        probs_long = np.zeros((n, 2))
        probs_long[:, 1] = 0.8

        result = _simulate_trades_core(
            df, probs_long, None, 1, None, 0.6, 0.6, 1, 1, _make_ctx(),
        )
        trades = result["trades"]
        assert len(trades) > 0
        for t in trades:
            assert "rv_at_entry" in t
            assert isinstance(t["rv_at_entry"], float)
            assert not np.isnan(t["rv_at_entry"])

    def test_rv_absent_when_no_rv_column(self):
        from fwbg.optimization.targets import _simulate_trades_core

        df = _make_df_with_rv(500).drop(columns=["vol_rv_20"])
        n = len(df)
        probs_long = np.zeros((n, 2))
        probs_long[:, 1] = 0.8

        result = _simulate_trades_core(
            df, probs_long, None, 1, None, 0.6, 0.6, 1, 1, _make_ctx(),
        )
        trades = result["trades"]
        assert len(trades) > 0
        for t in trades:
            assert "rv_at_entry" not in t


# =============================================================================
# TASK 2: Return-Based Metrics
# =============================================================================


class TestReturnBasedMetrics:

    def test_calmar_from_returns_positive(self):
        from fwbg.simulation.trade import calculate_calmar_from_returns
        returns = [0.02] * 60 + [-0.01] * 40
        calmar = calculate_calmar_from_returns(returns)
        assert calmar > 0

    def test_calmar_from_returns_matches_fixed(self):
        from fwbg.simulation.trade import (
            calculate_calmar_ratio, calculate_calmar_from_returns,
        )
        trades_binary = [1.0] * 60 + [-1.0] * 40
        fk, rrr = 0.02, 1.5
        fixed_calmar = calculate_calmar_ratio(trades_binary, fk, rrr)
        returns = [fk * rrr if t > 0 else -fk for t in trades_binary]
        returns_calmar = calculate_calmar_from_returns(returns)
        assert abs(fixed_calmar - returns_calmar) < 0.01

    def test_calmar_empty_returns(self):
        from fwbg.simulation.trade import calculate_calmar_from_returns
        assert calculate_calmar_from_returns([]) == 0.0

    def test_mc_equity_from_returns(self):
        from fwbg.simulation.trade import monte_carlo_equity_from_returns
        returns = [0.03] * 60 + [-0.02] * 40
        result = monte_carlo_equity_from_returns(returns, n_simulations=100)
        assert "median_equity" in result
        assert "bankruptcy_rate" in result
        assert "observed_equity" in result
        assert result["median_equity"] > 100
        assert result["n_simulations"] == 100

    def test_mc_equity_from_returns_matches_fixed(self):
        from fwbg.simulation.trade import (
            monte_carlo_equity_simulation,
            monte_carlo_equity_from_returns,
        )
        trades_binary = [1.0] * 60 + [-1.0] * 40
        fk, rrr = 0.02, 1.5
        fixed_mc = monte_carlo_equity_simulation(
            trades_binary, fk, rrr, n_simulations=500, random_seed=42,
        )
        returns = [fk * rrr if t > 0 else -fk for t in trades_binary]
        returns_mc = monte_carlo_equity_from_returns(
            returns, n_simulations=500, random_seed=42,
        )
        assert abs(fixed_mc["observed_equity"] - returns_mc["observed_equity"]) < 0.01

    def test_mc_equity_from_returns_few_trades(self):
        from fwbg.simulation.trade import monte_carlo_equity_from_returns
        result = monte_carlo_equity_from_returns([0.01] * 5)
        assert result["n_simulations"] == 0
        assert result["median_equity"] == 100.0


# =============================================================================
# TASK 3: Vol Targeted Kelly Plugin
# =============================================================================


class TestVolTargetedKelly:

    @pytest.fixture(autouse=True)
    def _load_plugin(self):
        from fwbg.plugins import import_plugin_module
        mod = import_plugin_module("fwbg-core", "risk_management", "vol_targeted_kelly")
        self.VTK = mod.VolTargetedKellyRiskManager

    def _make_trades_with_rv(self, n_wins=60, n_losses=40, rv_mean=15.0, rv_std=3.0):
        rng = np.random.default_rng(42)
        trades = [1.0] * n_wins + [-1.0] * n_losses
        rv_values = (rv_mean + rng.normal(0, rv_std, n_wins + n_losses)).clip(min=1.0).tolist()
        return trades, rv_values

    def test_returns_trade_returns(self):
        mgr = self.VTK()
        trades, rv_values = self._make_trades_with_rv()
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        assert "trade_returns" in result
        assert len(result["trade_returns"]) == len(trades)
        assert result["risk_per_trade"] > 0

    def test_scaling_low_vol_bigger_position(self):
        mgr = self.VTK()
        trades = [1.0] * 60 + [-1.0] * 40
        rv_values = [10.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        fk_base = result["risk_per_trade"]
        assert result["trade_returns"][0] > fk_base * 1.5 * 0.99

    def test_scaling_high_vol_smaller_position(self):
        mgr = self.VTK()
        trades = [1.0] * 60 + [-1.0] * 40
        rv_values = [30.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        fk_base = result["risk_per_trade"]
        assert result["trade_returns"][0] < fk_base * 1.5 * 1.01

    def test_scale_clamped(self):
        mgr = self.VTK()
        trades = [1.0] * 60 + [-1.0] * 40
        rv_values = [1.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
            max_scale=2.0,
        )
        fk_base = result["risk_per_trade"]
        assert result["trade_returns"][0] <= fk_base * 2.0 * 1.5 + 1e-10

    def test_fallback_without_rv(self):
        """Without rv_values, should produce same fixed-size returns as Kelly."""
        from fwbg.plugins import import_plugin_module
        kelly_mgr = import_plugin_module("fwbg-core", "risk_management", "kelly").KellyRiskManager()
        vtk_mgr = self.VTK()

        trades = [1.0] * 60 + [-1.0] * 40
        kelly_result = kelly_mgr.compute_risk_params(trades, 0.60, 1.5)
        vtk_result = vtk_mgr.compute_risk_params(trades, 0.60, 1.5)

        assert vtk_result["risk_per_trade"] == kelly_result["risk_per_trade"]
        assert vtk_result["trade_returns"] == kelly_result["trade_returns"]
        assert "vol_targeting" not in vtk_result

    def test_unprofitable_strategy(self):
        mgr = self.VTK()
        trades = [1.0] * 30 + [-1.0] * 70
        rv_values = [15.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.30, rrr=1.0,
            rv_values=rv_values, target_vol=15.0,
        )
        assert result["risk_per_trade"] == 0
        assert result["is_profitable"] is False

    def test_vol_targeting_metadata(self):
        mgr = self.VTK()
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
        from fwbg.core import get_risk_manager
        cls = get_risk_manager("vol_targeted_kelly")
        assert cls.__name__ == "VolTargetedKellyRiskManager"

    def test_default_params(self):
        defaults = self.VTK.get_default_params()
        assert defaults["target_vol"] == 15.0
        assert defaults["min_scale"] == 0.25
        assert defaults["max_scale"] == 2.0
