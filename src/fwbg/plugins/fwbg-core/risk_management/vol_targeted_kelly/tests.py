"""Tests for VolTargetedKellyRiskManager plugin."""
import numpy as np
import pytest

from fwbg.plugins import import_plugin_module

_vtk = import_plugin_module("fwbg-core", "risk_management", "vol_targeted_kelly")
if _vtk is None:
    pytest.skip("fwbg-core vol_targeted_kelly plugin not available", allow_module_level=True)

VolTargetedKellyRiskManager = _vtk.VolTargetedKellyRiskManager


class TestVolTargetedKelly:

    def _make_trades_with_rv(self, n_wins=60, n_losses=40, rv_mean=15.0, rv_std=3.0):
        rng = np.random.default_rng(42)
        trades = [1.0] * n_wins + [-1.0] * n_losses
        rv_values = (rv_mean + rng.normal(0, rv_std, n_wins + n_losses)).clip(min=1.0).tolist()
        return trades, rv_values

    def test_returns_trade_returns(self):
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
        mgr = VolTargetedKellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        rv_values = [10.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        fk_base = result["risk_per_trade"]
        assert result["trade_returns"][0] > fk_base * 1.5 * 0.99

    def test_scaling_high_vol_smaller_position(self):
        mgr = VolTargetedKellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        rv_values = [30.0] * 100
        result = mgr.compute_risk_params(
            trades, win_rate=0.60, rrr=1.5,
            rv_values=rv_values, target_vol=15.0,
        )
        fk_base = result["risk_per_trade"]
        assert result["trade_returns"][0] < fk_base * 1.5 * 1.01

    def test_scale_clamped(self):
        mgr = VolTargetedKellyRiskManager()
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
        kelly_mgr = import_plugin_module("fwbg-core", "risk_management", "kelly").KellyRiskManager()
        vtk_mgr = VolTargetedKellyRiskManager()

        trades = [1.0] * 60 + [-1.0] * 40
        kelly_result = kelly_mgr.compute_risk_params(trades, 0.60, 1.5)
        vtk_result = vtk_mgr.compute_risk_params(trades, 0.60, 1.5)

        assert vtk_result["risk_per_trade"] == kelly_result["risk_per_trade"]
        assert vtk_result["trade_returns"] == kelly_result["trade_returns"]
        assert "vol_targeting" not in vtk_result

    def test_unprofitable_strategy(self):
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
        from fwbg.core import get_risk_manager
        cls = get_risk_manager("vol_targeted_kelly")
        assert cls.__name__ == "VolTargetedKellyRiskManager"

    def test_default_params(self):
        defaults = VolTargetedKellyRiskManager.get_default_params()
        assert defaults["target_vol"] == 15.0
        assert defaults["min_scale"] == 0.25
        assert defaults["max_scale"] == 2.0
