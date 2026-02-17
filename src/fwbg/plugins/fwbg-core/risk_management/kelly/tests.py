"""Tests for KellyRiskManager plugin."""
import pytest
from fwbg.plugins import import_plugin_module

_kelly = import_plugin_module("fwbg-core", "risk_management", "kelly")
if _kelly is None:
    pytest.skip("fwbg-core kelly plugin not available", allow_module_level=True)

KellyRiskManager = _kelly.KellyRiskManager


class TestKellyBasic:
    def test_profitable_strategy(self):
        mgr = KellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        result = mgr.compute_risk_params(trades, win_rate=0.60, rrr=1.5)
        assert result["risk_per_trade"] > 0
        assert result["is_profitable"] is True
        assert "circuit_breaker" in result
        assert "risk_adjustment" in result

    def test_unprofitable_strategy(self):
        mgr = KellyRiskManager()
        trades = [1.0] * 30 + [-1.0] * 70
        result = mgr.compute_risk_params(trades, win_rate=0.30, rrr=1.0)
        assert result["risk_per_trade"] == 0
        assert result["is_profitable"] is False

    def test_max_risk_cap(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 80 + [-1.0] * 20, win_rate=0.80, rrr=3.0,
            max_risk=0.02
        )
        assert result["risk_per_trade"] <= 0.02

    def test_zero_rrr(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 50 + [-1.0] * 50, win_rate=0.50, rrr=0.0
        )
        assert result["risk_per_trade"] == 0

    def test_circuit_breaker_structure(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 60 + [-1.0] * 40, win_rate=0.60, rrr=1.5
        )
        cb = result["circuit_breaker"]
        assert "pause_after_losses" in cb
        assert "pause_bars" in cb
        assert "enabled" in cb

    def test_risk_adjustment_structure(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 60 + [-1.0] * 40, win_rate=0.60, rrr=1.5
        )
        ka = result["risk_adjustment"]
        assert "original_risk" in ka
        assert "scale_factor" in ka
        assert "target_dd" in ka

    def test_default_params(self):
        defaults = KellyRiskManager.get_default_params()
        assert defaults["kelly_fraction"] == 0.25
        assert defaults["max_risk"] == 0.05
        assert defaults["target_max_dd"] == 0.30


class TestKellyRegistry:
    def test_kelly_registered(self):
        from fwbg.core import get_risk_manager
        cls = get_risk_manager("kelly")
        assert cls.__name__ == "KellyRiskManager"

    def test_list_includes_kelly(self):
        from fwbg.core import list_risk_managers
        assert "kelly" in list_risk_managers()
