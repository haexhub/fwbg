"""Tests for Risk Management Plugins."""
import pytest
from fwbg.plugins import import_plugin_module

_kelly = import_plugin_module("fwbg-core", "risk_management", "kelly")
KellyRiskManager = _kelly.KellyRiskManager


def test_base_risk_manager_import():
    from fwbg.plugins import BaseRiskManager
    assert BaseRiskManager is not None


def test_registry_functions_exist():
    from fwbg.core import register_risk_manager, get_risk_manager, list_risk_managers
    assert callable(register_risk_manager)
    assert callable(get_risk_manager)
    assert callable(list_risk_managers)


def test_unknown_risk_manager_raises():
    from fwbg.core.registry import RISK_MANAGER_REGISTRY
    orig = dict(RISK_MANAGER_REGISTRY)
    RISK_MANAGER_REGISTRY.clear()
    try:
        from fwbg.core import get_risk_manager
        with pytest.raises(ValueError, match="Unknown risk manager"):
            get_risk_manager("nonexistent")
    finally:
        RISK_MANAGER_REGISTRY.update(orig)


class TestKellyBasic:
    def test_profitable_strategy(self):
        mgr = KellyRiskManager()
        trades = [1.0] * 60 + [-1.0] * 40
        result = mgr.compute_risk_params(trades, win_rate=0.60, rrr=1.5)
        assert result["kelly_risk"] > 0
        assert result["is_profitable"] is True
        assert "circuit_breaker" in result
        assert "kelly_adjustment" in result

    def test_unprofitable_strategy(self):
        mgr = KellyRiskManager()
        trades = [1.0] * 30 + [-1.0] * 70
        result = mgr.compute_risk_params(trades, win_rate=0.30, rrr=1.0)
        assert result["kelly_risk"] == 0
        assert result["is_profitable"] is False

    def test_max_risk_cap(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 80 + [-1.0] * 20, win_rate=0.80, rrr=3.0,
            max_risk=0.02
        )
        assert result["kelly_risk"] <= 0.02

    def test_zero_rrr(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 50 + [-1.0] * 50, win_rate=0.50, rrr=0.0
        )
        assert result["kelly_risk"] == 0

    def test_circuit_breaker_structure(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 60 + [-1.0] * 40, win_rate=0.60, rrr=1.5
        )
        cb = result["circuit_breaker"]
        assert "pause_after_losses" in cb
        assert "pause_bars" in cb
        assert "enabled" in cb

    def test_kelly_adjustment_structure(self):
        mgr = KellyRiskManager()
        result = mgr.compute_risk_params(
            [1.0] * 60 + [-1.0] * 40, win_rate=0.60, rrr=1.5
        )
        ka = result["kelly_adjustment"]
        assert "original_kelly" in ka
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
