"""Tests for the deprecated VolTargetedKellyRiskManager alias."""

import pytest

from fwbg.plugins import import_plugin_module

_vtk = import_plugin_module("fwbg-core", "risk_management", "vol_targeted_kelly")
if _vtk is None:
    pytest.skip(
        "fwbg-core vol_targeted_kelly plugin not available", allow_module_level=True
    )

VolTargetedKellyRiskManager = _vtk.VolTargetedKellyRiskManager

TRADES = [1.0] * 60 + [-1.0] * 40


def _kelly():
    return import_plugin_module(
        "fwbg-core", "risk_management", "kelly"
    ).KellyRiskManager()


class TestVolTargetedKellyAlias:
    def test_sizes_exactly_like_kelly_with_or_without_rv(self):
        kelly = _kelly().compute_risk_params(TRADES, 0.60, 1.5)
        for rv in (None, [10.0] * 100, [30.0] * 100):
            res = VolTargetedKellyRiskManager().compute_risk_params(
                TRADES, 0.60, 1.5, rv_values=rv
            )
            assert res["risk_per_trade"] == kelly["risk_per_trade"]
            assert res["trade_returns"] == kelly["trade_returns"]
            assert res["circuit_breaker"] == kelly["circuit_breaker"]
            assert "vol_targeting" not in res

    def test_trade_returns_use_the_fixed_risk(self):
        res = VolTargetedKellyRiskManager().compute_risk_params(
            TRADES, 0.60, 1.5, rv_values=[1.0] * 100
        )
        fk = res["risk_per_trade"]
        assert res["trade_returns"][0] == pytest.approx(fk * 1.5)
        assert res["trade_returns"][-1] == pytest.approx(-fk)

    def test_legacy_vol_params_warn_and_are_ignored(self):
        with pytest.warns(DeprecationWarning, match="target_vol"):
            res = VolTargetedKellyRiskManager().compute_risk_params(
                TRADES, 0.60, 1.5, target_vol=15.0, min_scale=0.25, max_scale=2.0
            )
        assert (
            res["trade_returns"]
            == _kelly().compute_risk_params(TRADES, 0.60, 1.5)["trade_returns"]
        )

    def test_unprofitable_strategy(self):
        trades = [1.0] * 30 + [-1.0] * 70
        res = VolTargetedKellyRiskManager().compute_risk_params(trades, 0.30, 1.0)
        assert res["risk_per_trade"] == 0
        assert res["is_profitable"] is False
        assert res["trade_returns"] == [0.0] * 100

    def test_registry_registered(self):
        from fwbg.core import get_risk_manager

        cls = get_risk_manager("vol_targeted_kelly")
        assert cls.__name__ == "VolTargetedKellyRiskManager"

    def test_default_params_have_no_vol_options(self):
        defaults = VolTargetedKellyRiskManager.get_default_params()
        assert not {"target_vol", "min_scale", "max_scale"} & set(defaults)
        assert not {"target_vol", "min_scale", "max_scale"} & set(
            VolTargetedKellyRiskManager.get_param_schema()
        )
