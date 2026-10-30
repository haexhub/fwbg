"""
Integration tests for risk management framework.

Plugin-specific tests are in each plugin's tests.py:
- src/fwbg/plugins/fwbg-core/risk_management/kelly/tests.py
- src/fwbg/plugins/fwbg-core/risk_management/vol_targeted_kelly/tests.py
"""
import pytest


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


class TestKellyInputs:
    """Kelly sees the realised payoff, not the nominal TP/SL ratio."""

    @staticmethod
    def _trades(*pairs):
        return [{"result": r, "pnl_raw": p} for r, p in pairs]

    def test_small_timeout_wins_lower_the_payoff(self):
        from fwbg.simulation.trade import kelly_inputs

        # 25 TP wins (+2R), 25 timeouts closing at +0.1R, 50 stops (-1R)
        trades = self._trades(*[(1.0, 2.0)] * 25, *[(1.0, 0.1)] * 25, *[(-1.0, -1.0)] * 50)

        binary, win_rate, payoff = kelly_inputs(trades, default_payoff=2.0)

        assert win_rate == 0.5
        assert payoff == pytest.approx(1.05)
        assert binary.count(1.0) == 50
        # full Kelly W - (1-W)/R: ~2.4 % instead of 25 % at the nominal RRR of 2
        assert win_rate - (1 - win_rate) / payoff == pytest.approx(0.0238, abs=1e-4)

    def test_noise_exit_counts_as_loss(self):
        from fwbg.simulation.trade import kelly_inputs

        trades = self._trades((1.0, 1.0), (-1.0, 1e-8), (-1.0, -1.0))

        binary, win_rate, payoff = kelly_inputs(trades, default_payoff=1.0)

        assert binary == [1.0, -1.0, -1.0]
        assert win_rate == pytest.approx(1 / 3)
        assert payoff == pytest.approx(2.0)  # 1.0 / mean(0, 1)

    def test_payoff_falls_back_without_losses(self):
        from fwbg.simulation.trade import kelly_inputs

        assert kelly_inputs(self._trades((1.0, 1.0)), default_payoff=1.5)[2] == 1.5
        assert kelly_inputs([], default_payoff=1.5) == ([], 0.0, 1.5)
