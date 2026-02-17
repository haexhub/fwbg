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
