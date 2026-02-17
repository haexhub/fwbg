"""Tests for regime indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_regime = import_plugin_module("fwbg-premium", "indicators", "regime")
if _regime is None:
    pytest.skip("fwbg-premium regime plugin not available", allow_module_level=True)


class TestRegimePlugin:
    """Tests für das RegimeIndicators Plugin."""

    def test_plugin_importable(self):
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-premium:regime")
        assert plugin_cls is not None

    def test_plugin_computes_features(self):
        """Plugin berechnet Regime-Features korrekt."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.randn(n) * 0.005))
        low = close * (1 - np.abs(np.random.randn(n) * 0.005))
        open_price = close * (1 + np.random.randn(n) * 0.002)

        df = pd.DataFrame({
            'O': open_price, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[50], vr_windows=[100], vr_lags=[5], step=10)

        assert "regime_hurst_100" in result.columns
        assert "regime_entropy_50" in result.columns
        assert "regime_vr_100_5" in result.columns

    def test_plugin_hurst_values_valid(self):
        """Hurst-Werte sind zwischen 0 und 1."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'O': close, 'H': close * 1.005, 'L': close * 0.995, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[], vr_windows=[], vr_lags=[], step=10)

        hurst = result["regime_hurst_100"].dropna()
        assert len(hurst) > 0
        assert all(0 <= v <= 1 for v in hurst)

    def test_plugin_uses_original_close(self):
        """Plugin nutzt _original_close wenn vorhanden (Frac-Diff)."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'O': close, 'H': close + 0.5, 'L': close - 0.5,
            'C': np.random.randn(n),
            '_original_close': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[], vr_windows=[], vr_lags=[], step=10)

        hurst = result["regime_hurst_100"].dropna()
        assert len(hurst) > 0
        assert all(0 <= v <= 1 for v in hurst)

    def test_plugin_benefits_from_stationary_false(self):
        """Regime-Plugin sollte benefits_from_stationary=False haben."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        assert plugin_cls.benefits_from_stationary is False
