"""
Tests für den SimplePoolManager.

Testet:
- Leere Items-Liste gibt leere Ergebnisse zurück
- Signal-Handling-Setup
- Resource-Info-Ausgabe
"""
import pytest
from unittest.mock import patch


class TestSimplePoolManager:
    """Tests für den SimplePoolManager."""

    def test_empty_items_returns_empty_list(self):
        """Leere Items-Liste sollte leere Ergebnisse zurückgeben ohne Fehler."""
        from fwbg.optimization.resource_manager import SimplePoolManager

        manager = SimplePoolManager(max_concurrent_assets=2)
        results = manager.map_adaptive(lambda x: x, [])

        assert results == []

    def test_default_max_concurrent_assets(self):
        """Default max_concurrent_assets sollte 1 sein."""
        from fwbg.optimization.resource_manager import SimplePoolManager

        manager = SimplePoolManager()
        assert manager.max_workers == 1

    def test_custom_max_concurrent_assets(self):
        """max_concurrent_assets sollte konfigurierbar sein."""
        from fwbg.optimization.resource_manager import SimplePoolManager

        manager = SimplePoolManager(max_concurrent_assets=4)
        assert manager.max_workers == 4

    def test_resource_info(self):
        """get_resource_info() sollte grundlegende Infos enthalten."""
        from fwbg.optimization.resource_manager import get_resource_info

        info = get_resource_info()
        assert "cpu_cores" in info
        assert "ram_total_gb" in info
        assert info["cpu_cores"] > 0
        assert info["ram_total_gb"] > 0
