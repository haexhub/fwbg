"""Tests für Plan 020 WP3: IG-/yfinance-Timeframe-Mapping fail-loud statt
stillem Fallback.

Läuft gegen die In-Tree-Kopie ``fwbg.adapters.broker.ig`` (Teil von ``fwbg``
selbst, immer installiert). Die gepackte Kopie ``fwbg_broker_ig`` erhält
denselben Patch, ist aber weder in ``pyproject.toml``s ``testpaths`` noch in
der Dev-/CI-Umgebung installiert (nur im Docker-Image, siehe Dockerfile) und
hat keine eigenen Tests — außerhalb des Scopes dieser Änderung.
"""
from unittest.mock import MagicMock

import pytest

from fwbg_sdk import Symbol, Timeframe

from fwbg.adapters.broker.ig import mappings as ig_mappings
from fwbg.adapters.broker.ig import adapter as ig_adapter_module


class TestYfIntervalMapCompleteness:
    """TIMEFRAME_TO_YF_INTERVAL: M30/W1 ergänzt, H2 bewusst ausgeschlossen."""

    def test_m30_mapped(self):
        assert ig_mappings.TIMEFRAME_TO_YF_INTERVAL[Timeframe.M30] == "30m"

    def test_w1_mapped(self):
        assert ig_mappings.TIMEFRAME_TO_YF_INTERVAL[Timeframe.W1] == "1wk"

    def test_h2_not_mapped(self):
        """yfinance kennt kein 2h-Intervall — H2 bleibt bewusst ungemappt."""
        assert Timeframe.H2 not in ig_mappings.TIMEFRAME_TO_YF_INTERVAL


class TestAdapterFailsLoudOnUnmappedTimeframe:
    """Kein stiller Fallback mehr auf 'HOUR'/'1h' — unbekannte Timeframes
    werfen eine ValueError statt falsch gelabelte Daten zu liefern."""

    pytestmark = pytest.mark.skipif(
        not ig_adapter_module.IG_AVAILABLE, reason="trading-ig nicht installiert"
    )

    def _adapter(self, monkeypatch):
        adapter = ig_adapter_module.IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        monkeypatch.setattr(adapter, "_ensure_session_valid", lambda: True)
        adapter._ig = MagicMock()
        return adapter

    def test_fetch_ig_historical_raises_for_unmapped_resolution(self, monkeypatch):
        """Simuliert einen zukünftigen Timeframe ohne Resolution-Mapping:
        die Auflösung muss werfen statt still auf 'HOUR' zu fallen."""
        adapter = self._adapter(monkeypatch)
        incomplete = {
            tf: v
            for tf, v in ig_adapter_module.TIMEFRAME_TO_RESOLUTION.items()
            if tf != Timeframe.H1
        }
        monkeypatch.setattr(ig_adapter_module, "TIMEFRAME_TO_RESOLUTION", incomplete)

        with pytest.raises(ValueError, match="unsupported timeframe"):
            adapter._fetch_ig_historical(Symbol.EURUSD, Timeframe.H1, limit=10)

    def test_fetch_yfinance_historical_raises_for_h2(self, monkeypatch):
        """H2 hat kein yfinance-Intervall — der Fallback-Pfad muss werfen,
        nicht still auf '1h' zurückfallen."""
        adapter = self._adapter(monkeypatch)
        monkeypatch.setattr(ig_adapter_module, "YFINANCE_AVAILABLE", True)

        with pytest.raises(ValueError, match="unsupported timeframe"):
            adapter._fetch_yfinance_historical(Symbol.EURUSD, Timeframe.H2, limit=10)

    def test_get_historical_bars_propagates_error_not_swallowed(self, monkeypatch):
        """get_historical_bars() darf die ValueError nicht schlucken: IG
        liefert leere Preise (kein Fehler), Fallback auf yfinance greift für
        H2 nicht und muss die Exception durchreichen."""
        adapter = self._adapter(monkeypatch)
        monkeypatch.setattr(ig_adapter_module, "YFINANCE_AVAILABLE", True)
        adapter._ig.fetch_historical_prices_by_epic.return_value = {"prices": []}

        with pytest.raises(ValueError, match="unsupported timeframe"):
            adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H2, limit=10)
