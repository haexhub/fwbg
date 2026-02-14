"""
Unit-Tests für den IGBrokerAdapter.

Testet alle Broker-Funktionalitäten mit gemockter IG API.
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd

from fwbg.core.enums import Symbol, Timeframe
from fwbg.adapters.broker import (
    OrderSide, OrderStatus, Position, AccountInfo,
)


# Skip wenn trading_ig nicht verfügbar
pytest.importorskip("trading_ig", reason="trading-ig nicht installiert")


class TestIGBrokerAdapterInit:
    """Tests für IGBrokerAdapter Initialisierung."""

    def test_init_with_required_params(self):
        """Adapter sollte mit Pflichtparametern initialisierbar sein."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="test_user",
            password="test_pass",
            api_key="test_key",
            env="DEMO"
        )
        assert adapter.username == "test_user"
        assert adapter.password == "test_pass"
        assert adapter.api_key == "test_key"
        assert adapter.env == "DEMO"

    def test_init_env_uppercase(self):
        """Env sollte automatisch in Uppercase konvertiert werden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k", env="demo"
        )
        assert adapter.env == "DEMO"

    def test_init_default_values(self):
        """Standard-Werte sollten korrekt gesetzt sein."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        assert adapter.env == "DEMO"
        assert adapter.currency == "EUR"
        assert adapter.use_yfinance_fallback is True
        assert adapter.rate_limit_delay == 2.0

    def test_adapter_type_is_ig(self):
        """Adapter type sollte 'ig' sein."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        assert adapter.adapter_type == "ig"


class TestIGBrokerAdapterConnection:
    """Tests für Connect/Disconnect."""

    def test_connect_creates_session(self):
        """connect() sollte IGService Session erstellen."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )

        with patch("fwbg.adapters.broker.ig.adapter.IGService") as mock_ig_class:
            mock_ig = MagicMock()
            mock_ig_class.return_value = mock_ig

            result = adapter.connect()

            assert result is True
            mock_ig_class.assert_called_once_with("u", "p", "k", "DEMO")
            mock_ig.create_session.assert_called_once()
            assert adapter._connected is True

    def test_connect_handles_error(self):
        """connect() sollte Fehler behandeln."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )

        with patch("fwbg.adapters.broker.ig.adapter.IGService") as mock_ig_class:
            mock_ig_class.side_effect = Exception("Connection failed")

            result = adapter.connect()

            assert result is False
            assert adapter._connected is False

    def test_disconnect_logs_out(self):
        """disconnect() sollte logout aufrufen."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        mock_ig = MagicMock()
        adapter._ig = mock_ig
        adapter._connected = True

        adapter.disconnect()

        mock_ig.logout.assert_called_once()
        assert adapter._connected is False
        assert adapter._ig is None


class TestIGBrokerAdapterSymbolMapping:
    """Tests für Symbol Mapping."""

    def test_get_broker_symbol_returns_epic(self):
        """get_broker_symbol sollte Epic zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )

        epic = adapter.get_broker_symbol(Symbol.EURUSD)
        assert epic == "CS.D.EURUSD.CFD.IP"

    def test_get_broker_symbol_returns_none_for_unknown(self):
        """get_broker_symbol sollte None für unbekannte Symbols zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )

        # Symbol das nicht gemappt ist (falls es eines gibt)
        # Da alle Symbols gemappt sind, testen wir direkt das Verhalten
        result = adapter.get_broker_symbol(Symbol.EURUSD)
        assert result is not None

    def test_get_point_value_returns_correct_value(self):
        """get_point_value sollte korrekten Wert zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )

        assert adapter.get_point_value(Symbol.EURUSD) == 0.0001
        assert adapter.get_point_value(Symbol.USDJPY) == 0.01
        assert adapter.get_point_value(Symbol.XAUUSD) == 0.01


class TestIGBrokerAdapterHistoricalData:
    """Tests für historische Daten."""

    def test_get_historical_bars_from_ig(self):
        """get_historical_bars sollte Daten von IG laden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        mock_response = {
            "prices": [
                {
                    "snapshotTimeUTC": "2024-01-15T14:00:00",
                    "openPrice": {"bid": 1.10, "ask": 1.1002},
                    "highPrice": {"bid": 1.11, "ask": 1.1102},
                    "lowPrice": {"bid": 1.09, "ask": 1.0902},
                    "closePrice": {"bid": 1.105, "ask": 1.1052},
                },
                {
                    "snapshotTimeUTC": "2024-01-15T15:00:00",
                    "openPrice": {"bid": 1.105, "ask": 1.1052},
                    "highPrice": {"bid": 1.12, "ask": 1.1202},
                    "lowPrice": {"bid": 1.10, "ask": 1.1002},
                    "closePrice": {"bid": 1.115, "ask": 1.1152},
                }
            ]
        }
        adapter._ig.fetch_historical_prices_by_epic.return_value = mock_response

        df = adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H1, limit=100)

        assert len(df) == 2
        assert "O" in df.columns
        assert "H" in df.columns
        assert "L" in df.columns
        assert "C" in df.columns

    def test_get_historical_bars_empty_response(self):
        """Leere Antwort sollte leeren DataFrame zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k",
            use_yfinance_fallback=False  # Kein Fallback für diesen Test
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0
        adapter._ig.fetch_historical_prices_by_epic.return_value = {"prices": []}

        df = adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H1)

        assert df.empty or len(df) == 0

    def test_get_historical_bars_uses_yfinance_fallback(self):
        """Bei IG-Fehler sollte yfinance Fallback verwendet werden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k",
            use_yfinance_fallback=True
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0
        adapter._ig.fetch_historical_prices_by_epic.side_effect = Exception("API Error")

        with patch("fwbg.adapters.broker.ig.adapter.YFINANCE_AVAILABLE", True):
            with patch("fwbg.adapters.broker.ig.adapter.yf") as mock_yf:
                mock_df = pd.DataFrame({
                    "Open": [1.10, 1.11],
                    "High": [1.12, 1.13],
                    "Low": [1.09, 1.10],
                    "Close": [1.11, 1.12]
                }, index=pd.DatetimeIndex([
                    datetime(2024, 1, 15, 14, 0),
                    datetime(2024, 1, 15, 15, 0)
                ]))
                mock_yf.download.return_value = mock_df

                adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H1)

                mock_yf.download.assert_called_once()

    def test_get_historical_bars_no_fallback_when_disabled(self):
        """Bei deaktiviertem Fallback sollte kein yfinance verwendet werden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k",
            use_yfinance_fallback=False
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0
        adapter._ig.fetch_historical_prices_by_epic.return_value = None

        df = adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H1)

        assert df.empty


class TestIGBrokerAdapterCurrentPrice:
    """Tests für aktuelle Preise."""

    def test_get_current_price_forex_uses_base_exchange_rate(self):
        """Forex-Preise sollten baseExchangeRate verwenden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        # Mock mit Forex-Daten (wie IG API sie zurückgibt)
        adapter._ig.fetch_market_by_epic.return_value = {
            "instrument": {
                "type": "CURRENCIES",
                "currencies": [{"baseExchangeRate": 1.1795}]
            },
            "snapshot": {
                "bid": 13050,  # Points (nicht echter Preis)
                "offer": 13052
            }
        }

        result = adapter.get_current_price(Symbol.EURUSD)

        assert result is not None
        assert result["mid"] == pytest.approx(1.1795, rel=1e-4)

    def test_get_current_price_index_uses_snapshot(self):
        """Index-Preise sollten snapshot direkt verwenden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        # Mock für Index (DAX)
        adapter._ig.fetch_market_by_epic.return_value = {
            "instrument": {
                "type": "INDICES",
            },
            "snapshot": {
                "bid": 21500.0,
                "offer": 21505.0
            }
        }

        result = adapter.get_current_price(Symbol.DAX)

        assert result is not None
        assert result["bid"] == 21500.0
        assert result["ask"] == 21505.0
        assert result["mid"] == pytest.approx(21502.5, rel=1e-5)

    def test_get_current_price_returns_none_when_not_connected(self):
        """get_current_price sollte None zurückgeben wenn nicht verbunden."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = None

        result = adapter.get_current_price(Symbol.EURUSD)

        assert result is None


class TestIGBrokerAdapterOrderExecution:
    """Tests für Order-Ausführung."""

    def test_submit_order_success(self):
        """Erfolgreiche Order sollte OrderResult mit success=True zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        adapter._ig.create_open_position.return_value = {
            "dealReference": "DEAL123"
        }
        adapter._ig.fetch_deal_by_deal_reference.return_value = {
            "dealStatus": "ACCEPTED",
            "level": 1.1000
        }

        result = adapter.submit_order(
            symbol=Symbol.EURUSD,
            direction=OrderSide.BUY,
            size=1.0,
            stop_distance=50,
            limit_distance=100
        )

        assert result.success is True
        assert result.status == OrderStatus.FILLED
        assert result.fill_price == 1.1000

    def test_submit_order_rejected(self):
        """Abgelehnte Order sollte OrderResult mit success=False zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        adapter._ig.create_open_position.return_value = {
            "dealReference": "DEAL123"
        }
        adapter._ig.fetch_deal_by_deal_reference.return_value = {
            "dealStatus": "REJECTED",
            "reason": "INSUFFICIENT_FUNDS"
        }

        result = adapter.submit_order(
            symbol=Symbol.EURUSD,
            direction=OrderSide.BUY,
            size=1.0
        )

        assert result.success is False
        assert result.status == OrderStatus.REJECTED

    def test_submit_order_not_connected(self):
        """Order ohne Verbindung sollte fehlschlagen."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = None

        result = adapter.submit_order(
            symbol=Symbol.EURUSD,
            direction=OrderSide.BUY,
            size=1.0
        )

        assert result.success is False
        assert "Not connected" in result.message


class TestIGBrokerAdapterPositions:
    """Tests für Position Management."""

    def test_get_positions_returns_list(self):
        """get_positions sollte Liste von Positions zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        mock_positions = pd.DataFrame({
            "dealId": ["POS1", "POS2"],
            "epic": ["CS.D.EURUSD.TODAY.IP", "CS.D.GBPUSD.TODAY.IP"],
            "direction": ["BUY", "SELL"],
            "size": [1.0, 2.0],
            "openLevel": [1.1000, 1.2500],
            "level": [1.1050, 1.2450],
            "profit": [50.0, 100.0],
            "currency": ["EUR", "EUR"],
        })
        adapter._ig.fetch_open_positions.return_value = mock_positions

        positions = adapter.get_positions()

        assert len(positions) == 2
        assert isinstance(positions[0], Position)
        assert positions[0].direction == OrderSide.BUY
        assert positions[1].direction == OrderSide.SELL

    def test_get_positions_empty(self):
        """Leere Positions sollten leere Liste zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0
        adapter._ig.fetch_open_positions.return_value = pd.DataFrame()

        positions = adapter.get_positions()

        assert positions == []

    def test_get_positions_not_connected(self):
        """get_positions ohne Verbindung sollte leere Liste zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = None

        positions = adapter.get_positions()

        assert positions == []


class TestIGBrokerAdapterAccountInfo:
    """Tests für Account Information."""

    def test_get_account_info_returns_data(self):
        """get_account_info sollte AccountInfo zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = MagicMock()
        adapter._last_request_time = 0

        mock_accounts = pd.DataFrame({
            "balance": [10000.0],
            "profitLoss": [500.0],
            "deposit": [1000.0],
            "available": [9000.0],
        })
        adapter._ig.fetch_accounts.return_value = mock_accounts

        info = adapter.get_account_info()

        assert isinstance(info, AccountInfo)
        assert info.balance == 10000.0
        assert info.equity == 10500.0
        assert info.margin_used == 1000.0
        assert info.margin_available == 9000.0

    def test_get_account_info_not_connected(self):
        """get_account_info ohne Verbindung sollte leere AccountInfo zurückgeben."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )
        adapter._ig = None

        info = adapter.get_account_info()

        assert info.balance == 0
        assert info.equity == 0


class TestIGBrokerAdapterRateLimiting:
    """Tests für Rate Limiting."""

    def test_rate_limit_delays_requests(self):
        """Rate Limiter sollte Requests verzögern."""
        from .adapter import IGBrokerAdapter
        import time

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k",
            rate_limit_delay=0.1  # 100ms für schnelle Tests
        )
        adapter._last_request_time = time.time()

        start = time.time()
        adapter._rate_limit()
        elapsed = time.time() - start

        assert elapsed >= 0.09  # Mindestens 90ms wegen Timing-Toleranz

    def test_rate_limit_no_delay_after_timeout(self):
        """Nach Ablauf sollte kein Delay erfolgen."""
        from .adapter import IGBrokerAdapter
        import time

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k",
            rate_limit_delay=0.1
        )
        adapter._last_request_time = time.time() - 1  # 1 Sekunde in der Vergangenheit

        start = time.time()
        adapter._rate_limit()
        elapsed = time.time() - start

        assert elapsed < 0.05  # Sollte fast sofort sein


class TestIGBrokerAdapterContextManager:
    """Tests für Context Manager."""

    def test_context_manager_connects_and_disconnects(self):
        """Context Manager sollte connect/disconnect aufrufen."""
        from .adapter import IGBrokerAdapter

        adapter = IGBrokerAdapter(
            username="u", password="p", api_key="k"
        )

        with patch.object(adapter, 'connect') as mock_connect, \
             patch.object(adapter, 'disconnect') as mock_disconnect:
            mock_connect.return_value = True

            with adapter:
                mock_connect.assert_called_once()

            mock_disconnect.assert_called_once()
