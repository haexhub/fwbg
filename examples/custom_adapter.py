"""
Beispiel: Eigenen Execution Adapter schreiben.

Dieser Adapter kann entweder:
1. Lokal verwendet werden (direkter Import)
2. Als eigenes Package auf PyPI veröffentlicht werden

Für PyPI-Veröffentlichung siehe: examples/adapter_package/
"""
from typing import List, Optional, Dict, Any
from fwbg.adapters import (
    ExecutionAdapter,
    Order, Position, AccountInfo,
    OrderType, OrderSide,
)
from fwbg.core.events import SignalEvent


class MyBrokerAdapter(ExecutionAdapter):
    """
    Beispiel-Adapter für einen fiktiven Broker.

    Empfängt SignalEvents automatisch über den MessageBus
    und führt Orders aus.
    """

    adapter_type = "mybroker"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._client = None

    def connect(self) -> bool:
        """Verbindung zum Broker herstellen."""
        try:
            # Hier: API-Client initialisieren
            # self._client = BrokerClient(self.api_key, self.api_secret)
            self._connected = True
            self.log_info(f"Connected to {'testnet' if self.testnet else 'live'}")
            return True
        except Exception as e:
            self.log_error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Verbindung trennen."""
        if self._client:
            # self._client.close()
            pass
        self._connected = False
        self.log_info("Disconnected")

    def signal_to_order(self, signal: SignalEvent) -> Optional[Order]:
        """
        Konvertiert SignalEvent zu Order.

        Hier kann eigene Logik für Position-Sizing,
        Risk-Management, etc. implementiert werden.
        """
        side = OrderSide.BUY if signal.direction == "BUY" else OrderSide.SELL

        # Beispiel: Feste Positionsgröße
        quantity = 0.01

        return Order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def submit_order(self, order: Order) -> bool:
        """Order an Broker senden."""
        try:
            self.log_info(f"Submitting: {order.side.value} {order.symbol} x{order.quantity}")

            # Hier: Order an Broker API senden
            # result = self._client.create_order(
            #     symbol=order.symbol,
            #     side=order.side.value,
            #     quantity=order.quantity,
            #     type="MARKET",
            # )

            # Simulierte Antwort
            result = {"orderId": "12345", "status": "FILLED", "price": 1.2345}

            if result.get("status") == "FILLED":
                # OrderFilledEvent veröffentlichen
                from fwbg.core.events import OrderFilledEvent
                self.publish(OrderFilledEvent(
                    symbol=order.symbol,
                    order_id=result["orderId"],
                    side=order.side.value,
                    quantity=order.quantity,
                    price=result["price"],
                    commission=0.0,
                ))
                return True

            return False

        except Exception as e:
            self.log_error(f"Order failed: {e}")
            from fwbg.core.events import OrderRejectedEvent
            self.publish(OrderRejectedEvent(
                symbol=order.symbol,
                order_id=order.order_id,
                reason=str(e),
            ))
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Order stornieren."""
        try:
            # self._client.cancel_order(order_id)
            self.log_info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            self.log_error(f"Cancel failed: {e}")
            return False

    def get_positions(self) -> List[Position]:
        """Offene Positionen abrufen."""
        try:
            # positions = self._client.get_positions()
            # return [Position(...) for p in positions]
            return []
        except Exception as e:
            self.log_error(f"Failed to get positions: {e}")
            return []

    def get_account_info(self) -> AccountInfo:
        """Kontoinformationen abrufen."""
        try:
            # account = self._client.get_account()
            return AccountInfo(
                balance=10000.0,
                equity=10000.0,
                margin_used=0.0,
                margin_available=10000.0,
                currency="USD",
            )
        except Exception as e:
            self.log_error(f"Failed to get account: {e}")
            return AccountInfo(balance=0, equity=0, currency="USD")


# ============================================================
# Verwendung
# ============================================================

if __name__ == "__main__":
    from fwbg.core.events import SignalEvent
    from fwbg.core.msgbus import get_message_bus

    # Adapter erstellen
    adapter = MyBrokerAdapter(
        api_key="your-api-key",
        api_secret="your-api-secret",
        testnet=True,
    )

    # Als Context Manager nutzen (connect/disconnect automatisch)
    with adapter:
        # Signal manuell senden (normalerweise von Strategie)
        bus = get_message_bus()

        signal = SignalEvent(
            symbol="BTCUSDT",
            direction="BUY",
            probability=0.75,
            stop_loss=100.0,
            take_profit=200.0,
        )

        # Signal veröffentlichen -> Adapter empfängt es automatisch
        bus.publish(signal)

        print("Signal gesendet!")
