"""
Execution Adapters - Führen Orders bei Brokern aus.

ExecutionAdapter ist die Basisklasse für alle Broker-Integrationen:
- Order-Übermittlung
- Position-Management
- Account-Information

Alle ExecutionAdapters:
- Empfangen SignalEvents vom MessageBus
- Veröffentlichen OrderEvents (Filled, Rejected, etc.)

Beispiel - Eigenen Broker-Adapter schreiben:

    from fwbg.adapters import ExecutionAdapter
    from fwbg.core.events import SignalEvent, OrderFilledEvent

    class MyBrokerAdapter(ExecutionAdapter):
        def connect(self):
            # API-Verbindung herstellen
            pass

        def submit_order(self, order):
            # Order an Broker senden
            pass
"""
from abc import abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..base import BaseAdapter
from fwbg.core.events import (
    SignalEvent, OrderEvent, OrderFilledEvent,
    OrderRejectedEvent, EventType
)
from fwbg.core.msgbus import get_message_bus


class OrderType(str, Enum):
    """Order-Typen."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Kauf/Verkauf."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Order-Objekt für Submission an Broker."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: str = ""
    client_order_id: str = ""
    time_in_force: str = "GTC"


@dataclass
class Position:
    """Aktive Position."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    position_id: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class AccountInfo:
    """Kontoinformationen."""
    balance: float
    equity: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    currency: str = "USD"


class ExecutionAdapter(BaseAdapter):
    """
    Basisklasse für Execution/Broker Adapters.

    Subklassen müssen implementieren:
    - connect(): Verbindung zum Broker
    - disconnect(): Verbindung trennen
    - submit_order(): Order absenden
    - cancel_order(): Order stornieren
    - get_positions(): Offene Positionen
    - get_account_info(): Kontoinformationen
    """

    adapter_type: str = "execution"

    def __init__(
        self,
        auto_subscribe_signals: bool = True,
        **kwargs
    ):
        """
        Args:
            auto_subscribe_signals: Automatisch SignalEvents abonnieren
        """
        super().__init__(**kwargs)
        self._auto_subscribe = auto_subscribe_signals
        self._pending_orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}

    def _on_start(self):
        """Registriert Signal-Handler beim Start."""
        if self._auto_subscribe:
            self._bus.subscribe(
                handler=self._on_signal,
                event_types=[EventType.SIGNAL],
            )
            self.log_info("Subscribed to SignalEvents")

    def _on_stop(self):
        """Entfernt Signal-Handler beim Stop."""
        if self._auto_subscribe:
            self._bus.unsubscribe(
                handler=self._on_signal,
                event_types=[EventType.SIGNAL],
            )

    def _on_signal(self, event: SignalEvent):
        """
        Verarbeitet eingehende Trading-Signale.

        Default: Erstellt Order aus Signal und submitted sie.
        Kann überschrieben werden für komplexere Logik.
        """
        order = self.signal_to_order(event)
        if order:
            self.submit_order(order)

    def signal_to_order(self, signal: SignalEvent) -> Optional[Order]:
        """
        Konvertiert SignalEvent zu Order.

        Default-Implementation. Kann überschrieben werden
        für Risk-Management, Position-Sizing, etc.

        Args:
            signal: Eingehendes Signal

        Returns:
            Order oder None wenn Signal ignoriert werden soll
        """
        side = OrderSide.BUY if signal.direction == "BUY" else OrderSide.SELL

        return Order(
            symbol=signal.symbol,
            side=side,
            quantity=1.0,  # Default - sollte vom Risk-Manager berechnet werden
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """
        Sendet Order an Broker.

        Args:
            order: Die zu sendende Order

        Returns:
            True wenn Order akzeptiert wurde
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Storniert eine Order.

        Args:
            order_id: ID der zu stornierenden Order

        Returns:
            True wenn erfolgreich
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Ruft offene Positionen ab.

        Returns:
            Liste von Position-Objekten
        """
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Ruft Kontoinformationen ab.

        Returns:
            AccountInfo-Objekt
        """
        pass

    def close_position(self, symbol: str) -> bool:
        """
        Schließt eine Position.

        Args:
            symbol: Symbol der zu schließenden Position

        Returns:
            True wenn erfolgreich
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                # Gegenorder erstellen
                close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
                order = Order(
                    symbol=symbol,
                    side=close_side,
                    quantity=pos.quantity,
                    order_type=OrderType.MARKET,
                )
                return self.submit_order(order)
        return False

    def close_all_positions(self) -> int:
        """
        Schließt alle offenen Positionen.

        Returns:
            Anzahl geschlossener Positionen
        """
        closed = 0
        for pos in self.get_positions():
            if self.close_position(pos.symbol):
                closed += 1
        return closed


from .ig_adapter import IGExecutionAdapter

__all__ = [
    "ExecutionAdapter",
    "Order",
    "Position",
    "AccountInfo",
    "OrderType",
    "OrderSide",
    "IGExecutionAdapter",
]
