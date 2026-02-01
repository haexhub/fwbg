"""
Event System - Basis für Event-Driven Architecture.

Events sind immutable Nachrichten, die Zustandsänderungen beschreiben.
Sie werden über den MessageBus verteilt und von Handlern verarbeitet.

Event-Kategorien:
- DataEvents: Marktdaten (Bars, Ticks, OrderBook)
- SignalEvents: Trading-Signale
- OrderEvents: Order-Lifecycle (Submitted, Filled, Cancelled)
- SystemEvents: System-Status (Connected, Disconnected, Error)
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from abc import ABC
import uuid


class EventType(str, Enum):
    """Kategorien von Events."""
    # Data Events
    BAR = "BAR"
    TICK = "TICK"
    QUOTE = "QUOTE"
    ORDER_BOOK = "ORDER_BOOK"

    # Signal Events
    SIGNAL = "SIGNAL"

    # Order Events
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_EXPIRED = "ORDER_EXPIRED"

    # Position Events
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CHANGED = "POSITION_CHANGED"
    POSITION_CLOSED = "POSITION_CLOSED"

    # System Events
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class Event(ABC):
    """
    Basis-Klasse für alle Events.

    Events sind immutable (frozen=True) und haben immer:
    - event_id: Eindeutige ID
    - timestamp: Wann das Event erstellt wurde
    - event_type: Typ des Events
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def event_type(self) -> EventType:
        """Muss von Subklassen überschrieben werden."""
        raise NotImplementedError


# =============================================================================
# Data Events
# =============================================================================

@dataclass(frozen=True)
class BarEvent(Event):
    """OHLCV Bar Event."""
    symbol: str = ""
    timeframe: str = "1H"
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    bar_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def event_type(self) -> EventType:
        return EventType.BAR


@dataclass(frozen=True)
class TickEvent(Event):
    """Trade Tick Event."""
    symbol: str = ""
    price: float = 0.0
    size: float = 0.0
    side: str = ""  # "BUY" or "SELL"

    @property
    def event_type(self) -> EventType:
        return EventType.TICK


@dataclass(frozen=True)
class QuoteEvent(Event):
    """Bid/Ask Quote Event."""
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def event_type(self) -> EventType:
        return EventType.QUOTE

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# =============================================================================
# Signal Events
# =============================================================================

@dataclass(frozen=True)
class SignalEvent(Event):
    """Trading Signal Event."""
    symbol: str = ""
    direction: str = ""  # "BUY" or "SELL"
    strength: float = 0.5  # 0-1
    strategy_id: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL


# =============================================================================
# Order Events
# =============================================================================

@dataclass(frozen=True)
class OrderEvent(Event):
    """Basis für Order-Events."""
    order_id: str = ""
    symbol: str = ""
    direction: str = ""
    quantity: float = 0.0
    order_type: str = "MARKET"
    price: Optional[float] = None

    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_SUBMITTED


@dataclass(frozen=True)
class OrderFilledEvent(Event):
    """Order wurde ausgeführt."""
    order_id: str = ""
    symbol: str = ""
    direction: str = ""
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    position_id: str = ""

    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_FILLED


@dataclass(frozen=True)
class OrderRejectedEvent(Event):
    """Order wurde abgelehnt."""
    order_id: str = ""
    symbol: str = ""
    reason: str = ""

    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_REJECTED


# =============================================================================
# System Events
# =============================================================================

@dataclass(frozen=True)
class SystemEvent(Event):
    """System-Status Event."""
    source: str = ""  # Adapter/Component name
    message: str = ""
    level: str = "INFO"  # INFO, WARNING, ERROR
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> EventType:
        if self.level == "ERROR":
            return EventType.ERROR
        elif self.level == "WARNING":
            return EventType.WARNING
        return EventType.CONNECTED


@dataclass(frozen=True)
class ConnectedEvent(Event):
    """Adapter verbunden."""
    adapter_id: str = ""
    adapter_type: str = ""

    @property
    def event_type(self) -> EventType:
        return EventType.CONNECTED


@dataclass(frozen=True)
class DisconnectedEvent(Event):
    """Adapter getrennt."""
    adapter_id: str = ""
    adapter_type: str = ""
    reason: str = ""

    @property
    def event_type(self) -> EventType:
        return EventType.DISCONNECTED


__all__ = [
    "EventType",
    "Event",
    # Data Events
    "BarEvent",
    "TickEvent",
    "QuoteEvent",
    # Signal Events
    "SignalEvent",
    # Order Events
    "OrderEvent",
    "OrderFilledEvent",
    "OrderRejectedEvent",
    # System Events
    "SystemEvent",
    "ConnectedEvent",
    "DisconnectedEvent",
]
