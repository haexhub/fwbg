"""
Broker Adapters - Unified Interface für Daten und Order-Ausführung.

BrokerAdapter ist die Basisklasse für alle Broker-Integrationen:
- Historische Marktdaten abrufen
- Live-Streaming (optional)
- Order-Ausführung
- Position-Management
- Account-Information

Beispiel - Eigenen Broker-Adapter schreiben:

    from fwbg.adapters.broker import BrokerAdapter

    class MyBrokerAdapter(BrokerAdapter):
        def connect(self) -> bool:
            # API-Verbindung herstellen
            pass

        def get_historical_bars(self, symbol, timeframe, limit) -> pd.DataFrame:
            # OHLC-Daten laden
            pass

        def _submit_order_impl(self, symbol, direction, size,
                               stop_distance, limit_distance, order_type) -> OrderResult:
            # Order ausführen — die Basisklasse erzwingt vorher den Stop-Loss-Gate
            pass
"""
from abc import abstractmethod
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from ..base import BaseAdapter
from fwbg_sdk import Timeframe, Symbol


class OrderSide(str, Enum):
    """Kauf/Verkauf."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order-Typen."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    """Order-Status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderResult:
    """Ergebnis einer Order-Ausführung."""
    success: bool
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    filled_quantity: float = 0.0
    message: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Offene Position."""
    symbol: str
    direction: OrderSide
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    position_id: str = ""
    stop_level: Optional[float] = None
    limit_level: Optional[float] = None
    currency: str = "EUR"
    # M6a telemetry — optional, populated by adapters that know SL/TP per position.
    # Adapters that don't surface these may leave them as None.
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: Optional[datetime] = None


@dataclass
class AccountInfo:
    """Kontoinformationen."""
    balance: float
    equity: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    currency: str = "EUR"


@dataclass
class BarData:
    """OHLC Bar-Daten."""
    symbol: Symbol
    timeframe: Timeframe
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class BrokerAdapter(BaseAdapter):
    """
    Abstrakte Basisklasse für Broker-Adapter.

    Kombiniert Daten- und Execution-Funktionalität in einem Interface.
    Alle Broker-Adapter (IG, OANDA, Binance, etc.) erben von dieser Klasse.

    Subklassen MÜSSEN implementieren:
    - connect() / disconnect()
    - get_historical_bars()
    - _submit_order_impl()  (submit_order() selbst ist der Gate, nicht überschreiben)
    - get_positions()
    - get_account_info()
    - get_symbol_mapping()

    Optional überschreiben:
    - subscribe_bars() für Live-Streaming
    - get_current_price() für Echtzeit-Preise
    """

    adapter_type: str = "broker"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bar_callbacks: Dict[Symbol, List[Callable[[BarData], None]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Der Stop-Loss-Gate lebt in submit_order(); Adapter implementieren
        # _submit_order_impl(). Ein Override von submit_order() würde den Gate
        # umgehen — das ist verboten (uniform, nicht per Adapter aushebelbar).
        if "submit_order" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} darf submit_order() nicht überschreiben — "
                f"implementiere _submit_order_impl(). submit_order() erzwingt den "
                f"verpflichtenden Stop-Loss-Gate und darf nicht umgangen werden."
            )

    # =========================================================================
    # Abstrakte Methoden - MÜSSEN implementiert werden
    # =========================================================================

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: Symbol,
        timeframe: Timeframe = Timeframe.H1,
        limit: int = 1000,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        Lädt historische OHLC-Daten.

        Args:
            symbol: Asset-Symbol (z.B. Symbol.EURUSD)
            timeframe: Timeframe (z.B. Timeframe.H1)
            limit: Maximum Anzahl Bars
            start: Optional Start-Zeitpunkt
            end: Optional End-Zeitpunkt

        Returns:
            DataFrame mit Spalten O, H, L, C und DatetimeIndex
            Leerer DataFrame bei Fehler
        """
        pass

    @abstractmethod
    def _submit_order_impl(
        self,
        symbol: Symbol,
        direction: OrderSide,
        size: float,
        stop_distance: float = None,
        limit_distance: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """
        Adapter-spezifische Order-Ausführung.

        NICHT direkt aufrufen — der öffentliche Einstieg ist submit_order(), das
        den verpflichtenden Stop-Loss-Gate erzwingt. Einzig close_position() ruft
        diese Methode direkt auf (Exits sind vom Gate ausgenommen und übergeben
        stop_distance=None).

        Args:
            symbol: Asset-Symbol (z.B. Symbol.EURUSD)
            direction: BUY oder SELL
            size: Positionsgröße
            stop_distance: Stop-Loss Distanz in Punkten (None nur bei Exits)
            limit_distance: Take-Profit Distanz in Punkten
            order_type: Order-Typ (default: MARKET)

        Returns:
            OrderResult mit Status und Details
        """
        pass

    def submit_order(
        self,
        symbol: Symbol,
        direction: OrderSide,
        size: float,
        stop_distance: float = None,
        limit_distance: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """
        Sendet eine Entry-Order an den Broker — mit verpflichtendem Stop-Loss-Gate.

        Deterministischer Gate an der Broker-Grenze: Jede Entry-Order MUSS einen
        positiven Stop-Loss haben. Orders ohne (oder mit <= 0) Stop werden hart
        abgelehnt, bevor sie den Adapter/Broker erreichen — uniform über alle
        Adapter und nicht per Adapter umgehbar (siehe __init_subclass__). Der Stop
        wird vom Adapter atomar im selben Broker-Request mit dem Entry gesendet.

        Exits (close_position) rufen _submit_order_impl() direkt auf und sind
        bewusst vom Gate ausgenommen.

        Args:
            symbol: Asset-Symbol (z.B. Symbol.EURUSD)
            direction: BUY oder SELL
            size: Positionsgröße
            stop_distance: Stop-Loss Distanz in Punkten (PFLICHT, > 0)
            limit_distance: Take-Profit Distanz in Punkten
            order_type: Order-Typ (default: MARKET)

        Returns:
            OrderResult — bei fehlendem/nicht-positivem Stop: success=False,
            status=REJECTED, ohne den Broker zu kontaktieren
        """
        if stop_distance is None or not stop_distance > 0:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message="Rejected: stop-loss is mandatory for entry orders",
            )
        return self._submit_order_impl(
            symbol=symbol,
            direction=direction,
            size=size,
            stop_distance=stop_distance,
            limit_distance=limit_distance,
            order_type=order_type,
        )

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

    @abstractmethod
    def get_broker_symbol(self, symbol: Symbol) -> Optional[str]:
        """
        Konvertiert ein Standard-Symbol zum broker-spezifischen Identifier.

        Args:
            symbol: Standard-Symbol (z.B. Symbol.EURUSD)

        Returns:
            Broker-spezifischer Identifier (z.B. IG Epic) oder None
        """
        pass

    @property
    def is_paper(self) -> bool:
        """
        True iff this adapter operates against a demo/paper account.

        Default: True (safe default — subclasses override for live trading).
        Used by read-side consumers and audit logs; the M6a telemetry writer
        does NOT gate on this — it writes for both paper and live mode.
        """
        return True

    # =========================================================================
    # Optionale Methoden - können überschrieben werden
    # =========================================================================

    def get_current_price(self, symbol: Symbol) -> Optional[Dict[str, float]]:
        """
        Ruft aktuellen Preis ab.

        Args:
            symbol: Asset-Symbol (z.B. Symbol.EURUSD)

        Returns:
            Dict mit "bid", "ask", "mid" oder None
        """
        # Default: Nicht implementiert
        return None

    def subscribe_bars(
        self,
        symbol: Symbol,
        timeframe: Timeframe = Timeframe.H1,
        callback: Callable[[BarData], None] = None,
    ) -> bool:
        """
        Abonniert Live-Bars für ein Symbol.

        Args:
            symbol: Asset-Symbol (z.B. Symbol.EURUSD)
            timeframe: Timeframe (z.B. Timeframe.H1)
            callback: Callback-Funktion für neue Bars

        Returns:
            True wenn erfolgreich
        """
        # Default: Nicht implementiert
        self.log_warning(f"subscribe_bars not implemented for {self.adapter_type}")
        return False

    def unsubscribe_bars(self, symbol: Symbol) -> bool:
        """
        Beendet Bar-Subscription.

        Args:
            symbol: Asset-Symbol (z.B. Symbol.EURUSD)

        Returns:
            True wenn erfolgreich
        """
        if symbol in self._bar_callbacks:
            del self._bar_callbacks[symbol]
            return True
        return False

    def close_position(self, position_id: str) -> OrderResult:
        """
        Schließt eine Position.

        Args:
            position_id: ID der zu schließenden Position

        Returns:
            OrderResult
        """
        # Default-Implementation: Finde Position und erstelle Gegenorder
        positions = self.get_positions()
        for pos in positions:
            if pos.position_id == position_id:
                close_direction = OrderSide.SELL if pos.direction == OrderSide.BUY else OrderSide.BUY
                # Exit: Gegenorder ohne Stop — bewusst am Stop-Loss-Gate vorbei
                # (submit_order würde eine Order ohne Stop ablehnen).
                return self._submit_order_impl(
                    symbol=pos.symbol,
                    direction=close_direction,
                    size=pos.size,
                )

        return OrderResult(
            success=False,
            status=OrderStatus.REJECTED,
            message=f"Position {position_id} not found"
        )

    def close_all_positions(self) -> List[OrderResult]:
        """
        Schließt alle offenen Positionen.

        Returns:
            Liste von OrderResults
        """
        results = []
        for pos in self.get_positions():
            result = self.close_position(pos.position_id)
            results.append(result)
        return results

    # =========================================================================
    # Helper-Methoden
    # =========================================================================

    def _notify_bar_callbacks(self, bar: BarData):
        """Benachrichtigt alle registrierten Callbacks für ein Symbol."""
        callbacks = self._bar_callbacks.get(bar.symbol, [])
        for callback in callbacks:
            try:
                callback(bar)
            except Exception as e:
                self.log_error(f"Bar callback error for {bar.symbol}: {e}")

    def add_bar_callback(self, symbol: Symbol, callback: Callable[[BarData], None]):
        """Registriert einen Callback für Bar-Updates."""
        if symbol not in self._bar_callbacks:
            self._bar_callbacks[symbol] = []
        self._bar_callbacks[symbol].append(callback)

    def remove_bar_callback(self, symbol: Symbol, callback: Callable[[BarData], None]):
        """Entfernt einen Callback."""
        if symbol in self._bar_callbacks:
            try:
                self._bar_callbacks[symbol].remove(callback)
            except ValueError:
                pass


# Import concrete implementations (nach Klassendefinition um zirkuläre Imports zu vermeiden)
from .ig import IGBrokerAdapter  # noqa: E402

__all__ = [
    # Base class
    "BrokerAdapter",
    # Order types
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "OrderResult",
    # Data types
    "Position",
    "AccountInfo",
    "BarData",
    # Implementations
    "IGBrokerAdapter",
    # Re-exported enums for convenience
    "Symbol",
    "Timeframe",
]
