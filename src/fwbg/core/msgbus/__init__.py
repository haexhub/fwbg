"""
Message Bus - Zentraler Event-Verteiler.

Der MessageBus ist das Herzstück der Event-Driven Architecture:
- Empfängt Events von Adaptern und Komponenten
- Verteilt Events an registrierte Handler
- Unterstützt Topic-basierte Subscriptions
- Thread-safe für parallele Verarbeitung

Patterns:
- Publish/Subscribe: Handler registrieren sich für Event-Typen
- Topic-basiert: Handler können auf spezifische Symbole filtern
- Async-ready: Unterstützt sowohl sync als auch async Handler
"""
from typing import Callable, Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import threading
import logging
import queue
import time

from ..events import Event, EventType

log = logging.getLogger(__name__)

# Type alias für Event Handler
EventHandler = Callable[[Event], None]


@dataclass
class Subscription:
    """Repräsentiert eine Event-Subscription."""
    handler: EventHandler
    event_types: Set[EventType]
    symbols: Optional[Set[str]] = None  # None = alle Symbole
    priority: int = 0  # Höher = früher aufgerufen


class MessageBus:
    """
    Zentraler Message Bus für Event-Verteilung.

    Features:
    - Topic-basierte Subscriptions (Event-Typ + optional Symbol)
    - Prioritäts-basierte Handler-Reihenfolge
    - Thread-safe Publishing
    - Event-History für Debugging
    - Async-Queue für nicht-blockierende Verarbeitung
    """

    def __init__(
        self,
        async_mode: bool = False,
        history_size: int = 1000,
    ):
        """
        Args:
            async_mode: Events asynchron verarbeiten (Queue-basiert)
            history_size: Anzahl Events in History behalten
        """
        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._all_handlers: List[Subscription] = []  # Handler für ALLE Events
        self._lock = threading.RLock()

        self._history: List[Event] = []
        self._history_size = history_size

        self._async_mode = async_mode
        self._event_queue: queue.Queue = queue.Queue() if async_mode else None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "errors": 0,
        }

    def subscribe(
        self,
        handler: EventHandler,
        event_types: Union[EventType, List[EventType], None] = None,
        symbols: Optional[List[str]] = None,
        priority: int = 0,
    ) -> str:
        """
        Registriert einen Handler für Events.

        Args:
            handler: Callback-Funktion die Events empfängt
            event_types: Event-Typen für die registriert wird (None = alle)
            symbols: Nur Events für diese Symbole (None = alle)
            priority: Höhere Priorität = früher aufgerufen

        Returns:
            Subscription ID für späteres Unsubscribe
        """
        if event_types is None:
            types_set = set()
        elif isinstance(event_types, EventType):
            types_set = {event_types}
        else:
            types_set = set(event_types)

        symbols_set = set(symbols) if symbols else None

        sub = Subscription(
            handler=handler,
            event_types=types_set,
            symbols=symbols_set,
            priority=priority,
        )

        with self._lock:
            if not types_set:
                # Handler für alle Events
                self._all_handlers.append(sub)
                self._all_handlers.sort(key=lambda s: -s.priority)
            else:
                for event_type in types_set:
                    self._subscriptions[event_type].append(sub)
                    self._subscriptions[event_type].sort(key=lambda s: -s.priority)

        sub_id = f"{id(handler)}_{id(sub)}"
        log.debug(f"Subscribed handler for {types_set or 'ALL'} events")
        return sub_id

    def unsubscribe(self, handler: EventHandler, event_types: Optional[List[EventType]] = None):
        """
        Entfernt einen Handler.

        Args:
            handler: Der zu entfernende Handler
            event_types: Nur von diesen Typen entfernen (None = alle)
        """
        with self._lock:
            # Aus all_handlers entfernen
            self._all_handlers = [s for s in self._all_handlers if s.handler != handler]

            # Aus spezifischen Subscriptions entfernen
            if event_types:
                for et in event_types:
                    self._subscriptions[et] = [
                        s for s in self._subscriptions[et] if s.handler != handler
                    ]
            else:
                for et in self._subscriptions:
                    self._subscriptions[et] = [
                        s for s in self._subscriptions[et] if s.handler != handler
                    ]

    def publish(self, event: Event):
        """
        Veröffentlicht ein Event an alle relevanten Handler.

        Args:
            event: Das zu veröffentlichende Event
        """
        self._stats["events_published"] += 1

        if self._async_mode and self._event_queue:
            self._event_queue.put(event)
        else:
            self._deliver_event(event)

    def _deliver_event(self, event: Event):
        """Liefert Event an alle passenden Handler."""
        # History aktualisieren
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        # Symbol extrahieren falls vorhanden
        symbol = getattr(event, "symbol", None)

        # Handler sammeln
        handlers_to_call: List[Subscription] = []

        with self._lock:
            # All-Event Handler
            handlers_to_call.extend(self._all_handlers)

            # Typ-spezifische Handler
            if event.event_type in self._subscriptions:
                handlers_to_call.extend(self._subscriptions[event.event_type])

        # Handler aufrufen (nach Priorität sortiert)
        handlers_to_call.sort(key=lambda s: -s.priority)

        for sub in handlers_to_call:
            # Symbol-Filter prüfen
            if sub.symbols and symbol and symbol not in sub.symbols:
                continue

            try:
                sub.handler(event)
                self._stats["events_delivered"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                log.error(f"Error in event handler: {e}")

    def _worker_loop(self):
        """Worker-Thread für async Event-Verarbeitung."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._deliver_event(event)
                self._event_queue.task_done()
            except queue.Empty:
                continue

    def start(self):
        """Startet den async Worker (nur wenn async_mode=True)."""
        if not self._async_mode:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="msgbus-worker"
        )
        self._worker_thread.start()
        log.info("MessageBus worker started")

    def stop(self):
        """Stoppt den async Worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            log.info("MessageBus worker stopped")

    @property
    def stats(self) -> Dict[str, int]:
        """Statistiken über Event-Verarbeitung."""
        return dict(self._stats)

    @property
    def history(self) -> List[Event]:
        """Letzte Events (Kopie)."""
        with self._lock:
            return list(self._history)

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Filtert Event-History.

        Args:
            event_type: Nur Events dieses Typs
            symbol: Nur Events für dieses Symbol
            limit: Maximum Anzahl Events
        """
        with self._lock:
            result = []
            for event in reversed(self._history):
                if event_type and event.event_type != event_type:
                    continue
                if symbol and getattr(event, "symbol", None) != symbol:
                    continue
                result.append(event)
                if len(result) >= limit:
                    break
            return list(reversed(result))


# Globale MessageBus-Instanz (Singleton-Pattern)
_default_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Gibt die globale MessageBus-Instanz zurück."""
    global _default_bus
    if _default_bus is None:
        _default_bus = MessageBus()
    return _default_bus


def set_message_bus(bus: MessageBus):
    """Setzt die globale MessageBus-Instanz."""
    global _default_bus
    _default_bus = bus


__all__ = [
    "MessageBus",
    "Subscription",
    "EventHandler",
    "get_message_bus",
    "set_message_bus",
]
