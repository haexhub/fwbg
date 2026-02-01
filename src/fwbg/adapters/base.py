"""
Base Adapter - Gemeinsame Funktionalität für alle Adapter.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import logging

from fwbg.core.msgbus import MessageBus, get_message_bus
from fwbg.core.events import Event, ConnectedEvent, DisconnectedEvent, SystemEvent

log = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Basisklasse für alle Adapter.

    Stellt gemeinsame Funktionalität bereit:
    - MessageBus Integration
    - Lifecycle Management (connect, disconnect, start, stop)
    - Status Tracking
    - Event Publishing
    """

    adapter_type: str = "base"

    def __init__(
        self,
        adapter_id: str = None,
        message_bus: MessageBus = None,
        config: Dict[str, Any] = None,
    ):
        """
        Args:
            adapter_id: Eindeutige ID für diesen Adapter
            message_bus: MessageBus für Event-Kommunikation
            config: Adapter-spezifische Konfiguration
        """
        self.adapter_id = adapter_id or f"{self.adapter_type}_{id(self)}"
        self._bus = message_bus or get_message_bus()
        self._config = config or {}

        self._connected = False
        self._running = False
        self._lock = threading.Lock()

        self._stats = {
            "events_published": 0,
            "errors": 0,
            "last_event_time": None,
        }

    @property
    def is_connected(self) -> bool:
        """Ist der Adapter verbunden?"""
        return self._connected

    @property
    def is_running(self) -> bool:
        """Läuft der Adapter?"""
        return self._running

    @property
    def config(self) -> Dict[str, Any]:
        """Adapter-Konfiguration."""
        return self._config

    @property
    def stats(self) -> Dict[str, Any]:
        """Adapter-Statistiken."""
        return dict(self._stats)

    def publish(self, event: Event):
        """
        Veröffentlicht ein Event über den MessageBus.

        Args:
            event: Das zu veröffentlichende Event
        """
        self._bus.publish(event)
        self._stats["events_published"] += 1
        self._stats["last_event_time"] = datetime.now()

    def log_info(self, message: str):
        """Logged Info-Nachricht und sendet SystemEvent."""
        log.info(f"[{self.adapter_id}] {message}")
        self.publish(SystemEvent(
            source=self.adapter_id,
            message=message,
            level="INFO",
        ))

    def log_warning(self, message: str):
        """Logged Warning und sendet SystemEvent."""
        log.warning(f"[{self.adapter_id}] {message}")
        self.publish(SystemEvent(
            source=self.adapter_id,
            message=message,
            level="WARNING",
        ))

    def log_error(self, message: str, error: Exception = None):
        """Logged Error und sendet SystemEvent."""
        self._stats["errors"] += 1
        full_message = f"{message}: {error}" if error else message
        log.error(f"[{self.adapter_id}] {full_message}")
        self.publish(SystemEvent(
            source=self.adapter_id,
            message=full_message,
            level="ERROR",
            details={"exception": str(error)} if error else {},
        ))

    @abstractmethod
    def connect(self) -> bool:
        """
        Stellt Verbindung her.

        Returns:
            True bei Erfolg, False bei Fehler
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Trennt die Verbindung."""
        pass

    def start(self):
        """Startet den Adapter."""
        with self._lock:
            if self._running:
                return

            if not self._connected:
                if not self.connect():
                    raise RuntimeError(f"Failed to connect adapter {self.adapter_id}")

            self._running = True
            self._on_start()

            self.publish(ConnectedEvent(
                adapter_id=self.adapter_id,
                adapter_type=self.adapter_type,
            ))
            log.info(f"[{self.adapter_id}] Started")

    def stop(self):
        """Stoppt den Adapter."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._on_stop()
            self.disconnect()

            self.publish(DisconnectedEvent(
                adapter_id=self.adapter_id,
                adapter_type=self.adapter_type,
                reason="Stopped by user",
            ))
            log.info(f"[{self.adapter_id}] Stopped")

    def _on_start(self):
        """Hook für Subklassen beim Start."""
        pass

    def _on_stop(self):
        """Hook für Subklassen beim Stop."""
        pass

    def __enter__(self):
        """Context Manager Support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support."""
        self.stop()
        return False


__all__ = ["BaseAdapter"]
