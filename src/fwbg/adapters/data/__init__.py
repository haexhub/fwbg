"""
Data Adapters - Liefern Marktdaten an das System.

DataAdapter ist die Basisklasse für alle Datenquellen:
- CSV-Dateien (Backtesting)
- REST APIs (historische Daten)
- WebSocket Feeds (Live-Daten)

Alle DataAdapters veröffentlichen Events über den MessageBus:
- BarEvent: OHLCV Kerzen
- TickEvent: Trade Ticks
- QuoteEvent: Bid/Ask Quotes
"""
from abc import abstractmethod
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from ..base import BaseAdapter
from fwbg.core.events import BarEvent, TickEvent, QuoteEvent


@dataclass
class BarData:
    """Container für Bar-Daten (intern, vor Event-Konvertierung)."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def to_event(self) -> BarEvent:
        """Konvertiert zu BarEvent."""
        return BarEvent(
            symbol=self.symbol,
            timeframe=self.timeframe,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            bar_timestamp=self.timestamp,
        )


class DataAdapter(BaseAdapter):
    """
    Basisklasse für Data Adapters.

    Subklassen müssen implementieren:
    - connect(): Verbindung zur Datenquelle
    - disconnect(): Verbindung trennen
    - get_historical_bars(): Historische Daten abrufen
    - subscribe_bars(): Live-Bars abonnieren (optional)
    """

    adapter_type: str = "data"

    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = "1H",
        **kwargs
    ):
        """
        Args:
            symbols: Liste von Symbolen die geladen werden
            timeframe: Standard-Timeframe
            **kwargs: Weitere Parameter für BaseAdapter
        """
        super().__init__(**kwargs)
        self._symbols = symbols or []
        self._timeframe = timeframe
        self._subscriptions: Dict[str, Dict[str, bool]] = {}

    @property
    def symbols(self) -> List[str]:
        """Konfigurierte Symbole."""
        return list(self._symbols)

    @property
    def timeframe(self) -> str:
        """Standard-Timeframe."""
        return self._timeframe

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = None,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Lädt historische Bar-Daten.

        Args:
            symbol: Asset-Symbol
            timeframe: Timeframe (default: self.timeframe)
            start: Start-Zeitpunkt
            end: End-Zeitpunkt
            limit: Maximum Anzahl Bars

        Returns:
            DataFrame mit Spalten: O, H, L, C, V und DatetimeIndex
        """
        pass

    def subscribe_bars(
        self,
        symbol: str,
        timeframe: str = None,
        callback: callable = None,
    ) -> bool:
        """
        Abonniert Live-Bars für ein Symbol.

        Default: Nicht implementiert (für historische Datenquellen).
        Live-Adapter überschreiben diese Methode.

        Args:
            symbol: Asset-Symbol
            timeframe: Timeframe
            callback: Optional callback für neue Bars

        Returns:
            True wenn erfolgreich
        """
        self.log_warning(f"subscribe_bars not implemented for {self.adapter_type}")
        return False

    def unsubscribe_bars(self, symbol: str, timeframe: str = None):
        """Beendet Bar-Subscription."""
        key = f"{symbol}_{timeframe or self._timeframe}"
        if key in self._subscriptions:
            del self._subscriptions[key]

    def stream_historical_bars(
        self,
        symbol: str,
        timeframe: str = None,
        start: datetime = None,
        end: datetime = None,
    ) -> Iterator[BarEvent]:
        """
        Streamt historische Bars als Events (für Backtesting).

        Args:
            symbol: Asset-Symbol
            timeframe: Timeframe
            start: Start-Zeitpunkt
            end: End-Zeitpunkt

        Yields:
            BarEvent für jede Bar
        """
        df = self.get_historical_bars(symbol, timeframe, start, end)
        tf = timeframe or self._timeframe

        for timestamp, row in df.iterrows():
            yield BarEvent(
                symbol=symbol,
                timeframe=tf,
                open=float(row.get("O", row.get("open", 0))),
                high=float(row.get("H", row.get("high", 0))),
                low=float(row.get("L", row.get("low", 0))),
                close=float(row.get("C", row.get("close", 0))),
                volume=float(row.get("V", row.get("volume", 0))),
                bar_timestamp=timestamp if isinstance(timestamp, datetime) else pd.to_datetime(timestamp),
            )

    def publish_bar(self, bar: BarData):
        """Veröffentlicht eine Bar als Event."""
        self.publish(bar.to_event())


# Re-export
from .csv_adapter import CSVDataAdapter

__all__ = [
    "DataAdapter",
    "BarData",
    "CSVDataAdapter",
]
