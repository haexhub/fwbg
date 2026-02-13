"""
Data Sources - Zentrale Konfiguration für Datenquellen.

Unterstützt verschiedene Quelltypen:
- CSV: Lokale CSV-Dateien
- REST: REST API Endpunkte
- WebSocket: WebSocket Streaming

Beispiel:
    from fwbg.core.data_sources import (
        get_data_source,
        register_csv_source,
        register_rest_source,
        register_websocket_source,
    )

    # CSV-Quelle (vorkonfiguriert)
    source = get_data_source("forexsb")
    adapter = source.create_adapter(timeframe="HOUR")

    # REST API registrieren
    register_rest_source(
        name="alphavantage",
        base_url="https://www.alphavantage.co/query",
        api_key="YOUR_KEY",
    )

    # WebSocket registrieren
    register_websocket_source(
        name="binance_ws",
        url="wss://stream.binance.com:9443/ws",
    )
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any
import logging

log = logging.getLogger(__name__)

# Default Basis-Pfad für Daten
DEFAULT_DATA_ROOT = Path("data")


class SourceType(str, Enum):
    """Typ der Datenquelle."""
    CSV = "csv"
    REST = "rest"
    WEBSOCKET = "websocket"


@dataclass
class DataSourceConfig(ABC):
    """Basis-Konfiguration für alle Datenquellen."""
    name: str
    source_type: SourceType = field(init=False)  # Wird in Subclass gesetzt
    description: str = ""

    @abstractmethod
    def create_adapter(self, **kwargs):
        """Erstellt einen passenden DataAdapter."""
        pass


@dataclass
class CSVSourceConfig(DataSourceConfig):
    """Konfiguration für CSV-Datenquellen."""
    path: Path = field(default_factory=lambda: Path("data"))
    file_pattern: str = "{symbol}.csv"
    timeframe_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.source_type = SourceType.CSV
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def get_file_path(self, symbol: str, timeframe: str = None) -> Path:
        """Gibt den vollständigen Dateipfad für ein Symbol zurück."""
        tf_value = self.timeframe_map.get(timeframe, timeframe) if timeframe else ""
        filename = self.file_pattern.format(symbol=symbol, timeframe=tf_value)
        return self.path / filename

    def exists(self) -> bool:
        """Prüft ob der Daten-Pfad existiert."""
        return self.path.exists()

    def list_files(self, pattern: str = "*.csv") -> List[Path]:
        """Listet alle Dateien im Daten-Verzeichnis."""
        if not self.exists():
            return []
        return list(self.path.glob(pattern))

    def create_adapter(self, timeframe: str = "HOUR", **kwargs):
        """
        Erstellt einen CSV DataAdapter.

        Hinweis: Der CSVDataAdapter muss separat implementiert
        oder als Plugin installiert werden.
        """
        from fwbg.core.registry import BROKER_ADAPTER_REGISTRY

        if "csv" in BROKER_ADAPTER_REGISTRY:
            adapter_cls = BROKER_ADAPTER_REGISTRY["csv"]
            tf_value = self.timeframe_map.get(timeframe, timeframe)
            pattern = self.file_pattern.format(symbol="{symbol}", timeframe=tf_value)
            return adapter_cls(
                data_path=str(self.path),
                file_pattern=pattern,
                timeframe=timeframe,
                **kwargs
            )

        raise NotImplementedError(
            f"CSV adapter not available. Install a CSV adapter plugin or "
            f"implement CSVDataAdapter for source '{self.name}'."
        )


@dataclass
class RESTSourceConfig(DataSourceConfig):
    """
    Konfiguration für REST API Datenquellen.

    Beispiel:
        config = RESTSourceConfig(
            name="alphavantage",
            base_url="https://www.alphavantage.co/query",
            api_key="YOUR_KEY",
            headers={"User-Agent": "FWBG/1.0"},
            endpoints={
                "historical": "/query?function=TIME_SERIES_INTRADAY&symbol={symbol}",
                "quote": "/query?function=GLOBAL_QUOTE&symbol={symbol}",
            },
        )
    """
    base_url: str = ""
    api_key: str = ""
    api_key_param: str = "apikey"  # Query-Parameter für API-Key
    api_key_header: str = ""  # Alternativ: Header für API-Key
    headers: Dict[str, str] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    rate_limit: float = 1.0  # Sekunden zwischen Requests
    timeout: float = 30.0

    def __post_init__(self):
        self.source_type = SourceType.REST

    def get_endpoint_url(self, endpoint: str, **params) -> str:
        """Erstellt die vollständige URL für einen Endpunkt."""
        path = self.endpoints.get(endpoint, endpoint)
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

        # Parameter einsetzen
        for key, value in params.items():
            url = url.replace(f"{{{key}}}", str(value))

        # API-Key als Query-Parameter hinzufügen
        if self.api_key and self.api_key_param:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{self.api_key_param}={self.api_key}"

        return url

    def get_headers(self) -> Dict[str, str]:
        """Gibt die Request-Headers zurück."""
        headers = dict(self.headers)
        if self.api_key and self.api_key_header:
            headers[self.api_key_header] = self.api_key
        return headers

    def create_adapter(self, **kwargs):
        """
        Erstellt einen REST DataAdapter.

        Hinweis: Der RESTDataAdapter muss separat implementiert
        oder als Plugin installiert werden.
        """
        # Versuche REST Adapter aus Registry zu laden
        from fwbg.core.registry import BROKER_ADAPTER_REGISTRY

        if "rest" in BROKER_ADAPTER_REGISTRY:
            adapter_cls = BROKER_ADAPTER_REGISTRY["rest"]
            return adapter_cls(config=self, **kwargs)

        raise NotImplementedError(
            f"REST adapter not available. Install a REST adapter plugin or "
            f"implement RESTDataAdapter for source '{self.name}'."
        )


@dataclass
class WebSocketSourceConfig(DataSourceConfig):
    """
    Konfiguration für WebSocket Streaming Datenquellen.

    Beispiel:
        config = WebSocketSourceConfig(
            name="binance_ws",
            url="wss://stream.binance.com:9443/ws",
            subscribe_message={
                "method": "SUBSCRIBE",
                "params": ["{symbol}@kline_{timeframe}"],
            },
        )
    """
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    subscribe_message: Dict[str, Any] = field(default_factory=dict)
    heartbeat_interval: float = 30.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10

    def __post_init__(self):
        self.source_type = SourceType.WEBSOCKET

    def get_subscribe_message(self, symbol: str, timeframe: str = "1m") -> Dict[str, Any]:
        """Erstellt die Subscribe-Nachricht mit eingesetzten Parametern."""
        import json
        msg_str = json.dumps(self.subscribe_message)
        msg_str = msg_str.replace("{symbol}", symbol.lower())
        msg_str = msg_str.replace("{timeframe}", timeframe)
        return json.loads(msg_str)

    def create_adapter(self, **kwargs):
        """
        Erstellt einen WebSocket DataAdapter.

        Hinweis: Der WebSocketDataAdapter muss separat implementiert
        oder als Plugin installiert werden.
        """
        from fwbg.core.registry import BROKER_ADAPTER_REGISTRY

        if "websocket" in BROKER_ADAPTER_REGISTRY:
            adapter_cls = BROKER_ADAPTER_REGISTRY["websocket"]
            return adapter_cls(config=self, **kwargs)

        raise NotImplementedError(
            f"WebSocket adapter not available. Install a WebSocket adapter plugin or "
            f"implement WebSocketDataAdapter for source '{self.name}'."
        )


# Typ-Alias für alle Source-Configs
DataSource = CSVSourceConfig | RESTSourceConfig | WebSocketSourceConfig

# Registry für Datenquellen
_DATA_SOURCES: Dict[str, DataSource] = {}


def register_csv_source(
    name: str,
    path: str | Path,
    file_pattern: str = "{symbol}.csv",
    description: str = "",
    timeframe_map: Dict[str, str] = None,
) -> CSVSourceConfig:
    """
    Registriert eine CSV-Datenquelle.

    Args:
        name: Eindeutiger Name
        path: Pfad zum Verzeichnis
        file_pattern: Dateinamens-Pattern ({symbol}, {timeframe})
        description: Optionale Beschreibung
        timeframe_map: Mapping von Timeframe-Namen zu Pattern-Werten

    Returns:
        Die registrierte CSVSourceConfig
    """
    source = CSVSourceConfig(
        name=name,
        path=Path(path),
        file_pattern=file_pattern,
        description=description,
        timeframe_map=timeframe_map or {},
    )
    _DATA_SOURCES[name] = source
    log.debug(f"Registered CSV source: {name} -> {path}")
    return source


def register_rest_source(
    name: str,
    base_url: str,
    api_key: str = "",
    api_key_param: str = "apikey",
    api_key_header: str = "",
    headers: Dict[str, str] = None,
    endpoints: Dict[str, str] = None,
    rate_limit: float = 1.0,
    description: str = "",
) -> RESTSourceConfig:
    """
    Registriert eine REST API Datenquelle.

    Args:
        name: Eindeutiger Name
        base_url: Basis-URL der API
        api_key: API-Schlüssel
        api_key_param: Query-Parameter für API-Key (default: "apikey")
        api_key_header: Header für API-Key (alternativ zu Query-Param)
        headers: Zusätzliche HTTP-Headers
        endpoints: Dict von Endpunkt-Namen zu Pfaden
        rate_limit: Sekunden zwischen Requests
        description: Optionale Beschreibung

    Returns:
        Die registrierte RESTSourceConfig
    """
    source = RESTSourceConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        api_key_param=api_key_param,
        api_key_header=api_key_header,
        headers=headers or {},
        endpoints=endpoints or {},
        rate_limit=rate_limit,
        description=description,
    )
    _DATA_SOURCES[name] = source
    log.debug(f"Registered REST source: {name} -> {base_url}")
    return source


def register_websocket_source(
    name: str,
    url: str,
    headers: Dict[str, str] = None,
    subscribe_message: Dict[str, Any] = None,
    heartbeat_interval: float = 30.0,
    reconnect_delay: float = 5.0,
    description: str = "",
) -> WebSocketSourceConfig:
    """
    Registriert eine WebSocket Datenquelle.

    Args:
        name: Eindeutiger Name
        url: WebSocket URL (wss://...)
        headers: HTTP-Headers für Verbindung
        subscribe_message: Template für Subscribe-Nachricht
        heartbeat_interval: Ping-Intervall in Sekunden
        reconnect_delay: Verzögerung bei Reconnect
        description: Optionale Beschreibung

    Returns:
        Die registrierte WebSocketSourceConfig
    """
    source = WebSocketSourceConfig(
        name=name,
        url=url,
        headers=headers or {},
        subscribe_message=subscribe_message or {},
        heartbeat_interval=heartbeat_interval,
        reconnect_delay=reconnect_delay,
        description=description,
    )
    _DATA_SOURCES[name] = source
    log.debug(f"Registered WebSocket source: {name} -> {url}")
    return source


def get_data_source(name: str) -> DataSource:
    """
    Gibt eine registrierte Datenquelle zurück.

    Args:
        name: Name der Datenquelle

    Returns:
        DataSourceConfig (CSV, REST oder WebSocket)

    Raises:
        ValueError: Wenn Quelle nicht gefunden
    """
    if name not in _DATA_SOURCES:
        available = list(_DATA_SOURCES.keys())
        raise ValueError(f"Unknown data source: '{name}'. Available: {available}")
    return _DATA_SOURCES[name]


def list_data_sources(source_type: SourceType = None) -> List[str]:
    """
    Listet alle registrierten Datenquellen.

    Args:
        source_type: Optional - nur Quellen dieses Typs

    Returns:
        Liste von Namen
    """
    if source_type is None:
        return list(_DATA_SOURCES.keys())
    return [
        name for name, source in _DATA_SOURCES.items()
        if source.source_type == source_type
    ]


def get_all_data_sources() -> Dict[str, DataSource]:
    """Gibt alle registrierten Datenquellen zurück."""
    return dict(_DATA_SOURCES)


def set_data_root(path: str | Path):
    """
    Setzt den Basis-Pfad für relative Daten-Pfade.

    Args:
        path: Neuer Basis-Pfad
    """
    global DEFAULT_DATA_ROOT
    DEFAULT_DATA_ROOT = Path(path)
    log.info(f"Data root set to: {DEFAULT_DATA_ROOT}")


def get_data_root() -> Path:
    """Gibt den aktuellen Basis-Pfad zurück."""
    return DEFAULT_DATA_ROOT


def _init_default_sources():
    """Initialisiert die Standard-Datenquellen."""
    root = DEFAULT_DATA_ROOT

    # === CSV Quellen ===

    # ForexSB - Forex Strategy Builder Daten
    register_csv_source(
        name="forexsb",
        path=root / "forexsb",
        file_pattern="{symbol}_{timeframe}.csv",
        description="Forex Strategy Builder CSV exports",
        timeframe_map={
            "1H": "HOUR",
            "H1": "HOUR",
            "HOUR": "HOUR",
            "15M": "MINUTE_15",
            "M15": "MINUTE_15",
            "MINUTE_15": "MINUTE_15",
            "30M": "MINUTE_30",
            "M30": "MINUTE_30",
            "MINUTE_30": "MINUTE_30",
        },
    )

    # Stooq - Stooq.com Daten
    register_csv_source(
        name="stooq",
        path=root / "stooq",
        file_pattern="{symbol}.csv",
        description="Stooq.com historical data",
    )

    # Downloads - Manuell heruntergeladene Daten
    register_csv_source(
        name="downloads",
        path=root / "downloads",
        file_pattern="{symbol}.csv",
        description="Manually downloaded data files",
    )

    # Yahoo - yfinance Daten
    register_csv_source(
        name="yahoo",
        path=root / "yahoo",
        file_pattern="{symbol}.csv",
        description="Yahoo Finance data via yfinance",
    )

    # === REST API Quellen (Beispiele - API-Key muss gesetzt werden) ===

    register_rest_source(
        name="alphavantage",
        base_url="https://www.alphavantage.co",
        api_key_param="apikey",
        endpoints={
            "intraday": "query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={timeframe}",
            "daily": "query?function=TIME_SERIES_DAILY&symbol={symbol}",
            "quote": "query?function=GLOBAL_QUOTE&symbol={symbol}",
        },
        rate_limit=12.0,  # 5 requests/minute = 12s zwischen Requests
        description="Alpha Vantage API (free tier: 5 calls/min)",
    )

    register_rest_source(
        name="polygon",
        base_url="https://api.polygon.io",
        api_key_param="apiKey",
        endpoints={
            "bars": "v2/aggs/ticker/{symbol}/range/{multiplier}/{timeframe}/{from}/{to}",
            "quote": "v2/last/trade/{symbol}",
        },
        rate_limit=0.2,  # 5 requests/second
        description="Polygon.io API",
    )

    # === WebSocket Quellen (Beispiele) ===

    register_websocket_source(
        name="binance_ws",
        url="wss://stream.binance.com:9443/ws",
        subscribe_message={
            "method": "SUBSCRIBE",
            "params": ["{symbol}@kline_{timeframe}"],
            "id": 1,
        },
        description="Binance WebSocket Streams",
    )

    register_websocket_source(
        name="finnhub_ws",
        url="wss://ws.finnhub.io",
        subscribe_message={
            "type": "subscribe",
            "symbol": "{symbol}",
        },
        description="Finnhub WebSocket (requires API key in URL)",
    )


# Initialisiere Standard-Quellen beim Import
_init_default_sources()


__all__ = [
    # Types
    "SourceType",
    "DataSourceConfig",
    "CSVSourceConfig",
    "RESTSourceConfig",
    "WebSocketSourceConfig",
    "DataSource",
    # Registration
    "register_csv_source",
    "register_rest_source",
    "register_websocket_source",
    # Getters
    "get_data_source",
    "list_data_sources",
    "get_all_data_sources",
    "set_data_root",
    "get_data_root",
]
