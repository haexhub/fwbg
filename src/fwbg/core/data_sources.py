"""
Data Sources - Zentrale Konfiguration für Datenquellen.

Persistenz über data/{name}/config.json — jede Quelle ist ein Verzeichnis.

Struktur:
    data/
      forexsb/
        config.json          ← Quell-Konfiguration
        raw/                 ← originale Uploads
        datasource/          ← aufbereitete CSV-Dateien
      alphavantage/
        config.json
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any
import json
import logging

import pandas as pd

log = logging.getLogger(__name__)

# Default Basis-Pfad für Daten
DEFAULT_DATA_ROOT = Path("data")


@dataclass
class LoadResult:
    """Standardisiertes Ergebnis von DataSource.load()."""
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_name: str = ""


class SourceType(str, Enum):
    """Typ der Datenquelle."""
    CSV = "csv"
    REST = "rest"
    WEBSOCKET = "websocket"
    DATABASE = "database"


@dataclass
class DataSourceConfig(ABC):
    """Basis-Konfiguration für alle Datenquellen."""
    name: str
    source_type: SourceType = field(init=False)
    description: str = ""

    @abstractmethod
    def create_adapter(self, **kwargs):
        """Erstellt einen passenden DataAdapter."""
        pass

    @abstractmethod
    def load(self, items: Dict[str, str], **params) -> "LoadResult":
        """Load named data items from this source."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        pass


@dataclass
class CSVSourceConfig(DataSourceConfig):
    """Konfiguration für CSV-Datenquellen."""
    path: Path = field(default_factory=lambda: Path("data"))
    file_pattern: str = "{symbol}_{timeframe}.csv"
    timeframe_map: Dict[str, str] = field(default_factory=dict)
    # ETL: raw → datasource conversion
    raw_path: Path = field(default=None)
    raw_pattern: str = "{raw_symbol}_m15.csv"
    timestamp_unit: str = ""          # "ms" for Unix milliseconds, "" for auto-detect
    symbol_map: Dict[str, str] = field(default_factory=dict)  # raw_prefix → symbol
    timezone: str = ""                # IANA timezone for UTC→local conversion, e.g. "Europe/Berlin"

    def __post_init__(self):
        self.source_type = SourceType.CSV
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if isinstance(self.raw_path, str):
            self.raw_path = Path(self.raw_path)

    def to_dict(self) -> dict:
        d = {
            "type": "csv",
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "file_pattern": self.file_pattern,
            "timeframe_map": self.timeframe_map,
        }
        if self.raw_path is not None:
            d["raw_path"] = str(self.raw_path)
            d["raw_pattern"] = self.raw_pattern
        if self.timestamp_unit:
            d["timestamp_unit"] = self.timestamp_unit
        if self.symbol_map:
            d["symbol_map"] = self.symbol_map
        if self.timezone:
            d["timezone"] = self.timezone
        return d

    def prepare(self) -> List[str]:
        """ETL: Konvertiert Rohdaten (raw_path) ins Standard-Format (path).

        Liest jede Datei aus raw_path, mappt den Dateinamen via symbol_map auf
        ein bekanntes Symbol, konvertiert Timestamps (z.B. Unix ms → ISO datetime)
        und schreibt das Ergebnis nach path/{symbol}_{timeframe}.csv.

        Returns:
            Liste der erfolgreich konvertierten Symbole.
        """
        if self.raw_path is None or not self.symbol_map:
            return []

        import pandas as pd

        raw_dir = Path(self.raw_path)
        if not raw_dir.exists():
            log.warning(f"prepare(): raw_path nicht gefunden: {raw_dir}")
            return []

        out_dir = Path(self.path)
        out_dir.mkdir(parents=True, exist_ok=True)

        converted = []
        suffix = "_m15.csv"  # Bestimmt aus raw_pattern
        raw_files = sorted(raw_dir.glob("*_m15.csv"))

        for raw_file in raw_files:
            stem = raw_file.stem                      # z.B. DE40_DAX_m15
            prefix = stem.rsplit("_m15", 1)[0]        # z.B. DE40_DAX
            symbol = self.symbol_map.get(prefix)
            if not symbol:
                log.debug(f"prepare(): kein Mapping für '{prefix}', übersprungen")
                continue

            try:
                df = pd.read_csv(raw_file)

                if "timestamp" not in df.columns:
                    log.warning(f"prepare(): Keine 'timestamp'-Spalte in {raw_file.name}")
                    continue

                if self.timestamp_unit == "ms":
                    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    if self.timezone:
                        ts = ts.dt.tz_convert(self.timezone)
                    df["T"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
                    if self.timezone:
                        # DST fall-back creates duplicate naive strings; keep earliest UTC bar.
                        # Affects ~4 M15 bars (02:00-02:45 local) on 1 Sunday/year in Oct.
                        # For exchange-traded indices these bars fall outside trading hours.
                        before = len(df)
                        df = df.drop_duplicates(subset=["T"], keep="first")
                        dropped = before - len(df)
                        if dropped:
                            log.warning(
                                f"prepare(): {symbol}: {dropped} Bars durch DST-Rückfall "
                                f"entfernt (doppelte naive Timestamps nach {self.timezone}-"
                                f"Konvertierung). Betrifft nur Sonntag 02:00-03:00 Uhr."
                            )
                else:
                    df["T"] = df["timestamp"].astype(str)

                cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
                out = df[["T"] + cols].copy()
                out.columns = ["T", "O", "H", "L", "C"]
                out["V"] = 0

                # Timeframe-Suffix aus file_pattern bestimmen
                # file_pattern: "{symbol}_MINUTE_15.csv" → Suffix = "MINUTE_15"
                tf_part = self.file_pattern.replace("{symbol}_", "").replace(".csv", "")
                dst = out_dir / f"{symbol}_{tf_part}.csv"
                out.to_csv(dst, index=False)
                log.info(f"prepare(): {raw_file.name} → {dst.name} ({len(out)} Bars)")
                converted.append(symbol)

            except Exception as e:
                log.warning(f"prepare(): Fehler bei {raw_file.name}: {e}")

        return converted

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

    def load(self, items: Dict[str, str], **params) -> LoadResult:
        """Load CSV files from self.path."""
        data = {}
        for filename, prefix in items.items():
            csv_path = self.path / f"{filename}.csv"
            if not csv_path.exists():
                log.debug(f"CSV not found: {csv_path}")
                continue
            try:
                raw_df = pd.read_csv(csv_path, nrows=1)
                cols = list(raw_df.columns)
                date_col = None
                for candidate in ["DATE", "Datetime", "datetime", "Time", "time", "Date"]:
                    if candidate in cols:
                        date_col = candidate
                        break
                if date_col:
                    df = pd.read_csv(csv_path, parse_dates=[date_col], index_col=date_col)
                else:
                    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
                data[prefix] = df
            except Exception as e:
                log.warning(f"Failed to load CSV {csv_path}: {e}")

        return LoadResult(data=data, source_name=self.name)

    def create_adapter(self, timeframe: str = "HOUR", **kwargs):
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
    """Konfiguration für REST API Datenquellen."""
    base_url: str = ""
    api_key: str = ""
    api_key_param: str = "apikey"
    api_key_header: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    rate_limit: float = 1.0
    timeout: float = 30.0

    def __post_init__(self):
        self.source_type = SourceType.REST

    def to_dict(self) -> dict:
        return {
            "type": "rest",
            "name": self.name,
            "description": self.description,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_key_param": self.api_key_param,
            "api_key_header": self.api_key_header,
            "headers": self.headers,
            "endpoints": self.endpoints,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
        }

    def get_endpoint_url(self, endpoint: str, **params) -> str:
        """Erstellt die vollständige URL für einen Endpunkt."""
        path = self.endpoints.get(endpoint, endpoint)
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        for key, value in params.items():
            url = url.replace(f"{{{key}}}", str(value))
        if self.api_key and self.api_key_param:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{self.api_key_param}={self.api_key}"
        return url

    def get_headers(self) -> Dict[str, str]:
        headers = dict(self.headers)
        if self.api_key and self.api_key_header:
            headers[self.api_key_header] = self.api_key
        return headers

    def load(self, items: Dict[str, str], **params) -> LoadResult:
        raise NotImplementedError(
            f"REST source '{self.name}' does not support batch loading. "
            f"Use create_adapter() for live data fetching."
        )

    def create_adapter(self, **kwargs):
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
    """Konfiguration für WebSocket Streaming Datenquellen."""
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    subscribe_message: Dict[str, Any] = field(default_factory=dict)
    heartbeat_interval: float = 30.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10

    def __post_init__(self):
        self.source_type = SourceType.WEBSOCKET

    def to_dict(self) -> dict:
        return {
            "type": "websocket",
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "headers": self.headers,
            "subscribe_message": self.subscribe_message,
            "heartbeat_interval": self.heartbeat_interval,
            "reconnect_delay": self.reconnect_delay,
            "max_reconnect_attempts": self.max_reconnect_attempts,
        }

    def get_subscribe_message(self, symbol: str, timeframe: str = "1m") -> Dict[str, Any]:
        msg_str = json.dumps(self.subscribe_message)
        msg_str = msg_str.replace("{symbol}", symbol.lower())
        msg_str = msg_str.replace("{timeframe}", timeframe)
        return json.loads(msg_str)

    def load(self, items: Dict[str, str], **params) -> LoadResult:
        raise NotImplementedError(
            f"WebSocket source '{self.name}' is streaming-only. "
            f"Use create_adapter() for real-time data."
        )

    def create_adapter(self, **kwargs):
        from fwbg.core.registry import BROKER_ADAPTER_REGISTRY

        if "websocket" in BROKER_ADAPTER_REGISTRY:
            adapter_cls = BROKER_ADAPTER_REGISTRY["websocket"]
            return adapter_cls(config=self, **kwargs)

        raise NotImplementedError(
            f"WebSocket adapter not available. Install a WebSocket adapter plugin or "
            f"implement WebSocketDataAdapter for source '{self.name}'."
        )


@dataclass
class DBSourceConfig(DataSourceConfig):
    """Konfiguration für Datenbank-Datenquellen."""
    connection_string: str = ""
    driver: str = "sqlalchemy"

    def __post_init__(self):
        self.source_type = SourceType.DATABASE

    def to_dict(self) -> dict:
        return {
            "type": "database",
            "name": self.name,
            "description": self.description,
            "connection_string": self.connection_string,
            "driver": self.driver,
        }

    def load(self, items: Dict[str, str], **params) -> LoadResult:
        try:
            import sqlalchemy
        except ImportError:
            raise ImportError("sqlalchemy is required for database sources")

        engine = sqlalchemy.create_engine(self.connection_string)
        data = {}
        for query_name, prefix in items.items():
            df = pd.read_sql(query_name, engine)
            data[prefix] = df
        return LoadResult(data=data, source_name=self.name)

    def create_adapter(self, **kwargs):
        raise NotImplementedError(
            f"Database source '{self.name}' does not support adapter creation."
        )


# Typ-Alias für alle Source-Configs
DataSource = CSVSourceConfig | RESTSourceConfig | WebSocketSourceConfig | DBSourceConfig

# Registry für Datenquellen
_DATA_SOURCES: Dict[str, DataSource] = {}


def source_from_dict(d: dict) -> DataSource:
    """Deserialize a source config from a dict."""
    t = d.get("type")
    if t == "csv":
        raw_path = d.get("raw_path")
        return CSVSourceConfig(
            name=d["name"],
            description=d.get("description", ""),
            path=Path(d.get("path", "data")),
            file_pattern=d.get("file_pattern", "{symbol}_{timeframe}.csv"),
            timeframe_map=d.get("timeframe_map", {}),
            raw_path=Path(raw_path) if raw_path else None,
            raw_pattern=d.get("raw_pattern", "{raw_symbol}_m15.csv"),
            timestamp_unit=d.get("timestamp_unit", ""),
            symbol_map=d.get("symbol_map", {}),
            timezone=d.get("timezone", ""),
        )
    elif t == "rest":
        return RESTSourceConfig(
            name=d["name"],
            description=d.get("description", ""),
            base_url=d.get("base_url", ""),
            api_key=d.get("api_key", ""),
            api_key_param=d.get("api_key_param", "apikey"),
            api_key_header=d.get("api_key_header", ""),
            headers=d.get("headers", {}),
            endpoints=d.get("endpoints", {}),
            rate_limit=d.get("rate_limit", 1.0),
            timeout=d.get("timeout", 30.0),
        )
    elif t == "websocket":
        return WebSocketSourceConfig(
            name=d["name"],
            description=d.get("description", ""),
            url=d.get("url", ""),
            headers=d.get("headers", {}),
            subscribe_message=d.get("subscribe_message", {}),
            heartbeat_interval=d.get("heartbeat_interval", 30.0),
            reconnect_delay=d.get("reconnect_delay", 5.0),
            max_reconnect_attempts=d.get("max_reconnect_attempts", 10),
        )
    elif t == "database":
        return DBSourceConfig(
            name=d["name"],
            description=d.get("description", ""),
            connection_string=d.get("connection_string", ""),
            driver=d.get("driver", "sqlalchemy"),
        )
    else:
        raise ValueError(f"Unknown source type: {t!r}")


def save_source_config(source: DataSource, data_root: Path = None) -> None:
    """Write source config to data/{name}/config.json."""
    root = data_root or DEFAULT_DATA_ROOT
    config_path = Path(root) / source.name / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(source.to_dict(), f, indent=2, ensure_ascii=False)
    log.debug(f"Saved config for source '{source.name}' to {config_path}")


def discover_sources(data_root: Path = None) -> None:
    """Scan data_root for source directories and register all found sources."""
    root = Path(data_root or DEFAULT_DATA_ROOT)
    if not root.exists():
        return
    for config_file in sorted(root.glob("*/config.json")):
        try:
            with open(config_file, encoding="utf-8") as f:
                d = json.load(f)
            source = source_from_dict(d)
            _DATA_SOURCES[source.name] = source
            log.debug(f"Discovered source: {source.name} ({source.source_type})")
        except Exception as e:
            log.warning(f"Failed to load source from {config_file}: {e}")
    log.info(f"Loaded {len(_DATA_SOURCES)} data sources from {root}")


def register_csv_source(
    name: str,
    path: str | Path,
    file_pattern: str = "{symbol}_{timeframe}.csv",
    description: str = "",
    timeframe_map: Dict[str, str] = None,
) -> CSVSourceConfig:
    """Registriert eine CSV-Datenquelle."""
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
    """Registriert eine REST API Datenquelle."""
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
    """Registriert eine WebSocket Datenquelle."""
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


def register_db_source(
    name: str,
    connection_string: str,
    driver: str = "sqlalchemy",
    description: str = "",
) -> DBSourceConfig:
    """Registriert eine Datenbank-Datenquelle."""
    source = DBSourceConfig(
        name=name,
        connection_string=connection_string,
        driver=driver,
        description=description,
    )
    _DATA_SOURCES[name] = source
    log.debug(f"Registered DB source: {name}")
    return source


def delete_data_source(name: str) -> None:
    """Remove a data source from the in-memory registry (does not delete files)."""
    if name not in _DATA_SOURCES:
        raise ValueError(f"Unknown data source: '{name}'")
    del _DATA_SOURCES[name]
    log.info(f"Unregistered data source: {name}")


def get_data_source(name: str) -> DataSource:
    """Gibt eine registrierte Datenquelle zurück."""
    if name not in _DATA_SOURCES:
        available = list(_DATA_SOURCES.keys())
        raise ValueError(f"Unknown data source: '{name}'. Available: {available}")
    return _DATA_SOURCES[name]


def list_data_sources(source_type: SourceType = None) -> List[str]:
    """Listet alle registrierten Datenquellen."""
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
    """Setzt den Basis-Pfad für relative Daten-Pfade."""
    global DEFAULT_DATA_ROOT
    DEFAULT_DATA_ROOT = Path(path)
    log.info(f"Data root set to: {DEFAULT_DATA_ROOT}")


def get_data_root() -> Path:
    """Gibt den aktuellen Basis-Pfad zurück."""
    return DEFAULT_DATA_ROOT


# Auto-discover sources from filesystem on import
discover_sources(DEFAULT_DATA_ROOT)


__all__ = [
    # Types
    "SourceType",
    "LoadResult",
    "DataSourceConfig",
    "CSVSourceConfig",
    "RESTSourceConfig",
    "WebSocketSourceConfig",
    "DBSourceConfig",
    "DataSource",
    # Serialization / persistence
    "source_from_dict",
    "save_source_config",
    "discover_sources",
    # Registration
    "register_csv_source",
    "register_rest_source",
    "register_websocket_source",
    "register_db_source",
    "delete_data_source",
    # Getters
    "get_data_source",
    "list_data_sources",
    "get_all_data_sources",
    "set_data_root",
    "get_data_root",
]
