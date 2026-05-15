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

# Default Basis-Pfad für Daten — resolved from workspace env vars at import time
def _default_data_root() -> Path:
    import os
    data_dir = os.environ.get("FWBG_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    workspace = os.environ.get("FWBG_WORKSPACE")
    if workspace:
        return Path(workspace) / "data"
    return Path.home() / "fwbg" / "data"

DEFAULT_DATA_ROOT = _default_data_root()


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
    timestamp_unit: str = ""          # "s", "ms", or "" for ISO/auto
    symbol_map: Dict[str, str] = field(default_factory=dict)  # raw_prefix → symbol
    timezone: str = ""                # IANA timezone, e.g. "Europe/Berlin"
    # Column mapping (raw CSV column names → standard TOHLCV)
    date_col: str = "timestamp"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = ""              # empty = no volume column, fill with 0

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
        # Column mapping — only persist non-defaults
        col_defaults = {"date_col": "timestamp", "open_col": "open", "high_col": "high",
                        "low_col": "low", "close_col": "close", "volume_col": ""}
        for attr, default in col_defaults.items():
            val = getattr(self, attr)
            if val != default:
                d[attr] = val
        return d

    def _build_glob_and_prefix_splitter(self):
        """Derive a glob pattern and a prefix-extraction function from raw_pattern.

        raw_pattern uses ``{raw_symbol}`` as placeholder, e.g.
        ``{raw_symbol}_m15.csv``.  The glob replaces the placeholder with ``*``
        and the splitter strips the suffix so we get the raw symbol prefix.
        """
        placeholder = "{raw_symbol}"
        if placeholder not in self.raw_pattern:
            # Treat the whole pattern as literal glob (user-defined)
            return self.raw_pattern, lambda stem: stem

        idx = self.raw_pattern.index(placeholder)
        suffix = self.raw_pattern[idx + len(placeholder):]  # e.g. "_m15.csv"
        suffix_no_ext = suffix.removesuffix(".csv")          # e.g. "_m15"
        glob_pat = self.raw_pattern.replace(placeholder, "*")

        def extract_prefix(stem: str) -> str:
            if suffix_no_ext and stem.endswith(suffix_no_ext):
                return stem[: -len(suffix_no_ext)]
            return stem

        return glob_pat, extract_prefix

    def prepare(self, glob_override: str = "", excludes: List[str] | None = None) -> List[str]:
        """ETL: Konvertiert Rohdaten (raw_path) ins Standard-Format (path).

        Uses raw_pattern (or glob_override) for file discovery, symbol_map for
        symbol assignment, and the date_col / open_col / … fields for column mapping.

        Args:
            glob_override: Optional glob pattern that overrides raw_pattern for
                file discovery. Passed directly to Path.glob().
            excludes: Optional list of filenames to exclude from processing.

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

        if glob_override:
            raw_files = sorted(raw_dir.glob(glob_override))
            # With a custom glob, prefix = full stem (user manages symbol_map accordingly)
            def extract_prefix(stem: str) -> str:
                return stem
        else:
            glob_pat, extract_prefix = self._build_glob_and_prefix_splitter()
            raw_files = sorted(raw_dir.glob(glob_pat))
        if not raw_files:
            log.warning(f"prepare(): keine Dateien gefunden in {raw_dir}")
            return []

        # Apply excludes
        exclude_set = set(excludes or [])
        if exclude_set:
            raw_files = [f for f in raw_files if f.name not in exclude_set]

        converted = []

        for raw_file in raw_files:
            prefix = extract_prefix(raw_file.stem)
            symbol = self.symbol_map.get(prefix)
            if not symbol:
                log.debug(f"prepare(): kein Mapping für '{prefix}', übersprungen")
                continue

            try:
                df = pd.read_csv(raw_file)

                # --- Timestamp ---
                if self.date_col not in df.columns:
                    log.warning(f"prepare(): Spalte '{self.date_col}' nicht in {raw_file.name}")
                    continue

                if self.timestamp_unit in ("s", "ms"):
                    ts = pd.to_datetime(df[self.date_col], unit=self.timestamp_unit, utc=True)
                    if self.timezone:
                        ts = ts.dt.tz_convert(self.timezone)
                    df["T"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
                    if self.timezone:
                        before = len(df)
                        df = df.drop_duplicates(subset=["T"], keep="first")
                        dropped = before - len(df)
                        if dropped:
                            log.warning(
                                f"prepare(): {symbol}: {dropped} Bars durch DST-Rückfall "
                                f"entfernt (doppelte naive Timestamps nach {self.timezone}-"
                                f"Konvertierung)."
                            )
                else:
                    df["T"] = df[self.date_col].astype(str)

                # --- OHLC columns ---
                ohlc_cols = [self.open_col, self.high_col, self.low_col, self.close_col]
                missing = [c for c in ohlc_cols if c not in df.columns]
                if missing:
                    log.warning(f"prepare(): {raw_file.name}: fehlende Spalten {missing}")
                    continue

                out = df[["T"] + ohlc_cols].copy()
                out.columns = ["T", "O", "H", "L", "C"]

                # --- Volume ---
                if self.volume_col and self.volume_col in df.columns:
                    out["V"] = df[self.volume_col].values
                else:
                    out["V"] = 0

                # --- Write output ---
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


def _resolve_path(p: str | Path, base_dir: Path | None) -> Path:
    """Resolve a path: absolute paths stay absolute, relative ones resolve against base_dir."""
    path = Path(p)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def source_from_dict(d: dict, base_dir: Path | None = None) -> DataSource:
    """Deserialize a source config from a dict.

    If *base_dir* is given, relative ``path`` / ``raw_path`` values are
    resolved against it (typically ``DEFAULT_DATA_ROOT.parent``).
    """
    t = d.get("type")
    if t == "csv":
        raw_path = d.get("raw_path")
        return CSVSourceConfig(
            name=d["name"],
            description=d.get("description", ""),
            path=_resolve_path(d.get("path", "data"), base_dir),
            file_pattern=d.get("file_pattern", "{symbol}_{timeframe}.csv"),
            timeframe_map=d.get("timeframe_map", {}),
            raw_path=_resolve_path(raw_path, base_dir) if raw_path else None,
            raw_pattern=d.get("raw_pattern", "{raw_symbol}_m15.csv"),
            timestamp_unit=d.get("timestamp_unit", ""),
            symbol_map=d.get("symbol_map", {}),
            timezone=d.get("timezone", ""),
            date_col=d.get("date_col", "timestamp"),
            open_col=d.get("open_col", "open"),
            high_col=d.get("high_col", "high"),
            low_col=d.get("low_col", "low"),
            close_col=d.get("close_col", "close"),
            volume_col=d.get("volume_col", ""),
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
            source = source_from_dict(d, base_dir=root.parent)
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
