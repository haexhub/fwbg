# Adapter System

FWBG nutzt ein flexibles Adapter-System für die Anbindung von Datenquellen und Brokern. Adapter werden automatisch über Python Entry Points entdeckt und können einfach per `pip install` hinzugefügt werden.

## Übersicht

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   DataAdapter   │────▶│   MessageBus    │◀────│ExecutionAdapter │
│  (Marktdaten)   │     │    (Events)     │     │   (Orders)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   BarEvent              SignalEvent             OrderFilledEvent
   TickEvent                                    OrderRejectedEvent
```

**Adapter-Typen:**
- **DataAdapter**: Liefert Marktdaten (CSV, REST APIs, WebSockets)
- **ExecutionAdapter**: Führt Orders aus (IG, Binance, etc.)

## Installation von Adaptern

```bash
# Built-in Adapter sind bereits enthalten
pip install fwbg

# Zusätzliche Adapter installieren (Beispiele - noch nicht verfügbar)
# pip install fwbg-adapter-binance
# pip install fwbg-adapter-mt5
```

Nach der Installation werden Adapter automatisch erkannt:

```python
from fwbg.core import discover_plugins, list_execution_adapters

discover_plugins()
print(list_execution_adapters())  # ['ig']
```

## Verwendung

### Execution Adapter

```python
from fwbg.core import discover_plugins, get_execution_adapter
from fwbg.core.events import SignalEvent
from fwbg.core.msgbus import get_message_bus

# Plugins laden
discover_plugins()

# Adapter über Registry holen
IGAdapter = get_execution_adapter("ig")

# Oder direkt importieren
from fwbg.adapters import IGExecutionAdapter

# Adapter konfigurieren
adapter = IGExecutionAdapter(
    username="your-username",
    password="your-password",
    api_key="your-api-key",
    env="DEMO",
)

# Als Context Manager nutzen
with adapter:
    # Adapter ist jetzt verbunden und subscribed zu SignalEvents

    # Signal senden (normalerweise von Strategie)
    bus = get_message_bus()
    bus.publish(SignalEvent(
        symbol="EURUSD",
        direction="BUY",
        probability=0.75,
        stop_loss=50.0,
        take_profit=100.0,
    ))

    # Adapter empfängt Signal automatisch und führt Order aus
```

### Data Adapter

```python
from fwbg.adapters import CSVDataAdapter

adapter = CSVDataAdapter(
    data_path="data/forexsb",
    file_pattern="{symbol}_HOUR.csv",
)

with adapter:
    # Historische Daten laden
    df = adapter.get_historical_bars("EURUSD")

    # Bars als Events streamen (für Backtesting)
    for bar_event in adapter.stream_historical_bars("EURUSD"):
        print(bar_event.close)
```

### Datenquellen-Konfiguration

FWBG bietet eine zentrale Konfiguration für Datenquellen mit vorkonfigurierten Pfaden:

```python
from fwbg.core import (
    list_data_sources,
    get_data_source,
    register_data_source,
    set_data_root,
)

# Verfügbare Quellen anzeigen
print(list_data_sources())  # ['forexsb', 'stooq', 'downloads', 'yahoo']

# Vorkonfigurierte Quelle nutzen
source = get_data_source("forexsb")
adapter = source.create_adapter(timeframe="HOUR")

with adapter:
    df = adapter.get_historical_bars("EURUSD")

# Eigene Quelle registrieren
register_data_source(
    name="custom",
    path="/pfad/zu/meinen/daten",
    file_pattern="{symbol}_{timeframe}.csv",
    timeframe_map={
        "1H": "hourly",
        "1D": "daily",
    },
)

# Basis-Pfad ändern (für alle relativen Pfade)
set_data_root("/andere/daten/basis")
```

**Vorkonfigurierte CSV-Quellen:**

| Name | Pfad | Pattern | Beschreibung |
|------|------|---------|--------------|
| `forexsb` | `data/forexsb/` | `{symbol}_{timeframe}.csv` | Forex Strategy Builder |
| `stooq` | `data/stooq/` | `{symbol}.csv` | Stooq.com Daten |
| `downloads` | `data/downloads/` | `{symbol}.csv` | Manuell heruntergeladen |
| `yahoo` | `data/yahoo/` | `{symbol}.csv` | Yahoo Finance |

### REST API Datenquellen

Für REST APIs können Quellen mit Endpunkten, Rate-Limits und API-Keys konfiguriert werden:

```python
from fwbg.core import (
    register_rest_source,
    get_data_source,
    SourceType,
    list_data_sources,
)

# REST-Quelle registrieren
register_rest_source(
    name="alphavantage",
    base_url="https://www.alphavantage.co",
    api_key="DEIN_API_KEY",
    api_key_param="apikey",  # Query-Parameter für API-Key
    endpoints={
        "intraday": "query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={timeframe}",
        "daily": "query?function=TIME_SERIES_DAILY&symbol={symbol}",
    },
    rate_limit=12.0,  # Sekunden zwischen Requests
)

# Alternative: API-Key im Header
register_rest_source(
    name="polygon",
    base_url="https://api.polygon.io",
    api_key="DEIN_API_KEY",
    api_key_header="Authorization",  # Header statt Query-Param
    endpoints={
        "bars": "v2/aggs/ticker/{symbol}/range/1/{timeframe}/{from}/{to}",
    },
)

# Quelle nutzen
source = get_data_source("alphavantage")
url = source.get_endpoint_url("daily", symbol="AAPL")
headers = source.get_headers()

# Nur REST-Quellen auflisten
rest_sources = list_data_sources(source_type=SourceType.REST)
```

**Vorkonfigurierte REST-Quellen:**

| Name | Base URL | Rate Limit | Beschreibung |
|------|----------|------------|--------------|
| `alphavantage` | `alphavantage.co` | 12s | Alpha Vantage (5 calls/min) |
| `polygon` | `api.polygon.io` | 0.2s | Polygon.io |

### WebSocket Streaming Datenquellen

Für Echtzeit-Daten via WebSocket:

```python
from fwbg.core import (
    register_websocket_source,
    get_data_source,
    SourceType,
)

# WebSocket-Quelle registrieren
register_websocket_source(
    name="binance_ws",
    url="wss://stream.binance.com:9443/ws",
    subscribe_message={
        "method": "SUBSCRIBE",
        "params": ["{symbol}@kline_{timeframe}"],
        "id": 1,
    },
    heartbeat_interval=30.0,
    reconnect_delay=5.0,
)

# Quelle nutzen
source = get_data_source("binance_ws")
msg = source.get_subscribe_message(symbol="btcusdt", timeframe="1m")
# -> {"method": "SUBSCRIBE", "params": ["btcusdt@kline_1m"], "id": 1}

# WebSocket-Quellen auflisten
ws_sources = list_data_sources(source_type=SourceType.WEBSOCKET)
```

**Vorkonfigurierte WebSocket-Quellen:**

| Name | URL | Beschreibung |
|------|-----|--------------|
| `binance_ws` | `wss://stream.binance.com:9443/ws` | Binance Kline Streams |
| `finnhub_ws` | `wss://ws.finnhub.io` | Finnhub Realtime |

### Quelltypen

```python
from fwbg.core import SourceType, list_data_sources

# Alle Quellen eines Typs
csv_sources = list_data_sources(source_type=SourceType.CSV)
rest_sources = list_data_sources(source_type=SourceType.REST)
ws_sources = list_data_sources(source_type=SourceType.WEBSOCKET)

# Alle Quellen
all_sources = list_data_sources()
```

---

## Eigenen Adapter schreiben

### Schritt 1: Adapter-Klasse erstellen

```python
from typing import List, Optional
from fwbg.adapters import (
    ExecutionAdapter,
    Order, Position, AccountInfo,
    OrderType, OrderSide,
)
from fwbg.core.events import SignalEvent, OrderFilledEvent, OrderRejectedEvent


class MyBrokerAdapter(ExecutionAdapter):
    """Adapter für MyBroker."""

    adapter_type = "mybroker"

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None

    def connect(self) -> bool:
        """Verbindung herstellen."""
        try:
            # self._client = MyBrokerClient(self.api_key, self.api_secret)
            self._connected = True
            self.log_info("Connected to MyBroker")
            return True
        except Exception as e:
            self.log_error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Verbindung trennen."""
        self._client = None
        self._connected = False

    def submit_order(self, order: Order) -> bool:
        """Order an Broker senden."""
        try:
            # result = self._client.create_order(...)

            # Bei Erfolg: Event veröffentlichen
            self.publish(OrderFilledEvent(
                symbol=order.symbol,
                order_id="12345",
                side=order.side.value,
                quantity=order.quantity,
                price=1.2345,
                commission=0.0,
            ))
            return True

        except Exception as e:
            self.publish(OrderRejectedEvent(
                symbol=order.symbol,
                order_id=order.order_id,
                reason=str(e),
            ))
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Order stornieren."""
        # self._client.cancel_order(order_id)
        return True

    def get_positions(self) -> List[Position]:
        """Offene Positionen abrufen."""
        return []

    def get_account_info(self) -> AccountInfo:
        """Kontoinformationen abrufen."""
        return AccountInfo(
            balance=10000.0,
            equity=10000.0,
            currency="USD",
        )
```

### Schritt 2: Optional - Signal-Konvertierung anpassen

```python
def signal_to_order(self, signal: SignalEvent) -> Optional[Order]:
    """
    Konvertiert SignalEvent zu Order.

    Hier kann eigene Logik implementiert werden:
    - Position Sizing
    - Risk Management
    - Symbol-Mapping
    """
    side = OrderSide.BUY if signal.direction == "BUY" else OrderSide.SELL

    # Beispiel: Risiko-basiertes Position Sizing
    account = self.get_account_info()
    risk_amount = account.balance * 0.02  # 2% Risiko

    if signal.stop_loss and signal.stop_loss > 0:
        quantity = risk_amount / signal.stop_loss
    else:
        quantity = 0.01  # Default

    return Order(
        symbol=signal.symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
    )
```

### Schritt 3: Lokal testen

```python
adapter = MyBrokerAdapter(api_key="test", api_secret="test")

with adapter:
    # Manuell Order senden
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=0.1,
    )
    adapter.submit_order(order)
```

---

## Als Package veröffentlichen

Damit andere Nutzer deinen Adapter installieren können, erstelle ein Python Package.

### Projektstruktur

```
fwbg-adapter-mybroker/
├── pyproject.toml
├── README.md
└── src/
    └── fwbg_adapter_mybroker/
        └── __init__.py       # Dein Adapter-Code
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "fwbg-adapter-mybroker"
version = "1.0.0"
description = "MyBroker Adapter for FWBG"
requires-python = ">=3.9"
dependencies = [
    "fwbg>=2.0.0",
    "mybroker-api>=1.0.0",  # Deine Broker-Library
]

# WICHTIG: Entry Point definieren
[project.entry-points."fwbg.execution_adapters"]
mybroker = "fwbg_adapter_mybroker:MyBrokerAdapter"

[tool.setuptools.packages.find]
where = ["src"]
```

### Veröffentlichen

```bash
# Lokal testen
pip install -e .

# Auf PyPI veröffentlichen
pip install build twine
python -m build
twine upload dist/*
```

Nach der Veröffentlichung können andere Nutzer installieren:

```bash
pip install fwbg-adapter-mybroker
```

---

## Data Adapter erstellen

Data Adapter funktionieren ähnlich, liefern aber Marktdaten statt Orders auszuführen.

```python
from fwbg.adapters import DataAdapter
from fwbg.core.events import BarEvent
import pandas as pd


class MyDataAdapter(DataAdapter):
    """Adapter für MyDataProvider."""

    adapter_type = "mydata"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self._client = None

    def connect(self) -> bool:
        # self._client = MyDataClient(self.api_key)
        self._connected = True
        return True

    def disconnect(self):
        self._client = None
        self._connected = False

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = None,
        start = None,
        end = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """Historische Bars laden."""
        # data = self._client.get_candles(symbol, timeframe, start, end)

        # DataFrame mit Spalten: O, H, L, C, V und DatetimeIndex
        return pd.DataFrame({
            "O": [...],
            "H": [...],
            "L": [...],
            "C": [...],
            "V": [...],
        }, index=pd.DatetimeIndex([...]))

    def subscribe_bars(self, symbol: str, timeframe: str = None, callback = None) -> bool:
        """Live-Bars abonnieren (optional)."""
        # Für Streaming-Datenquellen
        # self._client.subscribe(symbol, self._on_bar)
        return True

    def _on_bar(self, data):
        """Callback wenn neue Bar empfangen."""
        event = BarEvent(
            symbol=data["symbol"],
            timeframe="1H",
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
        )
        self.publish(event)
```

Entry Point für Data Adapter:

```toml
[project.entry-points."fwbg.data_adapters"]
mydata = "fwbg_adapter_mydata:MyDataAdapter"
```

---

## Verfügbare Built-in Adapter

### Execution Adapter

| Name | Beschreibung | Import |
|------|--------------|--------|
| `ig` | IG Markets CFD Broker | `fwbg.adapters.IGExecutionAdapter` |

### Data Adapter

| Name | Beschreibung | Import |
|------|--------------|--------|
| `csv` | CSV-Dateien | `fwbg.adapters.CSVDataAdapter` |

Zusätzlich können Datenquellen über `fwbg.core.get_data_source()` konfiguriert werden.

---

## API Referenz

### ExecutionAdapter

```python
class ExecutionAdapter(BaseAdapter):
    # Abstrakte Methoden (müssen implementiert werden)
    def connect(self) -> bool: ...
    def disconnect(self): ...
    def submit_order(self, order: Order) -> bool: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> List[Position]: ...
    def get_account_info(self) -> AccountInfo: ...

    # Optional überschreibbar
    def signal_to_order(self, signal: SignalEvent) -> Optional[Order]: ...

    # Hilfsmethoden
    def publish(self, event: Event): ...
    def log_info(self, msg: str): ...
    def log_warning(self, msg: str): ...
    def log_error(self, msg: str): ...
```

### Order

```python
@dataclass
class Order:
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
```

### Position

```python
@dataclass
class Position:
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    position_id: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
```

### AccountInfo

```python
@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    currency: str = "USD"
```
