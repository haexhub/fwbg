# fwbg-broker-template

Template für Broker-Adapter in [FWBG](https://github.com/yourorg/fwbg).

## Eigenen Broker-Adapter erstellen

### 1. Template kopieren

```bash
cp -r packages/fwbg-broker-template packages/fwbg-broker-mybroker
cd packages/fwbg-broker-mybroker
```

### 2. Dateien umbenennen

```bash
mv src/fwbg_broker_brokername src/fwbg_broker_mybroker
```

### 3. pyproject.toml anpassen

Ersetze alle Vorkommen von:
- `BROKERNAME` / `brokername` → Name deines Brokers
- `MyBrokerAdapter` → Name deiner Adapter-Klasse

### 4. Implementiere die abstrakten Methoden

In `adapter.py`:

```python
class MyBrokerAdapter(BrokerAdapter):
    adapter_type = "mybroker"

    def connect(self) -> bool:
        # Verbindung zum Broker herstellen
        pass

    def disconnect(self):
        # Verbindung trennen
        pass

    def get_broker_symbol(self, symbol: Symbol) -> Optional[str]:
        # Symbol-Mapping
        pass

    def get_historical_bars(self, symbol, timeframe, limit, start, end) -> pd.DataFrame:
        # OHLC-Daten abrufen
        pass

    def _submit_order_impl(self, symbol, direction, size, stop_distance, limit_distance, order_type) -> OrderResult:
        # Order ausführen — submit_order() der Basisklasse erzwingt vorher den Stop-Loss-Gate
        pass

    def get_positions(self) -> List[Position]:
        # Offene Positionen abrufen
        pass

    def get_account_info(self) -> AccountInfo:
        # Kontoinformationen abrufen
        pass
```

### 5. Symbol-Mappings erstellen

In `mappings.py`:

```python
SYMBOL_TO_BROKER: Dict[Symbol, str] = {
    Symbol.EURUSD: "EUR/USD",  # Broker-spezifisches Format
    Symbol.GBPUSD: "GBP/USD",
    # ...
}
```

### 6. Tests schreiben

Erstelle `tests/test_adapter.py`:

```python
import pytest
from fwbg_broker_mybroker import MyBrokerAdapter

def test_connect():
    adapter = MyBrokerAdapter(api_key="test", api_secret="test")
    # ...
```

### 7. Installieren und Testen

```bash
# Entwicklungsmodus
pip install -e .

# Tests ausführen
pytest

# Veröffentlichen (optional)
pip install build twine
python -m build
twine upload dist/*
```

## Entry Point

Der Entry Point in `pyproject.toml` registriert deinen Adapter automatisch bei fwbg:

```toml
[project.entry-points."fwbg.broker_adapters"]
mybroker = "fwbg_broker_mybroker:MyBrokerAdapter"
```

Nach der Installation kann fwbg deinen Adapter automatisch erkennen:

```python
# In fwbg
from fwbg.core.registry import get_broker_adapter

adapter = get_broker_adapter("mybroker", api_key="...", api_secret="...")
```

## Tipps

### Rate Limiting

Implementiere Rate Limiting um API-Limits nicht zu überschreiten:

```python
import time

def _rate_limit(self):
    elapsed = time.time() - self._last_request_time
    if elapsed < self.rate_limit_delay:
        time.sleep(self.rate_limit_delay - elapsed)
    self._last_request_time = time.time()
```

### Streaming

Wenn dein Broker WebSocket-Streaming unterstützt:

```python
def subscribe_bars(self, symbol, timeframe, callback) -> bool:
    # WebSocket-Subscription einrichten
    pass

def _on_bar_received(self, data):
    bar = BarData(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=data["time"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
    )
    self._notify_bar_callbacks(bar)
```

### Error Handling

Nutze die eingebauten Logging-Methoden:

```python
self.log_info("Connected")
self.log_warning("Rate limited, waiting...")
self.log_error(f"Order failed: {e}")
```

## Lizenz

MIT License
