# fwbg-broker-ig

IG Markets Broker Adapter für [FWBG](https://github.com/yourorg/fwbg).

## Installation

```bash
pip install fwbg-broker-ig
```

Mit optionalen Dependencies:

```bash
# Mit yfinance Fallback für historische Daten
pip install fwbg-broker-ig[yfinance]

# Mit Streaming-Unterstützung
pip install fwbg-broker-ig[streaming]

# Alles
pip install fwbg-broker-ig[full]
```

## Verwendung

```python
from fwbg_broker_ig import IGBrokerAdapter
from fwbg.core.enums import Symbol, Timeframe

# Adapter erstellen
adapter = IGBrokerAdapter(
    username="your_username",
    password="your_password",
    api_key="your_api_key",
    env="DEMO"  # oder "LIVE"
)

# Context Manager verwenden
with adapter:
    # Historische Daten abrufen
    df = adapter.get_historical_bars(
        symbol=Symbol.EURUSD,
        timeframe=Timeframe.H1,
        limit=1000
    )

    # Aktuellen Preis abrufen
    price = adapter.get_current_price(Symbol.EURUSD)
    print(f"EURUSD: Bid={price['bid']}, Ask={price['ask']}")

    # Positionen abrufen
    positions = adapter.get_positions()

    # Account Info
    info = adapter.get_account_info()
    print(f"Balance: {info.balance} {info.currency}")
```

## Features

- REST API für historische Daten
- Lightstreamer Streaming für Live-Daten
- yfinance Fallback bei IG Rate-Limiting
- Order-Ausführung (MARKET, LIMIT, STOP)
- Position- und Account-Management
- Automatisches Rate-Limiting

## Unterstützte Instrumente

### Forex
- Majors: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD
- Crosses: 21+ Währungspaare

### Indizes
- DAX, DOW30, NAS100, SPX500, FTSE100

### Commodities
- XAUUSD (Gold), XAGUSD (Silber), BRENT, WTI

### Crypto
- BTCUSD, ETHUSD

## Eigenen Broker-Adapter entwickeln

Dieses Paket kann als Vorlage für eigene Broker-Adapter dienen:

```python
from fwbg.adapters.broker import BrokerAdapter, Symbol, Timeframe

class MyBrokerAdapter(BrokerAdapter):
    adapter_type = "my_broker"

    def connect(self) -> bool:
        # Verbindung herstellen
        pass

    def disconnect(self):
        # Verbindung trennen
        pass

    def get_historical_bars(self, symbol, timeframe, limit, start, end):
        # OHLC-Daten laden
        pass

    def _submit_order_impl(self, symbol, direction, size, stop_distance, limit_distance, order_type):
        # Order ausführen — submit_order() der Basisklasse erzwingt vorher den Stop-Loss-Gate
        pass

    def get_positions(self):
        # Offene Positionen abrufen
        pass

    def get_account_info(self):
        # Kontoinformationen abrufen
        pass

    def get_broker_symbol(self, symbol):
        # Symbol-Mapping
        pass
```

Registriere deinen Adapter in `pyproject.toml`:

```toml
[project.entry-points."fwbg.broker_adapters"]
my_broker = "my_package:MyBrokerAdapter"
```

## Lizenz

MIT License
