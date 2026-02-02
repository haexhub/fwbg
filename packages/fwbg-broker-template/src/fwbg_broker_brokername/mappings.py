"""
Symbol Mappings für den Broker.

TODO: Füge Mappings für alle unterstützten Instrumente hinzu.
"""
from typing import Dict

from fwbg.core.enums import Symbol, Timeframe


# Symbol -> Broker-spezifischer Identifier
# TODO: Füge Mappings für alle unterstützten Instrumente hinzu
SYMBOL_TO_BROKER: Dict[Symbol, str] = {
    # Forex Majors
    Symbol.EURUSD: "EUR_USD",  # TODO: Anpassen an Broker-Format
    Symbol.GBPUSD: "GBP_USD",
    Symbol.USDJPY: "USD_JPY",
    # ...weitere Symbole hinzufügen
}


# Timeframe -> Broker-spezifisches Format
# TODO: Anpassen an Broker-API
TIMEFRAME_TO_BROKER: Dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}


# Point Value für Pip-Berechnung
# TODO: Anpassen an Broker-spezifische Werte
SYMBOL_POINT_VALUE: Dict[Symbol, float] = {
    Symbol.EURUSD: 0.0001,
    Symbol.GBPUSD: 0.0001,
    Symbol.USDJPY: 0.01,
    # ...weitere Symbole hinzufügen
}
