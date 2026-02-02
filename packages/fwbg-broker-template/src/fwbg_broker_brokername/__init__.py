"""
fwbg-broker-BROKERNAME: Broker Adapter Template.

Dieses Template zeigt, wie man einen eigenen Broker-Adapter für FWBG erstellt.

TODO:
1. Ersetze alle "BROKERNAME" / "brokername" durch deinen Broker-Namen
2. Implementiere die abstrakten Methoden in adapter.py
3. Erstelle Symbol-Mappings in mappings.py
4. Schreibe Tests
"""
from .adapter import MyBrokerAdapter

__version__ = "1.0.0"

__all__ = ["MyBrokerAdapter"]
