"""
Core Enums für FWBG.

Zentrale Definition aller Enumerationen die im Optimizer und Bot verwendet werden.
"""
from enum import Enum


class Timeframe(str, Enum):
    """Zeitrahmen für OHLC-Daten."""
    M1 = "1M"      # 1 Minute
    M5 = "5M"      # 5 Minuten
    M15 = "15M"    # 15 Minuten
    M30 = "30M"    # 30 Minuten
    H1 = "1H"      # 1 Stunde
    H2 = "2H"      # 2 Stunden
    H4 = "4H"      # 4 Stunden
    D1 = "1D"      # 1 Tag
    W1 = "1W"      # 1 Woche

    def __str__(self) -> str:
        return self.value


class AssetClass(str, Enum):
    """Asset-Klassen."""
    FOREX = "forex"
    INDEX = "index"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class Symbol(str, Enum):
    """
    Standardisierte Trading-Symbole.

    Diese Symbole sind broker-agnostisch. Der BrokerAdapter
    mappt sie auf broker-spezifische Identifier (z.B. IG EPICs).
    """
    # Forex Majors
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDJPY = "USDJPY"
    USDCHF = "USDCHF"
    USDCAD = "USDCAD"
    AUDUSD = "AUDUSD"
    NZDUSD = "NZDUSD"

    # Forex Crosses EUR
    EURCAD = "EURCAD"
    EURCHF = "EURCHF"
    EURGBP = "EURGBP"
    EURJPY = "EURJPY"
    EURAUD = "EURAUD"
    EURNZD = "EURNZD"

    # Forex Crosses GBP
    GBPAUD = "GBPAUD"
    GBPCAD = "GBPCAD"
    GBPCHF = "GBPCHF"
    GBPJPY = "GBPJPY"
    GBPNZD = "GBPNZD"

    # Forex Crosses AUD
    AUDCAD = "AUDCAD"
    AUDCHF = "AUDCHF"
    AUDJPY = "AUDJPY"
    AUDNZD = "AUDNZD"

    # Forex Crosses NZD
    NZDCAD = "NZDCAD"
    NZDCHF = "NZDCHF"
    NZDJPY = "NZDJPY"

    # Forex Crosses CAD/CHF
    CADCHF = "CADCHF"
    CADJPY = "CADJPY"
    CHFJPY = "CHFJPY"

    # Indizes
    DAX = "DAX"
    DOW30 = "DOW30"
    NAS100 = "NAS100"
    SPX500 = "SPX500"
    FTSE100 = "FTSE100"

    # Commodities
    XAUUSD = "XAUUSD"   # Gold
    XAGUSD = "XAGUSD"   # Silver
    BRENT = "BRENT"     # Brent Crude Oil
    WTI = "WTI"         # WTI Crude Oil

    # Crypto
    BTCUSD = "BTCUSD"
    ETHUSD = "ETHUSD"

    def __str__(self) -> str:
        return self.value

    @property
    def asset_class(self) -> AssetClass:
        """Gibt die Asset-Klasse für dieses Symbol zurück."""
        forex = {
            self.EURUSD, self.GBPUSD, self.USDJPY, self.USDCHF, self.USDCAD,
            self.AUDUSD, self.NZDUSD, self.EURCAD, self.EURCHF, self.EURGBP,
            self.EURJPY, self.EURAUD, self.EURNZD, self.GBPAUD, self.GBPCAD,
            self.GBPCHF, self.GBPJPY, self.GBPNZD, self.AUDCAD, self.AUDCHF,
            self.AUDJPY, self.AUDNZD, self.NZDCAD, self.NZDCHF, self.NZDJPY,
            self.CADCHF, self.CADJPY, self.CHFJPY,
        }
        indices = {self.DAX, self.DOW30, self.NAS100, self.SPX500, self.FTSE100}
        commodities = {self.XAUUSD, self.XAGUSD, self.BRENT, self.WTI}
        crypto = {self.BTCUSD, self.ETHUSD}

        if self in forex:
            return AssetClass.FOREX
        elif self in indices:
            return AssetClass.INDEX
        elif self in commodities:
            return AssetClass.COMMODITY
        elif self in crypto:
            return AssetClass.CRYPTO
        return AssetClass.FOREX  # Default


class Direction(str, Enum):
    """Trade-Richtung."""
    LONG = "LONG"
    SHORT = "SHORT"

    def __str__(self) -> str:
        return self.value


class SignalType(str, Enum):
    """Signal-Typen."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def __str__(self) -> str:
        return self.value


__all__ = [
    "Timeframe",
    "AssetClass",
    "Symbol",
    "Direction",
    "SignalType",
]
