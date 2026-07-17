"""
Core Enums für FWBG.

Zentrale Definition aller Enumerationen die im Optimizer und Bot verwendet werden.
"""
from enum import Enum


class Timeframe(str, Enum):
    """Zeitrahmen für OHLC-Daten.

    Zentrale Wahrheitsquelle für alle Timeframe-Vokabulare. Der ``value`` bleibt
    das kompakte Kürzel (``"1H"``); ``canonical`` liefert die im fwbg-Kern und in
    Dateinamen/Strategien genutzte Langform (``"HOUR_1"``). Über :meth:`from_str`
    werden beliebige Schreibweisen (``"HOUR_1"``, ``"HOUR"``, ``"1H"``, ``"H1"``,
    ``"1h"`` …) auf das Enum normalisiert; jede Datenquelle mappt anschließend vom
    Enum auf ihre eigene Bezeichnung.
    """
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

    @property
    def canonical(self) -> str:
        """fwbg-interne Langform, z.B. ``"HOUR_1"`` — Basis für Dateinamen."""
        return _TF_CANONICAL[self]

    @property
    def granularity(self) -> str:
        """Zeitfamilie: ``"MINUTE"`` | ``"HOUR"`` | ``"DAY"`` | ``"WEEK"``."""
        return self.canonical.split("_")[0]

    @property
    def minutes(self) -> int:
        """Balkendauer in Minuten."""
        return _TF_MINUTES[self]

    @property
    def resample_rule(self) -> str:
        """Pandas-Resample-Regel, z.B. ``"1h"``."""
        return _TF_RESAMPLE_RULE[self]

    @classmethod
    def from_str(cls, value: "str | Timeframe") -> "Timeframe":
        """Normalisiert eine beliebige Timeframe-Schreibweise auf das Enum.

        Akzeptiert kanonische Langform (``"HOUR_1"``), Enum-Namen (``"H1"``),
        Enum-Werte (``"1H"``), gängige Kurzformen (``"HOUR"``, ``"DAY"``) und
        quellenspezifische Varianten wie yfinance (``"1h"``, ``"1d"``).
        """
        if isinstance(value, cls):
            return value
        key = str(value).strip().upper()
        try:
            return _TF_ALIASES[key]
        except KeyError:
            raise ValueError(
                f"unknown timeframe {value!r}; valid: {sorted(_TF_ALIASES)}"
            ) from None


_TF_CANONICAL: dict[Timeframe, str] = {
    Timeframe.M1: "MINUTE_1",
    Timeframe.M5: "MINUTE_5",
    Timeframe.M15: "MINUTE_15",
    Timeframe.M30: "MINUTE_30",
    Timeframe.H1: "HOUR_1",
    Timeframe.H2: "HOUR_2",
    Timeframe.H4: "HOUR_4",
    Timeframe.D1: "DAY_1",
    Timeframe.W1: "WEEK_1",
}

_TF_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H2: 120,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
}

_TF_RESAMPLE_RULE: dict[Timeframe, str] = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H2: "2h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1D",
    Timeframe.W1: "1W",
}


def _build_tf_aliases() -> dict[str, Timeframe]:
    aliases: dict[str, Timeframe] = {}
    for tf in Timeframe:
        for key in (tf.name, tf.value, _TF_CANONICAL[tf]):
            aliases[key.upper()] = tf
    # Legacy-Kurzformen (Config, Charts, ältere Datenquellen)
    aliases.update({
        "MINUTE": Timeframe.M1,
        "HOUR": Timeframe.H1,
        "HOUR_SWING": Timeframe.H1,
        "DAY": Timeframe.D1,
        "WEEK": Timeframe.W1,
    })
    return aliases


_TF_ALIASES: dict[str, Timeframe] = _build_tf_aliases()


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
