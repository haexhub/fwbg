"""
AssetConfig - Konfiguration für einzelne Trading-Assets.

Enthält Asset-spezifische Parameter wie Spread, Point-Value, Währungen etc.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class AssetConfig:
    """Konfiguration für ein einzelnes Asset."""
    symbol: str
    asset_class: str  # FOREX, INDEX, COMMODITY, CRYPTO
    point: float  # Kleinste Preiseinheit
    spread: float  # Typischer Spread
    currencies: List[str]  # Beteiligte Währungen

    @classmethod
    def from_dict(cls, symbol: str, data: Dict) -> "AssetConfig":
        """Erstellt AssetConfig aus Dictionary."""
        return cls(
            symbol=symbol,
            asset_class=data.get("class", "FOREX"),
            point=data.get("point", 0.0001),
            spread=data.get("spread", 0.00020),
            currencies=data.get("currency", ["USD"]),
        )

    def spread_in_points(self) -> float:
        """Spread in Points."""
        return self.spread / self.point if self.point > 0 else 0


class AssetRegistry:
    """
    Registry für alle bekannten Assets.

    Singleton-Pattern für globalen Zugriff auf Asset-Konfigurationen.
    """
    _instance: Optional["AssetRegistry"] = None
    _assets: Dict[str, AssetConfig]

    # Default Asset-Konfigurationen
    # Spreads from IG Demo Account (Jan 2026)
    DEFAULT_ASSETS = {
        # FOREX - Majors
        "EURUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00018, "currency": ["EUR", "USD"]},   # 1.8 pips
        "GBPUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00058, "currency": ["GBP", "USD"]},   # 5.8 pips
        "USDJPY": {"class": "FOREX", "point": 0.01, "spread": 0.060, "currency": ["USD", "JPY"]},       # 6.0 pips
        "USDCHF": {"class": "FOREX", "point": 0.0001, "spread": 0.00029, "currency": ["USD", "CHF"]},   # 2.9 pips
        "USDCAD": {"class": "FOREX", "point": 0.0001, "spread": 0.00035, "currency": ["USD", "CAD"]},   # 3.5 pips
        "AUDUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["AUD", "USD"]},   # 3.0 pips
        "NZDUSD": {"class": "FOREX", "point": 0.0001, "spread": 0.00093, "currency": ["NZD", "USD"]},   # 9.3 pips
        # FOREX - Crosses
        "EURGBP": {"class": "FOREX", "point": 0.0001, "spread": 0.00040, "currency": ["EUR", "GBP"]},   # 4.0 pips
        "EURCAD": {"class": "FOREX", "point": 0.0001, "spread": 0.00117, "currency": ["EUR", "CAD"]},   # 11.7 pips
        "EURCHF": {"class": "FOREX", "point": 0.0001, "spread": 0.00030, "currency": ["EUR", "CHF"]},   # 3.0 pips
        "EURNZD": {"class": "FOREX", "point": 0.0001, "spread": 0.00180, "currency": ["EUR", "NZD"]},   # 18.0 pips
        # Indices
        "DAX": {"class": "INDEX", "point": 1.0, "spread": 7.0, "currency": ["EUR"]},                    # 7.0 points
        "DOW30": {"class": "INDEX", "point": 1.0, "spread": 4.8, "currency": ["USD"]},                  # 4.8 points
        "SPX500": {"class": "INDEX", "point": 0.1, "spread": 0.6, "currency": ["USD"]},                 # 0.6 points
        "NAS100": {"class": "INDEX", "point": 0.1, "spread": 2.0, "currency": ["USD"]},                 # 2.0 points
        "FTSE100": {"class": "INDEX", "point": 1.0, "spread": 4.0, "currency": ["GBP"]},                # 4.0 points
        "EU50":   {"class": "INDEX", "point": 1.0, "spread": 3.0, "currency": ["EUR"]},                 # 3.0 points
        "CAC40":  {"class": "INDEX", "point": 1.0, "spread": 2.0, "currency": ["EUR"]},                 # 2.0 points
        "JP225":  {"class": "INDEX", "point": 1.0, "spread": 20.0, "currency": ["JPY"]},                # 20.0 points
        "ASX200": {"class": "INDEX", "point": 1.0, "spread": 2.0, "currency": ["AUD"]},                 # 2.0 points
        "HK50":   {"class": "INDEX", "point": 1.0, "spread": 8.0, "currency": ["HKD"]},                 # 8.0 points
        # Commodities
        "XAUUSD": {"class": "COMMODITY", "point": 0.1, "spread": 0.60, "currency": ["USD"]},            # 0.60 USD
        "GOLD": {"class": "COMMODITY", "point": 0.1, "spread": 0.60, "currency": ["USD"]},              # 0.60 USD
        "XAGUSD": {"class": "COMMODITY", "point": 0.01, "spread": 0.040, "currency": ["USD"]},          # 0.04 USD
        "SILVER": {"class": "COMMODITY", "point": 0.01, "spread": 0.040, "currency": ["USD"]},          # 0.04 USD
        "BRENT": {"class": "COMMODITY", "point": 0.01, "spread": 0.078, "currency": ["USD"]},           # 7.8 cents
        # Crypto
        "BTCUSD": {"class": "CRYPTO", "point": 1.0, "spread": 581.0, "currency": ["USD"]},              # 581 USD
        "ETHUSD": {"class": "CRYPTO", "point": 0.1, "spread": 4.0, "currency": ["USD"]},                # 4.0 USD
        # Test
        "TESTUSD": {"class": "TEST", "point": 0.0001, "spread": 0.00018, "currency": ["USD"]},
    }

    def __new__(cls) -> "AssetRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._assets = {}
            cls._instance._load_defaults()
        return cls._instance

    def _load_defaults(self):
        """Lädt Standard-Asset-Konfigurationen."""
        for symbol, data in self.DEFAULT_ASSETS.items():
            self._assets[symbol] = AssetConfig.from_dict(symbol, data)

    def get(self, symbol: str) -> AssetConfig:
        """
        Gibt AssetConfig für ein Symbol zurück.

        Falls Symbol nicht bekannt, wird Default-FOREX zurückgegeben.
        """
        if symbol in self._assets:
            return self._assets[symbol]

        # Default für unbekannte Assets
        return AssetConfig(
            symbol=symbol,
            asset_class="FOREX",
            point=0.0001,
            spread=0.00020,
            currencies=["USD"]
        )

    def register(self, asset: AssetConfig):
        """Registriert ein neues Asset."""
        self._assets[asset.symbol] = asset

    def all_symbols(self) -> List[str]:
        """Gibt alle registrierten Symbole zurück."""
        return list(self._assets.keys())

    def symbols_by_class(self, asset_class: str) -> List[str]:
        """Gibt alle Symbole einer Asset-Klasse zurück."""
        return [s for s, a in self._assets.items() if a.asset_class == asset_class]


# Globale Instanz für einfachen Zugriff
def get_asset(symbol: str) -> AssetConfig:
    """Shortcut für AssetRegistry().get(symbol)."""
    return AssetRegistry().get(symbol)


