"""
AssetConfig - Konfiguration für einzelne Trading-Assets.

Enthält Asset-spezifische Parameter wie Spread, Point-Value, Währungen etc.
"""
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Dict, Optional


# ── Data-driven spread overrides ────────────────────────────────────────────────
# Real bid-ask spreads measured during data download (e.g. Dukascopy) are stored
# in ``<data_root>/_asset_meta.json`` and take precedence over the hand-tuned
# DEFAULT_ASSETS spreads below.  Cached by mtime so any process picks up updates.

_OVERRIDES_CACHE: Optional[Dict[str, dict]] = None
_OVERRIDES_MTIME: float = -1.0


def _asset_meta_path() -> Path:
    from fwbg.core.data_sources import get_data_root  # lazy: avoid import cycle

    return get_data_root() / "_asset_meta.json"


def load_asset_overrides() -> Dict[str, dict]:
    """Measured per-symbol overrides (currently ``{"spread": float}``)."""
    global _OVERRIDES_CACHE, _OVERRIDES_MTIME
    path = _asset_meta_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _OVERRIDES_CACHE, _OVERRIDES_MTIME = {}, -1.0
        return {}
    if _OVERRIDES_CACHE is None or mtime != _OVERRIDES_MTIME:
        try:
            _OVERRIDES_CACHE = json.loads(path.read_text())
        except (OSError, ValueError):
            _OVERRIDES_CACHE = {}
        _OVERRIDES_MTIME = mtime
    return _OVERRIDES_CACHE


def _update_asset_meta(symbol: str, key: str, value: Optional[float]) -> None:
    """Set (value>0) or remove (None) one field of a symbol's meta entry."""
    path = _asset_meta_path()
    data: Dict[str, dict] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
    entry = data.get(symbol, {})
    if value is None:
        entry.pop(key, None)
    else:
        entry[key] = float(value)
    if entry:
        data[symbol] = entry
    else:
        data.pop(symbol, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def save_asset_spread(symbol: str, spread: float, *, manual: bool = False) -> None:
    """Persist a spread for backtesting.

    ``manual=False`` records the value measured from data (e.g. Dukascopy p90);
    ``manual=True`` records a user override that wins over the measured value.
    The two are stored side by side so a re-download never clobbers an override.
    """
    if not spread or spread <= 0:
        return
    _update_asset_meta(symbol, "manual" if manual else "measured", float(spread))


def set_manual_spread(symbol: str, spread: Optional[float]) -> None:
    """Set (spread>0) or clear (None/≤0) the per-asset manual spread override."""
    _update_asset_meta(symbol, "manual", spread if spread and spread > 0 else None)


def _effective_spread(entry: dict) -> Optional[float]:
    """Manual override wins; otherwise the measured value (legacy ``spread`` too)."""
    for key in ("manual", "measured", "spread"):
        val = entry.get(key)
        if val and val > 0:
            return float(val)
    return None


def list_asset_spreads() -> List[dict]:
    """All symbols with a stored spread: ``{symbol, measured, manual, effective}``."""
    out: List[dict] = []
    for symbol, entry in sorted(load_asset_overrides().items()):
        measured = entry.get("measured") or entry.get("spread")
        manual = entry.get("manual")
        out.append(
            {
                "symbol": symbol,
                "measured": float(measured) if measured else None,
                "manual": float(manual) if manual else None,
                "effective": _effective_spread(entry),
            }
        )
    return out


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

    _instance_lock = threading.Lock()

    def __new__(cls) -> "AssetRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._assets = {}
                    instance._load_defaults()
                    # Publish only after full initialization so concurrent
                    # callers never see a half-built registry.
                    cls._instance = instance
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
            asset = self._assets[symbol]
        else:
            # Default für unbekannte Assets
            asset = AssetConfig(
                symbol=symbol,
                asset_class="FOREX",
                point=0.0001,
                spread=0.00020,
                currencies=["USD"],
            )

        # Data-measured / manually-set spread overrides the configured estimate.
        override = load_asset_overrides().get(symbol)
        if override:
            effective = _effective_spread(override)
            if effective:
                return replace(asset, spread=effective)
        return asset

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


