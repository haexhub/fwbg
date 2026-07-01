"""REST API for the asset registry (controlled asset vocabulary).

Single source of truth for asset classes and the symbol -> asset-class
classification. The dashboard dropdowns and the fwbg-agents researcher both
consume this instead of mirroring the mapping, so it can never drift.

Note: this is the *registry* view (which symbols/classes exist and how they are
classified). The *availability* view (which symbols actually have data, from
configured data sources) lives at ``/api/datasources/assets``.
"""
from fastapi import APIRouter

from fwbg.data.assets import AssetRegistry
from fwbg_sdk.enums import AssetClass

router = APIRouter()

# Canonical, tradeable asset classes. Derived from the AssetClass enum so the
# vocabulary stays in lock-step with the SDK; internal/test classes (e.g. the
# TESTUSD "TEST" bucket in the registry) are intentionally excluded.
_PUBLIC_CLASSES = [ac.name for ac in AssetClass]


@router.get("/assets/classes")
def list_asset_classes():
    """Controlled vocabulary of asset classes plus the known symbols per class."""
    registry = AssetRegistry()
    by_class = {cls: sorted(registry.symbols_by_class(cls)) for cls in _PUBLIC_CLASSES}
    return {"classes": _PUBLIC_CLASSES, "by_class": by_class}


@router.get("/assets")
def list_assets():
    """Flat list of known symbols with their asset class (symbol -> class lookup)."""
    registry = AssetRegistry()
    assets = [
        {
            "symbol": symbol,
            "asset_class": cfg.asset_class,
            "currencies": cfg.currencies,
        }
        for symbol in sorted(registry.all_symbols())
        for cfg in (registry.get(symbol),)
        if cfg.asset_class in _PUBLIC_CLASSES
    ]
    return {"assets": assets}
