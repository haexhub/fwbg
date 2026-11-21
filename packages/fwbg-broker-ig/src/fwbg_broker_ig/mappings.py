"""Compatibility import for the canonical FWBG IG mappings.

Re-exported like :mod:`fwbg_broker_ig.adapter`, so the package exports the
same symbol and timeframe mappings its adapter uses.
"""

from fwbg.adapters.broker.ig.mappings import (
    SYMBOL_POINT_VALUE,
    SYMBOL_TO_EPIC,
    SYMBOL_TO_YFINANCE,
    TIMEFRAME_TO_RESOLUTION,
    TIMEFRAME_TO_YF_INTERVAL,
)

__all__ = [
    "SYMBOL_POINT_VALUE",
    "SYMBOL_TO_EPIC",
    "SYMBOL_TO_YFINANCE",
    "TIMEFRAME_TO_RESOLUTION",
    "TIMEFRAME_TO_YF_INTERVAL",
]
