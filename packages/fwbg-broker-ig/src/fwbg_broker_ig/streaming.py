"""Compatibility import for the canonical FWBG IG streaming listener.

Re-exported like :mod:`fwbg_broker_ig.adapter`, so the package streams bars
with the same implementation its adapter uses.
"""

from fwbg.adapters.broker.ig.streaming import IGCandleListener

__all__ = ["IGCandleListener"]
