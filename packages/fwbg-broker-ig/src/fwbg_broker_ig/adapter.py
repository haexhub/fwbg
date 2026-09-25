"""Compatibility import for the canonical FWBG IG adapter.

The installable broker package and the monorepo import path intentionally
share one implementation. Keeping this module as a re-export prevents the
two entry points from drifting in SDK argument handling or safety checks.
"""

from fwbg.adapters.broker.ig.adapter import IGBrokerAdapter

__all__ = ["IGBrokerAdapter"]
