"""
FWBG - Forex/Trading Strategy Optimizer.

Plugin-basierte Architektur für Trading-Strategien.
"""

__version__ = "2.18.0"

from .core.registry import discover_plugins

# Plugin-Discovery beim Import
discover_plugins()
