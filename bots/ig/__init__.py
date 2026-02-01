"""
IG Markets Trading Bot.

Verwendet die FWBG Plugin-Infrastruktur für:
- Indicator-Berechnung
- Feature-Engineering
- Live-Trading via IG API

Usage:
    python -m bots.ig
"""

from .bot import EliteBot, STREAMING_AVAILABLE
from .streaming import StreamingManager, StreamingCacheManager

__all__ = ["EliteBot", "STREAMING_AVAILABLE", "StreamingManager", "StreamingCacheManager"]
