"""
Lookahead Bias Tests für Trend-Indikatoren.

Diese Tests stellen sicher, dass Features bei Bar i NUR auf Daten
von Bar i-1 oder früher basieren.
"""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import TrendIndicators


class TestTrendLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für TrendIndicators."""

    indicator_class = TrendIndicators
