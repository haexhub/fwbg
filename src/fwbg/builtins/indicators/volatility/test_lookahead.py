"""Lookahead Bias Tests für Volatility-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import VolatilityIndicators


class TestVolatilityLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für VolatilityIndicators."""

    indicator_class = VolatilityIndicators
