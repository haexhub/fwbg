"""Lookahead Bias Tests für Ichimoku-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import IchimokuIndicators


class TestIchimokuLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für IchimokuIndicators."""

    indicator_class = IchimokuIndicators
