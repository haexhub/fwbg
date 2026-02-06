"""Lookahead Bias Tests für Price Action-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import PriceActionIndicators


class TestPriceActionLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für PriceActionIndicators."""

    indicator_class = PriceActionIndicators
