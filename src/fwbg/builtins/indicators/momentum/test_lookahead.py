"""Lookahead Bias Tests für Momentum-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import MomentumIndicators


class TestMomentumLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für MomentumIndicators."""

    indicator_class = MomentumIndicators
