"""Lookahead Bias Tests für Dynamics-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import DynamicsIndicators


class TestDynamicsLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für DynamicsIndicators."""

    indicator_class = DynamicsIndicators
