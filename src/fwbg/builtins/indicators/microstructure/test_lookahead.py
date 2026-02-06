"""Lookahead Bias Tests für Microstructure-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import MicrostructureIndicator


class TestMicrostructureLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für MicrostructureIndicator."""

    indicator_class = MicrostructureIndicator
