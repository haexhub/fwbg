"""Lookahead Bias Tests für Risk-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import RiskIndicators


class TestRiskLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für RiskIndicators."""

    indicator_class = RiskIndicators
    test_data_size = 300  # Mehr Daten für CVaR und Vol-of-Vol
