"""Lookahead Bias Tests für Multi-Timeframe-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import MultiTimeframeIndicators


class TestMultiTimeframeLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für MultiTimeframeIndicators."""

    indicator_class = MultiTimeframeIndicators
    test_data_size = 300  # Mehr Daten für D1 features
