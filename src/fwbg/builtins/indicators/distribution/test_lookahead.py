"""Lookahead Bias Tests für Distribution-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import DistributionIndicators


class TestDistributionLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für DistributionIndicators."""

    indicator_class = DistributionIndicators
    test_data_size = 300  # Mehr Daten für z-score lookback
