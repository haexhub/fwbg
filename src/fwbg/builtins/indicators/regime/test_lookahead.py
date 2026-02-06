"""Lookahead Bias Tests für Regime-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import RegimeIndicators


class TestRegimeLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für RegimeIndicators."""

    indicator_class = RegimeIndicators
    test_data_size = 300  # Mehr Daten für Hurst-Exponent
