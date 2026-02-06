"""Lookahead Bias Tests für Structure-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import StructureIndicators


class TestStructureLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für StructureIndicators."""

    indicator_class = StructureIndicators
    test_data_size = 300  # Mehr Daten für FFT
