"""Lookahead Bias Tests für Cross-Feature-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import CrossFeatureIndicators


class TestCrossFeatureLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für CrossFeatureIndicators."""

    indicator_class = CrossFeatureIndicators
