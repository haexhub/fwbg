"""Lookahead Bias Tests für Macro Surprise-Indikatoren."""
from fwbg.builtins.indicators.test_utils import LookaheadBiasTestMixin
from . import MacroSurpriseIndicator


class TestMacroSurpriseLookaheadBias(LookaheadBiasTestMixin):
    """Lookahead Bias Tests für MacroSurpriseIndicator."""

    indicator_class = MacroSurpriseIndicator
