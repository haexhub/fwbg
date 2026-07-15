"""Tests für die sl_level-Auflösung/Validierung in targets.py.

Ein Suffix außerhalb der Schema-Choices (z. B. "range") matchte bisher per
endswith-Auto-Detect eine Distanz-Spalte und lieferte Range-HÖHEN als
absolute SL-PREISE (Run 20260715_042300_0b19fe).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fwbg_sdk import BaseExitStrategy, register_exit_strategy
from fwbg.optimization.targets import resolve_sl_levels


@register_exit_strategy("_test_sl_level_exit")
class _SlLevelExit(BaseExitStrategy):
    def compute_targets(self, df, ctx, **params):
        return np.zeros(len(df)), np.zeros(len(df))

    def resolve_distances(self, df, tp, sl, ctx):
        n = len(df)
        return np.full(n, tp), np.full(n, sl)

    @classmethod
    def get_param_schema(cls):
        return {
            "sl_level": {
                "type": "choice",
                "default": "none",
                "choices": ["none", "or_midpoint", "or_high", "or_low"],
            },
        }


@register_exit_strategy("_test_no_sl_level_exit")
class _NoSlLevelExit(BaseExitStrategy):
    def compute_targets(self, df, ctx, **params):
        return np.zeros(len(df)), np.zeros(len(df))

    def resolve_distances(self, df, tp, sl, ctx):
        n = len(df)
        return np.full(n, tp), np.full(n, sl)

    @classmethod
    def get_param_schema(cls):
        return {}


def _df():
    return pd.DataFrame({
        "orb_or_midpoint": [1.25, 1.26, 1.27],
        "orb_range": [0.005, 0.006, 0.004],
    })


def _ctx(strategy="_test_sl_level_exit"):
    return SimpleNamespace(exit_strategy=strategy)


class TestResolveSlLevels:
    def test_invalid_suffix_raises(self):
        with pytest.raises(ValueError, match="Invalid sl_level 'range'"):
            resolve_sl_levels(_df(), _ctx(), {"sl_level": "range"})

    def test_valid_suffix_returns_level_column(self):
        levels = resolve_sl_levels(_df(), _ctx(), {"sl_level": "or_midpoint"})
        assert levels is not None
        assert levels.tolist() == [1.25, 1.26, 1.27]

    def test_none_or_missing_returns_none(self):
        assert resolve_sl_levels(_df(), _ctx(), {}) is None
        assert resolve_sl_levels(_df(), _ctx(), {"sl_level": "none"}) is None

    def test_valid_suffix_without_matching_column_returns_none(self):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        assert resolve_sl_levels(df, _ctx(), {"sl_level": "or_high"}) is None

    def test_plugin_without_sl_level_schema_keeps_auto_detect(self):
        # Kein sl_level im Schema → keine Choices-Validierung (Altverhalten).
        levels = resolve_sl_levels(
            _df(), _ctx("_test_no_sl_level_exit"), {"sl_level": "or_midpoint"}
        )
        assert levels is not None
