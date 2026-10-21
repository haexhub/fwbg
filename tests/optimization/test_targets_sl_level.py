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
from fwbg.core.columns import AmbiguousColumnError
from fwbg.optimization.targets import resolve_sl_levels


@register_exit_strategy("_test_sl_level_exit")
class _SlLevelExit(BaseExitStrategy):
    def compute_targets(self, df, ctx, **params):
        return np.zeros(len(df)), np.zeros(len(df))

    def resolve_distances(self, df, tp, sl, ctx):
        n = len(df)
        return np.full(n, tp), np.full(n, sl)

    def get_cache_key(self, params):
        return "_test_sl_level_exit"

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


class TestAmbiguousSlLevel:
    """Several *_<sl_level> columns (one per ORB session) fail closed unless
    exit_params.sl_level_column names the one to use."""

    @staticmethod
    def _sessions_df():
        return pd.DataFrame({
            "rb4_orb_s00_or_high": [1.30, 1.31],
            "rb4_orb_s08_or_high": [1.40, 1.41],
        })

    def test_several_level_columns_are_refused(self):
        with pytest.raises(AmbiguousColumnError) as err:
            resolve_sl_levels(self._sessions_df(), _ctx(), {"sl_level": "or_high"})
        message = str(err.value)
        assert "rb4_orb_s00_or_high" in message and "rb4_orb_s08_or_high" in message
        assert "sl_level_column" in message

    def test_sl_level_column_resolves_the_ambiguity(self):
        levels = resolve_sl_levels(
            self._sessions_df(), _ctx(),
            {"sl_level": "or_high", "sl_level_column": "rb4_orb_s08_or_high"},
        )
        assert levels.tolist() == [1.40, 1.41]

    def test_sl_level_column_pattern_resolves_the_ambiguity(self):
        levels = resolve_sl_levels(
            self._sessions_df(), _ctx(),
            {"sl_level": "or_high", "sl_level_column": "*_orb_s08_or_high"},
        )
        assert levels.tolist() == [1.40, 1.41]


def test_simulate_trades_scales_the_level_distance_by_sl_mult():
    from fwbg.core.context import SimulationContext
    from fwbg.optimization.targets import simulate_trades

    n = 4
    df = pd.DataFrame(
        {
            "O": [1.2600] * n, "H": [1.2601] * n, "L": [1.2599] * n, "C": [1.2600] * n,
            "orb_or_low": [1.2580] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )
    ctx = SimulationContext(
        symbol="TEST", asset_class="FOREX", spread=0.0003, point=0.0001,
        exit_strategy="_test_sl_level_exit", exit_params={"sl_level": "or_low"},
    )
    probs = np.zeros((n, 2))
    probs[0, 1] = 1.0
    out = simulate_trades(
        df, probs, None, 1, None, 0.5, 0.5, tp=0.01, sl=1.5, ctx=ctx,
        return_detailed=True, timeout_bars=1,
    )
    [trade] = out["trades_detailed"]
    entry = trade["entry_price"]
    assert trade["sl_level"] == pytest.approx(entry - 1.5 * (entry - 1.2580))
