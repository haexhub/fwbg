"""Feature-selection targets preserve exit-column and TP-source resolution."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fwbg.core.context import SimulationContext
from fwbg.optimization import grid_search, targets


def test_select_features_carries_named_range_configuration(monkeypatch):
    captured = {}

    def fake_compute_targets(df, tp, sl, ctx, timeout_bars, exit_strategy_mode):
        captured["exit_params"] = ctx.exit_params
        return np.ones(len(df)), np.ones(len(df))

    monkeypatch.setattr(targets, "compute_targets_cached", fake_compute_targets)
    monkeypatch.setattr(
        grid_search,
        "select_features_from_fold",
        lambda *args, **kwargs: (None, None),
    )

    exit_config = SimpleNamespace(
        name="orb_based",
        params={
            "tp_mult": 0.8,
            "sl_mult": 2.0,
            "tp_mode": "range",
            "sl_dist_column": "chosen_sl_dist",
            "range_column": "chosen_range",
        },
    )
    ctx = SimulationContext(
        symbol="TEST",
        asset_class="index",
        spread=1.0,
        point=0.1,
        exit_strategies=[exit_config],
        exit_strategy="fixed",
        exit_params={},
        min_trades=2,
        long_enabled=True,
        short_enabled=True,
        feature_selection_plugins=None,
        required_features=[],
    )
    frame = pd.DataFrame({"f1": [1.0], "f2": [2.0], "f3": [3.0]})

    grid_search.select_features([(frame, frame)], list(frame.columns), ctx, "TEST")

    assert captured["exit_params"] == {
        "sl_dist_column": "chosen_sl_dist",
        "tp_mode": "range",
        "range_column": "chosen_range",
    }
