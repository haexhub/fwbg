"""Small signal-model fixtures shared by the signal selection and PBO tests."""

import numpy as np
import pandas as pd

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization.robust_validation import WalkForwardFold

SIGNALS = ["_composed_signal_long", "_composed_signal_short"]


def signal_ctx(**overrides) -> SimulationContext:
    """A signal model with exits TP 2 and TP 6 (CT 0.5 and 0.6), two inner windows."""
    defaults = dict(
        symbol="TEST",
        asset_class="forex",
        spread=1.0,
        point=0.01,
        min_trades=2,
        model_type="signal",
        n_inner_folds=2,
        embargo_bars=0,
        exit_strategies=[
            ExitStrategyConfig(name="fixed", params={"tp_mult": 2, "sl_mult": 4}),
            ExitStrategyConfig(name="fixed", params={"tp_mult": 6, "sl_mult": 4}, ct=[0.5, 0.6]),
        ],
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def signal_frame(n_bars, start="2024-01-01"):
    """Flat hourly bars with a long signal every 24th bar."""
    idx = pd.date_range(start, periods=n_bars, freq="h")
    signal = np.zeros(n_bars)
    signal[5::24] = 1.0
    return pd.DataFrame(
        {"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0, SIGNALS[0]: signal, SIGNALS[1]: 0.0},
        index=idx,
    )


def one_fold(train_df, test_df, fold_id=0):
    """A walk-forward fold over *train_df* followed by *test_df*."""
    return WalkForwardFold(
        fold_id=fold_id,
        train_start=0,
        train_end=len(train_df),
        test_start=len(train_df),
        test_end=len(train_df) + len(test_df),
        train_df=train_df,
        test_df=test_df,
    )
