"""A bar whose TP/SL distances cannot be computed opens no trade.

The per-trade distances of xgboost_rrr / xgboost_mfe are multiples of the ATR,
which is NaN while it warms up.  The NaN levels used to reach the simulator, where
no comparison with them is ever true: the trade stayed open to the end of the data.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import simulate_trades
from fwbg.plugins import import_plugin_module

WARMUP = 4
N = 30


def _df():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.5, N))
    return pd.DataFrame(
        {"O": close, "H": close + 0.4, "L": close - 0.4, "C": close},
        index=pd.date_range("2024-01-01", periods=N, freq="h"),
    )


def _atr():
    atr = np.full(N, 0.5)
    atr[:WARMUP] = np.nan
    return atr


def _ctx():
    return SimulationContext(
        symbol="T",
        asset_class="FOREX",
        spread=0.01,
        point=0.0001,
        long_enabled=True,
        short_enabled=False,
        min_trades=1,
        max_trade_bars=3,
    )


def test_rrr_distances_are_nan_exactly_where_the_atr_is():
    mod = import_plugin_module("fwbg-core", "models", "xgboost_rrr").XGBoostRRRModel()
    mod._selected_rrr = np.full(N, 2.0)
    mod._base_sl_atr = 1.5
    ptp = mod.get_per_trade_params(_df(), atr=_atr())
    assert np.isnan(ptp[:WARMUP]).all() and np.isfinite(ptp[WARMUP:]).all()


def test_mfe_distances_are_nan_exactly_where_the_atr_is():
    mod = import_plugin_module("fwbg-core", "models", "xgboost_mfe").XGBoostMFEModel()
    mod._predicted_mfe = np.full(N, 2.0)
    mod._selected_sl_atr = np.full(N, 1.5)
    ptp = mod.get_per_trade_params(_df(), atr=_atr())
    assert np.isnan(ptp[:WARMUP]).all() and np.isfinite(ptp[WARMUP:]).all()


@pytest.mark.parametrize("dist_tp, dist_sl", [(1.0, 0.5), (0.8, 0.6)])
def test_no_trade_opens_where_the_distances_are_nan(dist_tp, dist_sl):
    df = _df()
    probs = np.zeros((N, 2))
    probs[:, 1] = 1.0  # a signal on every bar
    ptp = np.empty((N, 2))
    ptp[:, 0], ptp[:, 1] = dist_tp, dist_sl
    ptp[:WARMUP] = np.nan
    trades = simulate_trades(
        df, probs, None, 1, None, 0.5, 0.5, 1, 1, _ctx(),
        return_detailed=True, per_trade_params=ptp,
    )["trades_detailed"]  # fmt: skip
    assert trades, "the tradable bars still trade"
    assert min(t["signal_idx"] for t in trades) == WARMUP
    for t in trades:
        assert np.isfinite(t["tp_level"]) and np.isfinite(t["sl_level"])
        assert np.isfinite(t["pnl_raw"])


def test_a_nan_warmup_does_not_hold_a_trade_to_the_end_of_the_data():
    df = _df()
    probs = np.zeros((N, 2))
    probs[0, 1] = 1.0  # only the warm-up bar signals
    ptp = np.full((N, 2), np.nan)
    res = simulate_trades(
        df, probs, None, 1, None, 0.5, 0.5, 1, 1, _ctx(),
        return_detailed=True, per_trade_params=ptp,
    )  # fmt: skip
    assert res["trades"] == []
