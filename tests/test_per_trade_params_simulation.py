"""Test that _simulate_trades_core uses per_trade_params when provided."""
import numpy as np
import pandas as pd

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import _simulate_trades_core


def _make_df(n=50):
    np.random.seed(42)
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "O": base,
        "H": base + abs(np.random.randn(n)) * 0.3,
        "L": base - abs(np.random.randn(n)) * 0.3,
        "C": base + np.random.randn(n) * 0.1,
        "_atr": np.full(n, 0.5),
        "_regime": np.full(n, 7, dtype=np.int8),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


def _make_probs(n, p_win=0.8):
    return np.column_stack([np.full(n, 1 - p_win), np.full(n, p_win)])


def _make_ctx():
    return SimulationContext(
        symbol="TEST", asset_class="forex", spread=0.0001, point=0.0001,
        grid_ct=[0.5],
        long_enabled=True, short_enabled=True,
        max_trade_bars=100, separate_long_short=False,
    )


class TestPerTradeParams:

    def test_per_trade_params_overrides_tp_sl(self):
        df = _make_df(50)
        n = len(df)
        probs = _make_probs(n)
        ctx = _make_ctx()

        # per_trade_params: tiny SL -> all trades should lose
        ptp = np.zeros((n, 2), dtype=np.float64)
        ptp[:, 0] = 1000.0  # TP unreachable
        ptp[:, 1] = 0.001   # SL instant stop

        result = _simulate_trades_core(
            df, probs, probs, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=2, sl=1, ctx=ctx,
            per_trade_params=ptp,
        )
        trades = result["trades"]
        if trades:
            assert all(t["result"] == -1.0 for t in trades)

    def test_none_per_trade_params_unchanged(self):
        df = _make_df(50)
        n = len(df)
        probs = _make_probs(n)
        ctx = _make_ctx()

        result1 = _simulate_trades_core(
            df, probs, probs, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=2, sl=1, ctx=ctx,
            per_trade_params=None,
        )
        result2 = _simulate_trades_core(
            df, probs, probs, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=2, sl=1, ctx=ctx,
        )
        assert len(result1["trades"]) == len(result2["trades"])
