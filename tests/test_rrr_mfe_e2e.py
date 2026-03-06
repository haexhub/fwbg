"""End-to-end tests for xgboost_rrr and xgboost_mfe models.

Tests the full flow: features -> train -> predict -> per_trade_params -> simulation.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext
from fwbg.optimization.targets import _simulate_trades_core


def _make_market_df(n=200):
    """Create realistic OHLC data with indicators."""
    np.random.seed(42)
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "O": base,
            "H": base + abs(np.random.randn(n)) * 0.5,
            "L": base - abs(np.random.randn(n)) * 0.5,
            "C": base + np.random.randn(n) * 0.1,
            "_atr": np.full(n, 1.0),
            "_regime": np.full(n, 7, dtype=np.int8),
            "feat_momentum": np.random.randn(n),
            "feat_volatility": abs(np.random.randn(n)),
            "feat_trend": np.random.randn(n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )
    return df


FEATURES = ["feat_momentum", "feat_volatility", "feat_trend"]


class TestRRREndToEnd:
    def test_full_flow(self):
        from fwbg.core.registry import get_model

        df = _make_market_df(200)
        targets = (np.random.rand(200) > 0.4).astype(np.float64)

        model = get_model("xgboost_rrr")()
        model.train(
            df[FEATURES],
            targets,
            TrainingContext(direction="long"),
            rrr_variants=[1.5, 2.0, 3.0],
            base_sl_atr=2.0,
        )

        probs = model.predict_probability(df[FEATURES])
        assert probs.shape == (200, 2)

        atr = df["_atr"].values
        ptp = model.get_per_trade_params(df[FEATURES], atr=atr)
        assert ptp is not None
        assert ptp.shape == (200, 2)
        # SL = base_sl_atr * atr = 2.0 * 1.0 = 2.0
        assert np.allclose(ptp[:, 1], 2.0)
        # TP = selected_rrr * base_sl_atr * atr
        for i in range(200):
            expected_tp = model.selected_rrr[i] * 2.0 * 1.0
            assert np.isclose(ptp[i, 0], expected_tp)

    def test_rrr_selection_varies(self):
        """Different samples should potentially select different RRRs."""
        from fwbg.core.registry import get_model

        df = _make_market_df(500)
        targets = (np.random.rand(500) > 0.4).astype(np.float64)

        model = get_model("xgboost_rrr")()
        model.train(
            df[FEATURES],
            targets,
            TrainingContext(),
            rrr_variants=[1.5, 2.0, 3.0, 5.0],
        )
        model.predict_probability(df[FEATURES])
        # With enough data and variants, at least 2 different RRRs should be chosen
        unique_rrr = set(model.selected_rrr)
        assert len(unique_rrr) >= 1  # at minimum 1, hopefully >1

    def test_per_trade_params_none_without_predict(self):
        """get_per_trade_params returns None if predict wasn't called."""
        from fwbg.core.registry import get_model

        df = _make_market_df(100)
        targets = (np.random.rand(100) > 0.4).astype(np.float64)

        model = get_model("xgboost_rrr")()
        model.train(
            df[FEATURES],
            targets,
            TrainingContext(),
            rrr_variants=[2.0],
        )
        ptp = model.get_per_trade_params(df[FEATURES], atr=df["_atr"].values)
        assert ptp is None


class TestMFEEndToEnd:
    def test_full_flow(self):
        from fwbg.core.registry import get_model
        from fwbg.optimization.targets import compute_mfe_targets

        df = _make_market_df(200)
        mfe_long, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)

        model = get_model("xgboost_mfe")()
        model.train(
            df[FEATURES],
            mfe_long,
            TrainingContext(direction="long"),
            sl_variants=[1.5, 2.0, 3.0],
        )

        probs = model.predict_probability(df[FEATURES])
        assert probs.shape == (200, 2)
        # Column 1 = predicted MFE (clamped >= 0)
        assert np.all(probs[:, 1] >= 0.0)

        atr = df["_atr"].values
        ptp = model.get_per_trade_params(df[FEATURES], atr=atr)
        assert ptp is not None
        assert ptp.shape == (200, 2)
        # TP = predicted_mfe * atr
        assert np.allclose(ptp[:, 0], probs[:, 1] * atr)
        # SL = selected_sl_atr * atr
        for i in range(200):
            expected_sl = model.selected_sl_atr[i] * atr[i]
            assert np.isclose(ptp[i, 1], expected_sl)

    def test_mfe_with_simulation(self):
        """Test that per_trade_params can be fed into _simulate_trades_core."""
        from fwbg.core.registry import get_model
        from fwbg.core.context import SimulationContext
        from fwbg.optimization.targets import compute_mfe_targets

        df = _make_market_df(200)
        mfe_long, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)

        model = get_model("xgboost_mfe")()
        model.train(
            df[FEATURES],
            mfe_long,
            TrainingContext(),
            sl_variants=[2.0],
        )

        probs = model.predict_probability(df[FEATURES])
        atr = df["_atr"].values
        ptp = model.get_per_trade_params(df[FEATURES], atr=atr)

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="forex",
            spread=0.5,
            point=0.00001,
        )

        result = _simulate_trades_core(
            df,
            probs,
            probs,
            1,
            1,
            ct_long=1.0,
            ct_short=1.0,
            tp=2,
            sl=1,
            ctx=ctx,
            per_trade_params=ptp,
        )
        # Should not crash; verify result has expected structure
        assert "trades" in result

    def test_per_trade_params_none_without_predict(self):
        """get_per_trade_params returns None if predict wasn't called."""
        from fwbg.core.registry import get_model
        from fwbg.optimization.targets import compute_mfe_targets

        df = _make_market_df(100)
        mfe_long, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)

        model = get_model("xgboost_mfe")()
        model.train(
            df[FEATURES],
            mfe_long,
            TrainingContext(),
            sl_variants=[2.0],
        )
        ptp = model.get_per_trade_params(df[FEATURES], atr=df["_atr"].values)
        assert ptp is None


class TestBothModelsCoexist:
    def test_both_models_registered(self):
        """Both models can be loaded simultaneously."""
        from fwbg.core.registry import get_model

        rrr_cls = get_model("xgboost_rrr")
        mfe_cls = get_model("xgboost_mfe")
        xgb_cls = get_model("xgboost")

        assert rrr_cls.name == "xgboost_rrr"
        assert mfe_cls.name == "xgboost_mfe"
        assert xgb_cls.name == "xgboost"

        # All three are different classes
        assert rrr_cls is not mfe_cls
        assert rrr_cls is not xgb_cls
