"""Test that the pipeline computes MFE targets for xgboost_mfe model type."""
import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.targets import compute_mfe_targets


class TestMFETargetIntegration:
    def test_mfe_targets_work_as_training_targets(self):
        """MFE targets can be passed to xgboost_mfe model.train()."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "O": 100.0 + np.random.randn(n) * 0.5,
                "H": 101.0 + abs(np.random.randn(n)) * 0.5,
                "L": 99.0 - abs(np.random.randn(n)) * 0.5,
                "C": 100.0 + np.random.randn(n) * 0.5,
                "_atr": np.full(n, 1.0),
                "feat_1": np.random.randn(n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="15min"),
        )

        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.5
        )

        # MFE targets should be continuous, not binary
        assert not np.all(np.isin(mfe_long, [0.0, 1.0]))

        from fwbg.core.registry import get_model
        from fwbg_sdk.models import TrainingContext

        model = get_model("xgboost_mfe")()
        model.train(
            df[["feat_1"]],
            mfe_long,
            TrainingContext(direction="long"),
            sl_variants=[2.0],
        )
        assert model.is_trained

        probs = model.predict_probability(df[["feat_1"]])
        assert probs.shape == (n, 2)
        assert np.all(probs[:, 1] >= 0.0)

    def test_binary_targets_still_work_for_xgboost(self):
        """Standard xgboost model still uses binary targets."""
        np.random.seed(42)
        n = 100
        features = pd.DataFrame(
            {
                "feat_1": np.random.randn(n),
            }
        )
        targets = (np.random.rand(n) > 0.4).astype(np.float64)

        from fwbg.core.registry import get_model
        from fwbg_sdk.models import TrainingContext

        model = get_model("xgboost")()
        model.train(features, targets, TrainingContext())
        assert model.is_trained
        probs = model.predict_probability(features)
        assert probs.shape == (n, 2)
