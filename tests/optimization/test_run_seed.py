"""The strategy's ``seed`` feeds every seeded component of a run.

Without a seed the built-in seed (42) applies exactly as before: a model's own
random_state and a selector's own seed parameter still win. A set seed is
authoritative for all of them.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import DEFAULT_RUN_SEED, StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization import nested_cv
from fwbg.optimization.causal_reporting import (
    bootstrap_terminal_equity,
    monte_carlo_summary,
)
from fwbg.simulation.trade import monte_carlo_permutation_test


def _ctx(seed=None, hyperparameters=None, plugins=None) -> SimulationContext:
    return SimulationContext(
        symbol="TEST",
        asset_class="FOREX",
        spread=0.0,
        point=0.0001,
        currencies=("EUR", "USD"),
        min_trades=4,
        model_hyperparameters=hyperparameters or {},
        feature_selection_plugins=plugins,
        seed=seed,
    )


class TestSeedConfig:
    def test_default_is_unset_and_means_the_builtin_seed(self):
        config = StrategyConfig.from_dict({"name": "x"})
        assert config.seed is None
        assert config.run_seed == DEFAULT_RUN_SEED == 42
        # An unset seed leaves the serialised config (and so its fingerprint) alone.
        assert "seed" not in config.to_dict()

    def test_seed_is_read_serialised_and_reaches_the_context(self):
        config = StrategyConfig.from_dict({"name": "x", "seed": 7})
        assert (config.seed, config.run_seed) == (7, 7)
        assert config.to_dict()["seed"] == 7
        assert _ctx(seed=config.seed).run_seed == 7

    def test_zero_is_a_valid_seed(self):
        assert StrategyConfig.from_dict({"seed": 0}).run_seed == 0

    @pytest.mark.parametrize("bad", [-1, 1.5, 7.0, "7", True, [1]])
    def test_invalid_seed_is_refused(self, bad):
        with pytest.raises(ValueError, match="seed must be an integer >= 0"):
            StrategyConfig.from_dict({"name": "x", "seed": bad})

    def test_dataclass_construction_is_validated_too(self):
        with pytest.raises(ValueError, match="seed must be an integer >= 0"):
            StrategyConfig(seed=-3)

    def test_context_create_carries_the_strategy_seed(self, monkeypatch):
        from fwbg.data.assets import get_asset

        asset = get_asset("EURUSD")
        unset = SimulationContext.create(asset, StrategyConfig.from_dict({}))
        seeded = SimulationContext.create(asset, StrategyConfig.from_dict({"seed": 5}))
        assert (unset.seed, unset.run_seed) == (None, 42)
        assert (seeded.seed, seeded.run_seed) == (5, 5)


class _RecordingSelector:
    seen: list = []

    def select_features(self, df, targets, max_features=None, seed=0, **params):
        type(self).seen.append(seed)
        return list(df.columns), {}


class TestFeatureSelectionSeed:
    @pytest.fixture(autouse=True)
    def _selector(self, monkeypatch):
        _RecordingSelector.seen = []
        monkeypatch.setattr(
            nested_cv, "get_feature_selector", lambda name: _RecordingSelector
        )

    def _select(self, plugins, ctx):
        df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)[::-1]})
        targets = np.array([0, 1] * 5)
        nested_cv.select_features_from_fold(
            df,
            targets,
            ["a", "b"],
            min_trades=2,
            feature_selection_plugins=plugins,
            **nested_cv.fold_selection_seed(ctx),
        )
        return _RecordingSelector.seen[-1]

    def test_default_keeps_the_builtin_seed(self):
        assert self._select([{"name": "s", "params": {}}], _ctx()) == 42

    def test_default_keeps_a_plugins_own_seed(self):
        plugins = [{"name": "s", "params": {"seed": 3}}]
        assert self._select(plugins, _ctx()) == 3

    def test_a_run_seed_changes_the_selector_seed_and_overrides_the_plugin(self):
        assert self._select([{"name": "s", "params": {}}], _ctx(seed=9)) == 9
        plugins = [{"name": "s", "params": {"seed": 3}}]
        assert self._select(plugins, _ctx(seed=9)) == 9


class _RecordingModel:
    params: dict = {}
    trained_classes = np.array([0, 1])

    def train(self, X, y, training_context, **params):
        type(self).params = params

    @classmethod
    def get_reduced_hyperparameters(cls, params):
        return dict(params)


class TestModelSeed:
    @pytest.fixture(autouse=True)
    def _model(self, monkeypatch):
        _RecordingModel.params = {}
        monkeypatch.setattr(nested_cv, "get_model", lambda name: _RecordingModel)

    def _train(self, ctx, **kwargs) -> dict:
        df = pd.DataFrame({"a": np.arange(20.0)})
        targets = np.array([0, 1] * 10)
        model = nested_cv.train_model(df, targets, ["a"], 4, ctx, **kwargs)
        assert model is not None
        return _RecordingModel.params

    def test_default_hands_the_model_its_hyperparameters_untouched(self):
        # The model plugins fall back to random_state=42 themselves.
        assert self._train(_ctx(hyperparameters={"max_depth": 3})) == {"max_depth": 3}
        assert self._train(_ctx(hyperparameters={"random_state": 11})) == {
            "random_state": 11
        }

    def test_run_seed_becomes_the_random_state_and_overrides_the_config(self):
        ctx = _ctx(seed=7, hyperparameters={"random_state": 42, "max_depth": 3})
        assert self._train(ctx) == {"random_state": 7, "max_depth": 3}
        assert self._train(ctx, use_reduced_params=True)["random_state"] == 7

    def test_seed_zero_is_applied(self):
        assert self._train(_ctx(seed=0))["random_state"] == 0


def test_default_seed_reproduces_the_unseeded_xgboost_model_and_another_seed_differs():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    targets = (df["a"] + rng.normal(scale=2.0, size=400) > 0).astype(int).to_numpy()
    hp = {"n_estimators": 20, "max_depth": 3, "subsample": 0.5, "colsample_bytree": 0.5}

    def probabilities(ctx):
        model = nested_cv.train_model(df, targets, list(df.columns), 10, ctx)
        return model.predict_probability(df[list(df.columns)])

    legacy = probabilities(_ctx(hyperparameters={**hp, "random_state": 42}))
    unset = probabilities(_ctx(hyperparameters=hp))
    explicit_default = probabilities(_ctx(seed=42, hyperparameters=hp))
    other = probabilities(_ctx(seed=1, hyperparameters=hp))
    np.testing.assert_array_equal(unset, legacy)
    np.testing.assert_array_equal(explicit_default, legacy)
    assert not np.array_equal(other, legacy)


class TestResamplingSeed:
    PNL = [3.0, -2.0, 1.5, -1.0, 2.5, -2.2, 0.8, 1.1, -0.9, 2.0, -1.4, 1.7]
    RETURNS = [p / 100 for p in PNL]

    def test_default_reproduces_the_previous_fixed_seed(self):
        previous = {
            **{
                k: v
                for k, v in monte_carlo_permutation_test(
                    self.PNL, n_permutations=1000, random_seed=42
                ).items()
                if k in ("p_value", "is_significant", "percentile")
            },
        }
        bootstrap = bootstrap_terminal_equity(
            self.RETURNS, n_simulations=500, random_seed=42
        )
        summary = monte_carlo_summary(self.PNL, self.RETURNS)
        assert {k: summary[k] for k in previous} == previous
        assert summary["equity_median"] == bootstrap["median_equity"]
        assert summary["equity_p5"] == bootstrap["p5_equity"]
        assert summary == monte_carlo_summary(self.PNL, self.RETURNS, seed=42)

    def test_another_seed_changes_the_bootstrap(self):
        base = monte_carlo_summary(self.PNL, self.RETURNS)
        other = monte_carlo_summary(self.PNL, self.RETURNS, seed=1)
        assert other["equity_p5"] != base["equity_p5"]
        assert other == monte_carlo_summary(self.PNL, self.RETURNS, seed=1)
