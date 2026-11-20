"""Training skips rows whose label window is still open at the end of the data.

The label kernels mark such rows NaN (``nan_incomplete``) instead of calling the
last bars losses.  Fitting, feature selection, the out-of-fold predictions of the
meta model and calibration must not see those rows, and a fold whose known labels
hold a single class trains no model instead of aborting the run.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from fwbg.core import GridParams
from fwbg.core.context import SimulationContext
from fwbg.optimization import nested_cv
from fwbg.optimization.nested_cv import (
    _generate_oof_predictions,
    _train_meta_model,
    select_features_from_fold,
    train_model,
)
from fwbg.optimization.targets import (
    _validate_targets,
    compute_targets_cached,
    count_positive_labels,
    labelled_rows,
    simulate_trades,
)
from fwbg.plugins import import_plugin_module
from fwbg_sdk.models import TrainingContext, UntrainableLabelsError

NAN = np.nan


class _RecordingModel:
    """Stands in for a model plugin and records what it was fitted on."""

    calls: list = []
    calibrations: list = []
    classes_ = np.array([0, 1])
    trained_classes = np.array([0, 1])

    @classmethod
    def get_reduced_hyperparameters(cls, params):
        return dict(params)

    def train(self, features, targets, training_context, **params):
        self.calls.append(
            (features.copy(), np.asarray(targets).copy(), training_context)
        )

    def calibrate(self, features, targets, method="isotonic"):
        self.calibrations.append((features.copy(), np.asarray(targets).copy()))

    def predict_probability(self, features):
        return np.full((len(features), 2), 0.5)


@pytest.fixture
def recorder(monkeypatch):
    _RecordingModel.calls = []
    _RecordingModel.calibrations = []
    monkeypatch.setattr(nested_cv, "get_model", lambda name: _RecordingModel)
    return _RecordingModel


def _ctx(**kw):
    defaults = {
        "symbol": "T",
        "asset_class": "FOREX",
        "spread": 0.1,
        "point": 0.0001,
        "min_trades": 2,
        "exit_strategy": "fixed",
        "model_type": "xgboost",
    }
    defaults.update(kw)
    return SimulationContext(**defaults)


def _train_frame(n=12):
    return pd.DataFrame(
        {"f1": np.arange(n, dtype=float), "f2": np.arange(n, dtype=float)[::-1]}
    )


# Two wins, four known losses, two unknown rows at the end.
TARGETS = np.array([1, 0, 0, 1, 0, 0, 0, 0, NAN, NAN, NAN, NAN])


class TestLabelCounting:
    def test_nan_is_not_a_known_label_or_a_win(self):
        assert labelled_rows(TARGETS).tolist() == [True] * 8 + [False] * 4
        assert count_positive_labels(TARGETS) == 2

    def test_open_rows_do_not_make_a_direction_tradeable(self):
        ctx = _ctx(min_trades=6)  # needs 3 wins per direction
        only_nan = np.array([0, 0, NAN, NAN, NAN, NAN])
        assert _validate_targets(only_nan, only_nan, ctx) == (False, False)


class TestTrainModel:
    def test_fit_sees_only_rows_with_a_known_label(self, recorder):
        weights = np.arange(12, dtype=float)
        model = train_model(
            _train_frame(), TARGETS, ["f1", "f2"], 2, _ctx(), sample_weight=weights
        )
        assert model is not None
        features, targets, context = recorder.calls[0]
        assert len(features) == 8 and len(targets) == 8
        assert not np.isnan(targets).any()
        assert features["f1"].tolist() == list(range(8))
        assert context.sample_weights.tolist() == list(range(8))

    def test_calibration_skips_them_too(self, recorder):
        ctx = _ctx(probability_calibration=True)
        train_model(_train_frame(), TARGETS, ["f1", "f2"], 2, ctx)
        features, targets = recorder.calibrations[0]
        assert len(features) == len(targets) == 8
        assert not np.isnan(targets).any()

    def test_a_fully_known_fold_is_fitted_unchanged(self, recorder):
        targets = np.array([1, 0, 1, 0, 0, 0], dtype=float)
        train_model(_train_frame(6), targets, ["f1"], 2, _ctx())
        features, fitted, _ = recorder.calls[0]
        assert len(features) == 6 and fitted.tolist() == targets.tolist()

    def test_open_rows_do_not_count_towards_min_trades(self, recorder):
        targets = np.array([1, 0, 0, 0, NAN, NAN, NAN, NAN], dtype=float)
        assert train_model(_train_frame(8), targets, ["f1"], 6, _ctx()) is None
        assert recorder.calls == []


class TestFeatureSelection:
    def test_selector_gets_the_labelled_rows_only(self, monkeypatch):
        seen = {}

        class Selector:
            def select_features(self, features, targets, max_features=None, **kw):
                seen["rows"], seen["targets"] = len(features), np.asarray(targets)
                return list(features.columns), {}

        monkeypatch.setattr(nested_cv, "get_feature_selector", lambda name: Selector)
        selected, _ = select_features_from_fold(
            _train_frame(),
            TARGETS,
            ["f1", "f2"],
            2,
            feature_selection_plugins=[{"name": "any"}],
        )
        assert selected == ["f1", "f2"]
        assert seen["rows"] == 8 and not np.isnan(seen["targets"]).any()


class TestMetaModel:
    def test_oof_fits_skip_open_rows(self, recorder):
        n = 40
        targets = np.tile([1.0, 0.0, 0.0, 1.0], n // 4)
        targets[-8:] = NAN
        df = pd.DataFrame({"f1": np.arange(n, dtype=float)})
        ctx = _ctx(model_hyperparameters={})
        oof = _generate_oof_predictions(df, targets, ["f1"], ctx, label_horizon=1)
        assert recorder.calls
        for _, fitted, _ in recorder.calls:
            assert not np.isnan(fitted).any()
        assert np.isfinite(oof).any()

    def test_meta_model_trains_on_known_labels_only(self, recorder):
        n = 40
        targets = np.tile([1.0, 0.0, 0.0, 1.0], n // 4)
        targets[-8:] = NAN
        df = pd.DataFrame({"f1": np.arange(n, dtype=float)})
        oof = np.full(n, 0.5)
        _train_meta_model(df, targets, ["f1"], oof, _ctx(model_hyperparameters={}))
        features, fitted, _ = recorder.calls[0]
        assert len(features) == len(fitted) == n - 8
        assert not np.isnan(fitted).any()


# --- the label side -----------------------------------------------------------


def _frame(rows):
    a = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"O": a[:, 0], "H": a[:, 1], "L": a[:, 2], "C": a[:, 3]},
        index=pd.date_range("2024-01-01", periods=len(a), freq="h"),
    )


FLAT = (100.0, 100.5, 99.5, 100.0)


def _fixed_ctx(**kw):
    # spread 0.1: tp=sl=30 -> 3.0 points, never touched by FLAT bars
    return _ctx(max_trade_bars=3, **kw)


class TestOpenWindowsAreNanInTheCoreStrategies:
    def test_fixed_marks_the_cut_windows_unknown(self):
        long_, short = compute_targets_cached(
            _frame([FLAT] * 6), 30, 30, _fixed_ctx(), exit_strategy_mode="fixed"
        )
        assert long_[:3].tolist() == [0.0, 0.0, 0.0]
        assert np.isnan(long_[3:]).all() and np.isnan(short[3:]).all()

    def test_a_settled_outcome_near_the_end_stays_a_label(self):
        rows = [FLAT] * 5 + [(100.0, 104.0, 99.9, 103.0)]  # TP on the last bar
        long_, _ = compute_targets_cached(
            _frame(rows), 30, 30, _fixed_ctx(), exit_strategy_mode="fixed"
        )
        assert long_[4] == 1.0

    def test_structural_rr_marks_them_too(self):
        structural = import_plugin_module(
            "fwbg-core", "exit_strategies", "structural_rr"
        )
        df = _frame([FLAT] * 6)
        df["tpm_sl_dist_long"] = 3.0
        long_, short = structural.StructuralRRExitStrategy().compute_targets(
            df,
            _fixed_ctx(exit_params={"min_tp_pips": 1, "min_sl_pips": 1}),
            params=GridParams(tp_value=1.0, sl_value=1.0),
        )
        assert long_[:3].tolist() == [0.0, 0.0, 0.0]
        assert np.isnan(long_[3:]).all() and np.isnan(short[3:]).all()

    def test_without_a_bar_limit_unresolved_trades_at_the_end_are_unknown(self):
        ctx = _ctx()  # max_trade_bars None: the window is the rest of the data
        long_, _ = compute_targets_cached(
            _frame([FLAT] * 4), 30, 30, ctx, exit_strategy_mode="fixed"
        )
        assert np.isnan(long_).all()


class TestEvaluationIsUnchanged:
    def test_trades_on_the_last_bars_are_still_simulated(self):
        """Only labels turn NaN: a signal on the open tail still trades and settles."""
        df = _frame([FLAT] * 6)
        probs = np.zeros((6, 2))
        probs[4, 1] = 1.0  # signal on the second to last bar
        ctx = _fixed_ctx(long_enabled=True, short_enabled=False)
        res = simulate_trades(
            df, probs, None, 1, None, 0.5, 0.5, 30, 30, ctx, return_detailed=True
        )
        assert len(res["trades"]) == 1
        trade = res["trades_detailed"][0]
        assert trade["entry_idx"] == 5 and trade["exit_idx"] == 5
        assert trade["exit_reason"] == "end_of_data"


# --- xgboost_rrr -------------------------------------------------------------


_rrr = import_plugin_module("fwbg-core", "models", "xgboost_rrr")


def _atr_frame(rows):
    df = _frame(rows)
    df["_atr"] = 1.0
    return df


class TestRrrEndWindows:
    def test_cut_windows_and_the_last_bar_are_unknown(self):
        df = _atr_frame([FLAT] * 6)  # SL 2 / TP 4 are never touched
        t = _rrr._compute_binary_targets(
            df, rrr=2.0, base_sl_atr=2.0, direction="long", max_bars=3
        )
        assert t[:3].tolist() == [0.0, 0.0, 0.0]
        assert np.isnan(t[3:]).all()

    def test_a_hit_inside_a_cut_window_is_a_label(self):
        rows = [FLAT] * 5 + [(100.0, 105.0, 99.9, 104.0)]
        t = _rrr._compute_binary_targets(
            _atr_frame(rows), rrr=2.0, base_sl_atr=2.0, direction="long", max_bars=3
        )
        assert t[4] == 1.0

    def test_stacked_training_drops_the_unknown_rows(self):
        n = 40
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1.5, n))
        df = pd.DataFrame(
            {
                "O": close,
                "H": close + 0.3,
                "L": close - 0.3,
                "C": close,
                "_atr": 1.0,
                "f1": rng.normal(size=n),
            }
        )
        model = _rrr.XGBoostRRRModel()
        model._variants = [2.0, 3.0]
        model._base_sl_atr = 1.0
        X, y, w = model._build_stacked_dataset(
            df[["f1"]],
            np.zeros(n),
            TrainingContext(sample_weights=np.arange(n, dtype=float)),
            df,
            "long",
        )
        assert len(X) == len(y) == len(w) < 2 * n
        assert not np.isnan(y).any()
        assert set(np.unique(y)) <= {0, 1}


# --- single-class folds ------------------------------------------------------


class TestSingleClassFold:
    def test_xgboost_refuses_a_constant_label(self):
        xgboost = import_plugin_module("fwbg-core", "models", "xgboost")
        with pytest.raises(UntrainableLabelsError):
            xgboost.XGBoostModel().train(
                _train_frame(8), np.zeros(8), TrainingContext()
            )

    def test_train_model_yields_no_model_and_warns(self, caplog):
        ctx = _ctx(min_trades=0)  # no positive-label minimum: only the class check
        with caplog.at_level(logging.WARNING, logger=nested_cv.__name__):
            model = train_model(_train_frame(8), np.zeros(8), ["f1", "f2"], 0, ctx)
        assert model is None
        assert any("No model for this fold" in r.message for r in caplog.records)

    def test_labels_that_are_single_class_after_dropping_nan(self):
        targets = np.array([0, 0, 0, 0, 0, 0, NAN, NAN], dtype=float)
        ctx = _ctx(min_trades=0)
        assert train_model(_train_frame(8), targets, ["f1"], 0, ctx) is None

    def test_stacked_classifier_with_constant_labels_yields_no_model(self):
        n = 30
        close = np.full(n, 100.0)  # nothing ever reaches TP or SL
        df = pd.DataFrame(
            {
                "O": close,
                "H": close + 0.1,
                "L": close - 0.1,
                "C": close,
                "_atr": 1.0,
                "f1": np.arange(n, dtype=float),
            }
        )
        ctx = _ctx(min_trades=0, model_type="xgboost_rrr", max_trade_bars=10)
        assert train_model(df, np.zeros(n), ["f1"], 0, ctx) is None

    def test_the_signal_model_still_trains_without_label_classes(self):
        ctx = _ctx(min_trades=0, model_type="signal")
        model = train_model(
            _train_frame(8), np.zeros(8), ["f1"], 0, ctx, direction="long"
        )
        assert model is not None


class TestGridSearchWithoutAWinner:
    """A fold where no trade wins is a data outcome, not a GridSearchError."""

    def test_all_zero_labels_run_through_grid_search_with_no_trades(self):
        from fwbg.core.config import ExitStrategyConfig
        from fwbg.optimization.grid_search import run_grid_search

        n = 1500
        rng = np.random.default_rng(1)
        close = 15000.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
        df = pd.DataFrame(
            {
                "O": close,
                "H": close * 1.001,
                "L": close * 0.999,
                "C": close,
                "V": 1000.0,
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
                "f3": rng.normal(size=n),
            },
            index=pd.date_range("2024-01-02", periods=n, freq="15min"),
        )
        # TP of 1000 spreads is never reached: every label is 0.
        strategies = [
            ExitStrategyConfig(
                name="fixed", params={"tp_mult": 1000.0, "sl_mult": 3.0}, ct=[0.5]
            )
        ]
        ctx = SimulationContext(
            symbol="TEST",
            asset_class="INDEX",
            spread=15.0,
            point=1.0,
            exit_strategies=strategies,
            exit_strategy="fixed",
            exit_params={},
            model_type="xgboost",
            min_trades=1,
            early_termination=False,
            first_fold_sanity_check=False,
            min_fold_stability=0.0,
        )
        ctx.n_inner_folds = 2
        folds = [
            (df.iloc[:500].copy(), df.iloc[500:800].copy()),
            (df.iloc[300:800].copy(), df.iloc[800:1100].copy()),
        ]
        candidates, grid_results = run_grid_search(
            full_pool=["f1", "f2", "f3"],
            inner_folds=folds,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=df,
        )
        assert candidates == []
