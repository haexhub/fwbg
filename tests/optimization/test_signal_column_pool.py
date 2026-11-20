"""Unshifted signal columns stay out of the ML feature pool.

Indicators return model features shifted by one bar (row i uses bars <= i-1).
Columns declared by ``get_signal_columns`` are deliberately unshifted: row i is
the decision at bar i's close. They are entry / exit inputs, not model features.
The pool is built in ``_prepare_fold_common``; these tests go through that real
code path with core plugins on synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization.process_fold import prepare_fold_data
from fwbg.optimization.signal_fold import prepare_signal_fold_data
from fwbg.optimization.robust_validation import WalkForwardFold
from fwbg.pipeline.features import (
    add_shifted_signal_columns,
    compute_indicator_pool,
    configured_signal_columns,
    get_feature_columns,
    ml_feature_pool,
)

ORB = {"name": "opening_range", "params": {"sessions": [8], "range_bars": 1}}
IBS = {"name": "ibs", "params": {}}
ZSCORE = {"name": "zscore", "params": {"lookbacks": [20]}}
MACD = {"name": "macd", "params": {}}
SIGNAL_INDICATORS = [ORB, IBS, ZSCORE]
ALL_INDICATORS = [ORB, IBS, ZSCORE, MACD]

N_BARS = 24 * 70
SPLIT = 24 * 50


def make_ohlc(n: int = N_BARS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.2, n))
    index = pd.date_range("2021-01-04", periods=n, freq="h")
    return pd.DataFrame(
        {
            "O": open_,
            "H": high,
            "L": low,
            "C": close,
            "V": rng.integers(100, 1000, n).astype(float),
        },
        index=index,
    )


def make_fold(df: pd.DataFrame) -> WalkForwardFold:
    return WalkForwardFold(
        fold_id=0,
        train_start=0,
        train_end=SPLIT,
        test_start=SPLIT,
        test_end=len(df),
        train_df=df.iloc[:SPLIT].copy(),
        test_df=df.iloc[SPLIT:].copy(),
    )


def make_ctx(indicators, **overrides) -> SimulationContext:
    fields = dict(
        symbol="TESTUSD",
        asset_class="FOREX",
        spread=0.0002,
        point=0.0001,
        min_trades=5,
        long_enabled=True,
        short_enabled=True,
        indicator_plugins=indicators,
    )
    fields.update(overrides)
    return SimulationContext(**fields)


def prepare_pool(df: pd.DataFrame, indicators, **ctx_overrides):
    """(train_df, test_df, pool) from the real ML fold preparation."""
    ctx = make_ctx(indicators, **ctx_overrides)
    prepared = prepare_fold_data(make_fold(df), indicators, None, [], ctx, "TEST")
    assert prepared is not None
    return prepared


@pytest.fixture(scope="module")
def ohlc() -> pd.DataFrame:
    return make_ohlc()


@pytest.fixture(scope="module")
def declared() -> set:
    """Columns the configured signal indicators declare (default settings)."""
    return set(configured_signal_columns(SIGNAL_INDICATORS))


class TestPoolExcludesSignalColumns:
    def test_declared_columns_are_real_plugin_columns(self, declared):
        assert {"ibs", "ibs_down", "ibs_above_prev_high", "zscore_20"} <= declared
        orb = {c for c in declared if "orb_s08" in c}
        assert any(c.endswith("_breakout_up") for c in orb)
        assert any(c.endswith("_retest_bull") for c in orb)

    def test_ml_pool_drops_signal_columns_and_keeps_shifted_features(
        self, ohlc, declared
    ):
        train_df, _, pool = prepare_pool(ohlc, ALL_INDICATORS)
        assert pool, "the shifted MACD features must stay"
        assert not declared & set(pool)
        assert any(c.startswith("macd_") for c in pool)

    def test_signal_columns_stay_in_the_frames_for_rules_and_exits(self, ohlc):
        train_df, test_df, _ = prepare_pool(ohlc, ALL_INDICATORS)
        for frame in (train_df, test_df):
            for col in ("ibs", "ibs_down", "zscore_20"):
                assert col in frame.columns

    def test_old_pool_would_have_contained_them(self, ohlc, declared):
        """get_feature_columns (the pre-fix pool) takes every non-underscore column."""
        frame = compute_indicator_pool(ohlc, indicators=SIGNAL_INDICATORS)
        assert {"ibs", "ibs_down", "zscore_20"} <= set(get_feature_columns(frame))
        assert not declared & set(ml_feature_pool(frame, SIGNAL_INDICATORS))

    def test_calendar_columns_stay_in_the_pool(self, ohlc):
        indicators = [{"name": "time_season", "params": {}}, MACD]
        train_df, _, pool = prepare_pool(ohlc, indicators)
        assert "time_session_ny" in train_df.columns
        assert "time_session_ny" in pool
        assert configured_signal_columns(indicators) == []

    def test_model_with_signal_columns_only_has_no_pool(self, ohlc):
        """A strategy whose only columns are signals has nothing for a model."""
        ctx = make_ctx([IBS])
        prepared = prepare_fold_data(make_fold(ohlc), [IBS], None, [], ctx, "TEST")
        assert prepared is None

    def test_required_feature_naming_a_signal_column_is_rejected(self, ohlc):
        with pytest.raises(ValueError, match="unshifted signal column"):
            prepare_pool(ohlc, ALL_INDICATORS, required_features=["ibs"])

    def test_signal_model_keeps_its_required_signal_columns(self, ohlc):
        ctx = make_ctx(
            ALL_INDICATORS, model_type="signal", required_features=["ibs_down"]
        )
        prepared = prepare_signal_fold_data(
            make_fold(ohlc), ALL_INDICATORS, None, [], ctx, "TEST"
        )
        assert prepared is not None
        _, _, pool = prepared
        assert "ibs_down" in pool


class TestOptIn:
    def indicators(self, flag: bool):
        return [{**IBS, "signal_columns_as_features": flag}, ZSCORE, MACD]

    def test_opt_in_adds_only_shifted_copies(self, ohlc):
        indicators = self.indicators(True)
        train_df, _, pool = prepare_pool(ohlc, indicators)
        assert {"ibs_prev", "ibs_down_prev", "ibs_above_prev_high_prev"} <= set(pool)
        for original in ("ibs", "ibs_down", "ibs_above_prev_high"):
            assert original not in pool
            # the fold frame lost its warm-up rows, so its first row has no predecessor
            pd.testing.assert_series_equal(
                train_df[f"{original}_prev"].iloc[1:],
                train_df[original].shift(1).iloc[1:],
                check_names=False,
            )
        # zscore did not opt in: no copies, and its unshifted columns stay out
        assert not any(c.startswith("zscore") for c in pool)

    def test_default_adds_no_copies(self, ohlc):
        _, _, pool = prepare_pool(ohlc, self.indicators(False))
        assert not any(c.endswith("_prev") for c in pool)

    def test_copy_equals_shift_of_the_computed_column(self, ohlc):
        frame = compute_indicator_pool(ohlc, indicators=self.indicators(True))
        for col in ("ibs", "ibs_down", "ibs_above_prev_high"):
            pd.testing.assert_series_equal(
                frame[f"{col}_prev"], frame[col].shift(1), check_names=False
            )
            assert pd.isna(frame[f"{col}_prev"].iloc[0])

    def test_name_collision_is_an_error(self, ohlc):
        frame = compute_indicator_pool(ohlc, indicators=[IBS])
        frame["ibs_prev"] = 0.0
        with pytest.raises(ValueError, match="already in use"):
            add_shifted_signal_columns(
                frame, [{**IBS, "signal_columns_as_features": True}]
            )

    def test_calendar_columns_get_no_copy(self, ohlc):
        indicator = {"name": "time_season", "signal_columns_as_features": True}
        frame = compute_indicator_pool(ohlc, indicators=[indicator])
        assert not any(c.endswith("_prev") for c in frame.columns)

    def test_non_boolean_flag_is_rejected(self, ohlc):
        for bad in ("true", "false", 1, None):
            with pytest.raises(ValueError, match="signal_columns_as_features"):
                compute_indicator_pool(
                    ohlc, indicators=[{**IBS, "signal_columns_as_features": bad}]
                )


class TestStrategyConfigBoundary:
    def strategy(self, flag):
        return {
            "name": "t",
            "pipeline": {
                "indicators": [
                    {"name": "ibs", "params": {}, "signal_columns_as_features": flag}
                ]
            },
        }

    def test_boolean_flag_is_accepted(self):
        for flag in (True, False):
            cfg = StrategyConfig.from_dict(self.strategy(flag))
            assert cfg.get_indicators()[0]["signal_columns_as_features"] is flag

    @pytest.mark.parametrize("flag", ["true", "yes", 1, 0, None, []])
    def test_non_boolean_flag_fails_at_load(self, flag):
        with pytest.raises(
            ValueError, match="signal_columns_as_features must be true or false"
        ):
            StrategyConfig.from_dict(self.strategy(flag))

    def test_flag_is_optional(self):
        cfg = StrategyConfig.from_dict(
            {"name": "t", "pipeline": {"indicators": [{"name": "ibs"}]}}
        )
        assert cfg.get_indicators() == [{"name": "ibs"}]


class TestModelFeaturesNeverDependOnTheOwnBar:
    """Row i of every model feature may use bars <= i-1: perturb bar i, row i is unchanged.

    The pool comes from the real fold preparation. A dense window of rows is
    perturbed one at a time (not a single cut).
    """

    WINDOW = range(SPLIT - 28, SPLIT - 12)

    @staticmethod
    def perturbed(df: pd.DataFrame, row: int) -> pd.DataFrame:
        out = df.copy()
        out.iloc[row, out.columns.get_loc("C")] += 7.0
        out.iloc[row, out.columns.get_loc("H")] += 8.0
        out.iloc[row, out.columns.get_loc("L")] -= 2.0
        out.iloc[row, out.columns.get_loc("O")] += 1.0
        return out

    @staticmethod
    def changed_pool_columns(base, other, pool, label) -> set:
        changed = set()
        for col in pool:
            a, b = base.loc[label, col], other.loc[label, col]
            if not (pd.isna(a) and pd.isna(b)) and a != b:
                changed.add(col)
        return changed

    @pytest.mark.parametrize("opt_in", [False, True])
    def test_pool_columns_ignore_the_own_bar(self, ohlc, opt_in):
        indicators = [
            {**ib, "signal_columns_as_features": opt_in} for ib in ALL_INDICATORS
        ]
        base_train, _, pool = prepare_pool(ohlc, indicators)
        assert pool
        leaks: dict = {}
        for row in self.WINDOW:
            label = ohlc.index[row]
            other_train, _, other_pool = prepare_pool(
                self.perturbed(ohlc, row), indicators
            )
            assert other_pool == pool
            for col in self.changed_pool_columns(base_train, other_train, pool, label):
                leaks.setdefault(col, []).append(row)
        assert not leaks, f"model features read their own bar: {sorted(leaks)}"

    def test_detector_flags_the_pre_fix_pool(self, ohlc):
        """The same probe on the old pool (every non-underscore column) finds the leak."""
        indicators = SIGNAL_INDICATORS
        base = compute_indicator_pool(ohlc, indicators=indicators)
        old_pool = get_feature_columns(base)
        leaking = set()
        for row in self.WINDOW:
            other = compute_indicator_pool(
                self.perturbed(ohlc, row), indicators=indicators
            )
            leaking |= self.changed_pool_columns(base, other, old_pool, ohlc.index[row])
        assert {"ibs", "zscore_20"} <= leaking
        assert not leaking & set(ml_feature_pool(base, indicators))
