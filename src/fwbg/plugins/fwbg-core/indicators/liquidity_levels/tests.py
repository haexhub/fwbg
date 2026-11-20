"""Tests for liquidity_levels indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_liq = import_plugin_module("fwbg-core", "indicators", "liquidity_levels")
if _liq is None:
    pytest.skip("liquidity_levels plugin not available", allow_module_level=True)


def _make_ohlc(n=500, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "O": close * (1 + rng.normal(0, 0.001, n)),
            "H": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "L": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "C": close,
        },
        index=idx,
    )
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ranging_df(n=500):
    """Create range-bound data that should produce equal highs/lows."""
    rng = np.random.default_rng(55)
    oscillation = 5 * np.sin(np.linspace(0, 20 * np.pi, n))
    prices = 105 + oscillation + rng.normal(0, 0.2, n)
    df = pd.DataFrame(
        {
            "O": prices + rng.normal(0, 0.1, n),
            "H": prices + np.abs(rng.normal(0, 0.3, n)) + 0.3,
            "L": prices - np.abs(rng.normal(0, 0.3, n)) - 0.3,
            "C": prices,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


def _get_indicator():
    return _liq.LiquidityLevelsIndicator()


class TestLiquidityFeatures:
    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values_in_ranging(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        late = result.iloc[100:]
        for col in ["liq_eqh_active", "liq_eql_active"]:
            non_null = late[col].dropna()
            assert non_null.sum() > 0, f"{col} never activated in ranging data"

    def test_binary_features(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in [
            "liq_eqh_active",
            "liq_eql_active",
            "liq_sweep_up",
            "liq_sweep_down",
        ]:
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_distances_positive_when_active(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        for col in ["liq_eqh_dist", "liq_eql_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals > 0).all(), f"{col} should be positive"

    def test_counts_at_least_min_touches(self):
        ind = _get_indicator()
        result = ind.compute(_make_ranging_df())
        for col in ["liq_eqh_count", "liq_eql_count"]:
            vals = result[col].dropna()
            active_vals = vals[vals > 0]
            if len(active_vals) > 0:
                assert (active_vals >= 2).all(), f"{col} should be >= min_touches"


class TestLiquidityShiftAndInf:
    def test_shift_applied(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted"

    def test_no_inf_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=1000))
        for col in ind.get_feature_columns():
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} has inf"

    def test_no_undeclared_features(self):
        ind = _get_indicator()
        df = _make_ohlc()
        original = set(df.columns)
        result = ind.compute(df)
        undeclared = set(result.columns) - original - set(ind.get_feature_columns())
        assert not undeclared, f"Undeclared: {undeclared}"

    def test_feature_count(self):
        ind = _get_indicator()
        assert len(ind.get_feature_columns()) == 8


class TestLiquidityParameters:
    def test_get_default_params(self):
        params = _liq.LiquidityLevelsIndicator.get_default_params()
        assert params["swing_lookback"] == 10
        assert params["min_touches"] == 2

    def test_get_param_schema(self):
        schema = _liq.LiquidityLevelsIndicator.get_param_schema()
        assert "swing_lookback" in schema
        assert "tolerance_atr_mult" in schema

    def test_custom_params(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(), swing_lookback=5, min_touches=3)
        assert "liq_eqh_dist" in result.columns


class TestLiquidityHelpers:
    def test_find_equal_levels_basic(self):
        prices = np.array([100.0, 100.1, 100.05, 105.0, 105.1])
        indices = np.array([0, 10, 20, 30, 40])
        clusters = _liq._find_equal_levels(
            prices, indices, tolerance=0.2, min_touches=2
        )
        assert len(clusters) >= 2

    def test_find_equal_levels_no_clusters(self):
        prices = np.array([100.0, 110.0, 120.0])
        indices = np.array([0, 10, 20])
        clusters = _liq._find_equal_levels(
            prices, indices, tolerance=0.5, min_touches=2
        )
        assert len(clusters) == 0

    def test_find_equal_levels_empty(self):
        clusters = _liq._find_equal_levels(
            np.array([]), np.array([]), tolerance=0.5, min_touches=2
        )
        assert len(clusters) == 0


class TestLiquidityDiscovery:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator

        discover_plugins()
        cls = get_indicator("liquidity_levels")
        assert cls is not None


# ---------------------------------------------------------------------------
# Causality, level lifecycle, tied pivots
# ---------------------------------------------------------------------------

_K = 3
_PARAMS = {
    "swing_lookback": _K,
    "tolerance_atr_mult": 0.3,
    "atr_period": 3,
    "min_touches": 2,
    "lookback_window": 100,
}


def _flat_df(n=60, highs=None, closes=None):
    """Flat market (C=100, H=100.5, L=99.5, TR=ATR=1) with chosen spikes.

    ``highs`` maps bar -> high, ``closes`` maps bar -> close.  Lows stay at 99.5
    for every bar so they tie and never form a swing.
    """
    c = np.full(n, 100.0)
    h = np.full(n, 100.5)
    for bar, val in (closes or {}).items():
        c[bar] = val
    for bar, val in (highs or {}).items():
        h[bar] = val
    h = np.maximum(h, c + 0.5)
    low = np.full(n, 99.5)
    low = np.minimum(low, c - 0.5)
    return pd.DataFrame(
        {"O": c, "H": h, "L": low, "C": c},
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def _compute(df, **over):
    return _get_indicator().compute(df, **{**_PARAMS, **over})


class TestCausality:
    def test_swing_only_used_after_confirmation(self):
        """Equal highs at bars 10 and 20 become a level at bar 20 + k, not before."""
        df = _flat_df(highs={10: 103.0, 20: 103.1})
        res = _compute(df)
        # Swing bar 20 is confirmed at raw row 20 + k = 23; the 1-bar shift
        # makes it visible at row 24.
        assert res["liq_eqh_active"].iloc[: 20 + _K + 1].fillna(0).sum() == 0.0
        assert res["liq_eqh_active"].iloc[20 + _K + 1] == 1.0
        assert res["liq_eqh_count"].iloc[20 + _K + 1] == 2.0
        # level = mean(103.0, 103.1) = 103.05, close 100, ATR 1
        assert res["liq_eqh_dist"].iloc[20 + _K + 1] == pytest.approx(3.05)

    def test_warmup_rows_are_nan(self):
        res = _compute(_flat_df())
        first_valid_raw = max(2 * _K, _PARAMS["atr_period"] - 1)
        for col in _get_indicator().get_feature_columns():
            assert res[col].iloc[: first_valid_raw + 1].isna().all(), col
        # counts / flags are real zeros once the warm-up is over
        assert res["liq_eqh_count"].iloc[first_valid_raw + 1] == 0.0
        assert res["liq_sweep_down"].iloc[first_valid_raw + 1] == 0.0

    @pytest.mark.parametrize(
        "make,params",
        [
            (lambda: _make_ranging_df(n=260), {}),
            (lambda: _make_ranging_df(n=260), {"swing_lookback": 5, "min_touches": 3}),
            (lambda: _make_ohlc(n=260, seed=3), {"tolerance_atr_mult": 0.6}),
        ],
    )
    def test_truncation_invariance_every_cut(self, make, params):
        """Rows < cut must not depend on bars >= cut (dense cut points)."""
        df = make()
        ind = _get_indicator()
        kw = {
            "swing_lookback": 4,
            "atr_period": 5,
            "tolerance_atr_mult": 0.4,
            "lookback_window": 80,
            **params,
        }
        full = ind.compute(df, **kw)
        cols = ind.get_feature_columns()
        for cut in range(150, 211):
            part = ind.compute(df.iloc[:cut], **kw)
            np.testing.assert_array_equal(
                part[cols].to_numpy(),
                full[cols].iloc[:cut].to_numpy(),
                err_msg=f"cut={cut}",
            )


class TestLevelLifecycle:
    def test_sweep_fires_once_and_consumes_the_level(self):
        # Bar 30 wicks through the 103.05 level and closes back below it.
        # Bar 40 wicks to 103.5, which would re-sweep a level that was kept.
        df = _flat_df(highs={10: 103.0, 20: 103.1, 30: 104.0, 40: 103.5})
        res = _compute(df)
        sweeps = np.flatnonzero(res["liq_sweep_up"].fillna(0).to_numpy() == 1.0)
        assert sweeps.tolist() == [30 + 1]
        # swept level is gone: no equal-high liquidity left to report
        assert res["liq_eqh_active"].iloc[32:].eq(0.0).all()

    def test_level_closed_through_is_removed(self):
        # Bar 30 closes above the level; once price drops back the level
        # must not come back as liquidity above.
        df = _flat_df(highs={10: 103.0, 20: 103.1}, closes={30: 105.0})
        res = _compute(df)
        assert res["liq_eqh_active"].iloc[24:31].eq(1.0).all()
        assert res["liq_eqh_active"].iloc[31:].eq(0.0).all()
        assert res["liq_sweep_up"].iloc[24:].eq(0.0).all()

    def test_adjacent_tied_pivots_are_one_touch(self):
        # Equal highs on adjacent bars are a single pivot, not a 2-touch level.
        df = _flat_df(highs={10: 103.0, 11: 103.0})
        res = _compute(df)
        assert res["liq_eqh_active"].fillna(0).sum() == 0.0


class TestMirrorSymmetry:
    def test_lows_mirror_highs(self):
        df = _flat_df(highs={10: 103.0, 20: 103.1, 30: 104.0, 40: 103.5})
        mirror = pd.DataFrame(
            {
                "O": 200.0 - df["O"],
                "H": 200.0 - df["L"],
                "L": 200.0 - df["H"],
                "C": 200.0 - df["C"],
            },
            index=df.index,
        )
        up = _compute(df)
        down = _compute(mirror)
        for hi, lo in [
            ("liq_eqh_dist", "liq_eql_dist"),
            ("liq_eqh_count", "liq_eql_count"),
            ("liq_eqh_active", "liq_eql_active"),
            ("liq_sweep_up", "liq_sweep_down"),
        ]:
            np.testing.assert_allclose(
                up[hi].to_numpy(), down[lo].to_numpy(), err_msg=hi
            )
