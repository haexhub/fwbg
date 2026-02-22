"""Grid search correctness tests with SignalModel.

Verifies that the full pipeline (indicator → SignalModel → grid search → trades)
works correctly end-to-end:

1. SignalModel reads the correct signal column and produces correct probabilities
2. Different TP/SL combos produce different trade outcomes (targets differ)
3. Trades only fire on bars where the signal column = 1.0
4. Grid search correctly identifies the best TP/SL combo
5. evaluate_on_validation correctly handles the 10-trade minimum
6. The full inner CV loop works with SignalModel and enough data
"""
import dataclasses
import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import (
    _simulate_trades_core,
    compute_targets_cached,
    evaluate_on_validation,
    simulate_trades_sequential,
)
from fwbg.optimization.nested_cv import train_model, _evaluate_single_fold
from fwbg.optimization.grid_search import (
    _compute_cached_targets,
    _build_combo_tuples,
    run_grid_search,
    select_features,
)
from fwbg.plugins import import_plugin_module

_pdl_mod = import_plugin_module("fwbg-core", "indicators", "previous_day_levels")
_signal_mod = import_plugin_module("fwbg-core", "models", "signal")
if _pdl_mod is None or _signal_mod is None:
    pytest.skip("Required plugins not available", allow_module_level=True)

from fwbg_sdk.models import TrainingContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(**overrides) -> SimulationContext:
    """SimulationContext for grid search testing with fixed exit strategy."""
    defaults = dict(
        symbol="TEST",
        asset_class="forex",
        spread=1.0,
        point=0.01,
        min_trades=1,
        long_enabled=True,
        short_enabled=True,
        exit_strategy="fixed",
        model_type="signal",
        model_hyperparameters={
            "signal_column_long": "pdl_retest_bull",
            "signal_column_short": "pdl_retest_bear",
        },
        grid_ct=[0.5],
        grid_tp=[2, 4, 6],
        grid_sl=[2, 4],
        required_features=["pdl_retest_bull", "pdl_retest_bear"],
        early_pruning_enabled=False,
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def _make_multi_day_bull_data(n_days=10):
    """Create n_days of hourly data with repeating PDH/PDL bull retest pattern.

    Each pair of days:
      - Even day: range 90..110 (establishes PDH=110, PDL=90, midpoint=100)
      - Odd day: breakout above 110, retracement to midpoint=100

    VARYING OUTCOMES by signal number:
      - Signals 0, 2, 4, ... → strong rally (108-115) → TP hit for narrow AND wide TP
      - Signals 1, 3, 5, ... → weak rally (101-103) → TP hit only for narrow TP
    This ensures different TP values produce different targets.
    """
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    signal_num = 0
    for day in range(n_days):
        base = day * 24
        if day % 2 == 0:
            # Range-setting day: 90..110
            for i in range(base, base + 24):
                close[i] = 100.0
                high[i] = 110.0
                low[i] = 90.0
                opn[i] = 100.0
        else:
            # Determine if this signal day is a "strong" or "weak" rally
            strong = signal_num % 2 == 0
            signal_num += 1

            for i in range(base, base + 24):
                h = idx[i].hour
                if h < 9:
                    close[i] = 105.0; high[i] = 106.0; low[i] = 104.0; opn[i] = 105.0
                elif h == 9:
                    # Breakout above PDH=110
                    close[i] = 115.0; high[i] = 116.0; low[i] = 108.0; opn[i] = 108.0
                elif h <= 11:
                    close[i] = 112.0; high[i] = 113.0; low[i] = 111.0; opn[i] = 112.0
                elif h == 12:
                    # Retracement to midpoint=100
                    close[i] = 100.0; high[i] = 105.0; low[i] = 99.0; opn[i] = 105.0
                elif h == 13:
                    # Post-retest bar — entry bar
                    close[i] = 101.0; high[i] = 102.0; low[i] = 100.0; opn[i] = 100.0
                else:
                    if strong:
                        # Strong rally → hits both narrow and wide TP
                        close[i] = 108.0 + (h - 14) * 0.5
                        high[i] = close[i] + 2.0
                        low[i] = close[i] - 1.0
                        opn[i] = close[i] - 0.5
                    else:
                        # Weak rally → only narrow TP hit, then drops to SL
                        if h == 14:
                            close[i] = 102.0; high[i] = 103.0; low[i] = 101.0; opn[i] = 101.0
                        elif h == 15:
                            close[i] = 101.5; high[i] = 102.0; low[i] = 100.5; opn[i] = 102.0
                        else:
                            # Price drops — triggers SL for wide-TP combos
                            close[i] = 98.0; high[i] = 99.0; low[i] = 97.0; opn[i] = 99.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


def _make_large_bull_data():
    """60-day data with ~30 signal days for grid search tests.

    Needed because evaluate_on_validation requires ≥10 trades per CT
    (hardcoded minimum). With inner CV folds, each validation fold needs
    enough signal bars.
    """
    return _make_multi_day_bull_data(60)


def _compute_indicator(df):
    """Compute previous_day_levels indicator on df, return augmented df."""
    ind = _pdl_mod.PreviousDayLevelsIndicator()
    return ind.compute(df.copy(), retest_atr_width=0.5, enable_retest=True)


def _train_signal_model(features_df, ctx, direction="long"):
    """Train a SignalModel for the given direction."""
    model = _signal_mod.SignalModel()
    feature_cols = [c for c in features_df.columns if c.startswith("pdl_")]
    dummy_targets = np.zeros(len(features_df))
    hp = ctx.model_hyperparameters
    model.train(
        features_df[feature_cols], dummy_targets,
        TrainingContext(direction=direction), **hp,
    )
    return model, feature_cols


# ===========================================================================
# Test Class 1: SignalModel mechanics
# ===========================================================================

class TestSignalModelMechanics:
    """Verify SignalModel reads the correct column and produces correct probs."""

    def test_signal_model_uses_correct_column_for_long(self):
        """Long model should read pdl_retest_bull."""
        df = pd.DataFrame({
            "pdl_retest_bull": [0, 0, 1, 0, 0],
            "pdl_retest_bear": [0, 1, 0, 0, 0],
        })
        model = _signal_mod.SignalModel()
        model.train(
            df, np.zeros(5), TrainingContext(direction="long"),
            signal_column_long="pdl_retest_bull",
            signal_column_short="pdl_retest_bear",
        )
        probs = model.predict_probability(df)
        np.testing.assert_array_equal(probs[:, 1], [0, 0, 1, 0, 0])

    def test_signal_model_uses_correct_column_for_short(self):
        """Short model should read pdl_retest_bear."""
        df = pd.DataFrame({
            "pdl_retest_bull": [0, 0, 1, 0, 0],
            "pdl_retest_bear": [0, 1, 0, 0, 0],
        })
        model = _signal_mod.SignalModel()
        model.train(
            df, np.zeros(5), TrainingContext(direction="short"),
            signal_column_long="pdl_retest_bull",
            signal_column_short="pdl_retest_bear",
        )
        probs = model.predict_probability(df)
        np.testing.assert_array_equal(probs[:, 1], [0, 1, 0, 0, 0])

    def test_signal_model_nan_treated_as_zero(self):
        """NaN in signal column should be treated as 0 (no signal)."""
        df = pd.DataFrame({"sig": [np.nan, 1.0, np.nan, 0.0]})
        model = _signal_mod.SignalModel()
        model.train(
            df, np.zeros(4), TrainingContext(direction="long"),
            signal_column_long="sig",
        )
        probs = model.predict_probability(df)
        np.testing.assert_array_equal(probs[:, 1], [0, 1, 0, 0])

    def test_signal_model_probs_sum_to_one(self):
        """Probabilities should always sum to 1.0 per row."""
        df = pd.DataFrame({"sig": [0, 1, 0, 1, 0]})
        model = _signal_mod.SignalModel()
        model.train(
            df, np.zeros(5), TrainingContext(direction="long"),
            signal_column_long="sig",
        )
        probs = model.predict_probability(df)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_signal_model_training_is_noop(self):
        """SignalModel.train ignores features and targets completely."""
        df = pd.DataFrame({"sig": [0, 1, 0, 1, 0], "noise": np.random.randn(5)})
        model = _signal_mod.SignalModel()
        # Train with random targets — should produce same result
        model.train(
            df, np.array([1, 0, 1, 0, 1]), TrainingContext(direction="long"),
            signal_column_long="sig",
        )
        probs = model.predict_probability(df)
        # Result depends ONLY on signal column, NOT on targets
        np.testing.assert_array_equal(probs[:, 1], [0, 1, 0, 1, 0])


# ===========================================================================
# Test Class 2: Target computation differs per TP/SL
# ===========================================================================

class TestTargetsDifferPerTPSL:
    """Verify that different TP/SL values produce different targets."""

    def test_different_tp_produces_different_targets(self):
        """Narrow TP hits more easily than wide TP → different target labels.

        With varying outcome data: strong rallies hit both narrow and wide TP,
        weak rallies only hit narrow TP. So targets should differ.
        """
        df = _make_multi_day_bull_data(10)
        df_feat = _compute_indicator(df)
        ctx_narrow = _ctx(spread=1.0)
        ctx_wide = _ctx(spread=1.0)

        # tp=2, spread=1 → tp_dist=2 (easy to hit)
        tgt_l_narrow, _ = compute_targets_cached(
            df_feat, 2, 4, ctx_narrow, exit_strategy_mode="fixed",
        )
        # tp=8, spread=1 → tp_dist=8 (hard to hit on weak rally days)
        tgt_l_wide, _ = compute_targets_cached(
            df_feat, 8, 4, ctx_wide, exit_strategy_mode="fixed",
        )

        wins_narrow = np.nansum(tgt_l_narrow)
        wins_wide = np.nansum(tgt_l_wide)
        assert wins_narrow > wins_wide, (
            f"Narrow TP ({wins_narrow} wins) should have more wins than wide TP ({wins_wide})"
        )

    def test_targets_not_all_zero(self):
        """Targets should have some wins (not all 0)."""
        df = _make_multi_day_bull_data(10)
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=1.0)

        tgt_long, tgt_short = compute_targets_cached(
            df_feat, 3, 4, ctx, exit_strategy_mode="fixed",
        )
        assert np.nansum(tgt_long) > 0, "Long targets should have some wins"


# ===========================================================================
# Test Class 3: Trades fire only on signal bars
# ===========================================================================

class TestTradesOnlyOnSignalBars:
    """Verify _simulate_trades_core only enters on bars where signal=1."""

    def test_trades_only_where_signal_is_one(self):
        """Every trade's signal bar must have signal=1 in the indicator output."""
        df = _make_multi_day_bull_data(6)
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=0.5)

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")

        probs_long = model_long.predict_probability(df_feat[feat_cols])
        probs_short = model_short.predict_probability(df_feat[feat_cols])

        result = _simulate_trades_core(
            df_feat, probs_long, probs_short,
            long_win_idx=1, short_win_idx=1,
            ct_long=0.5, ct_short=0.5,
            tp=4, sl=4, ctx=ctx,
            return_detailed=True,
        )

        trades = result.get("trades_detailed", [])
        assert len(trades) > 0, "Should produce at least some trades"

        for t in trades:
            sig_time = pd.Timestamp(t["signal_time"])
            sig_idx = df_feat.index.get_loc(sig_time)
            if t["direction"] == "LONG":
                assert probs_long[sig_idx, 1] >= 0.5, (
                    f"Long trade at {sig_time}: signal prob={probs_long[sig_idx, 1]:.2f} < 0.5"
                )
            else:
                assert probs_short[sig_idx, 1] >= 0.5, (
                    f"Short trade at {sig_time}: signal prob={probs_short[sig_idx, 1]:.2f} < 0.5"
                )

    def test_no_trades_when_signal_column_is_empty(self):
        """If signal column has all zeros, no trades should fire."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "O": np.full(n, 100.0),
            "H": np.full(n, 101.0),
            "L": np.full(n, 99.0),
            "C": np.full(n, 100.0),
            "pdl_retest_bull": np.zeros(n),
            "pdl_retest_bear": np.zeros(n),
        }, index=idx)

        ctx = _ctx(spread=0.5)
        model_long = _signal_mod.SignalModel()
        model_long.train(
            df, np.zeros(n), TrainingContext(direction="long"),
            **ctx.model_hyperparameters,
        )
        probs_long = model_long.predict_probability(df)

        result = _simulate_trades_core(
            df, probs_long, None, 1, None,
            ct_long=0.5, ct_short=0.5,
            tp=4, sl=4, ctx=ctx,
        )
        assert len(result["trades"]) == 0

    def test_trade_count_matches_signal_count(self):
        """Number of trades should be <= number of signal=1 bars."""
        df = _make_multi_day_bull_data(6)
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=0.5)

        signal_count_bull = (df_feat["pdl_retest_bull"].fillna(0) > 0).sum()
        signal_count_bear = (df_feat["pdl_retest_bear"].fillna(0) > 0).sum()
        total_signals = signal_count_bull + signal_count_bear

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")
        probs_long = model_long.predict_probability(df_feat[feat_cols])
        probs_short = model_short.predict_probability(df_feat[feat_cols])

        result = _simulate_trades_core(
            df_feat, probs_long, probs_short, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=4, sl=4, ctx=ctx,
        )
        trade_count = len(result["trades"])
        assert trade_count <= total_signals, (
            f"Trades ({trade_count}) should be <= signals ({total_signals})"
        )
        assert trade_count > 0, "Should produce at least some trades"


# ===========================================================================
# Test Class 4: evaluate_on_validation — hardcoded 10-trade minimum
# ===========================================================================

class TestEvaluateOnValidation:
    """Verify evaluate_on_validation behavior with SignalModel."""

    def test_evaluation_produces_trades_at_ct(self):
        """evaluate_on_validation should produce trades at ct=0.5 for valid signals."""
        df = _make_multi_day_bull_data(10)
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=0.5, grid_ct=[0.5], min_trades=1)

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")

        _, _, trades_by_ct = evaluate_on_validation(
            df_feat, model_long, model_short,
            feat_cols, feat_cols,
            tp=4, sl=4, ctx=ctx,
        )

        assert 0.5 in trades_by_ct, "Expected trades at ct=0.5"
        assert len(trades_by_ct[0.5]) > 0, "Expected >0 trades at ct=0.5"

    def test_pnl_is_neg_inf_when_under_min_eval_trades(self):
        """With trades < min_eval_trades, PnL is -inf.

        With default min_eval_trades=10 and only ~3 trades from 6 days,
        all combos get -inf → no candidate survives.
        """
        df = _make_multi_day_bull_data(6)  # Only 3 signal days → ~3 trades
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=0.5, grid_ct=[0.5], min_trades=1, min_eval_trades=10)

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")

        best_ct, best_pnl, trades_by_ct = evaluate_on_validation(
            df_feat, model_long, model_short,
            feat_cols, feat_cols,
            tp=4, sl=4, ctx=ctx,
        )

        trades = trades_by_ct.get(0.5, [])
        if len(trades) < 10:
            assert best_ct is None
            assert best_pnl == float("-inf")

    def test_lower_min_eval_trades_produces_real_pnl(self):
        """With min_eval_trades=3, even small datasets produce real PnL.

        This is the fix for signal-model strategies with rare events:
        lower the inner CV evaluation threshold so combos aren't rejected.
        """
        df = _make_multi_day_bull_data(10)  # 5 signal days → ~5 trades
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=0.5, grid_ct=[0.5], min_trades=1, min_eval_trades=3)

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")

        best_ct, best_pnl, trades_by_ct = evaluate_on_validation(
            df_feat, model_long, model_short,
            feat_cols, feat_cols,
            tp=4, sl=4, ctx=ctx,
        )

        trades = trades_by_ct.get(0.5, [])
        assert len(trades) >= 3, f"Expected ≥3 trades, got {len(trades)}"
        assert best_pnl != float("-inf"), (
            f"With min_eval_trades=3 and {len(trades)} trades, PnL should be real"
        )
        assert best_ct is not None

    def test_pnl_is_real_when_enough_trades(self):
        """With ≥10 trades, PnL should be a real number."""
        df = _make_large_bull_data()  # 60 days → ~30 signals
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=0.5, grid_ct=[0.5], min_trades=1)

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")

        best_ct, best_pnl, trades_by_ct = evaluate_on_validation(
            df_feat, model_long, model_short,
            feat_cols, feat_cols,
            tp=3, sl=4, ctx=ctx,
        )

        trades = trades_by_ct.get(0.5, [])
        assert len(trades) >= 10, (
            f"Expected ≥10 trades with 60 days of data, got {len(trades)}"
        )
        assert best_pnl != float("-inf"), (
            f"With {len(trades)} trades (≥10), PnL should be a real number"
        )
        assert best_ct is not None, "best_ct should not be None with enough trades"

    def test_different_tp_sl_different_pnl_with_enough_data(self):
        """Different TP/SL should produce different PnL when there are enough trades."""
        df = _make_large_bull_data()  # 60 days
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=1.0, grid_ct=[0.5], min_trades=1)

        model_long, feat_cols = _train_signal_model(df_feat, ctx, "long")
        model_short, _ = _train_signal_model(df_feat, ctx, "short")

        results = {}
        for tp, sl in [(2, 4), (4, 4), (8, 4)]:
            _, pnl, trades_by_ct = evaluate_on_validation(
                df_feat, model_long, model_short,
                feat_cols, feat_cols,
                tp=tp, sl=sl, ctx=ctx,
            )
            trades = trades_by_ct.get(0.5, [])
            results[(tp, sl)] = {"pnl": pnl, "n_trades": len(trades)}

        real_pnls = [r["pnl"] for r in results.values() if r["pnl"] != float("-inf")]
        assert len(real_pnls) >= 2, (
            f"At least 2 combos should have real PnL (enough trades): {results}"
        )
        assert len(set(real_pnls)) > 1, (
            f"TP/SL combos should produce different PnL: {results}"
        )


# ===========================================================================
# Test Class 5: _evaluate_single_fold with cached targets
# ===========================================================================

class TestEvaluateSingleFold:
    """Verify _evaluate_single_fold works with SignalModel and cached targets."""

    def _make_fold_data(self):
        """Create train/val split from large dataset (needs ≥10 trades per fold)."""
        df = _make_large_bull_data()  # 60 days
        df_feat = _compute_indicator(df)
        # Split: first 30 days train, next 30 days validation
        split_point = 30 * 24
        train_df = df_feat.iloc[:split_point]
        val_df = df_feat.iloc[split_point:]
        return df_feat, train_df, val_df

    def test_fold_evaluation_succeeds(self):
        """Single fold evaluation should succeed with enough data."""
        df_feat, train_df, val_df = self._make_fold_data()
        ctx = _ctx(spread=0.5, grid_ct=[0.5], min_trades=1)

        feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]
        inner_folds = [(train_df, val_df)]

        cached = _compute_cached_targets(
            4, 4, None, inner_folds, df_feat, ctx,
        )

        result = _evaluate_single_fold(
            fold_idx=0,
            train_df=train_df,
            val_df=val_df,
            group_features=feature_cols,
            tp=4, sl=4, ctx=ctx,
            cached_targets=cached,
            selected_features_long=feature_cols,
            selected_features_short=feature_cols,
        )

        assert result["success"], (
            f"Fold evaluation failed. This means too few trades or model training failed. "
            f"Train size={len(train_df)}, Val size={len(val_df)}"
        )
        assert "pnl" in result
        assert "best_ct" in result

    def test_cached_targets_per_tp_sl_differ_with_varying_data(self):
        """Cached targets differ between narrow and wide TP on varying-outcome data."""
        df = _make_multi_day_bull_data(20)  # More data, varying outcomes
        df_feat = _compute_indicator(df)
        ctx = _ctx(spread=1.0)
        split = 10 * 24
        train_df = df_feat.iloc[:split]
        val_df = df_feat.iloc[split:]
        inner_folds = [(train_df, val_df)]

        # tp=2 (tp_dist=2) vs tp=8 (tp_dist=8)
        cached_narrow = _compute_cached_targets(2, 4, None, inner_folds, df_feat, ctx)
        cached_wide = _compute_cached_targets(8, 4, None, inner_folds, df_feat, ctx)

        tgt_narrow = cached_narrow[0][0]  # fold 0, targets_long
        tgt_wide = cached_wide[0][0]

        wins_narrow = np.nansum(tgt_narrow)
        wins_wide = np.nansum(tgt_wide)
        assert wins_narrow > wins_wide, (
            f"Narrow TP should have more wins ({wins_narrow}) than wide TP ({wins_wide})"
        )


# ===========================================================================
# Test Class 6: Full grid search flow
# ===========================================================================

class TestFullGridSearch:
    """Verify run_grid_search produces correct results with SignalModel."""

    def _setup_grid_data(self):
        """Create dataset large enough for grid search (≥10 trades per fold)."""
        df = _make_large_bull_data()  # 60 days
        df_feat = _compute_indicator(df)
        # 2 inner folds: train on first half, validate on second half
        split1 = 20 * 24
        split2 = 40 * 24
        train1 = df_feat.iloc[:split1]
        val1 = df_feat.iloc[split1:split2]
        train2 = df_feat.iloc[:split2]
        val2 = df_feat.iloc[split2:]
        inner_folds = [(train1, val1), (train2, val2)]
        return df_feat, inner_folds

    def test_grid_search_produces_candidates(self):
        """Grid search should produce at least one candidate with enough data."""
        df_feat, inner_folds = self._setup_grid_data()
        feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]
        ctx = _ctx(
            spread=1.0, grid_ct=[0.5], min_trades=1,
            grid_tp=[2, 4], grid_sl=[4],
            early_pruning_enabled=False,
        )

        from fwbg.core.config import GridConfig
        grid = GridConfig(tp=[2, 4], sl=[4], ct=[0.5])

        candidates, grid_results = run_grid_search(
            full_pool=feature_cols,
            inner_folds=inner_folds,
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=df_feat,
        )

        assert len(candidates) > 0, (
            f"Grid search should produce candidates with 60 days of data. "
            f"Grid results: {grid_results}"
        )

    def test_grid_search_candidates_have_different_params(self):
        """Multiple candidates should have different TP/SL params."""
        df_feat, inner_folds = self._setup_grid_data()
        feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]
        ctx = _ctx(
            spread=1.0, grid_ct=[0.5], min_trades=1,
            grid_tp=[2, 4, 8], grid_sl=[4],
            early_pruning_enabled=False,
        )

        from fwbg.core.config import GridConfig
        grid = GridConfig(tp=[2, 4, 8], sl=[4], ct=[0.5])

        candidates, _ = run_grid_search(
            full_pool=feature_cols,
            inner_folds=inner_folds,
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=df_feat,
        )

        if len(candidates) > 1:
            param_sets = [c["params"] for c in candidates]
            unique_params = set(param_sets)
            assert len(unique_params) > 1, (
                f"Multiple candidates should have different params: {param_sets}"
            )

    def test_grid_search_model_hyperparameters_stored(self):
        """Each candidate should store model_hyperparameters."""
        df_feat, inner_folds = self._setup_grid_data()
        feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]
        ctx = _ctx(
            spread=1.0, grid_ct=[0.5], min_trades=1,
            grid_tp=[3], grid_sl=[4],
            early_pruning_enabled=False,
        )

        from fwbg.core.config import GridConfig
        grid = GridConfig(tp=[3], sl=[4], ct=[0.5])

        candidates, _ = run_grid_search(
            full_pool=feature_cols,
            inner_folds=inner_folds,
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=df_feat,
        )

        for c in candidates:
            assert "model_hyperparameters" in c
            hp = c["model_hyperparameters"]
            assert hp.get("signal_column_long") == "pdl_retest_bull"
            assert hp.get("signal_column_short") == "pdl_retest_bear"

    def test_grid_search_no_candidates_with_tiny_data(self):
        """With too few signal bars, grid search returns 0 candidates.

        This documents the current behavior: evaluate_on_validation requires
        ≥10 trades per CT (hardcoded). With rare event signals and small
        validation folds, this minimum is not met → all combos rejected.
        """
        df = _make_multi_day_bull_data(6)  # Only 3 signal days
        df_feat = _compute_indicator(df)
        feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]

        split = 3 * 24
        train = df_feat.iloc[:split]
        val = df_feat.iloc[split:]
        inner_folds = [(train, val)]

        ctx = _ctx(
            spread=0.5, grid_ct=[0.5], min_trades=1,
            grid_tp=[4], grid_sl=[4],
            early_pruning_enabled=False,
        )

        from fwbg.core.config import GridConfig
        grid = GridConfig(tp=[4], sl=[4], ct=[0.5])

        candidates, _ = run_grid_search(
            full_pool=feature_cols,
            inner_folds=inner_folds,
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=df_feat,
        )

        # Document: with ~2 signal bars in validation, 0 candidates expected
        assert len(candidates) == 0, (
            f"Expected 0 candidates with tiny data (insufficient trades for 10-trade minimum)"
        )


# ===========================================================================
# Test Class 7: Combo building
# ===========================================================================

class TestComboBuilding:
    """Verify _build_combo_tuples builds correct number of combos."""

    def test_combo_count_matches_grid_dimensions(self):
        """Number of combos should be TP × SL × timeout × modifiers × model_hp."""
        from fwbg.core.config import GridConfig
        grid = GridConfig(tp=[2, 4, 6], sl=[2, 4], ct=[0.5])
        ctx = _ctx(grid_tp=[2, 4, 6], grid_sl=[2, 4])

        combos, skipped = _build_combo_tuples(
            grid, ctx,
            timeout_values=[None],
            features=["pdl_retest_bull", "pdl_retest_bear"],
            inner_folds=[],
            regime_config={},
            total_grid_combos=6,
            inner_df=None,
            selected_features_long=["pdl_retest_bull"],
            selected_features_short=["pdl_retest_bear"],
            sym="TEST",
        )
        assert len(combos) == 6
        assert skipped == 0

    def test_rrr_filter_skips_combos(self):
        """Combos with RRR < min_rrr should be skipped."""
        from fwbg.core.config import GridConfig
        grid = GridConfig(tp=[2, 4], sl=[4], ct=[0.5])
        ctx = _ctx(grid_tp=[2, 4], grid_sl=[4], min_rrr=1.0)

        combos, skipped = _build_combo_tuples(
            grid, ctx,
            timeout_values=[None],
            features=["pdl_retest_bull"],
            inner_folds=[],
            regime_config={},
            total_grid_combos=2,
            inner_df=None,
            selected_features_long=["pdl_retest_bull"],
            selected_features_short=["pdl_retest_bear"],
            sym="TEST",
        )
        assert len(combos) == 1
        assert skipped == 1


# ===========================================================================
# Test Class 8: Feature selection with required_features
# ===========================================================================

class TestFeatureSelectionWithSignalModel:
    """Verify that required_features are always included."""

    def test_required_features_always_present(self):
        """Signal columns from required_features must survive feature selection."""
        df = _make_multi_day_bull_data(8)
        df_feat = _compute_indicator(df)
        ctx = _ctx(
            spread=0.5, min_trades=1,
            required_features=["pdl_retest_bull", "pdl_retest_bear"],
        )

        split = 4 * 24
        train_df = df_feat.iloc[:split]
        val_df = df_feat.iloc[split:]
        inner_folds = [(train_df, val_df)]

        feature_cols = [c for c in df_feat.columns if c.startswith("pdl_")]
        selected_long, selected_short = select_features(
            inner_folds, feature_cols, ctx, "TEST",
        )

        if selected_long is not None:
            assert "pdl_retest_bull" in selected_long
        if selected_short is not None:
            assert "pdl_retest_bear" in selected_short
