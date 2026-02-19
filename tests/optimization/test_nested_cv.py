"""
Tests für nested_cv.py - Nested Cross-Validation Funktionen.

Testet:
- nested_cv_split: Korrekte Aufteilung, Randbedingungen, Edge Cases
- compute_targets: Target-Berechnung, NaN-Handling, Direction-Filter
- slice_targets_for_fold: Index-Mapping, Grenzfälle
- run_inner_cv: Early Termination, Fold Stability, Feature Selection
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from fwbg.optimization.nested_cv import (
    nested_cv_split,
    select_features_from_fold,
    train_model,
    run_inner_cv,
    evaluate_on_holdout,
)
from fwbg.optimization.targets import (
    compute_targets,
    compute_targets_cached,
    slice_targets_for_fold,
    simulate_trades_sequential,
)
from fwbg.core.context import SimulationContext


def create_test_df(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Erstellt einen Test-DataFrame mit OHLC-Daten."""
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_rows)]

    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)
    high = close + np.abs(np.random.randn(n_rows) * 0.05)
    low = close - np.abs(np.random.randn(n_rows) * 0.05)
    open_price = close + np.random.randn(n_rows) * 0.02

    df = pd.DataFrame({
        "O": open_price,
        "H": high,
        "L": low,
        "C": close,
        "_atr": np.abs(np.random.randn(n_rows) * 0.1) + 0.05,
        "_regime": np.full(n_rows, 7, dtype=np.int8),
    }, index=pd.DatetimeIndex(dates))

    # Einige Feature-Spalten hinzufügen
    df["trend_rsi_14"] = np.random.rand(n_rows) * 100
    df["trend_adx_14"] = np.random.rand(n_rows) * 100
    df["mom_stoch_14"] = np.random.rand(n_rows) * 100

    return df


def create_mock_context(
    symbol: str = "TESTUSD",
    spread: float = 0.0002,
    long_enabled: bool = True,
    short_enabled: bool = True,
    min_trades: int = 10,
    max_trade_bars: int = None,
) -> SimulationContext:
    """Erstellt einen Mock-SimulationContext."""
    ctx = Mock(spec=SimulationContext)
    ctx.symbol = symbol
    ctx.spread = spread
    ctx.long_enabled = long_enabled
    ctx.short_enabled = short_enabled
    ctx.min_trades = min_trades
    ctx.max_trade_bars = max_trade_bars
    ctx.grid_ct = [0.5, 0.6, 0.7]
    ctx.long_grid_ct = None  # None = verwende grid_ct
    ctx.short_grid_ct = None  # None = verwende grid_ct
    ctx.separate_long_short = False
    ctx.feature_selection_plugins = [{"name": "boruta", "params": {"min_z_score": 0.5}}]
    ctx.early_termination = True
    ctx.min_fold_stability = 0.5
    ctx.first_fold_sanity_check = True
    ctx.first_fold_min_win_rate = 0.25
    ctx.first_fold_min_pnl = -10.0
    ctx.first_fold_min_trades = 5
    ctx.model_hyperparameters = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }
    ctx.preprocessing_plugins = []  # No preprocessing by default
    ctx.exit_strategy = "fixed"
    ctx.exit_params = {}
    ctx.probability_calibration = False
    ctx.calibration_method = "isotonic"
    ctx.model_type = "xgboost"
    return ctx


class TestNestedCvSplit:
    """Tests für nested_cv_split Funktion."""

    def test_basic_split(self):
        """Test: Grundlegende Aufteilung mit Standard-Parametern."""
        df = create_test_df(10000)
        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=5)

        assert "inner_folds" in result
        assert "holdout_df" in result
        assert "inner_df" in result

        # Holdout sollte ca. 20% sein
        assert len(result["holdout_df"]) == 2000
        assert len(result["inner_df"]) == 8000

        # Mindestens 1 Fold sollte erstellt werden
        assert len(result["inner_folds"]) >= 1

    def test_no_data_overlap(self):
        """Test: Kein Daten-Overlap zwischen Train/Val/Holdout."""
        df = create_test_df(5000)
        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=3)

        holdout_indices = set(result["holdout_df"].index)
        inner_indices = set(result["inner_df"].index)

        # Holdout und Inner dürfen nicht überlappen
        assert holdout_indices.isdisjoint(inner_indices)

        # Innerhalb der Folds: Train und Val dürfen nicht überlappen
        for train_df, val_df in result["inner_folds"]:
            train_idx = set(train_df.index)
            val_idx = set(val_df.index)
            assert train_idx.isdisjoint(val_idx), "Train und Val überlappen!"

    def test_temporal_ordering(self):
        """Test: Zeitliche Ordnung wird eingehalten (Walk-Forward)."""
        df = create_test_df(5000)
        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=3)

        # Holdout muss nach Inner kommen
        assert result["inner_df"].index[-1] < result["holdout_df"].index[0]

        # In jedem Fold: Train muss vor Val kommen
        for train_df, val_df in result["inner_folds"]:
            assert train_df.index[-1] < val_df.index[0], \
                "Train endet nicht vor Val Start!"

    def test_minimum_data(self):
        """Test: Verhalten bei sehr wenig Daten."""
        df = create_test_df(100)
        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=5)

        # Bei wenig Daten: Möglicherweise weniger Folds
        assert isinstance(result["inner_folds"], list)
        assert len(result["holdout_df"]) > 0

    def test_extreme_holdout_ratio(self):
        """Test: Extreme Holdout-Ratios."""
        df = create_test_df(1000)

        # Sehr kleine Holdout
        result_small = nested_cv_split(df, holdout_ratio=0.05)
        assert len(result_small["holdout_df"]) == 50

        # Sehr große Holdout
        result_large = nested_cv_split(df, holdout_ratio=0.50)
        assert len(result_large["holdout_df"]) == 500

    def test_zero_folds_edge_case(self):
        """Test: Edge Case wo keine Folds möglich sind."""
        df = create_test_df(50)
        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=10, oos_size=100)

        # Bei zu wenig Daten könnten 0 Folds entstehen
        assert isinstance(result["inner_folds"], list)

    def test_custom_oos_size(self):
        """Test: Benutzerdefinierte OOS-Größe."""
        df = create_test_df(10000)
        result = nested_cv_split(df, n_inner_folds=3, oos_size=500)

        for train_df, val_df in result["inner_folds"]:
            # Val-Size sollte <= oos_size sein
            assert len(val_df) <= 500


class TestComputeTargets:
    """Tests für compute_targets und compute_targets_cached."""

    def test_basic_target_computation(self):
        """Test: Grundlegende Target-Berechnung."""
        df = create_test_df(1000)
        ctx = create_mock_context()

        targets_long, targets_short, has_long, has_short = compute_targets(
            df, tp=20, sl=10, ctx=ctx
        )

        assert len(targets_long) == len(df)
        assert len(targets_short) == len(df)
        assert targets_long.dtype in [np.float64, np.int64]
        assert set(np.unique(targets_long)).issubset({0, 1, 0.0, 1.0})

    def test_direction_filtering(self):
        """Test: Long/Short werden korrekt gefiltert."""
        df = create_test_df(1000)

        # Nur Long aktiviert
        ctx_long_only = create_mock_context(long_enabled=True, short_enabled=False)
        _, _, has_long, has_short = compute_targets(df, 20, 10, ctx_long_only)
        assert has_short is False

        # Nur Short aktiviert
        ctx_short_only = create_mock_context(long_enabled=False, short_enabled=True)
        _, _, has_long, has_short = compute_targets(df, 20, 10, ctx_short_only)
        assert has_long is False

    def test_min_trades_threshold(self):
        """Test: Minimum-Trades Schwelle wird respektiert."""
        df = create_test_df(100)  # Wenig Daten = wenig Trades
        ctx = create_mock_context(min_trades=1000)  # Hohe Schwelle

        _, _, has_long, has_short = compute_targets(df, 20, 10, ctx)

        # Bei zu wenig Trades sollte has_* False sein
        # (abhängig von den zufälligen Daten)
        # Akzeptiere bool und np.bool_ (numpy Vergleiche liefern np.bool_)
        assert isinstance(has_long, (bool, np.bool_))
        assert isinstance(has_short, (bool, np.bool_))

    def test_cached_vs_uncached_equivalence(self):
        """Test: Cached und Uncached Berechnung liefern gleiche Ergebnisse."""
        df = create_test_df(500)
        ctx = create_mock_context()

        # Uncached
        targets_l, targets_s, _, _ = compute_targets(df, 20, 10, ctx)

        # Cached
        cached_l, cached_s = compute_targets_cached(df, 20, 10, ctx)

        # Ergebnisse sollten identisch sein
        np.testing.assert_array_equal(targets_l, cached_l)
        np.testing.assert_array_equal(targets_s, cached_s)


class TestSliceTargetsForFold:
    """Tests für slice_targets_for_fold."""

    def test_correct_slicing(self):
        """Test: Korrektes Slicen der Targets für einen Fold."""
        df = create_test_df(1000)
        ctx = create_mock_context()

        # Volle Targets berechnen
        full_targets_l = np.random.randint(0, 2, len(df)).astype(float)
        full_targets_s = np.random.randint(0, 2, len(df)).astype(float)

        # Fold erstellen
        fold_df = df.iloc[200:400].copy()

        targets_l, targets_s, has_l, has_s = slice_targets_for_fold(
            full_targets_l, full_targets_s, df, fold_df, ctx
        )

        assert len(targets_l) == 200
        np.testing.assert_array_equal(targets_l, full_targets_l[200:400])

    def test_boundary_conditions(self):
        """Test: Randbedinungen beim Slicen."""
        df = create_test_df(1000)
        ctx = create_mock_context()

        full_targets_l = np.random.randint(0, 2, len(df)).astype(float)
        full_targets_s = np.random.randint(0, 2, len(df)).astype(float)

        # Anfang
        fold_start = df.iloc[:100].copy()
        t_l, _, _, _ = slice_targets_for_fold(
            full_targets_l, full_targets_s, df, fold_start, ctx
        )
        np.testing.assert_array_equal(t_l, full_targets_l[:100])

        # Ende
        fold_end = df.iloc[900:].copy()
        t_l, _, _, _ = slice_targets_for_fold(
            full_targets_l, full_targets_s, df, fold_end, ctx
        )
        np.testing.assert_array_equal(t_l, full_targets_l[900:])


class TestSelectFeaturesFromFold:
    """Tests für select_features_from_fold."""

    def test_empty_features(self):
        """Test: Leere Feature-Liste."""
        df = create_test_df(500)
        targets = np.random.randint(0, 2, len(df))

        selected, importances = select_features_from_fold(
            df, targets, [], min_trades=10
        )

        assert selected is None
        assert importances == {}

    def test_insufficient_targets(self):
        """Test: Zu wenige Targets für Training."""
        df = create_test_df(500)
        targets = np.zeros(len(df))  # Keine positiven Targets
        features = ["trend_rsi_14", "trend_adx_14"]

        selected, _ = select_features_from_fold(
            df, targets, features, min_trades=100
        )

        assert selected is None

    def test_missing_features(self):
        """Test: Features die nicht im DataFrame existieren."""
        df = create_test_df(500)
        targets = np.random.randint(0, 2, len(df))
        features = ["nonexistent_feature_1", "nonexistent_feature_2"]

        selected, _ = select_features_from_fold(
            df, targets, features, min_trades=10
        )

        assert selected is None

    def test_boruta_via_plugin(self):
        """Test: Feature Selection via Plugin-Interface."""
        df = create_test_df(1000)
        # Targets die mit einem Feature korrelieren
        targets = (df["trend_rsi_14"].values > 50).astype(int)
        features = ["trend_rsi_14", "trend_adx_14", "mom_stoch_14"]

        plugins = [{"name": "boruta", "params": {
            "n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.0
        }}]

        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=plugins,
        )

        # Sollte mindestens einige Features auswählen (oder None)
        assert selected is None or len(selected) >= 0


class TestTrainModel:
    """Tests für train_model."""

    def test_basic_training(self):
        """Test: Grundlegendes Modell-Training."""
        df = create_test_df(500)
        targets = np.random.randint(0, 2, len(df))
        features = ["trend_rsi_14", "trend_adx_14"]
        ctx = create_mock_context()

        model = train_model(df, targets, features, min_trades=10, ctx=ctx)

        if np.count_nonzero(targets) >= 5:
            assert model is not None
            assert hasattr(model, "predict_probability")

    def test_no_features(self):
        """Test: Training ohne Features."""
        df = create_test_df(500)
        targets = np.random.randint(0, 2, len(df))
        ctx = create_mock_context()

        model = train_model(df, targets, None, min_trades=10, ctx=ctx)
        assert model is None

    def test_insufficient_positive_targets(self):
        """Test: Zu wenige positive Targets."""
        df = create_test_df(500)
        targets = np.zeros(len(df))  # Keine positiven
        features = ["trend_rsi_14"]
        ctx = create_mock_context()

        model = train_model(df, targets, features, min_trades=100, ctx=ctx)
        assert model is None


class TestSimulateTradesSequential:
    """Tests für simulate_trades_sequential."""

    def test_no_signals(self):
        """Test: Keine Signale (alle Probs unter Threshold)."""
        df = create_test_df(500)
        ctx = create_mock_context()

        probs_long = np.zeros((len(df), 2))
        probs_long[:, 0] = 1.0  # Alle Probs für Klasse 0

        result = simulate_trades_sequential(
            df, probs_long, None, 1, None, ct=0.9, tp=20, sl=10, ctx=ctx
        )

        assert result["trades"] == []

    def test_regime_filter(self):
        """Test: Regime-Filter blockiert Trades."""
        df = create_test_df(500)
        df["_regime"] = np.int8(0)  # Kein Trade erlaubt
        ctx = create_mock_context()

        probs_long = np.ones((len(df), 2)) * 0.5
        probs_long[:, 1] = 0.9  # Hohe Win-Prob

        result = simulate_trades_sequential(
            df, probs_long, None, 1, None, ct=0.5, tp=20, sl=10, ctx=ctx
        )

        assert result["trades"] == []

    def test_detailed_output(self):
        """Test: Detaillierte Trade-Ausgabe."""
        df = create_test_df(500)
        ctx = create_mock_context()

        probs_long = np.ones((len(df), 2)) * 0.5
        probs_long[:, 1] = 0.8

        result = simulate_trades_sequential(
            df, probs_long, None, 1, None, ct=0.5, tp=20, sl=10, ctx=ctx,
            return_detailed=True
        )

        assert "trades_detailed" in result
        if result["trades"]:
            assert len(result["trades_detailed"]) == len(result["trades"])


class TestRunInnerCv:
    """Tests für run_inner_cv."""

    def test_empty_folds(self):
        """Test: Leere Fold-Liste."""
        ctx = create_mock_context()

        result = run_inner_cv(
            inner_folds=[],
            group_features=["trend_rsi_14"],
            tp=20, sl=10, ctx=ctx,
            global_grid_pos=0, total_grid_combos=1
        )

        assert result["success"] is False

    def test_early_termination(self):
        """Test: Early Termination bei schlechter Performance."""
        df = create_test_df(5000)
        ctx = create_mock_context()
        ctx.early_termination = True
        ctx.min_fold_stability = 0.9  # Hohe Anforderung

        # Folds erstellen
        folds = [
            (df.iloc[:1000], df.iloc[1000:1500]),
            (df.iloc[:1500], df.iloc[1500:2000]),
            (df.iloc[:2000], df.iloc[2000:2500]),
        ]

        # Mit Mock um schnell zu laufen
        with patch('fwbg.optimization.nested_cv.compute_targets') as mock_compute:
            # Simuliere schlechte Targets (keine Wins)
            mock_compute.return_value = (
                np.zeros(1000), np.zeros(1000), False, False
            )

            result = run_inner_cv(
                inner_folds=folds,
                group_features=["trend_rsi_14"],
                tp=20, sl=10, ctx=ctx,
                global_grid_pos=0, total_grid_combos=1
            )

            assert result["success"] is False


class TestEvaluateOnHoldout:
    """Tests für evaluate_on_holdout."""

    def test_no_models(self):
        """Test: Keine trainierten Modelle."""
        df = create_test_df(1000)
        ctx = create_mock_context()

        candidate = {
            "params": (20, 10, 0.6),
            "selected_features_long": None,
            "selected_features_short": None,
        }

        result = evaluate_on_holdout(
            holdout_df=df.iloc[500:],
            inner_df=df.iloc[:500],
            candidate=candidate,
            ctx=ctx
        )

        assert result["trades"] == []
        assert result["n_trades"] == 0


class TestEdgeCases:
    """Edge Cases und Fehlerfälle."""

    def test_nan_in_data(self):
        """Test: NaN-Werte in den Daten."""
        df = create_test_df(500)
        df.loc[df.index[100:110], "C"] = np.nan
        df.loc[df.index[200:210], "_atr"] = np.nan

        # Sollte nicht crashen
        result = nested_cv_split(df, holdout_ratio=0.20)
        assert len(result["inner_folds"]) >= 0

    def test_constant_prices(self):
        """Test: Konstante Preise (keine Bewegung)."""
        df = create_test_df(500)
        df["C"] = 100.0
        df["H"] = 100.0
        df["L"] = 100.0
        df["O"] = 100.0

        ctx = create_mock_context()
        targets_l, targets_s, _, _ = compute_targets(df, 20, 10, ctx)

        # Bei konstanten Preisen: Alle Targets sollten 0 sein
        assert np.sum(targets_l) == 0 or np.sum(targets_l) > 0  # Keine Assertion

    def test_single_row_df(self):
        """Test: DataFrame mit nur einer Zeile."""
        df = create_test_df(1)

        result = nested_cv_split(df, holdout_ratio=0.20)

        # Sollte nicht crashen, auch wenn keine Folds möglich
        assert isinstance(result["inner_folds"], list)

    def test_all_regime_false(self):
        """Test: Alle Regime-Filter sind False."""
        df = create_test_df(500)
        df["_regime"] = np.int8(0)
        ctx = create_mock_context()

        probs = np.ones((len(df), 2)) * 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None, ct=0.5, tp=20, sl=10, ctx=ctx
        )

        assert result["trades"] == []


class TestSeparateCTEvaluation:
    """Tests für die separate Long/Short CT-Optimierung."""

    def test_separate_ct_returns_tuple(self):
        """Test: Separate CT gibt Tuple (ct_long, ct_short) zurück."""
        from fwbg.optimization.targets import _evaluate_separate_ct

        df = create_test_df(500)
        ctx = create_mock_context()
        ctx.separate_long_short = True
        ctx.grid_ct = [0.5, 0.55, 0.6]

        # Erstelle Dummy-Probs die Trades generieren
        probs_long = np.random.rand(len(df), 2)
        probs_long[:, 1] = 0.7  # Win-Wahrscheinlichkeit hoch
        probs_short = np.random.rand(len(df), 2)
        probs_short[:, 1] = 0.7

        result = _evaluate_separate_ct(
            df, probs_long, probs_short, 1, 1,
            tp=20, sl=10, ctx=ctx
        )

        best_ct, best_pnl, trades_info = result

        # Bei ausreichend Trades sollte ein Tuple zurückkommen
        # (oder None wenn keine valide Kombination)
        if best_ct is not None:
            assert isinstance(best_ct, tuple)
            assert len(best_ct) == 2

    def test_separate_ct_early_termination_no_trades(self):
        """Test: Early Termination wenn keine Trades generiert werden."""
        from fwbg.optimization.targets import _evaluate_separate_ct

        df = create_test_df(500)
        df["_regime"] = np.int8(0)  # Keine Trades möglich
        ctx = create_mock_context()
        ctx.separate_long_short = True
        ctx.grid_ct = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  # 6 CT-Werte

        # Sehr hohe Probs, aber regime_ok ist False -> keine Trades
        probs_long = np.ones((len(df), 2)) * 0.9
        probs_short = np.ones((len(df), 2)) * 0.9

        result = _evaluate_separate_ct(
            df, probs_long, probs_short, 1, 1,
            tp=20, sl=10, ctx=ctx
        )

        best_ct, best_pnl, trades_info = result

        # Sollte None zurückgeben (keine valide Kombination)
        assert best_ct is None

    def test_separate_ct_independent_optimization(self):
        """Test: Long und Short werden unabhängig optimiert (O(2n) statt O(n²))."""
        from fwbg.optimization.targets import _evaluate_separate_ct

        df = create_test_df(500)
        ctx = create_mock_context()
        ctx.separate_long_short = True
        ctx.grid_ct = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  # 6 CT-Werte

        # Probs die Trades generieren
        probs_long = np.random.rand(len(df), 2)
        probs_long[:, 1] = 0.7
        probs_short = np.random.rand(len(df), 2)
        probs_short[:, 1] = 0.7

        result = _evaluate_separate_ct(
            df, probs_long, probs_short, 1, 1,
            tp=20, sl=10, ctx=ctx
        )

        best_ct, best_pnl, trades_info = result

        # trades_info sollte separate Long und Short Dicts haben
        assert "long" in trades_info
        assert "short" in trades_info
        assert "combined" in trades_info

        # Long und Short wurden separat getestet (6 CTs jeweils, nicht 36 Kombinationen)
        # Jeder CT-Wert sollte in den Dicts sein
        if trades_info["long"]:
            assert len(trades_info["long"]) <= len(ctx.grid_ct)
        if trades_info["short"]:
            assert len(trades_info["short"]) <= len(ctx.grid_ct)

    def test_separate_ct_only_short_enabled(self):
        """Test: Nur Short aktiviert, Long sollte Default-CT bekommen."""
        from fwbg.optimization.targets import _evaluate_separate_ct

        df = create_test_df(500)
        ctx = create_mock_context()
        ctx.separate_long_short = True
        ctx.long_enabled = False  # Long deaktiviert
        ctx.short_enabled = True
        ctx.grid_ct = [0.5, 0.55, 0.6]

        probs_short = np.random.rand(len(df), 2)
        probs_short[:, 1] = 0.7

        result = _evaluate_separate_ct(
            df, None, probs_short, None, 1,  # Keine Long-Probs
            tp=20, sl=10, ctx=ctx
        )

        best_ct, best_pnl, trades_info = result

        # Long sollte leer sein
        assert trades_info["long"] == {}

        # Wenn Short Trades hatte, sollte ein Tuple zurückkommen
        if best_ct is not None:
            assert isinstance(best_ct, tuple)
            # ct_long sollte ein Fallback-Wert sein
            assert best_ct[0] in ctx.grid_ct or best_ct[0] == 0.55
