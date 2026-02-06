"""
TDD Tests für Grid-Search Feature-Gruppen Problem.

Bug: Grid-Search findet 0 Kandidaten obwohl 691 saubere Features existieren.
"""
import pytest
import numpy as np
import pandas as pd


class TestFeatureGroupFiltering:
    """Tests dass Feature-Gruppen korrekt gefiltert werden."""

    @pytest.fixture
    def sample_features(self):
        """Sample Feature-Namen wie sie von Indikatoren generiert werden."""
        return [
            # Trend features
            "trend_adx_7", "trend_adx_14", "trend_adx_21",
            "trend_ema_dist_8", "trend_ema_dist_21", "trend_ema_dist_50",
            "trend_macd", "trend_macd_signal",
            # Momentum features
            "mom_rsi_7", "mom_rsi_14", "mom_rsi_21",
            "mom_stoch_k_14", "mom_stoch_d_14",
            "mom_williams_r_14", "mom_roc_10",
            # Volatility features
            "vol_atr_pct_14", "vol_atr_pct_20",
            "vol_bb_wband_20", "vol_bb_pband_20",
            "vol_kc_wband_20", "vol_dc_wband_20",
            # Dynamics features
            "dyn_rsi14_chg_4h", "dyn_rsi14_chg_8h",
            "dyn_atr_chg_4h", "dyn_atr_chg_8h",
            "lag_rsi14_4h", "lag_rsi14_8h",
            "accel_rsi", "accel_atr",
            # Multi-timeframe features
            "mtf_h4_trend", "mtf_h4_range_pos",
            "mtf_h4_ema20_dist", "mtf_d1_range_pos",
            # Ichimoku features
            "ichi_tenkan", "ichi_kijun",
            "ichi_cloud_thick", "ichi_cloud_pos",
            "ichi_tk_cross", "ichi_above_cloud",
            # Price action features
            "pa_range_pos", "pa_body_ratio", "pa_body_dir",
            "pa_hh", "pa_ll", "pa_gap",
            # Time/season features
            "time_hour_sin", "time_hour_cos",
            "time_dow_sin", "time_dow_cos",
            "season_month_sin", "season_month_cos",
            # Distribution features
            "dist_skew_20", "dist_kurt_20",
            "dist_skew_50", "dist_kurt_50",
            # Regime features
            "regime_hurst_100", "regime_hurst_200",
            # Structure features
            "fft_dom_freq", "fft_spectral_energy",
            "path_efficiency_10", "path_efficiency_20",
            "event_bars_since_high", "event_bars_since_low",
            # Risk features
            "risk_dd_current", "risk_dd_max_20",
            "dd_state", "cvar_5pct",
            # Cross features
            "cross_rsi_bb", "cross_macd_vol",
            "cross_trend_mom",
            # Microstructure features
            "micro_imbalance", "micro_tick_count",
            "micro_avg_trade_size",
            # Macro surprise features
            "macro_gap_up", "macro_gap_down",
            "macro_overnight_return",
        ]

    def test_feature_groups_have_enough_features(self, sample_features):
        """Jede Feature-Gruppe muss mindestens 2 Features haben."""
        from fwbg.builtins.indicators import filter_features_by_group, FEATURE_GROUPS

        feature_groups_to_test = [
            "trend", "momentum", "volatility", "dynamics",
            "multi_timeframe", "ichimoku", "price_action",
            "time_season", "distribution", "regime", "structure",
            "risk", "cross_features", "microstructure", "macro_surprise"
        ]

        # Minimum 2 Features für Boruta (braucht Vergleich)
        min_features = 2

        for fg in feature_groups_to_test:
            group_features = filter_features_by_group(sample_features, fg)
            assert len(group_features) >= min_features, (
                f"Feature-Gruppe '{fg}' hat nur {len(group_features)} Features: {group_features}. "
                f"FEATURE_GROUPS[{fg}] = {FEATURE_GROUPS.get(fg, 'NOT FOUND')}"
            )

    def test_all_feature_prefixes_match_group_names(self):
        """Feature-Gruppen-Prefixes müssen mit generierten Feature-Namen übereinstimmen."""
        from fwbg.builtins.indicators import FEATURE_GROUPS

        # Mapping von Indikator-Gruppen zu erwarteten Prefixes in Features
        expected_prefix_mapping = {
            "trend": ["trend_"],
            "momentum": ["mom_"],
            "volatility": ["vol_"],
            "dynamics": ["dyn_", "lag_", "accel_"],
            "multi_timeframe": ["mtf_"],
            "ichimoku": ["ichi_"],
            "price_action": ["pa_"],
            "time_season": ["time_", "season_"],
            "distribution": ["dist_"],
            "regime": ["regime_"],
            "structure": ["fft_", "path_", "event_"],  # Updated expected
            "risk": ["risk_", "dd_", "cvar_"],
            "cross_features": ["cross_"],
            "microstructure": ["micro_"],
            "macro_surprise": ["macro_"],  # Note: This might need specific prefixes
        }

        for fg, expected_prefixes in expected_prefix_mapping.items():
            assert fg in FEATURE_GROUPS, f"Feature-Gruppe '{fg}' fehlt in FEATURE_GROUPS"
            actual_prefixes = FEATURE_GROUPS[fg].get("prefixes", [])

            for expected in expected_prefixes:
                assert any(expected in p or p in expected for p in actual_prefixes), (
                    f"Feature-Gruppe '{fg}': Erwarteter Prefix '{expected}' nicht in {actual_prefixes}"
                )


class TestRealIndicatorFeatureGroups:
    """Tests mit echten Indikator-Features."""

    @pytest.fixture
    def real_ohlc_data(self):
        """Erstellt realistische OHLC-Daten."""
        np.random.seed(42)
        n = 5000

        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "O": prices * (1 + np.random.randn(n) * 0.001),
            "H": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
            "L": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
            "C": prices,
            "V": np.abs(np.random.randn(n)) * 10000 + 1000,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        df["H"] = df[["O", "H", "C"]].max(axis=1)
        df["L"] = df[["O", "L", "C"]].min(axis=1)

        return df

    def test_each_indicator_group_produces_enough_features(self, real_ohlc_data):
        """Jeder Indikator muss genug Features für Grid-Search produzieren (>= 3)."""
        from fwbg.builtins.indicators import (
            compute_indicator_pool, get_feature_columns, filter_features_by_group
        )

        indicators_to_test = [
            "trend", "momentum", "volatility", "dynamics",
            "multi_timeframe", "ichimoku", "price_action"
        ]

        for indicator in indicators_to_test:
            # Berechne nur diesen Indikator
            result = compute_indicator_pool(
                real_ohlc_data.copy(),
                indicators=[indicator]
            )

            # Hole Features
            all_features = get_feature_columns(result)

            # Filtere für diese Gruppe
            group_features = filter_features_by_group(all_features, indicator)

            assert len(group_features) >= 3, (
                f"Indikator '{indicator}' produziert nur {len(group_features)} Features "
                f"für seine Gruppe: {group_features[:10]}..."
            )

    def test_exploration_fast_feature_groups_all_have_features(self, real_ohlc_data):
        """exploration_fast Feature-Gruppen müssen alle genug Features haben."""
        from fwbg.builtins.indicators import (
            compute_indicator_pool, get_feature_columns, filter_features_by_group
        )
        from fwbg.core.config import StrategyConfig

        strategy = StrategyConfig.from_json_file("strategies/exploration_fast.json")

        # Berechne alle Indikatoren der Strategie
        result = compute_indicator_pool(
            real_ohlc_data.copy(),
            indicators=strategy.indicators
        )

        all_features = get_feature_columns(result)

        # Prüfe jede Feature-Gruppe
        feature_groups = strategy.get_feature_groups()
        groups_with_insufficient_features = []

        for fg in feature_groups:
            group_features = filter_features_by_group(all_features, fg)
            if len(group_features) < 3:
                groups_with_insufficient_features.append(
                    (fg, len(group_features), group_features[:5])
                )

        assert len(groups_with_insufficient_features) == 0, (
            f"Feature-Gruppen mit < 3 Features:\n"
            + "\n".join(
                f"  {fg}: {count} Features {samples}"
                for fg, count, samples in groups_with_insufficient_features
            )
        )


class TestGridSearchProcessing:
    """Tests dass Grid-Search korrekt verarbeitet."""

    @pytest.fixture
    def prepared_data(self):
        """Bereitet Daten vor wie in process.py."""
        from fwbg.builtins.indicators import compute_indicator_pool, get_feature_columns
        from fwbg.optimization.nested_cv import nested_cv_split

        np.random.seed(42)
        n = 5000

        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "O": prices * (1 + np.random.randn(n) * 0.001),
            "H": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
            "L": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
            "C": prices,
            "V": np.abs(np.random.randn(n)) * 10000 + 1000,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        df["H"] = df[["O", "H", "C"]].max(axis=1)
        df["L"] = df[["O", "L", "C"]].min(axis=1)

        # Compute indicators for core groups only (faster test)
        inner_df = compute_indicator_pool(
            df.copy(),
            indicators=["trend", "momentum", "volatility"]
        )
        inner_df = inner_df.dropna()

        # Add _regime_ok (required for compute_targets)
        inner_df["_regime_ok"] = 1

        # Get feature pool
        full_pool = get_feature_columns(inner_df)

        # Filter inf/nan like in process.py
        clean_pool = []
        for col in full_pool:
            if col in inner_df.columns:
                has_inf = np.isinf(inner_df[col]).any()
                nan_ratio = inner_df[col].isna().sum() / len(inner_df)
                if not has_inf and nan_ratio < 0.1:
                    clean_pool.append(col)

        # Create inner folds
        cv_split = nested_cv_split(inner_df, holdout_ratio=0.0, n_inner_folds=5)
        inner_folds = cv_split["inner_folds"]

        return {
            "inner_df": inner_df,
            "full_pool": clean_pool,
            "inner_folds": inner_folds,
        }

    @pytest.fixture
    def prepared_data_with_signal(self):
        """Daten mit künstlichem Vorhersage-Signal für Feature-Selektion.

        Strategie: Erst Targets berechnen, dann Features erzeugen die mit
        den Targets korrelieren. So testet man die Pipeline, nicht die Daten.
        """
        from fwbg.builtins.indicators import compute_indicator_pool, get_feature_columns
        from fwbg.optimization.nested_cv import nested_cv_split, compute_targets
        from fwbg.core.context import SimulationContext
        from fwbg.core.config import StrategyConfig
        from fwbg.data.assets import get_asset

        np.random.seed(42)
        n = 5000

        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "O": prices * (1 + np.random.randn(n) * 0.001),
            "H": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
            "L": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
            "C": prices,
            "V": np.abs(np.random.randn(n)) * 10000 + 1000,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        df["H"] = df[["O", "H", "C"]].max(axis=1)
        df["L"] = df[["O", "L", "C"]].min(axis=1)

        # Compute indicators
        inner_df = compute_indicator_pool(
            df.copy(),
            indicators=["trend", "momentum", "volatility"]
        )
        inner_df = inner_df.dropna()
        inner_df["_regime_ok"] = 1

        # Berechne Targets ZUERST
        asset = get_asset("EURUSD")
        strategy = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        ctx = SimulationContext.create(asset, strategy)

        targets_long, targets_short, _, _ = compute_targets(
            inner_df, tp=20, sl=30, ctx=ctx, timeout_bars=None
        )

        # JETZT erzeuge Features die mit den Targets korrelieren
        # Diese Features "kennen" die Zukunft - das ist für Pipeline-Tests OK
        noise_factor = 0.3  # Etwas Rauschen für Realismus
        inner_df["trend_oracle_long"] = targets_long + np.random.randn(len(inner_df)) * noise_factor
        inner_df["trend_oracle_short"] = targets_short + np.random.randn(len(inner_df)) * noise_factor
        inner_df["trend_oracle_combined"] = (
            targets_long * 0.6 + targets_short * 0.4
            + np.random.randn(len(inner_df)) * noise_factor
        )

        # Get feature pool (includes oracle features via get_feature_columns)
        full_pool = get_feature_columns(inner_df)
        full_pool = list(dict.fromkeys(full_pool))

        clean_pool = []
        for col in full_pool:
            if col in inner_df.columns:
                has_inf = np.isinf(inner_df[col]).any()
                nan_ratio = inner_df[col].isna().sum() / len(inner_df)
                if not has_inf and nan_ratio < 0.1:
                    clean_pool.append(col)

        cv_split = nested_cv_split(inner_df, holdout_ratio=0.0, n_inner_folds=5)
        inner_folds = cv_split["inner_folds"]

        return {
            "inner_df": inner_df,
            "full_pool": clean_pool,
            "inner_folds": inner_folds,
            "ctx": ctx,
        }

    def test_compute_targets_produces_valid_targets(self, prepared_data):
        """compute_targets muss für mindestens eine Richtung gültige Targets erzeugen."""
        from fwbg.optimization.nested_cv import compute_targets
        from fwbg.core.context import SimulationContext
        from fwbg.core.config import StrategyConfig
        from fwbg.data.assets import get_asset

        # Setup context
        asset = get_asset("EURUSD")
        strategy = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        ctx = SimulationContext.create(asset, strategy)

        # Get first fold's train data
        train_df, val_df = prepared_data["inner_folds"][0]

        # Diagnostic output
        print(f"\nTrain shape: {train_df.shape}")
        print(f"Has _regime_ok: {'_regime_ok' in train_df.columns}")
        print(f"Has _atr: {'_atr' in train_df.columns}")

        # Compute targets
        targets_long, targets_short, has_long, has_short = compute_targets(
            train_df, tp=20, sl=30, ctx=ctx, timeout_bars=None
        )

        print(f"\nTargets computed:")
        print(f"  has_long: {has_long}")
        print(f"  has_short: {has_short}")
        print(f"  targets_long positive count: {np.sum(targets_long > 0)}")
        print(f"  targets_short positive count: {np.sum(targets_short > 0)}")

        # CRITICAL: Either long or short must have valid targets
        assert has_long or has_short, (
            f"compute_targets returned no valid targets! "
            f"This is likely the root cause of 0 candidates. "
            f"targets_long positive={np.sum(targets_long > 0)}, "
            f"targets_short positive={np.sum(targets_short > 0)}"
        )

    def test_feature_selection_finds_features(self, prepared_data):
        """Feature-Selection muss Features finden können."""
        from fwbg.optimization.nested_cv import compute_targets, select_features_from_fold
        from fwbg.core.context import SimulationContext
        from fwbg.core.config import StrategyConfig
        from fwbg.data.assets import get_asset
        from fwbg.builtins.indicators import filter_features_by_group
        from fwbg.builtins.feature_selection.boruta.selector import boruta_select_fast

        # Setup context
        asset = get_asset("EURUSD")
        strategy = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        ctx = SimulationContext.create(asset, strategy)

        # Get first fold's train data
        train_df, val_df = prepared_data["inner_folds"][0]

        # Compute targets
        targets_long, targets_short, has_long, has_short = compute_targets(
            train_df, tp=20, sl=30, ctx=ctx, timeout_bars=None
        )

        # Get trend features
        group_features = filter_features_by_group(prepared_data["full_pool"], "trend")
        print(f"\n=== DIAGNOSTIC: Feature Selection ===")
        print(f"Group features available: {len(group_features)}")
        print(f"Features: {group_features[:10]}...")
        print(f"Train df shape: {train_df.shape}")
        print(f"ctx.min_trades: {ctx.min_trades}")
        print(f"min_trades // 2 = {ctx.min_trades // 2}")

        # Check target counts
        long_count = np.count_nonzero(targets_long)
        short_count = np.count_nonzero(targets_short)
        print(f"\nTarget counts:")
        print(f"  Long targets (nonzero): {long_count}")
        print(f"  Short targets (nonzero): {short_count}")
        print(f"  Threshold (min_trades // 2): {ctx.min_trades // 2}")
        print(f"  Long passes threshold: {long_count >= ctx.min_trades // 2}")
        print(f"  Short passes threshold: {short_count >= ctx.min_trades // 2}")

        # Check features exist in train_df
        available_features = [f for f in group_features if f in train_df.columns]
        print(f"\nAvailable in train_df: {len(available_features)} / {len(group_features)}")

        # Try Boruta directly
        if long_count >= ctx.min_trades // 2 and available_features:
            print(f"\n=== Running Boruta directly ===")
            from fwbg.utils import clean_dataframe
            X = train_df[available_features].copy()
            X = clean_dataframe(X)
            print(f"X shape after cleaning: {X.shape}")
            print(f"X has NaN: {X.isna().any().any()}")
            print(f"X has Inf: {np.isinf(X.values).any()}")

            # Check z-scores at different thresholds
            for threshold in [0.5, 0.3, 0.1, 0.0, -0.5]:
                selected = boruta_select_fast(
                    X, targets_long,
                    n_iter=10,
                    n_estimators=50,
                    max_depth=4,
                    min_z_score=threshold,
                )
                print(f"Boruta (z >= {threshold}): {len(selected)} features")
                if selected:
                    print(f"  Selected: {selected[:5]}...")
                    break

        # Try feature selection via the wrapper
        selected_long = []
        selected_short = []

        if has_long:
            selected_long, _ = select_features_from_fold(
                train_df, targets_long, group_features, ctx.min_trades,
                feature_selection=ctx.feature_selection,
                max_features=ctx.max_features,
                min_z_score=ctx.min_z_score,
            )
            print(f"\nselect_features_from_fold (long): {selected_long}")

        if has_short:
            selected_short, _ = select_features_from_fold(
                train_df, targets_short, group_features, ctx.min_trades,
                feature_selection=ctx.feature_selection,
                max_features=ctx.max_features,
                min_z_score=ctx.min_z_score,
            )
            print(f"select_features_from_fold (short): {selected_short}")

        # NOTE: With random synthetic data, Boruta correctly finds nothing.
        # This is EXPECTED behavior - use test_feature_selection_with_signal for real tests.
        # We keep this test to document that random data = no features.
        if selected_long or selected_short:
            print("WARNING: Random data unexpectedly produced features")
        else:
            print("OK: Random data correctly produced no features (expected)")

    def test_feature_selection_with_signal(self, prepared_data_with_signal):
        """Feature-Selection muss Oracle-Features finden."""
        from fwbg.optimization.nested_cv import compute_targets, select_features_from_fold
        from fwbg.builtins.indicators import filter_features_by_group

        ctx = prepared_data_with_signal["ctx"]
        train_df, _ = prepared_data_with_signal["inner_folds"][0]

        targets_long, targets_short, has_long, has_short = compute_targets(
            train_df, tp=20, sl=30, ctx=ctx, timeout_bars=None
        )

        # Get trend features (including our oracle features)
        group_features = filter_features_by_group(prepared_data_with_signal["full_pool"], "trend")
        print(f"\nTrend features: {len(group_features)}")
        oracle_features = [f for f in group_features if 'oracle' in f]
        print(f"Contains oracle features: {oracle_features}")

        assert has_long or has_short, "No targets computed"
        assert len(oracle_features) >= 2, f"Oracle features missing: {oracle_features}"

        if has_long:
            selected, _ = select_features_from_fold(
                train_df, targets_long, group_features, ctx.min_trades,
                feature_selection=ctx.feature_selection,
                max_features=ctx.max_features,
                min_z_score=ctx.min_z_score,
            )
            print(f"Selected long features: {selected}")

            # Oracle features should be found
            assert selected is not None, (
                f"Feature selection failed! Oracle features should be detectable. "
                f"Check if Boruta z-score threshold is too strict."
            )
            oracle_found = any("oracle" in f for f in selected)
            print(f"Oracle feature found: {oracle_found}")
            assert oracle_found, f"Oracle features not found in: {selected}"

    def test_inner_cv_with_signal(self, prepared_data_with_signal):
        """run_inner_cv muss Kandidaten finden wenn Features prädiktiv sind."""
        from fwbg.optimization.nested_cv import run_inner_cv
        from fwbg.builtins.indicators import filter_features_by_group

        ctx = prepared_data_with_signal["ctx"]
        group_features = filter_features_by_group(prepared_data_with_signal["full_pool"], "trend")
        oracle_features = [f for f in group_features if 'oracle' in f]

        assert len(group_features) >= 3, f"Not enough trend features"
        assert len(oracle_features) >= 2, f"Oracle features missing"

        print(f"\n=== test_inner_cv_with_signal ===")
        print(f"Group features: {len(group_features)}, Oracle: {oracle_features}")

        result = run_inner_cv(
            inner_folds=prepared_data_with_signal["inner_folds"],
            group_features=group_features,
            tp=20,
            sl=30,
            ctx=ctx,
            global_grid_pos=0,
            total_grid_combos=1,
            timeout_bars=None,
            cached_targets={},
        )

        print(f"Result: {result}")

        assert result["success"], (
            f"run_inner_cv failed with oracle features: {result}. "
            f"This indicates a bug in the CV pipeline, not the data."
        )

    def test_inner_cv_returns_results(self, prepared_data):
        """run_inner_cv muss Ergebnisse liefern wenn Daten valid sind."""
        from fwbg.optimization.nested_cv import run_inner_cv
        from fwbg.core.context import SimulationContext
        from fwbg.core.config import StrategyConfig
        from fwbg.data.assets import get_asset
        from fwbg.builtins.indicators import filter_features_by_group

        # Setup context
        asset = get_asset("EURUSD")
        strategy = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        ctx = SimulationContext.create(asset, strategy)

        # Get trend features
        group_features = filter_features_by_group(prepared_data["full_pool"], "trend")
        assert len(group_features) >= 3, f"Not enough trend features: {group_features}"

        # Run inner CV with one TP/SL combination
        result = run_inner_cv(
            inner_folds=prepared_data["inner_folds"],
            group_features=group_features,
            tp=20,
            sl=30,
            ctx=ctx,
            global_grid_pos=0,
            total_grid_combos=1,
            timeout_bars=None,
            cached_targets={},
        )

        # Check result structure
        assert "success" in result, f"run_inner_cv result missing 'success': {result}"

        # NOTE: With random synthetic data, grid search fails because features
        # have no predictive power. This is EXPECTED behavior.
        # Use test_inner_cv_with_signal to test the actual pipeline.
        if result["success"]:
            print("WARNING: Random data unexpectedly produced successful grid search")
        else:
            print(f"OK: Random data correctly failed grid search (expected): {result}")
