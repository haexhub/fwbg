"""
CRITICAL BIAS DETECTION TESTS

Diese Tests MÜSSEN GRÜN sein, sonst ist etwas systematisch falsch im Code.
Wenn diese Tests fehlschlagen, deutet das auf ein fundamentales Problem hin
(z.B. Lookahead Bias, Feature Leakage, etc.).

Diese Tests sollten im CI/CD laufen und Development BLOCKIEREN wenn sie failen.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fwbg_sdk import shift_features
from fwbg.optimization.targets import compute_targets
from fwbg.optimization.nested_cv import train_model
from fwbg.core.context import SimulationContext
from fwbg.simulation.numba_core import _simulate_trade_numba


class TestNoLookaheadBias:
    """
    KRITISCHE Tests: Stellen sicher dass kein Lookahead Bias existiert.

    Wenn diese Tests fehlschlagen, ist etwas fundamental falsch im Code.
    """

    def test_future_spike_cannot_be_predicted(self):
        """
        KRITISCH: Features dürfen zukünftige Spikes nicht vorhersehen.

        Test erstellt flat data mit einem einzelnen Spike bei Bar 500.
        Features bei Bars VOR dem Spike dürfen den Spike nicht enthalten.

        WENN DIESER TEST FEHLSCHLÄGT: Lookahead Bias im Feature Calculation!
        """
        np.random.seed(42)
        n = 1000
        prices = np.ones(n) * 100.0
        prices = prices + np.random.randn(n) * 0.1

        # Spike at bar 500
        spike_bar = 500
        prices[spike_bar] = 200.0  # 100% spike

        df = pd.DataFrame({
            'O': prices,
            'H': prices * 1.001,
            'L': prices * 0.999,
            'C': prices,
        })

        # Calculate rolling max (should NOT see future spike)
        features = {'rolling_max_50': df['C'].rolling(50).max()}
        features_df = shift_features(features, df.index)
        df = pd.concat([df, features_df], axis=1)

        # Check bars BEFORE spike
        bars_before = [spike_bar - 1, spike_bar - 2, spike_bar - 5, spike_bar - 10]

        for bar in bars_before:
            feat_val = df.loc[bar, 'rolling_max_50']
            if pd.notna(feat_val) and feat_val > 150:
                pytest.fail(
                    f"LOOKAHEAD BIAS DETECTED! "
                    f"Feature at bar {bar} (before spike at {spike_bar}) "
                    f"has value {feat_val:.2f} which includes the spike (200.0). "
                    f"This means features can see the future!"
                )

    def test_random_features_give_random_performance(self):
        """
        KRITISCH: Rein zufällige Features sollten ~50% Win-Rate haben.

        Wenn zufällige Features >65% Win-Rate erreichen, gibt es Lookahead Bias.

        WENN DIESER TEST FEHLSCHLÄGT: Systematischer Bias im Training/Evaluation!
        """
        np.random.seed(42)
        n = 10000

        # Random walk prices
        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'O': prices,
            'H': prices * 1.01,
            'L': prices * 0.99,
            'C': prices,
            '_atr': np.ones(n) * 2.0,
            '_regime': np.full(n, 7, dtype=np.int8),
        })

        # Create RANDOM features (should have NO predictive power)
        num_features = 10
        for i in range(num_features):
            feature_values = np.random.randn(n)
            features = {f'random_{i}': pd.Series(feature_values, index=df.index)}
            features_df = shift_features(features, df.index)
            df = pd.concat([df, features_df], axis=1)

        df = df.dropna()

        # Split data
        train_size = int(len(df) * 0.7)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # Setup context
        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            point=0.0001,
            spread=0.0,
            model_hyperparameters={
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
            }
        )

        # Compute targets
        targets_long, _, _, _ = compute_targets(
            train_df, tp=10, sl=20, ctx=ctx, timeout_bars=None
        )

        # Train model on random features
        random_features = [f'random_{i}' for i in range(num_features)]
        model = train_model(
            train_df, targets_long, random_features,
            min_trades=10, ctx=ctx, use_reduced_params=False
        )

        if model is None:
            # If model couldn't train, that's OK (not enough trades)
            return

        # Test on holdout
        X_test = test_df[random_features]
        y_pred_proba = model.predict_probability(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.6).astype(int)

        test_targets, _, _, _ = compute_targets(
            test_df, tp=10, sl=20, ctx=ctx, timeout_bars=None
        )

        # Check win-rate on predicted trades
        predicted_trades = y_pred == 1
        if predicted_trades.sum() > 30:  # Need enough trades
            y_true = pd.Series(test_targets, index=test_df.index).loc[X_test.index]
            win_rate = y_true[predicted_trades].mean()

            # Random features should give ~50% win-rate (±15% tolerance)
            if win_rate > 0.65:
                pytest.fail(
                    f"SYSTEMATIC BIAS DETECTED! "
                    f"Random features achieved {win_rate*100:.1f}% win-rate "
                    f"(expected ~50%). This indicates lookahead bias or data leakage."
                )

    def test_feature_target_timing_alignment(self):
        """
        KRITISCH: Features[i] dürfen nur Daten bis Bar i-1 enthalten.

        Target[i] ist das Ergebnis eines Trades der bei Bar i+1 startet.
        Feature[i] darf KEINE Informationen von Bar i oder später enthalten.

        WENN DIESER TEST FEHLSCHLÄGT: Timing-Fehler in Feature Calculation!
        """
        np.random.seed(42)
        n = 1000

        # Create uptrend then downtrend
        uptrend = np.linspace(100, 150, 500)
        downtrend = np.linspace(150, 100, 500)
        prices = np.concatenate([uptrend, downtrend])
        prices += np.random.randn(n) * 0.5

        df = pd.DataFrame({
            'O': prices,
            'H': prices * 1.01,
            'L': prices * 0.99,
            'C': prices,
        })

        # Calculate SMA and price_vs_sma
        sma_20 = df['C'].rolling(20).mean()
        features = {
            'sma_20': sma_20,
            'price_vs_sma': df['C'] - sma_20
        }

        features_df = shift_features(features, df.index)
        df = pd.concat([df, features_df], axis=1)
        df = df.dropna()

        # Test specific bar
        test_bar = 100

        # Feature at bar 100 should use data from bar 99 and earlier
        # NOT from bar 100 or later
        feature_val = df.loc[test_bar, 'price_vs_sma']

        # Manually calculate what it SHOULD be
        # SMA calculated at bar 99, then shifted to bar 100
        window_end = test_bar - 1  # Bar 99
        window_start = window_end - 20 + 1  # 20-bar window

        if window_start >= 0:
            expected_sma = df['C'].iloc[window_start:window_end+1].mean()
            price_at_99 = df.loc[test_bar - 1, 'C']
            expected_feature = price_at_99 - expected_sma

            # Check if feature used bar 100's price (lookahead!)
            price_at_100 = df.loc[test_bar, 'C']
            if abs(feature_val - (price_at_100 - expected_sma)) < 0.01:
                pytest.fail(
                    f"LOOKAHEAD BIAS DETECTED! "
                    f"Feature at bar {test_bar} uses price from bar {test_bar}, "
                    f"but should only use data up to bar {test_bar-1}!"
                )


class TestSystematicBiasInResults:
    """
    Tests die auf systematischen Bias in aktuellen Results prüfen.

    Diese Tests verwenden die AKTUELLEN Optimization Results und prüfen
    ob systematischer Bias vorliegt (z.B. alle Assets zu gut).

    WENN DIESE TESTS FEHLSCHLAGEN: Code hat systematisches Problem!
    """

    @pytest.fixture
    def latest_results_dir(self):
        """Find latest test results directory."""
        results_base = Path("/home/haex/Projekte/fwbg/test_results")
        if not results_base.exists():
            pytest.skip("No test results directory found")

        result_dirs = sorted(results_base.glob("*"))
        if not result_dirs:
            pytest.skip("No test results found")

        latest = result_dirs[-1]
        grid_details = latest / "grid_details"

        if not grid_details.exists():
            pytest.skip("No grid_details found in latest results")

        return grid_details

    def test_no_systematic_sample_bias_in_results(self, latest_results_dir):
        """
        KRITISCH: Prüft ob aktuelle Results systematischen Sample Bias zeigen.

        Checkt ob MEAN bias ratio über alle folds <1.5 ist.
        Alle Results MÜSSEN walk-forward sein, sonst FAIL.

        WENN DIESER TEST FEHLSCHLÄGT: Code hat systematischen Bias!
        """
        import json

        sym_dirs = [d for d in latest_results_dir.iterdir() if d.is_dir()]
        if len(sym_dirs) < 3:
            pytest.skip("Not enough results to test (need at least 3 assets)")

        biased_assets = []
        total_assets = 0
        non_walk_forward = []

        # Statuses where no walk-forward result is expected (strategy found no edge / no data)
        _SKIP_STATUSES = {
            'no_successful_folds', 'no_data', 'insufficient_data',
            'insufficient_data_for_folds', 'macro_asset', 'error',
        }

        for sym_dir in sym_dirs:
            cfg_file = sym_dir / "config.json"
            fold_file = sym_dir / "fold_results.json"
            if not cfg_file.exists():
                continue

            with open(cfg_file) as f:
                data = json.load(f)

            # Assets with no successful folds are valid optimizer outcomes, not legacy format
            if data.get('status') in _SKIP_STATUSES:
                continue

            if fold_file.exists():
                with open(fold_file) as f:
                    fold_data = json.load(f)
                wf_results = fold_data.get('walk_forward', {})
            else:
                wf_results = {}

            if not wf_results or 'mean_bias_ratio' not in wf_results:
                # Results müssen walk-forward sein!
                non_walk_forward.append(data['symbol'])
                continue

            # Check mean bias ratio
            mean_bias = wf_results['mean_bias_ratio']
            total_assets += 1

            if mean_bias > 1.5:
                biased_assets.append({
                    'symbol': data['symbol'],
                    'mean_bias_ratio': mean_bias,
                    'n_folds': wf_results.get('n_folds', 0),
                    'bias_ratios': wf_results.get('bias_ratios', []),
                })

        if non_walk_forward:
            pytest.fail(
                f"NON-WALK-FORWARD RESULTS DETECTED!\n"
                f"{len(non_walk_forward)} assets don't have walk-forward results:\n"
                + "\n".join(f"  - {s}" for s in non_walk_forward) + "\n\n"
                f"All results MUST use walk-forward validation. "
                f"Legacy single holdout is NOT supported."
            )

        if total_assets == 0:
            pytest.skip("No valid assets to analyze")

        bias_percentage = len(biased_assets) / total_assets

        # Walk-forward threshold: 20%
        if bias_percentage > 0.2:
            bias_details = "\n".join([
                f"  - {a['symbol']}: Mean bias {a['mean_bias_ratio']:.2f}x "
                f"({a['n_folds']} folds, ratios: {[f'{r:.2f}' for r in a['bias_ratios']]})"
                for a in biased_assets
            ])
            pytest.fail(
                f"SYSTEMATIC SAMPLE BIAS IN WALK-FORWARD RESULTS!\n"
                f"{len(biased_assets)}/{total_assets} assets ({bias_percentage*100:.0f}%) "
                f"have mean bias ratio >1.5x across folds.\n\n"
                f"Biased assets:\n{bias_details}\n\n"
                f"Walk-forward should prevent this! Check for:\n"
                f"  1. Lookahead bias in feature calculation\n"
                f"  2. Data leakage in preprocessing\n"
                f"  3. Walk-forward implementation bugs"
            )

    def test_no_unrealistic_winrates_in_results(self, latest_results_dir):
        """
        KRITISCH: Prüft ob Win-Rates realistisch sind für gegebenes RRR.

        Wenn MEHR ALS 50% der Assets eine Win-Rate haben die >15% über
        dem Break-Even liegt, ist das verdächtig.

        WENN DIESER TEST FEHLSCHLÄGT: Code hat systematischen Bias!
        """
        import json

        sym_dirs = [d for d in latest_results_dir.iterdir() if d.is_dir()]
        if len(sym_dirs) < 3:
            pytest.skip("Not enough results to test")

        unrealistic_assets = []
        total_assets = 0

        for sym_dir in sym_dirs:
            cfg_file = sym_dir / "config.json"
            if not cfg_file.exists():
                continue
            with open(cfg_file) as f:
                data = json.load(f)

            holdout_metrics = data.get('holdout_metrics', {})
            selected_config = data.get('selected_config', {})

            win_rate = holdout_metrics.get('win_rate', 0)
            rrr = holdout_metrics.get('rrr', 0)
            trades = holdout_metrics.get('trades', 0)

            if trades > 50 and rrr > 0:  # Need reasonable sample size
                breakeven_wr = 1.0 / (1.0 + rrr)
                excess = win_rate - breakeven_wr

                total_assets += 1

                if excess > 0.15:  # More than 15% above breakeven
                    unrealistic_assets.append({
                        'symbol': data['symbol'],
                        'win_rate': win_rate,
                        'breakeven': breakeven_wr,
                        'excess': excess,
                        'rrr': rrr,
                        'trades': trades,
                    })

        if total_assets == 0:
            pytest.skip("No valid assets to analyze")

        unrealistic_percentage = len(unrealistic_assets) / total_assets

        if unrealistic_percentage > 0.5:
            details = "\n".join([
                f"  - {a['symbol']}: WR={a['win_rate']*100:.1f}% "
                f"(breakeven={a['breakeven']*100:.1f}%, excess=+{a['excess']*100:.1f}%, "
                f"RRR={a['rrr']:.2f}, trades={a['trades']})"
                for a in unrealistic_assets
            ])

            pytest.fail(
                f"SYSTEMATIC UNREALISTIC WIN-RATES DETECTED!\n"
                f"{len(unrealistic_assets)}/{total_assets} assets ({unrealistic_percentage*100:.0f}%) "
                f"have win-rates >15% above break-even.\n"
                f"This is statistically implausible and indicates systematic bias.\n\n"
                f"Unrealistic assets:\n{details}\n\n"
                f"ACTION REQUIRED: Investigate code for bias sources."
            )


    def test_walk_forward_implementation(self, latest_results_dir):
        """
        KRITISCH: Prüft ob Walk-Forward korrekt implementiert ist.

        Verifiziert dass:
        1. ALLE assets walk-forward verwenden
        2. Multiple folds verwendet werden (>=5)
        3. Bias ratios über folds variieren (nicht alle gleich)
        4. Aggregierte Metriken vorhanden sind

        WENN DIESER TEST FEHLSCHLÄGT: Walk-Forward nicht korrekt!
        """
        import json

        sym_dirs = [d for d in latest_results_dir.iterdir() if d.is_dir()]
        if len(sym_dirs) == 0:
            pytest.skip("No results found")

        walk_forward_assets = []
        non_walk_forward = []
        skipped_assets = []  # Assets with no_candidates or error status

        for sym_dir in sym_dirs:
            cfg_file = sym_dir / "config.json"
            fold_file = sym_dir / "fold_results.json"
            if not cfg_file.exists():
                continue
            with open(cfg_file) as f:
                data = json.load(f)

            # Skip assets that had no candidates or errors - they can't have walk-forward data
            status = data.get('status', '')
            _NO_WF_STATUSES = {
                'no_candidates', 'error', 'skipped', 'no_successful_folds',
                'no_data', 'insufficient_data', 'insufficient_data_for_folds',
                'macro_asset',
            }
            if status in _NO_WF_STATUSES:
                skipped_assets.append(data['symbol'])
                continue

            if fold_file.exists():
                with open(fold_file) as f:
                    fold_data = json.load(f)
                wf_results = fold_data.get('walk_forward', {})
            else:
                wf_results = {}

            if wf_results and 'n_folds' in wf_results:
                walk_forward_assets.append({
                    'symbol': data['symbol'],
                    'n_folds': wf_results['n_folds'],
                    'bias_ratios': wf_results.get('bias_ratios', []),
                    'mean_wr': wf_results.get('mean_win_rate', 0),
                    'std_wr': wf_results.get('std_win_rate', 0),
                })
            else:
                non_walk_forward.append(data['symbol'])

        # FAIL if ANY asset doesn't use walk-forward
        if non_walk_forward:
            pytest.fail(
                f"NON-WALK-FORWARD ASSETS DETECTED!\n"
                f"{len(non_walk_forward)} assets don't use walk-forward:\n"
                + "\n".join(f"  - {s}" for s in non_walk_forward) + "\n\n"
                f"ALL assets MUST use walk-forward validation."
            )

        # Check each walk-forward asset for implementation issues
        issues = []

        for asset in walk_forward_assets:
            symbol = asset['symbol']
            n_folds = asset['n_folds']
            bias_ratios = asset['bias_ratios']

            # Check 1: Minimum 5 folds
            if n_folds < 5:
                issues.append(f"{symbol}: Only {n_folds} folds (need >=5)")

            # Check 2: Bias ratios should vary (not all identical)
            if len(bias_ratios) >= 2:
                if len(set(bias_ratios)) == 1:
                    issues.append(f"{symbol}: All bias ratios identical ({bias_ratios[0]:.2f}x) - suspicious!")

            # Check 3: Aggregated metrics present
            if asset['std_wr'] == 0 and n_folds > 1:
                issues.append(f"{symbol}: Std WR is 0 with {n_folds} folds - aggregation broken?")

        if issues:
            pytest.fail(
                f"WALK-FORWARD IMPLEMENTATION ISSUES DETECTED!\n\n"
                f"Problems found:\n" + "\n".join(f"  - {i}" for i in issues)
            )


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short", "-x"])
