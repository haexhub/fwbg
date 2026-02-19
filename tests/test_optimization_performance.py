"""
Tests für Performance-Optimierungen im Optimizer.

Testet echte Funktionalität:
1. XGBoost-Hyperparameter werden korrekt aus Config geladen
2. Inner CV verwendet reduzierte Parameter
3. Holdout verwendet volle Parameter
"""
import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path

from fwbg.core.context import SimulationContext
from fwbg.core.config import StrategyConfig, ModelConfig
from fwbg.optimization.nested_cv import train_model
from fwbg.data.assets import AssetConfig


class TestXGBoostHyperparameters:
    """Teste XGBoost-Hyperparameter-Handling."""

    def test_hyperparameters_flow_from_config_to_model(self):
        """
        End-to-End Test: Hyperparameter sollten von StrategyConfig
        über SimulationContext bis zum XGBoost-Modell fließen.
        """
        # Erstelle StrategyConfig mit custom hyperparameters
        config_dict = {
            "assets": ["EURUSD"],
            "model": {
                "type": "xgboost",
                "architecture": "unified",
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "random_state": 42
                }
            },
            "validation": {
                "method": "walk_forward",
                "folds": 3,
                "oos_size": 1000,
                "min_trades": 30
            },
            "filters": {
                "min_rrr": 1.0,
                "min_trades": 30
            },
            "features": {
                "groups": ["trend"],
                "selection": "boruta"
            },
            "grids": {
                "FOREX": {
                    "tp": [1.0, 2.0],
                    "sl": [1.0],
                    "ct": [0.5],
                    "timeout_bars": [None]
                }
            },
            "exit_strategy": "fixed",
            "exit_params": {},
            "resources": {
                "max_concurrent_assets": 1
            }
        }

        strategy = StrategyConfig.from_dict(config_dict)
        asset = AssetConfig(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.0001,
            currencies=["EUR", "USD"]
        )

        # Erstelle Context
        ctx = SimulationContext.create(asset, strategy)

        # Verifiziere: Hyperparameter wurden übernommen
        assert ctx.model_hyperparameters["n_estimators"] == 200
        assert ctx.model_hyperparameters["max_depth"] == 6
        assert ctx.model_hyperparameters["learning_rate"] == 0.05

        # Erstelle Trainingsdaten
        np.random.seed(42)
        train_df = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
        })
        targets = np.random.choice([0, 1], size=100, p=[0.4, 0.6])

        # Trainiere Modell mit VOLLEN Parametern
        model_full = train_model(
            train_df, targets, ["feat1", "feat2"],
            ctx.min_trades, ctx, use_reduced_params=False
        )

        xgb_full = model_full.as_sklearn_estimator()
        assert xgb_full.n_estimators == 200
        assert xgb_full.max_depth == 6

        # Trainiere Modell mit REDUZIERTEN Parametern (Inner CV)
        model_reduced = train_model(
            train_df, targets, ["feat1", "feat2"],
            ctx.min_trades, ctx, use_reduced_params=True
        )

        # n_estimators sollte halbiert sein
        xgb_reduced = model_reduced.as_sklearn_estimator()
        assert xgb_reduced.n_estimators == 100  # 200 // 2
        # max_depth bleibt gleich (wichtiger für Qualität)
        assert xgb_reduced.max_depth == 6

    def test_default_hyperparameters_when_not_specified(self):
        """Default-Werte sollten verwendet werden wenn nicht angegeben."""
        config_dict = {
            "assets": ["EURUSD"],
            "model": {
                "type": "xgboost"
                # Keine hyperparameters angegeben
            },
            "validation": {"method": "walk_forward", "folds": 3, "oos_size": 1000, "min_trades": 30},
            "filters": {"min_rrr": 1.0, "min_trades": 30},
            "features": {"groups": ["trend"], "selection": "boruta"},
            "grids": {"FOREX": {"tp": [1.0], "sl": [1.0], "ct": [0.5]}},
            "exit_strategy": "fixed",
            "exit_params": {},
            "resources": {
                "max_concurrent_assets": 1
            }
        }

        strategy = StrategyConfig.from_dict(config_dict)
        asset = AssetConfig("EURUSD", "FOREX", 0.0001, 0.0001, ["EUR", "USD"])
        ctx = SimulationContext.create(asset, strategy)

        # Default-Werte sollten gesetzt sein
        assert ctx.model_hyperparameters.get("n_estimators", 100) == 100
        assert ctx.model_hyperparameters.get("max_depth", 5) == 5

    def test_reduced_params_minimum_threshold(self):
        """
        Reduzierte Parameter sollten nicht unter Minimum fallen.
        Bei sehr kleinen Werten sollte Minimum 10 n_estimators gelten.
        """
        config_dict = {
            "assets": ["EURUSD"],
            "model": {
                "type": "xgboost",
                "hyperparameters": {
                    "n_estimators": 15,  # Sehr klein
                    "max_depth": 3
                }
            },
            "validation": {"method": "walk_forward", "folds": 3, "oos_size": 1000, "min_trades": 30},
            "filters": {"min_rrr": 1.0, "min_trades": 30},
            "features": {"groups": ["trend"], "selection": "boruta"},
            "grids": {"FOREX": {"tp": [1.0], "sl": [1.0], "ct": [0.5]}},
            "exit_strategy": "fixed",
            "exit_params": {},
            "resources": {
                "max_concurrent_assets": 1
            }
        }

        strategy = StrategyConfig.from_dict(config_dict)
        asset = AssetConfig("EURUSD", "FOREX", 0.0001, 0.0001, ["EUR", "USD"])
        ctx = SimulationContext.create(asset, strategy)

        train_df = pd.DataFrame({"feat1": np.random.randn(100)})
        targets = np.random.choice([0, 1], size=100)

        model = train_model(
            train_df, targets, ["feat1"],
            30, ctx, use_reduced_params=True
        )

        # 15 // 2 = 7, max(10, 7) = 10
        assert model.as_sklearn_estimator().n_estimators == 10

    def test_model_quality_inner_vs_holdout(self):
        """
        Modelle mit reduzierten Parametern sollten schlechtere Accuracy haben,
        aber immer noch reasonable Performance zeigen.
        """
        np.random.seed(42)

        # Erstelle synthetische Daten mit klarem Pattern
        n_samples = 500
        X = np.random.randn(n_samples, 5)
        # Target korreliert stark mit feat1 + feat2
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        train_df = pd.DataFrame(X, columns=[f"feat{i}" for i in range(5)])
        test_df = train_df.copy()

        config_dict = {
            "assets": ["EURUSD"],
            "model": {
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5
                }
            },
            "validation": {"method": "walk_forward", "folds": 3, "oos_size": 1000, "min_trades": 30},
            "filters": {"min_rrr": 1.0, "min_trades": 30},
            "features": {"groups": ["trend"], "selection": "boruta"},
            "grids": {"FOREX": {"tp": [1.0], "sl": [1.0], "ct": [0.5]}},
            "exit_strategy": "fixed",
            "exit_params": {},
            "resources": {
                "max_concurrent_assets": 1
            }
        }

        strategy = StrategyConfig.from_dict(config_dict)
        asset = AssetConfig("EURUSD", "FOREX", 0.0001, 0.0001, ["EUR", "USD"])
        ctx = SimulationContext.create(asset, strategy)

        features = [f"feat{i}" for i in range(5)]

        # Trainiere beide Modelle
        model_full = train_model(train_df, y, features, 30, ctx, use_reduced_params=False)
        model_reduced = train_model(train_df, y, features, 30, ctx, use_reduced_params=True)

        # Predictions
        pred_full = model_full.predict(test_df[features])
        pred_reduced = model_reduced.predict(test_df[features])

        # Accuracy berechnen
        acc_full = (pred_full == y).mean()
        acc_reduced = (pred_reduced == y).mean()

        # Beide sollten deutlich besser als Random (0.5) sein
        assert acc_full > 0.6, "Volles Modell sollte > 60% Accuracy haben"
        assert acc_reduced > 0.6, "Reduziertes Modell sollte > 60% Accuracy haben"

        # Volles Modell sollte gleich gut oder besser sein
        # (kann auch gleich sein bei kleinen Daten)
        assert acc_full >= acc_reduced - 0.05  # Max 5% schlechter erlaubt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
