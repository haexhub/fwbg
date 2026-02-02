"""
Integration-Tests für die CLI.

Testet:
- CLI-Argument-Parsing
- Strategy-Loading
- Account/Timeframe-Overrides
- Feature-Groups Listing
- Runs Listing
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set PYTHONPATH before imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCLIArgumentParsing:
    """Tests für CLI-Argument-Parsing."""

    def test_help_command(self):
        """Test: --help zeigt Hilfe."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "fwbg.cli", "--help"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        assert result.returncode == 0
        assert "FWBG Strategy Backtester" in result.stdout

    def test_list_features_command(self):
        """Test: --list-features zeigt Feature-Gruppen."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "fwbg.cli", "--list-features"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        assert result.returncode == 0
        assert "FEATURE-GRUPPEN" in result.stdout
        assert "trend" in result.stdout
        assert "momentum" in result.stdout

    def test_list_runs_command(self):
        """Test: --list zeigt vorhandene Runs oder 'keine gefunden' Meldung."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "fwbg.cli", "--list"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        assert result.returncode == 0
        # Entweder gibt es Runs oder die "keine gefunden" Meldung
        assert "TEST-RUNS" in result.stdout or "Keine Test-Runs" in result.stdout


class TestStrategyLoading:
    """Tests für Strategy-Loading aus JSON."""

    def test_load_valid_strategy(self):
        """Test: Gültige Strategy-Datei laden."""
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "TestStrategy",
            "indicators": ["trend", "momentum"],
            "grids": {
                "FOREX": {
                    "tp": [1.5, 2.0],
                    "sl": [1.0],
                    "ct": [0.55]
                }
            }
        }

        config = StrategyConfig.from_dict(strategy_data)
        assert config.name == "TestStrategy"
        assert "trend" in config.indicators
        assert "momentum" in config.indicators
        forex_grid = config.get_grid_for_class("FOREX")
        assert 1.5 in forex_grid.tp
        assert 2.0 in forex_grid.tp

    def test_load_strategy_with_account(self):
        """Test: Strategy mit Account-Konfiguration."""
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "TestWithAccount",
            "version": "1.0",
            "account": "custom_account",
            "timeframe": "MINUTE_15",
            "indicators": ["trend"],
            "grid": {"tp": [2.0], "sl": [1.0], "ct": [0.5]}
        }

        config = StrategyConfig.from_dict(strategy_data)
        # Account wird nicht in StrategyConfig gespeichert, sondern separat behandelt
        assert config.name == "TestWithAccount"

    def test_load_strategy_with_preprocessing(self):
        """Test: Strategy mit Preprocessing-Plugins."""
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "TestPreprocessing",
            "version": "1.0",
            "preprocessing": ["fractional_diff"],
            "preprocessing_params": {
                "fractional_diff": {"d": 0.4}
            },
            "indicators": ["trend"],
            "grid": {"tp": [2.0], "sl": [1.0], "ct": [0.5]}
        }

        config = StrategyConfig.from_dict(strategy_data)
        assert config.preprocessing == ["fractional_diff"]
        assert config.preprocessing_params.get("fractional_diff") == {"d": 0.4}

    def test_load_invalid_json(self):
        """Test: Ungültige JSON-Daten."""
        from fwbg.core.config import StrategyConfig

        # from_dict erwartet ein dict - ungültige Eingabe sollte TypeError werfen
        with pytest.raises((TypeError, AttributeError)):
            StrategyConfig.from_dict("not a dict")

    def test_load_empty_dict(self):
        """Test: Leeres Dictionary - verwendet Defaults."""
        from fwbg.core.config import StrategyConfig

        # Sollte mit Defaults funktionieren
        config = StrategyConfig.from_dict({})
        assert config.name == "Default Strategy"
        assert config.exit_strategy == "atr_based"
        assert config.preprocessing == []


class TestAccountTimeframeOverride:
    """Tests für Account/Timeframe CLI-Overrides."""

    def test_timeframe_override(self):
        """Test: --timeframe überschreibt Default."""
        import fwbg.data.config as data_config

        original_tf = data_config.TIMEFRAME

        # Simuliere CLI-Override
        data_config.TIMEFRAME = "MINUTE_15"
        tf_cfg = data_config.TIMEFRAME_CONFIG.get("MINUTE_15")
        data_config.OOS_SIZE = tf_cfg["oos_size"]

        assert data_config.TIMEFRAME == "MINUTE_15"
        assert data_config.OOS_SIZE == 8000

        # Restore
        data_config.TIMEFRAME = original_tf
        tf_cfg = data_config.TIMEFRAME_CONFIG.get(original_tf, data_config.TIMEFRAME_CONFIG["HOUR"])
        data_config.OOS_SIZE = tf_cfg["oos_size"]


class TestPreprocessingPluginFormat:
    """Tests für das neue Preprocessing-Plugin-Format."""

    def test_preprocessing_is_list(self):
        """Test: preprocessing sollte Liste von Plugin-Namen sein."""
        from fwbg.core.config import StrategyConfig

        config = StrategyConfig.from_dict({
            "name": "Test",
            "version": "1.0",
            "preprocessing": ["fractional_diff", "normalize"],
            "preprocessing_params": {"fractional_diff": {"d": 0.5}}
        })

        assert isinstance(config.preprocessing, list)
        assert "fractional_diff" in config.preprocessing
        assert "normalize" in config.preprocessing

    def test_empty_preprocessing(self):
        """Test: Leere Preprocessing-Liste."""
        from fwbg.core.config import StrategyConfig

        config = StrategyConfig.from_dict({"name": "Test", "version": "1.0"})

        assert config.preprocessing == []
        assert not config.preprocessing  # Evaluiert zu False


class TestFeatureGroups:
    """Tests für Feature-Groups."""

    def test_feature_groups_structure(self):
        """Test: FEATURE_GROUPS hat korrekte Struktur."""
        from fwbg.builtins.indicators import FEATURE_GROUPS

        assert "trend" in FEATURE_GROUPS
        assert "momentum" in FEATURE_GROUPS
        assert "volatility" in FEATURE_GROUPS

        for name, group in FEATURE_GROUPS.items():
            assert "name" in group, f"{name} fehlt 'name'"
            assert "prefixes" in group, f"{name} fehlt 'prefixes'"
            assert isinstance(group["prefixes"], list), f"{name} prefixes ist keine Liste"

    def test_filter_features_by_group(self):
        """Test: filter_features_by_group funktioniert."""
        from fwbg.builtins.indicators import filter_features_by_group

        all_features = [
            "trend_adx_14", "trend_ema_20",
            "mom_rsi_14", "mom_stoch_14",
            "vol_atr_14", "vol_bb_upper"
        ]

        trend_features = filter_features_by_group(all_features, "trend")
        assert "trend_adx_14" in trend_features
        assert "trend_ema_20" in trend_features
        assert "mom_rsi_14" not in trend_features

        mom_features = filter_features_by_group(all_features, "momentum")
        assert "mom_rsi_14" in mom_features
        assert "trend_adx_14" not in mom_features

    def test_unknown_group_returns_all(self):
        """Test: Unbekannte Gruppe gibt alle Features zurück."""
        from fwbg.builtins.indicators import filter_features_by_group

        all_features = ["trend_adx_14", "mom_rsi_14"]
        result = filter_features_by_group(all_features, "nonexistent_group")

        assert result == all_features


class TestComputeIndicatorPool:
    """Tests für compute_indicator_pool."""

    def test_compute_basic(self):
        """Test: Grundlegende Indikator-Berechnung."""
        import pandas as pd
        import numpy as np
        from fwbg.builtins.indicators import compute_indicator_pool

        # Erstelle Test-DataFrame
        n = 500
        df = pd.DataFrame({
            "O": np.random.randn(n).cumsum() + 100,
            "H": np.random.randn(n).cumsum() + 101,
            "L": np.random.randn(n).cumsum() + 99,
            "C": np.random.randn(n).cumsum() + 100,
        })
        df["H"] = df[["O", "H", "C"]].max(axis=1) + 0.1
        df["L"] = df[["O", "L", "C"]].min(axis=1) - 0.1
        original_cols = len(df.columns)

        result = compute_indicator_pool(df)

        # compute_indicator_pool gibt den DataFrame MIT Indikatoren zurück
        # Es sollte mindestens Indikator-Spalten hinzugefügt haben
        indicator_cols = [c for c in result.columns if c not in ["O", "H", "L", "C"]]
        assert len(indicator_cols) > 0, "Keine Indikator-Spalten hinzugefügt"
        assert "trend_adx_14" in result.columns or any("trend" in c for c in result.columns)

    def test_compute_no_symbol_arg(self):
        """Test: compute_indicator_pool akzeptiert kein symbol Argument."""
        import pandas as pd
        import numpy as np
        from fwbg.builtins.indicators import compute_indicator_pool
        import inspect

        sig = inspect.signature(compute_indicator_pool)
        param_names = list(sig.parameters.keys())

        assert "symbol" not in param_names, \
            "compute_indicator_pool sollte kein 'symbol' Argument haben"


class TestExitStrategies:
    """Tests für Exit-Strategies."""

    def test_get_strategy(self):
        """Test: get_strategy gibt korrekte Klasse zurück."""
        from fwbg.builtins.exit_strategies import get_strategy

        atr_strategy = get_strategy("atr_based")
        assert atr_strategy is not None

        fixed_strategy = get_strategy("fixed")
        assert fixed_strategy is not None

    def test_get_nonexistent_strategy(self):
        """Test: get_strategy mit unbekanntem Namen wirft ValueError."""
        from fwbg.builtins.exit_strategies import get_strategy

        with pytest.raises(ValueError):
            get_strategy("nonexistent_strategy")

    def test_grid_params(self):
        """Test: GridParams Klasse."""
        from fwbg.builtins.exit_strategies.base import GridParams

        params = GridParams(tp_value=2.0, sl_value=1.0)

        assert params.tp_value == 2.0
        assert params.sl_value == 1.0
        assert params.rrr == 2.0

    def test_grid_params_to_dict(self):
        """Test: GridParams.to_dict()."""
        from fwbg.builtins.exit_strategies.base import GridParams

        params = GridParams(tp_value=2.0, sl_value=1.0, timeout_bars=50)
        d = params.to_dict()

        assert d["tp_mult"] == 2.0
        assert d["sl_mult"] == 1.0
        assert d["timeout_bars"] == 50


class TestAssetRegistry:
    """Tests für AssetRegistry."""

    def test_get_asset(self):
        """Test: get_asset gibt korrekte Config zurück."""
        from fwbg.data.assets import get_asset

        eurusd = get_asset("EURUSD")

        assert eurusd.symbol == "EURUSD"
        assert eurusd.asset_class == "FOREX"
        assert eurusd.point == 0.0001

    def test_get_unknown_asset(self):
        """Test: Unbekanntes Asset gibt Default zurück."""
        from fwbg.data.assets import get_asset

        unknown = get_asset("UNKNOWN_XYZ")

        assert unknown.symbol == "UNKNOWN_XYZ"
        assert unknown.asset_class == "FOREX"  # Default

    def test_symbols_by_class(self):
        """Test: symbols_by_class filtert korrekt."""
        from fwbg.data.assets import AssetRegistry

        registry = AssetRegistry()

        forex = registry.symbols_by_class("FOREX")
        assert "EURUSD" in forex
        assert "GBPUSD" in forex

        commodities = registry.symbols_by_class("COMMODITY")
        assert "XAUUSD" in commodities or "GOLD" in commodities


class TestDataConfig:
    """Tests für data/config.py."""

    def test_config_values_exist(self):
        """Test: Alle wichtigen Config-Werte existieren."""
        from fwbg.data.config import (
            DATA_PATH, TIMEFRAME, OOS_SIZE,
            WALK_FORWARD_FOLDS, CORR_THRESHOLD, MIN_TRADES
        )

        assert DATA_PATH is not None
        assert TIMEFRAME is not None
        assert OOS_SIZE > 0
        assert WALK_FORWARD_FOLDS > 0
        assert 0 < CORR_THRESHOLD < 1
        assert MIN_TRADES > 0

    def test_timeframe_config(self):
        """Test: TIMEFRAME_CONFIG hat alle Timeframes."""
        from fwbg.data.config import TIMEFRAME_CONFIG

        expected = ["HOUR", "MINUTE_15", "MINUTE_5", "DAY"]
        for tf in expected:
            assert tf in TIMEFRAME_CONFIG, f"Timeframe {tf} fehlt"
            assert "oos_size" in TIMEFRAME_CONFIG[tf]
            assert "bars_per_hour" in TIMEFRAME_CONFIG[tf]
