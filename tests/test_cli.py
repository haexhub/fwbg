"""
Integration-Tests für die CLI.

Testet:
- CLI-Argument-Parsing
- Strategy-Loading
- Account/Timeframe-Overrides
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
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "fwbg.cli", "--help"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        assert result.returncode == 0
        assert "FWBG Strategy Backtester" in result.stdout

    def test_list_runs_command(self):
        """Test: --list zeigt vorhandene Runs oder 'keine gefunden' Meldung."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "fwbg.cli", "--list"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        assert result.returncode == 0
        # Entweder gibt es Runs oder die "keine gefunden" Meldung
        assert "TEST-RUNS" in result.stdout or "Keine Test-Runs" in result.stdout


class TestStrategyLoading:
    """Tests für Strategy-Loading aus JSON."""

    def test_load_valid_strategy(self):
        """Test: Gueltige Strategy-Datei laden."""
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "TestStrategy",
            "pipeline": {
                "indicators": [
                    {"name": "trend", "params": {}},
                    {"name": "momentum", "params": {}},
                ]
            },
            "exit_strategies": [
                {"name": "fixed", "params": {"tp_mult": 1.5, "sl_mult": 1.0}, "ct": [0.55]},
                {"name": "fixed", "params": {"tp_mult": 2.0, "sl_mult": 1.0}, "ct": [0.55]},
            ],
        }

        config = StrategyConfig.from_dict(strategy_data)
        assert config.name == "TestStrategy"
        indicators = config.get_indicators()
        assert len(indicators) == 2
        assert indicators[0]["name"] == "trend"
        assert indicators[1]["name"] == "momentum"
        assert len(config.exit_strategies) == 2
        assert config.exit_strategies[0].params["tp_mult"] == 1.5
        assert config.exit_strategies[1].params["tp_mult"] == 2.0

    def test_load_strategy_with_account(self):
        """Test: Strategy mit Account-Konfiguration."""
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "TestWithAccount",
            "version": "1.0",
            "account": "custom_account",
            "timeframe": "MINUTE_15",
            "pipeline": {
                "indicators": [{"name": "trend", "params": {}}]
            },
            "grid": {"tp": [2.0], "sl": [1.0], "ct": [0.5]}
        }

        config = StrategyConfig.from_dict(strategy_data)
        # Account wird nicht in StrategyConfig gespeichert, sondern separat behandelt
        assert config.name == "TestWithAccount"

    def test_load_strategy_with_preprocessing(self):
        """Test: Strategy mit Preprocessing-Plugins (Pipeline-Format)."""
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "TestPreprocessing",
            "version": "1.0",
            "pipeline": {
                "preprocessing": [
                    {"name": "fractional_diff", "params": {"d": 0.4}}
                ],
                "indicators": [{"name": "trend", "params": {}}]
            },
            "grid": {"tp": [2.0], "sl": [1.0], "ct": [0.5]}
        }

        config = StrategyConfig.from_dict(strategy_data)
        preprocessing = config.get_preprocessing()
        assert len(preprocessing) == 1
        assert preprocessing[0]["name"] == "fractional_diff"
        assert preprocessing[0]["params"]["d"] == 0.4

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
        assert config.exit_strategies == []
        assert config.get_preprocessing() == []


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
    """Tests für das neue Preprocessing-Plugin-Format (Pipeline)."""

    def test_preprocessing_is_list(self):
        """Test: preprocessing sollte Liste von Plugin-Configs sein."""
        from fwbg.core.config import StrategyConfig

        config = StrategyConfig.from_dict({
            "name": "Test",
            "version": "1.0",
            "pipeline": {
                "preprocessing": [
                    {"name": "fractional_diff", "params": {"d": 0.5}},
                    {"name": "normalize", "params": {}}
                ]
            }
        })

        preprocessing = config.get_preprocessing()
        assert isinstance(preprocessing, list)
        assert len(preprocessing) == 2
        assert preprocessing[0]["name"] == "fractional_diff"
        assert preprocessing[1]["name"] == "normalize"

    def test_empty_preprocessing(self):
        """Test: Leere Preprocessing-Liste."""
        from fwbg.core.config import StrategyConfig

        config = StrategyConfig.from_dict({"name": "Test", "version": "1.0"})

        preprocessing = config.get_preprocessing()
        assert preprocessing == []
        assert not preprocessing  # Evaluiert zu False


class TestComputeIndicatorPool:
    """Tests für compute_indicator_pool."""

    def test_compute_basic(self):
        """Test: Grundlegende Indikator-Berechnung."""
        import pandas as pd
        import numpy as np
        from fwbg.pipeline import compute_indicator_pool

        # Erstelle Test-DataFrame mit DateTimeIndex (benötigt für time_season)
        n = 500
        df = pd.DataFrame({
            "O": np.random.randn(n).cumsum() + 100,
            "H": np.random.randn(n).cumsum() + 101,
            "L": np.random.randn(n).cumsum() + 99,
            "C": np.random.randn(n).cumsum() + 100,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
        df["H"] = df[["O", "H", "C"]].max(axis=1) + 0.1
        df["L"] = df[["O", "L", "C"]].min(axis=1) - 0.1

        # Test mit ein paar Trend-Indikatoren (nach dem Split aus 'trend')
        result = compute_indicator_pool(df, indicators=["ema", "adx"])

        # compute_indicator_pool gibt den DataFrame MIT Indikatoren zurück
        indicator_cols = [c for c in result.columns if c not in ["O", "H", "L", "C"]]
        assert len(indicator_cols) > 0, "Keine Indikator-Spalten hinzugefügt"
        assert any("adx_" in c for c in result.columns) or any("ema" in c.lower() for c in result.columns)

    def test_compute_no_symbol_arg(self):
        """Test: compute_indicator_pool akzeptiert kein symbol Argument."""
        import pandas as pd
        import numpy as np
        from fwbg.pipeline import compute_indicator_pool
        import inspect

        sig = inspect.signature(compute_indicator_pool)
        param_names = list(sig.parameters.keys())

        assert "symbol" not in param_names, \
            "compute_indicator_pool sollte kein 'symbol' Argument haben"


class TestExitStrategies:
    """Tests für Exit-Strategies."""

    def test_get_exit_strategy(self):
        """Test: get_exit_strategy gibt korrekte Klasse zurück."""
        from fwbg.core import get_exit_strategy

        atr_strategy = get_exit_strategy("atr_based")
        assert atr_strategy is not None

        fixed_strategy = get_exit_strategy("fixed")
        assert fixed_strategy is not None

    def test_get_nonexistent_exit_strategy(self):
        """Test: get_exit_strategy mit unbekanntem Namen wirft ValueError."""
        from fwbg.core import get_exit_strategy

        with pytest.raises(ValueError):
            get_exit_strategy("nonexistent_strategy")

    def test_grid_params(self):
        """Test: GridParams Klasse."""
        from fwbg.core import GridParams

        params = GridParams(tp_value=2.0, sl_value=1.0)

        assert params.tp_value == 2.0
        assert params.sl_value == 1.0
        assert params.rrr == 2.0

    def test_grid_params_to_dict(self):
        """Test: GridParams.to_dict()."""
        from fwbg.core import GridParams

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
            TIMEFRAME, OOS_SIZE,
            WALK_FORWARD_FOLDS, CORR_THRESHOLD, MIN_TRADES
        )

        # DATA_PATH is None by default (set via strategy.datasource or --data-path)
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
