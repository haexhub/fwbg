"""
Tests for preset loading, OptimizationConfig, and ExitStrategyConfig.

Verifies:
1. _load_json_preset loads JSON files and raises on missing
2. ExitStrategyConfig.from_dict parses ct, params, modifier
3. OptimizationConfig.from_dict parses regime_filter_grid, model_hyperparameters_grid
4. StrategyConfig.from_dict integrates optimization + exit_strategies
"""
import json
import os
import pytest

from fwbg.core.config import (
    ExitStrategyConfig,
    OptimizationConfig,
    RegimeFilterGridConfig,
    StrategyConfig,
    _load_json_preset,
)


# -- Fixtures --

SAMPLE_REGIME_FILTER = {
    "condition_grids": [
        {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [None, 25],
            "directions": 6,
            "else_directions": 0,
        }
    ]
}


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ============================================================
# TestPresetLoading
# ============================================================


class TestPresetLoading:
    """Tests for _load_json_preset."""

    def test_load_preset_file(self, tmp_path):
        """Basic JSON load returns correct dict."""
        presets_dir = tmp_path / "grids"
        presets_dir.mkdir()
        preset_data = {"tp": [1.0, 2.0], "sl": [1.5], "ct": [0.55]}
        _write_json(str(presets_dir / "forex_wide.json"), preset_data)

        result = _load_json_preset("forex_wide", str(presets_dir))
        assert result == preset_data

    def test_preset_not_found_raises(self, tmp_path):
        """FileNotFoundError with clear message when preset missing."""
        presets_dir = tmp_path / "grids"
        presets_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _load_json_preset("nonexistent", str(presets_dir))

    def test_meta_stripped_from_preset(self, tmp_path):
        """_meta key should be stripped from loaded presets."""
        presets_dir = tmp_path / "test_presets"
        presets_dir.mkdir()
        preset_data = {"key": "value", "_meta": {"author": "test"}}
        _write_json(str(presets_dir / "with_meta.json"), preset_data)

        result = _load_json_preset("with_meta", str(presets_dir))
        assert "_meta" not in result
        assert result == {"key": "value"}


# ============================================================
# TestExitStrategyConfig
# ============================================================


class TestExitStrategyConfig:
    """Tests for ExitStrategyConfig.from_dict."""

    def test_default_values(self):
        """Default ExitStrategyConfig has sensible defaults."""
        cfg = ExitStrategyConfig()
        assert cfg.name == "fixed"
        assert cfg.params == {}
        assert cfg.ct == [0.5]
        assert cfg.long_ct is None
        assert cfg.short_ct is None
        assert cfg.min_rrr == 0
        assert cfg.exit_modifier is None
        assert cfg.exit_modifier_params == {}

    def test_from_dict_minimal(self):
        """Minimal dict creates config with defaults."""
        cfg = ExitStrategyConfig.from_dict({"name": "fixed"})
        assert cfg.name == "fixed"
        assert cfg.ct == [0.5]

    def test_scalar_ct_wrapped(self):
        """Scalar ct value should be wrapped in a list."""
        cfg = ExitStrategyConfig.from_dict({"name": "fixed", "ct": 0.6})
        assert cfg.ct == [0.6]

    def test_ct_list_preserved(self):
        """List ct value should pass through."""
        cfg = ExitStrategyConfig.from_dict({"name": "fixed", "ct": [0.5, 0.55, 0.6]})
        assert cfg.ct == [0.5, 0.55, 0.6]

    def test_long_short_ct(self):
        """long_ct and short_ct parsed from dict."""
        cfg = ExitStrategyConfig.from_dict({
            "name": "fixed",
            "ct": [0.5],
            "long_ct": [0.6, 0.65],
            "short_ct": 0.55,
        })
        assert cfg.long_ct == [0.6, 0.65]
        assert cfg.short_ct == [0.55]

    def test_params_preserved(self):
        """params dict passed through."""
        cfg = ExitStrategyConfig.from_dict({
            "name": "atr_based",
            "params": {"atr_period": 14, "tp_mult": 2.0, "sl_mult": 1.5},
        })
        assert cfg.params["atr_period"] == 14
        assert cfg.params["tp_mult"] == 2.0

    def test_exit_modifier(self):
        """exit_modifier and exit_modifier_params parsed."""
        cfg = ExitStrategyConfig.from_dict({
            "name": "atr_based",
            "params": {},
            "exit_modifier": "trailing_stop",
            "exit_modifier_params": {"breakeven_trigger": 0.5},
        })
        assert cfg.exit_modifier == "trailing_stop"
        assert cfg.exit_modifier_params == {"breakeven_trigger": 0.5}

    def test_min_rrr(self):
        """min_rrr parsed from dict."""
        cfg = ExitStrategyConfig.from_dict({
            "name": "fixed",
            "params": {},
            "min_rrr": 1.5,
        })
        assert cfg.min_rrr == 1.5


# ============================================================
# TestOptimizationConfig
# ============================================================


class TestOptimizationConfig:
    """Tests for OptimizationConfig.from_dict."""

    def test_from_dict_none_returns_default(self):
        """None input returns default config."""
        cfg = OptimizationConfig.from_dict(None)
        assert cfg.regime_filter_grid.total_combinations() == 1
        assert cfg.model_hyperparameters_grid is None

    def test_regime_filter_grid_parsed(self):
        """regime_filter_grid dict should be parsed into RegimeFilterGridConfig."""
        cfg = OptimizationConfig.from_dict({
            "regime_filter_grid": SAMPLE_REGIME_FILTER,
        })
        assert cfg.regime_filter_grid.total_combinations() == 2

    def test_no_regime_filter_grid(self):
        """Without regime_filter_grid, default (1 combo) is used."""
        cfg = OptimizationConfig.from_dict({})
        assert cfg.regime_filter_grid.total_combinations() == 1

    def test_model_hyperparameters_grid_list(self):
        """model_hyperparameters_grid list is preserved."""
        cfg = OptimizationConfig.from_dict({
            "model_hyperparameters_grid": [
                {"signal_column_long": "col_a"},
                {"signal_column_long": "col_b"},
            ],
        })
        assert len(cfg.model_hyperparameters_grid) == 2

    def test_model_hyperparameters_grid_single_dict_wrapped(self):
        """Single dict model_hyperparameters_grid is wrapped in a list."""
        cfg = OptimizationConfig.from_dict({
            "model_hyperparameters_grid": {"signal_column_long": "col_a"},
        })
        assert cfg.model_hyperparameters_grid == [{"signal_column_long": "col_a"}]


# ============================================================
# TestStrategyOptimizationIntegration
# ============================================================


class TestStrategyOptimizationIntegration:
    """Tests that StrategyConfig.from_dict integrates optimization + exit_strategies."""

    def test_optimization_parsed(self):
        """optimization section parsed into OptimizationConfig."""
        config = StrategyConfig.from_dict({
            "name": "Test",
            "optimization": {
                "regime_filter_grid": SAMPLE_REGIME_FILTER,
            },
        })
        assert isinstance(config.optimization, OptimizationConfig)
        assert config.optimization.regime_filter_grid.total_combinations() == 2

    def test_no_optimization_uses_defaults(self):
        """Without optimization key, defaults are used."""
        config = StrategyConfig.from_dict({"name": "NoOpt"})
        assert config.optimization.regime_filter_grid.total_combinations() == 1

    def test_exit_strategies_parsed(self):
        """exit_strategies array parsed into ExitStrategyConfig list."""
        config = StrategyConfig.from_dict({
            "name": "Test",
            "exit_strategies": [
                {"name": "fixed", "params": {"tp_mult": 2.0, "sl_mult": 1.0}, "ct": [0.5, 0.55]},
                {"name": "atr_based", "params": {"atr_period": 14}, "ct": [0.5]},
            ],
        })
        assert len(config.exit_strategies) == 2
        assert config.exit_strategies[0].name == "fixed"
        assert config.exit_strategies[0].params["tp_mult"] == 2.0
        assert config.exit_strategies[0].ct == [0.5, 0.55]
        assert config.exit_strategies[1].name == "atr_based"
        assert config.exit_strategies[1].ct == [0.5]

    def test_no_exit_strategies_defaults_to_empty(self):
        """Without exit_strategies, defaults to empty list."""
        config = StrategyConfig.from_dict({"name": "NoExit"})
        assert config.exit_strategies == []
