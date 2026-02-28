"""
Tests for preset loading and OptimizationConfig.

Verifies:
1. _load_json_preset loads JSON files and raises on missing
2. _normalize_exit_params converts scalars to arrays
3. OptimizationConfig.from_dict parses ct, regime, exit_modifier_params_grid
4. StrategyConfig.from_dict integrates optimization section
"""
import json
import os
import pytest

from fwbg.core.config import (
    OptimizationConfig,
    RegimeFilterGridConfig,
    StrategyConfig,
    _load_json_preset,
    _normalize_exit_params,
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
# TestNormalizeExitParams
# ============================================================


class TestNormalizeExitParams:
    """Tests for _normalize_exit_params."""

    def test_scalar_values_wrapped_in_list(self):
        """Scalar values should be converted to single-element lists."""
        params = {"tp_mult": 2.0, "sl_mult": 1.5, "atr_period": 14}
        result = _normalize_exit_params(params)
        assert result == {"tp_mult": [2.0], "sl_mult": [1.5], "atr_period": [14]}

    def test_list_values_unchanged(self):
        """List values should pass through unchanged."""
        params = {"tp_mult": [1.0, 2.0], "sl_mult": [1.5]}
        result = _normalize_exit_params(params)
        assert result == params

    def test_mixed_scalar_and_list(self):
        """Mix of scalars and lists."""
        params = {"tp_mult": [1.0, 2.0], "sl_mult": 1.5, "timeout_bars": [None, 24]}
        result = _normalize_exit_params(params)
        assert result == {
            "tp_mult": [1.0, 2.0],
            "sl_mult": [1.5],
            "timeout_bars": [None, 24],
        }

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert _normalize_exit_params({}) == {}


# ============================================================
# TestOptimizationConfig
# ============================================================


class TestOptimizationConfig:
    """Tests for OptimizationConfig.from_dict."""

    def test_default_ct(self):
        """Default ct should be [0.5]."""
        cfg = OptimizationConfig()
        assert cfg.ct == [0.5]

    def test_from_dict_none_returns_default(self):
        """None input returns default config."""
        cfg = OptimizationConfig.from_dict(None)
        assert cfg.ct == [0.5]
        assert cfg.regime_filter_grid.total_combinations() == 1

    def test_ct_scalar_wrapped(self):
        """Scalar ct value should be wrapped in a list."""
        cfg = OptimizationConfig.from_dict({"ct": 0.6})
        assert cfg.ct == [0.6]

    def test_ct_list_preserved(self):
        """List ct value should pass through."""
        cfg = OptimizationConfig.from_dict({"ct": [0.5, 0.55, 0.6]})
        assert cfg.ct == [0.5, 0.55, 0.6]

    def test_long_short_ct(self):
        """long_ct and short_ct parsed from dict."""
        cfg = OptimizationConfig.from_dict({
            "ct": [0.5],
            "long_ct": [0.6, 0.65],
            "short_ct": 0.55,
        })
        assert cfg.long_ct == [0.6, 0.65]
        assert cfg.short_ct == [0.55]

    def test_regime_filter_grid_parsed(self):
        """regime_filter_grid dict should be parsed into RegimeFilterGridConfig."""
        cfg = OptimizationConfig.from_dict({
            "regime_filter_grid": SAMPLE_REGIME_FILTER,
        })
        assert cfg.regime_filter_grid.total_combinations() == 2

    def test_no_regime_filter_grid(self):
        """Without regime_filter_grid, default (1 combo) is used."""
        cfg = OptimizationConfig.from_dict({"ct": [0.5]})
        assert cfg.regime_filter_grid.total_combinations() == 1

    def test_absent_exit_modifier_params_grid_defaults_to_none(self):
        """Without exit_modifier_params_grid, default is None."""
        cfg = OptimizationConfig.from_dict({"ct": [0.5]})
        assert cfg.exit_modifier_params_grid is None

    def test_list_exit_modifier_params_grid_parsed_as_is(self):
        """List is preserved unchanged."""
        cfg = OptimizationConfig.from_dict({
            "exit_modifier_params_grid": [
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
        })
        assert len(cfg.exit_modifier_params_grid) == 2
        assert cfg.exit_modifier_params_grid[0] == {
            "breakeven_trigger": 0.0, "trail_atr_mult": 0.0
        }
        assert cfg.exit_modifier_params_grid[1] == {
            "breakeven_trigger": 0.5, "trail_atr_mult": 0.5
        }

    def test_single_dict_exit_modifier_params_grid_wrapped_in_list(self):
        """Single dict is wrapped in a list."""
        cfg = OptimizationConfig.from_dict({
            "exit_modifier_params_grid": {
                "breakeven_trigger": 0.5, "trail_atr_mult": 0.5
            },
        })
        assert cfg.exit_modifier_params_grid == [
            {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5}
        ]

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
    """Tests that StrategyConfig.from_dict integrates optimization correctly."""

    def test_optimization_parsed(self):
        """optimization section parsed into OptimizationConfig."""
        config = StrategyConfig.from_dict({
            "name": "Test",
            "optimization": {
                "ct": [0.5, 0.55],
                "regime_filter_grid": SAMPLE_REGIME_FILTER,
            },
        })
        assert isinstance(config.optimization, OptimizationConfig)
        assert config.optimization.ct == [0.5, 0.55]
        assert config.optimization.regime_filter_grid.total_combinations() == 2

    def test_no_optimization_uses_defaults(self):
        """Without optimization key, defaults are used."""
        config = StrategyConfig.from_dict({"name": "NoOpt"})
        assert config.optimization.ct == [0.5]
        assert config.optimization.regime_filter_grid.total_combinations() == 1

    def test_exit_params_normalized_to_arrays(self):
        """exit_params values are normalized to arrays."""
        config = StrategyConfig.from_dict({
            "name": "Test",
            "exit_params": {
                "tp_mult": [1.5, 2.0],
                "sl_mult": 1.0,
                "timeout_bars": [None, 24],
            },
        })
        assert config.exit_params["tp_mult"] == [1.5, 2.0]
        assert config.exit_params["sl_mult"] == [1.0]
        assert config.exit_params["timeout_bars"] == [None, 24]
