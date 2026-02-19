"""
Tests for config section preset resolution.

Verifies:
1. _resolve_section resolves strings to preset files, passes through dict/None
2. StrategyConfig.from_dict resolves all section presets
3. Backward compatibility with all-inline format
4. from_json_file resolves presets relative to strategy file directory
"""
import json
import os
import pytest

from fwbg.core.config import (
    FilterConfig,
    ModelConfig,
    ResourceConfig,
    StrategyConfig,
    ValidationConfig,
    _resolve_section,
)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ============================================================
# TestResolveSection
# ============================================================


class TestResolveSection:
    """Tests for _resolve_section helper function."""

    def test_string_loads_from_preset_dir(self, tmp_path):
        """String value loads JSON file from section_dir."""
        section_dir = tmp_path / "pipelines"
        section_dir.mkdir()
        preset_data = {"indicators": [{"name": "trend", "params": {}}]}
        _write_json(str(section_dir / "exploration.json"), preset_data)

        # strategy_dir points to e.g. /project/strategies/
        # os.path.dirname gives /project/ (the base)
        strategy_dir = str(tmp_path / "strategies")

        result = _resolve_section("exploration", "pipelines", strategy_dir)
        assert result == preset_data

    def test_dict_passes_through(self):
        """Dict value is returned as-is."""
        data = {"indicators": [{"name": "trend", "params": {}}]}
        result = _resolve_section(data, "pipelines", None)
        assert result is data

    def test_none_passes_through(self):
        """None value is returned as-is."""
        result = _resolve_section(None, "pipelines", None)
        assert result is None

    def test_preset_not_found_raises(self, tmp_path):
        """FileNotFoundError when preset file does not exist."""
        section_dir = tmp_path / "pipelines"
        section_dir.mkdir()
        strategy_dir = str(tmp_path / "strategies")

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _resolve_section("nonexistent", "pipelines", strategy_dir)


# ============================================================
# TestStrategyWithPresets
# ============================================================


class TestStrategyWithPresets:
    """Tests for StrategyConfig.from_dict with preset string references."""

    def test_pipeline_string_ref(self, tmp_path):
        """pipeline: "exploration" loads pipelines/exploration.json."""
        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()
        pipeline_data = {
            "indicators": [{"name": "trend", "params": {"period": 14}}],
            "preprocessing": [{"name": "normalize", "params": {}}],
        }
        _write_json(str(pipelines_dir / "exploration.json"), pipeline_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "PipelinePresetTest",
            "pipeline": "exploration",
            "_strategy_dir": strategy_dir,
        })

        assert config.name == "PipelinePresetTest"
        assert config.pipeline == pipeline_data
        assert len(config.get_indicators()) == 1
        assert config.get_indicators()[0]["name"] == "trend"

    def test_model_string_ref(self, tmp_path):
        """model: "xgboost_depth6" loads models/xgboost_depth6.json."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_data = {
            "type": "xgboost",
            "architecture": "unified",
            "hyperparameters": {"n_estimators": 200, "max_depth": 6, "random_state": 42},
            "trade_directions": ["long", "short"],
        }
        _write_json(str(models_dir / "xgboost_depth6.json"), model_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "ModelPresetTest",
            "model": "xgboost_depth6",
            "_strategy_dir": strategy_dir,
        })

        assert isinstance(config.model, ModelConfig)
        assert config.model.type == "xgboost"
        assert config.model.hyperparameters["max_depth"] == 6
        assert config.model.hyperparameters["n_estimators"] == 200

    def test_validation_string_ref(self, tmp_path):
        """validation: "walk_forward" loads validations/walk_forward.json."""
        validations_dir = tmp_path / "validations"
        validations_dir.mkdir()
        validation_data = {
            "method": "walk_forward",
            "folds": 10,
            "oos_size": 5000,
            "min_trades": 100,
        }
        _write_json(str(validations_dir / "walk_forward.json"), validation_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "ValidationPresetTest",
            "validation": "walk_forward",
            "_strategy_dir": strategy_dir,
        })

        assert isinstance(config.validation, ValidationConfig)
        assert config.validation.folds == 10
        assert config.validation.oos_size == 5000
        assert config.validation.min_trades == 100

    def test_exit_params_string_ref(self, tmp_path):
        """exit_params: "atr_standard" loads exit_params/atr_standard.json."""
        exit_params_dir = tmp_path / "exit_params"
        exit_params_dir.mkdir()
        exit_params_data = {
            "atr_period": 14,
            "atr_multiplier": 2.0,
        }
        _write_json(str(exit_params_dir / "atr_standard.json"), exit_params_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "ExitParamsPresetTest",
            "exit_params": "atr_standard",
            "_strategy_dir": strategy_dir,
        })

        assert config.exit_params == exit_params_data
        assert config.exit_params["atr_period"] == 14

    def test_filters_string_ref(self, tmp_path):
        """filters: "permissive" loads filters/permissive.json."""
        filters_dir = tmp_path / "filters"
        filters_dir.mkdir()
        filters_data = {
            "min_rrr": 0.5,
            "min_trades": 20,
            "min_annual_return": 5.0,
            "max_drawdown": 0.8,
            "min_sharpe": 0.3,
        }
        _write_json(str(filters_dir / "permissive.json"), filters_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "FiltersPresetTest",
            "filters": "permissive",
            "_strategy_dir": strategy_dir,
        })

        assert isinstance(config.filters, FilterConfig)
        assert config.filters.min_rrr == 0.5
        assert config.filters.min_trades == 20
        assert config.filters.min_annual_return == 5.0

    def test_resources_string_ref(self, tmp_path):
        """resources: "standard" loads resources/standard.json."""
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()
        resources_data = {"max_concurrent_assets": 4}
        _write_json(str(resources_dir / "standard.json"), resources_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "ResourcesPresetTest",
            "resources": "standard",
            "_strategy_dir": strategy_dir,
        })

        assert isinstance(config.resources, ResourceConfig)
        assert config.resources.max_concurrent_assets == 4

    def test_risk_params_string_ref(self, tmp_path):
        """risk_params: "conservative" loads risk_params/conservative.json."""
        risk_params_dir = tmp_path / "risk_params"
        risk_params_dir.mkdir()
        risk_params_data = {"max_risk_per_trade": 0.01}
        _write_json(str(risk_params_dir / "conservative.json"), risk_params_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "RiskParamsPresetTest",
            "risk_params": "conservative",
            "_strategy_dir": strategy_dir,
        })

        assert config.risk_params == risk_params_data

    def test_mixed_inline_and_preset(self, tmp_path):
        """Some sections inline, some preset — both resolve correctly."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_data = {
            "type": "xgboost",
            "hyperparameters": {"n_estimators": 300, "max_depth": 8, "random_state": 42},
        }
        _write_json(str(models_dir / "deep_xgb.json"), model_data)

        validations_dir = tmp_path / "validations"
        validations_dir.mkdir()
        validation_data = {"method": "walk_forward", "folds": 12, "oos_size": 6000}
        _write_json(str(validations_dir / "long_wf.json"), validation_data)

        strategy_dir = str(tmp_path / "strategies")
        config = StrategyConfig.from_dict({
            "name": "MixedTest",
            "pipeline": {"indicators": [{"name": "momentum", "params": {}}]},
            "model": "deep_xgb",
            "validation": "long_wf",
            "filters": {"min_rrr": 1.0, "min_trades": 30},
            "_strategy_dir": strategy_dir,
        })

        # Inline pipeline
        assert len(config.get_indicators()) == 1
        assert config.get_indicators()[0]["name"] == "momentum"
        # Preset model
        assert config.model.hyperparameters["max_depth"] == 8
        assert config.model.hyperparameters["n_estimators"] == 300
        # Preset validation
        assert config.validation.folds == 12
        assert config.validation.oos_size == 6000
        # Inline filters
        assert config.filters.min_rrr == 1.0
        assert config.filters.min_trades == 30


# ============================================================
# TestBackwardCompatibility
# ============================================================


class TestBackwardCompatibility:
    """Tests ensuring existing inline format works unchanged."""

    def test_all_inline_still_works(self):
        """Existing all-inline strategy format unchanged."""
        config = StrategyConfig.from_dict({
            "name": "InlineTest",
            "pipeline": {"indicators": [{"name": "trend", "params": {}}]},
            "exit_strategy": "atr_based",
            "exit_params": {"atr_period": 14},
            "model": {"type": "xgboost", "hyperparameters": {"max_depth": 5}},
            "validation": {"folds": 8, "oos_size": 4000},
            "filters": {"min_rrr": 0.5, "min_trades": 50},
            "resources": {"max_concurrent_assets": 2},
            "risk_params": {"max_risk_per_trade": 0.02},
        })

        assert config.name == "InlineTest"
        assert config.pipeline == {"indicators": [{"name": "trend", "params": {}}]}
        assert config.exit_params == {"atr_period": 14}
        assert config.model.hyperparameters["max_depth"] == 5
        assert config.validation.folds == 8
        assert config.filters.min_rrr == 0.5
        assert config.resources.max_concurrent_assets == 2
        assert config.risk_params == {"max_risk_per_trade": 0.02}

    def test_from_json_file_with_presets(self, tmp_path):
        """from_json_file resolves presets relative to strategy file directory."""
        # Directory structure:
        #   tmp_path/
        #     strategies/test_strategy.json
        #     pipelines/exploration.json
        #     models/default_xgb.json
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()
        pipeline_data = {"indicators": [{"name": "trend", "params": {}}]}
        _write_json(str(pipelines_dir / "exploration.json"), pipeline_data)

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_data = {"type": "xgboost", "hyperparameters": {"max_depth": 7}}
        _write_json(str(models_dir / "default_xgb.json"), model_data)

        strategy = {
            "name": "FilePresetTest",
            "pipeline": "exploration",
            "model": "default_xgb",
        }
        strategy_path = str(strategies_dir / "test_strategy.json")
        _write_json(strategy_path, strategy)

        # from_json_file sets _strategy_dir to os.path.dirname(os.path.abspath(path))
        # which is .../strategies/ — os.path.dirname of that is tmp_path
        config = StrategyConfig.from_json_file(strategy_path)

        assert config.name == "FilePresetTest"
        assert config.pipeline == pipeline_data
        assert config.model.hyperparameters["max_depth"] == 7
