# tests/pipeline/test_config.py
"""Tests for PluginConfig and PipelineConfig parsers."""
import pytest

from fwbg.pipeline.config import PluginConfig, PipelineConfig, parse_pipeline_config


class TestPluginConfig:
    """Tests for PluginConfig dataclass."""

    def test_plugin_config_from_dict(self):
        """Parse basic plugin config with name and params."""
        data = {
            "name": "sma_indicator",
            "params": {"window": 20, "column": "close"},
        }

        config = PluginConfig.from_dict(data)

        assert config.name == "sma_indicator"
        assert config.params == {"window": 20, "column": "close"}
        assert config.stateful is None
        assert config.cacheable is None

    def test_plugin_config_with_overrides(self):
        """Parse plugin config with stateful/cacheable overrides."""
        data = {
            "name": "ml_model",
            "params": {"model_type": "xgboost"},
            "stateful": True,
            "cacheable": False,
        }

        config = PluginConfig.from_dict(data)

        assert config.name == "ml_model"
        assert config.params == {"model_type": "xgboost"}
        assert config.stateful is True
        assert config.cacheable is False

    def test_plugin_config_empty_params(self):
        """Handle missing params - should default to empty dict."""
        data = {"name": "simple_plugin"}

        config = PluginConfig.from_dict(data)

        assert config.name == "simple_plugin"
        assert config.params == {}
        assert config.stateful is None
        assert config.cacheable is None


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_pipeline_config_parse(self):
        """Parse full pipeline config with all phases."""
        data = {
            "pipeline": {
                "data_loading": [
                    {"name": "csv_loader", "params": {"path": "data.csv"}}
                ],
                "preprocessing": [
                    {"name": "normalizer", "params": {"method": "zscore"}}
                ],
                "indicators": [
                    {"name": "sma", "params": {"window": 14}},
                    {"name": "rsi", "params": {"window": 14}},
                ],
                "feature_selection": [
                    {"name": "correlation_filter", "params": {"threshold": 0.9}}
                ],
                "labeling": [
                    {"name": "binary_labeler", "params": {"horizon": 5}}
                ],
                "model": [
                    {"name": "xgboost", "params": {"n_estimators": 100}, "stateful": True}
                ],
                "validation": [
                    {"name": "walk_forward", "params": {"n_splits": 5}}
                ],
            }
        }

        config = parse_pipeline_config(data)

        # Verify all phases parsed
        assert len(config.data_loading) == 1
        assert config.data_loading[0].name == "csv_loader"

        assert len(config.preprocessing) == 1
        assert config.preprocessing[0].name == "normalizer"

        assert len(config.indicators) == 2
        assert config.indicators[0].name in ("sma", "fwbg-core:sma")
        assert config.indicators[1].name in ("rsi", "fwbg-core:rsi")

        assert len(config.feature_selection) == 1
        assert config.feature_selection[0].name in (
            "correlation_filter", "fwbg-premium:correlation_filter"
        )

        assert len(config.labeling) == 1
        assert config.labeling[0].name == "binary_labeler"

        assert len(config.model) == 1
        assert config.model[0].name in ("xgboost", "fwbg-core:xgboost")
        assert config.model[0].stateful is True

        assert len(config.validation) == 1
        assert config.validation[0].name == "walk_forward"

        # Test get_phase method
        indicators = config.get_phase("indicators")
        assert len(indicators) == 2
        assert indicators[0].name in ("sma", "fwbg-core:sma")

        # Test all_plugins method
        all_plugins = config.all_plugins()
        assert len(all_plugins) == 8  # Total plugins across all phases

    def test_pipeline_config_empty_phases(self):
        """Handle missing phases - should default to empty lists."""
        data = {
            "pipeline": {
                "indicators": [
                    {"name": "sma", "params": {"window": 20}}
                ]
            }
        }

        config = parse_pipeline_config(data)

        # Specified phase should have plugins
        assert len(config.indicators) == 1
        assert config.indicators[0].name in ("sma", "fwbg-core:sma")

        # Missing phases should be empty lists
        assert config.data_loading == []
        assert config.preprocessing == []
        assert config.feature_selection == []
        assert config.labeling == []
        assert config.model == []
        assert config.validation == []

        # get_phase for non-existent phase should return empty list
        assert config.get_phase("unknown_phase") == []

    def test_pipeline_config_validation_error(self):
        """Reject invalid plugin entries (missing name)."""
        data = {
            "pipeline": {
                "indicators": [
                    {"params": {"window": 20}}  # Missing 'name'
                ]
            }
        }

        with pytest.raises(ValueError, match="name"):
            parse_pipeline_config(data)
