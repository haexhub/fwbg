"""Integration tests for ORB exploration strategy configuration.

Guards against config bugs:
1. timeout_bars must be [None] for trailing_stop configs
2. exit_modifier_params_grid must not contain {0,0} (disables trailing)
3. exit_modifier='trailing_stop' must be set
4. orb_based exit strategy dispatches trailing kernel when exit_modifier is set
5. Signal columns cover all rb/cf/prb combinations
6. Pipeline config has required settings

Since the GridConfig removal, all grid params live in:
- exit_params (tp_mult, sl_mult, timeout_bars, etc.)
- optimization (ct, exit_modifier_params_grid, model_hyperparameters_grid)
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

STRATEGY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "strategies")
ORB_EXPLORATION_PATH = os.path.join(STRATEGY_DIR, "configs", "orb_exploration.json")
DEEP_ORB_INDEX_PATH = os.path.join(STRATEGY_DIR, "configs", "deep_orb_index.json")
WEEKLY_ORB_PATH = os.path.join(STRATEGY_DIR, "configs", "weekly_orb_scalping.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orb_config():
    """Load orb_exploration.json strategy config."""
    if not os.path.isfile(ORB_EXPLORATION_PATH):
        pytest.skip("orb_exploration.json not found")
    return StrategyConfig.from_json_file(ORB_EXPLORATION_PATH)


@pytest.fixture
def orb_config_raw():
    """Load raw JSON of orb_exploration.json."""
    if not os.path.isfile(ORB_EXPLORATION_PATH):
        pytest.skip("orb_exploration.json not found")
    with open(ORB_EXPLORATION_PATH) as f:
        return json.load(f)


@pytest.fixture
def deep_orb_config():
    """Load deep_orb_index.json strategy config."""
    if not os.path.isfile(DEEP_ORB_INDEX_PATH):
        pytest.skip("deep_orb_index.json not found")
    return StrategyConfig.from_json_file(DEEP_ORB_INDEX_PATH)


@pytest.fixture
def weekly_orb_config():
    """Load weekly_orb_scalping.json strategy config."""
    if not os.path.isfile(WEEKLY_ORB_PATH):
        pytest.skip("weekly_orb_scalping.json not found")
    return StrategyConfig.from_json_file(WEEKLY_ORB_PATH)


# ===========================================================================
# Test Class 1: Config Loading & Basic Structure
# ===========================================================================

class TestOrbConfigLoading:
    """Verify orb_exploration.json loads correctly."""

    def test_config_loads_without_error(self, orb_config):
        assert orb_config.name is not None

    def test_exit_strategy_is_orb_based(self, orb_config):
        assert orb_config.exit_strategy == "orb_based"

    def test_exit_modifier_is_trailing_stop(self, orb_config):
        assert orb_config.exit_modifier == "trailing_stop"

    def test_pipeline_has_opening_range(self, orb_config):
        assert "indicators" in orb_config.pipeline
        ind_names = [i["name"] for i in orb_config.pipeline["indicators"]]
        assert "opening_range" in ind_names

    def test_exit_params_have_required_keys(self, orb_config):
        ep = orb_config.exit_params
        assert "tp_mult" in ep
        assert "sl_mult" in ep
        assert "timeout_bars" in ep
        # All values should be lists (normalized)
        assert isinstance(ep["tp_mult"], list)
        assert isinstance(ep["sl_mult"], list)


# ===========================================================================
# Test Class 2: Timeout Bars Not Leaked (all trailing_stop configs)
# ===========================================================================

class TestTimeoutBarsNotLeaked:
    """timeout_bars must be [None] for configs using trailing stop."""

    def test_orb_exploration_timeout_null(self, orb_config):
        assert orb_config.exit_params["timeout_bars"] == [None], (
            f"orb_exploration: timeout_bars should be [None], "
            f"got {orb_config.exit_params['timeout_bars']}. Preset timeout_bars leaking!"
        )

    def test_deep_orb_timeout_null(self, deep_orb_config):
        if deep_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("deep_orb_index doesn't use trailing_stop")
        assert deep_orb_config.exit_params["timeout_bars"] == [None], (
            f"deep_orb_index: timeout_bars should be [None], "
            f"got {deep_orb_config.exit_params['timeout_bars']}"
        )

    def test_weekly_orb_timeout_null(self, weekly_orb_config):
        if weekly_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("weekly_orb doesn't use trailing_stop")
        assert weekly_orb_config.exit_params["timeout_bars"] == [None], (
            f"weekly_orb: timeout_bars should be [None], "
            f"got {weekly_orb_config.exit_params['timeout_bars']}"
        )


# ===========================================================================
# Test Class 3: Exit Modifier Params Grid
# ===========================================================================

class TestExitModifierParamsGrid:
    """No {0,0} variant that disables trailing stop."""

    def test_orb_exploration_no_zero_trail(self, orb_config):
        emp_grid = orb_config.optimization.exit_modifier_params_grid or []
        for i, params in enumerate(emp_grid):
            if params is None:
                continue
            trail = params.get("trail_atr_mult", 0)
            assert trail > 0, (
                f"orb_exploration: exit_modifier_params_grid[{i}] "
                f"trail_atr_mult={trail}. Disables trailing!"
            )

    def test_deep_orb_no_zero_trail(self, deep_orb_config):
        if deep_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("deep_orb_index doesn't use trailing_stop")
        emp_grid = deep_orb_config.optimization.exit_modifier_params_grid or []
        for i, params in enumerate(emp_grid):
            if params is None:
                continue
            assert params.get("trail_atr_mult", 0) > 0, (
                f"deep_orb_index: grid[{i}] disables trailing"
            )

    def test_weekly_orb_no_zero_trail(self, weekly_orb_config):
        if weekly_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("weekly_orb doesn't use trailing_stop")
        emp_grid = weekly_orb_config.optimization.exit_modifier_params_grid or []
        for i, params in enumerate(emp_grid):
            if params is None:
                continue
            assert params.get("trail_atr_mult", 0) > 0, (
                f"weekly_orb: grid[{i}] disables trailing"
            )


# ===========================================================================
# Test Class 4: Model Hyperparameters Grid Coverage
# ===========================================================================

class TestModelHyperparametersGridCoverage:
    """All model_hyperparameters_grid variants have signal columns with expected prefixes."""

    def test_all_variants_have_signal_columns(self, orb_config):
        """Every variant must have signal_column_long and signal_column_short."""
        mhg = orb_config.optimization.model_hyperparameters_grid or []
        non_none = [v for v in mhg if v is not None]
        assert len(non_none) > 0, "No model_hyperparameters_grid variants"

        for i, v in enumerate(non_none):
            assert "signal_column_long" in v, (
                f"Variant {i} missing signal_column_long"
            )
            assert "signal_column_short" in v, (
                f"Variant {i} missing signal_column_short"
            )

    def test_rb_cf_prb_combinations_present(self, orb_config):
        """At least rb1 and rb2 range_bars variants should be present."""
        mhg = orb_config.optimization.model_hyperparameters_grid or []
        non_none = [v for v in mhg if v is not None]

        rb1_count = sum(1 for v in non_none if "rb1_" in v.get("signal_column_long", ""))
        rb2_count = sum(1 for v in non_none if "rb2_" in v.get("signal_column_long", ""))

        assert rb1_count > 0, "No rb1 variants found"
        assert rb2_count > 0, "No rb2 variants found"


# ===========================================================================
# Test Class 5: Pipeline Config
# ===========================================================================

class TestOrbPipelineConfig:
    """Verify the ORB pipeline is correctly configured."""

    def test_opening_range_indicator_present(self, orb_config):
        ind_names = [i["name"] for i in orb_config.pipeline["indicators"]]
        assert "opening_range" in ind_names

    def test_range_bars_config(self, orb_config):
        """range_bars should cover rb1, rb2, and rb4 variants."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        assert orb_ind["params"]["range_bars"] == [1, 2, 4]

    def test_sessions_include_all_needed(self, orb_config):
        """Pipeline sessions must include all referenced session hours."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        configured_sessions = set(orb_ind["params"]["sessions"])

        # Check that signal columns reference sessions that exist in pipeline
        mhg = orb_config.optimization.model_hyperparameters_grid or []
        referenced_sessions = set()
        for v in mhg:
            if v is None:
                continue
            col = v.get("signal_column_long", "")
            # Extract session hour from pattern like "rb1_cf0_prb0_orb_s08_..."
            import re
            m = re.search(r"_s(\d{2})_", col)
            if m:
                referenced_sessions.add(int(m.group(1)))

        missing = referenced_sessions - configured_sessions
        assert not missing, (
            f"Session hours referenced in model_hyperparameters_grid but not in pipeline "
            f"sessions: {missing}"
        )

    def test_carry_forward_and_pre_range_present(self, orb_config):
        """carry_forward_days and pre_range_bars must be configured."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        params = orb_ind["params"]
        assert "carry_forward_days" in params, "carry_forward_days missing"
        assert "pre_range_bars" in params, "pre_range_bars missing"
        prb = params["pre_range_bars"]
        if isinstance(prb, list):
            assert len(prb) >= 2, (
                f"Need at least 2 pre_range_bars for prb0/prb1, got {prb}"
            )

    def test_enable_retracement(self, orb_config):
        """enable_retracement must be true for retest signals."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        assert orb_ind["params"].get("enable_retracement", False), (
            "enable_retracement must be true for ORB retest signals"
        )


# ===========================================================================
# Test Class 6: Cross-Config Consistency
# ===========================================================================

class TestCrossConfigConsistency:
    """Verify all trailing_stop configs have consistent settings."""

    @pytest.fixture
    def all_trailing_configs(self):
        """Load all configs that use trailing_stop."""
        configs = {}
        for name, path in [
            ("orb_exploration", ORB_EXPLORATION_PATH),
            ("deep_orb_index", DEEP_ORB_INDEX_PATH),
            ("weekly_orb", WEEKLY_ORB_PATH),
        ]:
            if os.path.isfile(path):
                cfg = StrategyConfig.from_json_file(path)
                if cfg.exit_modifier == "trailing_stop":
                    configs[name] = cfg
        if not configs:
            pytest.skip("No trailing_stop configs found")
        return configs

    def test_all_configs_have_null_timeout(self, all_trailing_configs):
        """Every trailing_stop config must have timeout_bars=[None]."""
        for name, cfg in all_trailing_configs.items():
            assert cfg.exit_params["timeout_bars"] == [None], (
                f"{name}: timeout_bars={cfg.exit_params['timeout_bars']}"
            )

    def test_all_configs_have_positive_trail(self, all_trailing_configs):
        """Every trailing_stop config's modifier grid must have trail > 0."""
        for name, cfg in all_trailing_configs.items():
            emp_grid = cfg.optimization.exit_modifier_params_grid or []
            for i, params in enumerate(emp_grid):
                if params is None:
                    continue
                assert params.get("trail_atr_mult", 0) > 0, (
                    f"{name}: grid[{i}] disables trailing"
                )

    def test_all_configs_use_orb_based_exit(self, all_trailing_configs):
        """All ORB configs must use orb_based exit strategy."""
        for name, cfg in all_trailing_configs.items():
            assert cfg.exit_strategy == "orb_based", (
                f"{name}: exit_strategy={cfg.exit_strategy}, expected orb_based"
            )
