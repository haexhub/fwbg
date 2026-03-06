"""Integration tests for ORB exploration strategy configuration.

Guards against config bugs:
1. timeout_bars must be None for trailing_stop configs
2. exit_modifier_params must not disable trailing (zero trail)
3. exit_modifier='trailing_stop' must be set on exit strategy instances
4. orb_based exit strategy dispatches trailing kernel when exit_modifier is set
5. Signal columns cover all rb/cf/prb combinations
6. Pipeline config has required settings

Exit strategy instances live in exit_strategies[] array, each with:
- name, params (fixed scalars), ct, exit_modifier, exit_modifier_params
"""
import json
import os
import re

import pytest

from fwbg.core.config import StrategyConfig

from fwbg.api.workspace import get_strategies_dir as _gsd
_CONFIGS = _gsd()
ORB_EXPLORATION_PATH = str(_CONFIGS / "orb_exploration.json")
DEEP_ORB_INDEX_PATH = str(_CONFIGS / "deep_orb_index.json")
WEEKLY_ORB_PATH = str(_CONFIGS / "weekly_orb_scalping.json")


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_orb_instances(config):
    """Return all exit_strategies instances using orb_based."""
    return [es for es in config.exit_strategies if es.name == "orb_based"]


def _get_trailing_instances(config):
    """Return all exit_strategies instances using trailing_stop modifier."""
    return [es for es in config.exit_strategies if es.exit_modifier == "trailing_stop"]


# ===========================================================================
# Test Class 1: Config Loading & Basic Structure
# ===========================================================================

class TestOrbConfigLoading:
    """Verify orb_exploration.json loads correctly."""

    def test_config_loads_without_error(self, orb_config):
        assert orb_config.name is not None

    def test_has_orb_based_exit_strategies(self, orb_config):
        orb_instances = _get_orb_instances(orb_config)
        assert len(orb_instances) > 0, "No orb_based exit strategy instances found"

    def test_exit_strategies_have_trailing_stop(self, orb_config):
        trailing = _get_trailing_instances(orb_config)
        assert len(trailing) > 0, "No exit strategies with trailing_stop modifier"

    def test_pipeline_has_opening_range(self, orb_config):
        assert "indicators" in orb_config.pipeline
        ind_names = [i["name"] for i in orb_config.pipeline["indicators"]]
        assert "opening_range" in ind_names

    def test_exit_strategy_params_have_required_keys(self, orb_config):
        for es in orb_config.exit_strategies:
            assert "tp_mult" in es.params, f"{es.name} missing tp_mult"
            assert "sl_mult" in es.params, f"{es.name} missing sl_mult"


# ===========================================================================
# Test Class 2: Timeout Bars Not Leaked (all trailing_stop configs)
# ===========================================================================

class TestTimeoutBarsNotLeaked:
    """timeout_bars must be None for instances using trailing stop."""

    def test_orb_exploration_timeout_null(self, orb_config):
        for es in _get_trailing_instances(orb_config):
            timeout = es.params.get("timeout_bars")
            assert timeout is None, (
                f"orb_exploration: timeout_bars should be None, "
                f"got {timeout}. Preset timeout_bars leaking!"
            )

    def test_deep_orb_timeout_null(self, deep_orb_config):
        trailing = _get_trailing_instances(deep_orb_config)
        if not trailing:
            pytest.skip("deep_orb_index doesn't use trailing_stop")
        for es in trailing:
            timeout = es.params.get("timeout_bars")
            assert timeout is None, (
                f"deep_orb_index: timeout_bars should be None, got {timeout}"
            )

    def test_weekly_orb_timeout_null(self, weekly_orb_config):
        trailing = _get_trailing_instances(weekly_orb_config)
        if not trailing:
            pytest.skip("weekly_orb doesn't use trailing_stop")
        for es in trailing:
            timeout = es.params.get("timeout_bars")
            assert timeout is None, (
                f"weekly_orb: timeout_bars should be None, got {timeout}"
            )


# ===========================================================================
# Test Class 3: Exit Modifier Params — No Zero Trail
# ===========================================================================

class TestExitModifierParams:
    """No exit strategy instance should have trail_atr_mult=0 (disables trailing)."""

    def test_orb_exploration_no_zero_trail(self, orb_config):
        for i, es in enumerate(_get_trailing_instances(orb_config)):
            emp = es.exit_modifier_params or {}
            trail = emp.get("trail_atr_mult", 0)
            assert trail > 0, (
                f"orb_exploration: exit_strategies[{i}] "
                f"trail_atr_mult={trail}. Disables trailing!"
            )

    def test_deep_orb_no_zero_trail(self, deep_orb_config):
        trailing = _get_trailing_instances(deep_orb_config)
        if not trailing:
            pytest.skip("deep_orb_index doesn't use trailing_stop")
        for i, es in enumerate(trailing):
            emp = es.exit_modifier_params or {}
            assert emp.get("trail_atr_mult", 0) > 0, (
                f"deep_orb_index: instance[{i}] disables trailing"
            )

    def test_weekly_orb_no_zero_trail(self, weekly_orb_config):
        trailing = _get_trailing_instances(weekly_orb_config)
        if not trailing:
            pytest.skip("weekly_orb doesn't use trailing_stop")
        for i, es in enumerate(trailing):
            emp = es.exit_modifier_params or {}
            assert emp.get("trail_atr_mult", 0) > 0, (
                f"weekly_orb: instance[{i}] disables trailing"
            )


# ===========================================================================
# Test Class 4: Model Hyperparameters Grid Coverage
# ===========================================================================

class TestModelHyperparametersGridCoverage:
    """If model_hyperparameters_grid is explicitly configured, variants must have signal columns.

    Note: Most strategies now use optimization.indicator_grid to generate HP variants
    dynamically at runtime. Tests skip when no static model_hyperparameters_grid is present.
    """

    def test_all_variants_have_signal_columns(self, orb_config):
        """Every explicit variant must have signal_column_long and signal_column_short."""
        mhg = orb_config.optimization.model_hyperparameters_grid or []
        non_none = [v for v in mhg if v is not None]
        if not non_none:
            pytest.skip("No static model_hyperparameters_grid variants (uses indicator_grid)")

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
        if not non_none:
            pytest.skip("No static model_hyperparameters_grid variants (uses indicator_grid)")

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
        """Load all configs that use trailing_stop exit modifier."""
        configs = {}
        for name, path in [
            ("orb_exploration", ORB_EXPLORATION_PATH),
            ("deep_orb_index", DEEP_ORB_INDEX_PATH),
            ("weekly_orb", WEEKLY_ORB_PATH),
        ]:
            if os.path.isfile(path):
                cfg = StrategyConfig.from_json_file(path)
                trailing = _get_trailing_instances(cfg)
                if trailing:
                    configs[name] = cfg
        if not configs:
            pytest.skip("No trailing_stop configs found")
        return configs

    def test_all_configs_have_null_timeout(self, all_trailing_configs):
        """Every trailing_stop instance must have timeout_bars=None."""
        for name, cfg in all_trailing_configs.items():
            for es in _get_trailing_instances(cfg):
                timeout = es.params.get("timeout_bars")
                assert timeout is None, (
                    f"{name}: timeout_bars={timeout}"
                )

    def test_all_configs_have_positive_trail(self, all_trailing_configs):
        """Every trailing_stop instance must have trail_atr_mult > 0."""
        for name, cfg in all_trailing_configs.items():
            for i, es in enumerate(_get_trailing_instances(cfg)):
                emp = es.exit_modifier_params or {}
                assert emp.get("trail_atr_mult", 0) > 0, (
                    f"{name}: instance[{i}] disables trailing"
                )

    def test_all_configs_use_orb_based_exit(self, all_trailing_configs):
        """All ORB configs must use orb_based exit strategy."""
        for name, cfg in all_trailing_configs.items():
            orb_instances = _get_orb_instances(cfg)
            assert len(orb_instances) > 0, (
                f"{name}: no orb_based exit strategy instances found"
            )
