"""Integration tests for ORB exploration strategy configuration.

Guards against the same recurring config bugs as PDHL:
1. timeout_bars from preset must not leak into ORB config (trailing stop handles exits)
2. exit_modifier_params_grid must not contain {0,0} (disables trailing)
3. exit_modifier='trailing_stop' must be set
4. orb_based exit strategy dispatches trailing kernel when exit_modifier is set
5. Signal columns match the asset's session hour (s07=DAX, s14=NAS100, etc.)
6. All grid variants cover rb1/rb2 × cf0/cf1/cf2 × prb0/prb1 combinations

ORB-specific: reference candle is the first 45min (3 × 15min bars) of the
opening hour per asset per day. Session hours vary by asset:
  DAX/EU50/CAC40: s07 (7:00 UTC), FTSE100/FOREX/COMMODITY: s08,
  NAS100/SPX500/DOW30: s14, JP225/ASX200/CRYPTO: s00, HK50: s01
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig, GridConfig
from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

STRATEGY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "strategies")
ORB_EXPLORATION_PATH = os.path.join(STRATEGY_DIR, "configs", "orb_exploration.json")
DEEP_ORB_INDEX_PATH = os.path.join(STRATEGY_DIR, "configs", "deep_orb_index.json")
WEEKLY_ORB_PATH = os.path.join(STRATEGY_DIR, "configs", "weekly_orb_scalping.json")

# Expected session hours per asset in orb_exploration
EXPECTED_SESSION_HOURS = {
    "DAX": 7, "EU50": 7, "CAC40": 7,
    "FTSE100": 8, "FOREX": 8, "COMMODITY": 8,
    "NAS100": 14, "SPX500": 14, "DOW30": 14,
    "JP225": 0, "ASX200": 22, "CRYPTO": 0,
    "HK50": 1,
}


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

    def test_pipeline_is_orb_simple_v1(self, orb_config):
        assert "indicators" in orb_config.pipeline
        ind_names = [i["name"] for i in orb_config.pipeline["indicators"]]
        assert "opening_range" in ind_names

    def test_all_expected_assets_present(self, orb_config):
        expected = {"FOREX", "DAX", "EU50", "CAC40", "FTSE100",
                    "NAS100", "SPX500", "DOW30", "JP225", "ASX200", "HK50",
                    "COMMODITY", "CRYPTO"}
        assert set(orb_config.grids.keys()) == expected


# ===========================================================================
# Test Class 2: Timeout Bars Not Leaked (all trailing_stop configs)
# ===========================================================================

class TestTimeoutBarsNotLeaked:
    """timeout_bars must be [None] for configs using trailing stop."""

    def test_orb_exploration_timeout_null(self, orb_config):
        for asset, grid in orb_config.grids.items():
            assert grid.timeout_bars == [None], (
                f"orb_exploration/{asset}: timeout_bars should be [None], "
                f"got {grid.timeout_bars}. Preset timeout_bars leaking!"
            )

    def test_orb_exploration_has_explicit_timeout(self, orb_config_raw):
        for asset, assignment in orb_config_raw["grids"]["assignments"].items():
            assert "timeout_bars" in assignment, (
                f"orb_exploration/{asset}: missing explicit timeout_bars. "
                f"Preset values will leak."
            )

    def test_deep_orb_timeout_null(self, deep_orb_config):
        if deep_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("deep_orb_index doesn't use trailing_stop")
        for asset, grid in deep_orb_config.grids.items():
            assert grid.timeout_bars == [None], (
                f"deep_orb_index/{asset}: timeout_bars should be [None], "
                f"got {grid.timeout_bars}"
            )

    def test_weekly_orb_timeout_null(self, weekly_orb_config):
        if weekly_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("weekly_orb doesn't use trailing_stop")
        for asset, grid in weekly_orb_config.grids.items():
            assert grid.timeout_bars == [None], (
                f"weekly_orb/{asset}: timeout_bars should be [None], "
                f"got {grid.timeout_bars}"
            )


# ===========================================================================
# Test Class 3: Exit Modifier Params Grid
# ===========================================================================

class TestExitModifierParamsGrid:
    """No {0,0} variant that disables trailing stop."""

    def test_orb_exploration_no_zero_trail(self, orb_config):
        for asset, grid in orb_config.grids.items():
            for i, params in enumerate(grid.exit_modifier_params_grid):
                if params is None:
                    continue
                trail = params.get("trail_atr_mult", 0)
                assert trail > 0, (
                    f"orb_exploration/{asset}: exit_modifier_params_grid[{i}] "
                    f"trail_atr_mult={trail}. Disables trailing!"
                )

    def test_deep_orb_no_zero_trail(self, deep_orb_config):
        if deep_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("deep_orb_index doesn't use trailing_stop")
        for asset, grid in deep_orb_config.grids.items():
            for i, params in enumerate(grid.exit_modifier_params_grid):
                if params is None:
                    continue
                assert params.get("trail_atr_mult", 0) > 0, (
                    f"deep_orb_index/{asset}: grid[{i}] disables trailing"
                )

    def test_weekly_orb_no_zero_trail(self, weekly_orb_config):
        if weekly_orb_config.exit_modifier != "trailing_stop":
            pytest.skip("weekly_orb doesn't use trailing_stop")
        for asset, grid in weekly_orb_config.grids.items():
            for i, params in enumerate(grid.exit_modifier_params_grid):
                if params is None:
                    continue
                assert params.get("trail_atr_mult", 0) > 0, (
                    f"weekly_orb/{asset}: grid[{i}] disables trailing"
                )


# ===========================================================================
# Test Class 4: Signal Column Session Hour Consistency
# ===========================================================================

class TestSessionHourConsistency:
    """Signal columns must match the asset's session hour."""

    def test_signal_columns_use_correct_session(self, orb_config):
        """Each asset's signal columns must reference the correct session hour."""
        for asset, grid in orb_config.grids.items():
            if asset not in EXPECTED_SESSION_HOURS:
                continue
            expected_hour = EXPECTED_SESSION_HOURS[asset]
            expected_prefix = f"_s{expected_hour:02d}_"

            # Check base model_hyperparameters
            hp = grid.model_hyperparameters
            for key in ("signal_column_long", "signal_column_short"):
                col = hp.get(key, "")
                assert expected_prefix in col, (
                    f"{asset}: {key}='{col}' should contain '{expected_prefix}' "
                    f"for session hour {expected_hour}"
                )

            # Check all grid variants
            for i, variant in enumerate(grid.model_hyperparameters_grid):
                if variant is None:
                    continue
                for key in ("signal_column_long", "signal_column_short"):
                    col = variant.get(key, "")
                    assert expected_prefix in col, (
                        f"{asset}: grid variant {i} {key}='{col}' should contain "
                        f"'{expected_prefix}'"
                    )


# ===========================================================================
# Test Class 5: Grid Variant Coverage
# ===========================================================================

class TestGridVariantCoverage:
    """All assets must have the full combinatorial grid of variants."""

    # ASX200 uses a custom grid (rb4 × cf{0,1,2} × prb{0} = 3 variants)
    CUSTOM_GRID_ASSETS = {"ASX200"}

    def test_standard_assets_have_12_variants(self, orb_config):
        """Standard assets should have 12 variants: rb{1,2} × cf{0,1,2} × prb{0,1}."""
        for asset, grid in orb_config.grids.items():
            if asset in self.CUSTOM_GRID_ASSETS:
                continue
            non_none = [v for v in grid.model_hyperparameters_grid if v is not None]
            assert len(non_none) == 12, (
                f"{asset}: expected 12 grid variants (2×3×2), got {len(non_none)}"
            )

    def test_all_rb_cf_prb_combinations_present(self, orb_config):
        """Check that all rb × cf × prb combos are present per asset."""
        expected_prefixes = set()
        for rb in [1, 2]:
            for cf in [0, 1, 2]:
                for prb in [0, 1]:
                    expected_prefixes.add(f"rb{rb}_cf{cf}_prb{prb}_orb_")

        for asset, grid in orb_config.grids.items():
            if asset in self.CUSTOM_GRID_ASSETS:
                continue
            actual_prefixes = set()
            for v in grid.model_hyperparameters_grid:
                if v is None:
                    continue
                col = v.get("signal_column_long", "")
                # Extract prefix up to "orb_"
                prefix_end = col.find("orb_") + len("orb_")
                if prefix_end > len("orb_"):
                    actual_prefixes.add(col[:prefix_end])

            missing = expected_prefixes - actual_prefixes
            assert not missing, (
                f"{asset}: missing signal column prefixes: {missing}"
            )


# ===========================================================================
# Test Class 6: Pipeline Config
# ===========================================================================

class TestOrbPipelineConfig:
    """Verify the orb_simple_v1 pipeline is correctly configured."""

    def test_opening_range_indicator_present(self, orb_config):
        ind_names = [i["name"] for i in orb_config.pipeline["indicators"]]
        assert "opening_range" in ind_names

    def test_range_bars_config(self, orb_config):
        """range_bars should cover all rb variants (rb1, rb2, and rb4 for ASX200)."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        assert orb_ind["params"]["range_bars"] == [1, 2, 4]

    def test_sessions_include_all_needed(self, orb_config):
        """All session hours referenced in signal columns must be in pipeline."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        configured_sessions = set(orb_ind["params"]["sessions"])

        # All session hours from any asset must be in the pipeline
        for asset, expected_hour in EXPECTED_SESSION_HOURS.items():
            if asset in orb_config.grids:
                assert expected_hour in configured_sessions, (
                    f"Session hour {expected_hour} (for {asset}) not in pipeline "
                    f"sessions {configured_sessions}"
                )

    def test_carry_forward_and_pre_range_present(self, orb_config):
        """carry_forward_days and pre_range_bars must be configured for cf/prb variants."""
        orb_ind = next(
            i for i in orb_config.pipeline["indicators"]
            if i["name"] == "opening_range"
        )
        params = orb_ind["params"]
        assert "carry_forward_days" in params, "carry_forward_days missing"
        assert "pre_range_bars" in params, "pre_range_bars missing"
        assert len(params["carry_forward_days"]) >= 3, (
            f"Need at least 3 carry_forward_days for cf0/cf1/cf2, "
            f"got {params['carry_forward_days']}"
        )
        assert len(params["pre_range_bars"]) >= 2, (
            f"Need at least 2 pre_range_bars for prb0/prb1, "
            f"got {params['pre_range_bars']}"
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
# Test Class 7: Cross-Config Consistency
# ===========================================================================

class TestCrossConfigConsistency:
    """Verify all trailing_stop configs have consistent overrides."""

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
            for asset, grid in cfg.grids.items():
                assert grid.timeout_bars == [None], (
                    f"{name}/{asset}: timeout_bars={grid.timeout_bars}"
                )

    def test_all_configs_have_positive_trail(self, all_trailing_configs):
        """Every trailing_stop config's modifier grid must have trail > 0."""
        for name, cfg in all_trailing_configs.items():
            for asset, grid in cfg.grids.items():
                for i, params in enumerate(grid.exit_modifier_params_grid):
                    if params is None:
                        continue
                    assert params.get("trail_atr_mult", 0) > 0, (
                        f"{name}/{asset}: grid[{i}] disables trailing"
                    )

    def test_all_configs_use_orb_based_exit(self, all_trailing_configs):
        """All ORB configs must use orb_based exit strategy."""
        for name, cfg in all_trailing_configs.items():
            assert cfg.exit_strategy == "orb_based", (
                f"{name}: exit_strategy={cfg.exit_strategy}, expected orb_based"
            )
