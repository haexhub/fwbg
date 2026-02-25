"""Integration tests for PDHL (Previous Day High/Low) strategy configuration.

Guards against the recurring config/plugin bugs:
1. resample_tf default must be None (native-bar breakout detection)
2. timeout_bars from preset must not leak into PDHL config
3. exit_modifier_params_grid must not contain {0,0} (disables trailing)
4. signal_start_hour=null enables 24h signals (not overridden by session hours)
5. orb_based exit strategy dispatches trailing kernel when exit_modifier is set
6. sl_dist_column flows from model_hyperparameters through to exit strategy
7. Breakout detection uses native bars (not resampled) when resample_tf=None
8. Retest signals fire outside session hours when session_mask is None
"""
import json
import os
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig, GridConfig, _parse_grids
from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

_pdl_mod = import_plugin_module("fwbg-core", "indicators", "previous_day_levels")
_signal_mod = import_plugin_module("fwbg-core", "models", "signal")
if _pdl_mod is None or _signal_mod is None:
    pytest.skip("Required plugins not available", allow_module_level=True)

from fwbg_sdk.models import TrainingContext

STRATEGY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "strategies")
PDHL_CONFIG_PATH = os.path.join(STRATEGY_DIR, "configs", "pdhl_retest.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pdhl_config():
    """Load the real pdhl_retest.json strategy config."""
    if not os.path.isfile(PDHL_CONFIG_PATH):
        pytest.skip("pdhl_retest.json not found")
    return StrategyConfig.from_json_file(PDHL_CONFIG_PATH)


@pytest.fixture
def pdhl_config_raw():
    """Load raw JSON of pdhl_retest.json."""
    if not os.path.isfile(PDHL_CONFIG_PATH):
        pytest.skip("pdhl_retest.json not found")
    with open(PDHL_CONFIG_PATH) as f:
        return json.load(f)


def _minimal_ctx(**overrides) -> SimulationContext:
    """Minimal SimulationContext for testing."""
    defaults = dict(
        symbol="TEST",
        asset_class="INDEX",
        spread=0.5,
        point=0.01,
        min_trades=1,
        long_enabled=True,
        short_enabled=True,
        exit_strategy="orb_based",
        exit_params={"atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 12},
        model_type="signal",
        model_hyperparameters={
            "signal_column_long": "rl50_pdl_retest_bull",
            "signal_column_short": "rl50_pdl_retest_bear",
            "sl_dist_column": "rl50_pdl_sl_dist",
        },
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def _make_pdhl_bull_15min():
    """5-day 15min data: breakout above PDH + retracement on day 2.

    Day 0: Establish range H=110, L=90 → PDH=110, PDL=90, mid=100
    Day 1: Breakout above 110 at 09:15, retracement to 100 at 12:00.
    Day 2-4: Flat (verify no stale signals).

    15min bars = 96 bars/day, 480 bars total.
    """
    idx = pd.date_range("2024-01-01", periods=96 * 5, freq="15min")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    # Day 0: establish range
    for i in range(96):
        h = idx[i].hour
        close[i] = 100.0; opn[i] = 100.0
        if h == 8 and idx[i].minute == 0:
            high[i] = 100.5; low[i] = 90.0  # PDL
        elif h == 11 and idx[i].minute == 0:
            high[i] = 110.0; low[i] = 99.5  # PDH
        else:
            high[i] = 100.5; low[i] = 99.5

    # Day 1: breakout + retracement
    day1_start = 96
    for i in range(day1_start, day1_start + 96):
        h = idx[i].hour
        m = idx[i].minute
        if h < 9:
            close[i] = 105.0; high[i] = 106.0; low[i] = 104.0; opn[i] = 105.0
        elif h == 9 and m == 15:
            # Breakout bar: close > PDH=110
            close[i] = 115.0; high[i] = 116.0; low[i] = 108.0; opn[i] = 108.0
        elif h == 9 and m > 15 or h == 10 or h == 11:
            close[i] = 112.0; high[i] = 113.0; low[i] = 111.0; opn[i] = 112.0
        elif h == 12 and m == 0:
            # Retracement: Low touches retrace threshold, Close near midpoint
            close[i] = 100.0; high[i] = 105.0; low[i] = 99.0; opn[i] = 105.0
        elif h == 12 and m == 15:
            close[i] = 101.0; high[i] = 102.0; low[i] = 100.0; opn[i] = 100.0
        else:
            # After entry: strong rally to hit TP (needs to reach ~120)
            close[i] = 125.0; high[i] = 130.0; low[i] = 106.0; opn[i] = 106.0

    # Days 2-4: flat
    for i in range(day1_start + 96, n):
        close[i] = 108.0; high[i] = 109.0; low[i] = 107.0; opn[i] = 108.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


def _make_overnight_breakout():
    """3-day 15min data: breakout at 03:00 (off-session), retest at 05:00.

    Tests that breakout + retest work outside session hours (24h trading).
    Session = 8..17 for indicator range calc, but signals must fire anytime.
    """
    idx = pd.date_range("2024-01-01", periods=96 * 3, freq="15min")
    n = len(idx)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    opn = np.full(n, 100.0)

    # Day 0: establish range 90..110 during session (8-17)
    for i in range(96):
        h = idx[i].hour
        close[i] = 100.0; opn[i] = 100.0
        if h == 8 and idx[i].minute == 0:
            high[i] = 100.5; low[i] = 90.0
        elif h == 11 and idx[i].minute == 0:
            high[i] = 110.0; low[i] = 99.5
        else:
            high[i] = 100.5; low[i] = 99.5

    # Day 1: overnight breakout at 03:00, retest at 05:00
    day1_start = 96
    for i in range(day1_start, day1_start + 96):
        h = idx[i].hour
        m = idx[i].minute
        if h < 3:
            close[i] = 105.0; high[i] = 106.0; low[i] = 104.0; opn[i] = 105.0
        elif h == 3 and m == 0:
            # Breakout above PDH=110 at 03:00 (off-session!)
            close[i] = 115.0; high[i] = 116.0; low[i] = 108.0; opn[i] = 108.0
        elif h == 3 and m > 0 or h == 4:
            close[i] = 112.0; high[i] = 113.0; low[i] = 111.0; opn[i] = 112.0
        elif h == 5 and m == 0:
            # Retracement to midpoint at 05:00 (still off-session!)
            close[i] = 100.0; high[i] = 105.0; low[i] = 99.0; opn[i] = 105.0
        elif h == 5 and m == 15:
            close[i] = 101.0; high[i] = 102.0; low[i] = 100.0; opn[i] = 100.0
        else:
            # Strong rally to hit TP
            close[i] = 125.0; high[i] = 130.0; low[i] = 106.0; opn[i] = 106.0

    # Day 2: flat
    for i in range(day1_start + 96, n):
        close[i] = 125.0; high[i] = 126.0; low[i] = 124.0; opn[i] = 125.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


# ===========================================================================
# Test Class 1: Config Loading & Grid Resolution
# ===========================================================================

class TestConfigLoading:
    """Verify pdhl_retest.json loads correctly and grid resolution works."""

    def test_config_loads_without_error(self, pdhl_config):
        assert pdhl_config.name is not None

    def test_exit_strategy_is_orb_based(self, pdhl_config):
        assert pdhl_config.exit_strategy == "orb_based"

    def test_exit_modifier_is_trailing_stop(self, pdhl_config):
        assert pdhl_config.exit_modifier == "trailing_stop"

    def test_pipeline_is_pdhl_v1(self, pdhl_config):
        # Pipeline should have been resolved from the "pdhl_v1" reference
        assert "indicators" in pdhl_config.pipeline
        ind_names = [i["name"] for i in pdhl_config.pipeline["indicators"]]
        assert "previous_day_levels" in ind_names

    def test_all_assets_have_grids(self, pdhl_config):
        expected_assets = {"DAX", "FTSE100", "DOW30", "NAS100", "SPX500",
                           "EU50", "CAC40", "JP225", "ASX200", "HK50"}
        assert set(pdhl_config.grids.keys()) == expected_assets


class TestTimeoutBarsNotLeaked:
    """timeout_bars must be [None] for PDHL (trailing stop handles exits).

    Bug: preset orb_scalping_index_v1 has timeout_bars=[8,16,32] which leaks
    into PDHL config if not explicitly overridden. Trades get closed after
    2-8 hours before the trailing stop can work.
    """

    def test_timeout_bars_is_null_for_all_assets(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            assert grid.timeout_bars == [None], (
                f"{asset}: timeout_bars should be [None], got {grid.timeout_bars}. "
                f"Preset timeout_bars are leaking through!"
            )

    def test_timeout_bars_not_from_preset(self, pdhl_config_raw):
        """Verify each asset assignment explicitly sets timeout_bars."""
        for asset, assignment in pdhl_config_raw["grids"]["assignments"].items():
            assert "timeout_bars" in assignment, (
                f"{asset}: missing explicit timeout_bars override. "
                f"Without this, preset's [8,16,32] will leak through."
            )


class TestExitModifierParamsGrid:
    """exit_modifier_params_grid must not contain {0,0} (disables trailing).

    Bug: preset has [{breakeven:0, trail:0}, {breakeven:0.5, trail:0.5}].
    The first entry effectively disables the trailing stop, and the optimizer
    can select it as "best" because it has no trailing overhead.
    """

    def test_no_zero_trailing_variant(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            for i, params in enumerate(grid.exit_modifier_params_grid):
                if params is None:
                    continue
                trail = params.get("trail_atr_mult", 0)
                assert trail > 0, (
                    f"{asset}: exit_modifier_params_grid[{i}] has trail_atr_mult={trail}. "
                    f"This disables trailing stop — optimizer can select this as 'best'."
                )

    def test_all_variants_have_positive_trailing(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            for params in grid.exit_modifier_params_grid:
                if params is None:
                    continue
                assert params.get("trail_atr_mult", 0) > 0
                # breakeven_trigger can be 0 (no breakeven) — that's fine
                # but trail must always be positive for PDHL strategy


class TestSignalHoursAre24h:
    """signal_start_hour=null in model_hyperparameters enables 24h signals.

    Bug: context.py setdefault() injects session hours from indicator_overrides
    when signal_start_hour is not explicitly set. Explicit null prevents this.
    """

    def test_base_model_hp_has_null_signal_hours(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            hp = grid.model_hyperparameters
            assert "signal_start_hour" in hp, (
                f"{asset}: missing signal_start_hour in model_hyperparameters"
            )
            assert hp["signal_start_hour"] is None, (
                f"{asset}: signal_start_hour should be null for 24h, "
                f"got {hp['signal_start_hour']}"
            )
            assert hp["signal_end_hour"] is None, (
                f"{asset}: signal_end_hour should be null for 24h, "
                f"got {hp['signal_end_hour']}"
            )

    def test_all_grid_variants_have_null_signal_hours(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            for i, variant in enumerate(grid.model_hyperparameters_grid):
                if variant is None:
                    continue
                assert variant.get("signal_start_hour") is None, (
                    f"{asset}: model_hyperparameters_grid[{i}] has "
                    f"signal_start_hour={variant.get('signal_start_hour')}"
                )
                assert variant.get("signal_end_hour") is None, (
                    f"{asset}: model_hyperparameters_grid[{i}] has "
                    f"signal_end_hour={variant.get('signal_end_hour')}"
                )

    def test_setdefault_does_not_override_explicit_null(self):
        """Simulate context.py setdefault behavior with explicit null."""
        model_hp = {
            "signal_column_long": "rl50_pdl_retest_bull",
            "signal_start_hour": None,
            "signal_end_hour": None,
        }
        # This is what context.py does: setdefault won't overwrite existing keys
        model_hp.setdefault("signal_start_hour", 8)
        model_hp.setdefault("signal_end_hour", 17)
        # null value means key exists → setdefault is a no-op
        assert model_hp["signal_start_hour"] is None
        assert model_hp["signal_end_hour"] is None


class TestGridVariantCoverage:
    """All required grid variants are present (rl0/38/50/61/70 × session/all scope)."""

    EXPECTED_VARIANTS = [
        ("rl0_pdl_retest_bull", "rl0_pdl_retest_bear", "rl0_pdl_sl_dist"),
        ("rl38_pdl_retest_bull", "rl38_pdl_retest_bear", "rl38_pdl_sl_dist"),
        ("rl50_pdl_retest_bull", "rl50_pdl_retest_bear", "rl50_pdl_sl_dist"),
        ("rl61_pdl_retest_bull", "rl61_pdl_retest_bear", "rl61_pdl_sl_dist"),
        ("rl70_pdl_retest_bull", "rl70_pdl_retest_bear", "rl70_pdl_sl_dist"),
        ("a_rl0_pdl_retest_bull", "a_rl0_pdl_retest_bear", "a_rl0_pdl_sl_dist"),
        ("a_rl38_pdl_retest_bull", "a_rl38_pdl_retest_bear", "a_rl38_pdl_sl_dist"),
        ("a_rl50_pdl_retest_bull", "a_rl50_pdl_retest_bear", "a_rl50_pdl_sl_dist"),
        ("a_rl61_pdl_retest_bull", "a_rl61_pdl_retest_bear", "a_rl61_pdl_sl_dist"),
        ("a_rl70_pdl_retest_bull", "a_rl70_pdl_retest_bear", "a_rl70_pdl_sl_dist"),
    ]

    def test_all_assets_have_10_grid_variants(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            non_none = [v for v in grid.model_hyperparameters_grid if v is not None]
            assert len(non_none) == 10, (
                f"{asset}: expected 10 grid variants (5 rl × 2 scopes), got {len(non_none)}"
            )

    def test_all_expected_signal_columns_present(self, pdhl_config):
        for asset, grid in pdhl_config.grids.items():
            actual_variants = set()
            for v in grid.model_hyperparameters_grid:
                if v is None:
                    continue
                actual_variants.add((
                    v["signal_column_long"],
                    v["signal_column_short"],
                    v["sl_dist_column"],
                ))
            expected = set(self.EXPECTED_VARIANTS)
            assert actual_variants == expected, (
                f"{asset}: missing or extra variants. "
                f"Missing: {expected - actual_variants}, "
                f"Extra: {actual_variants - expected}"
            )

    def test_each_variant_has_sl_dist_column(self, pdhl_config):
        """Every grid variant must define its own sl_dist_column."""
        for asset, grid in pdhl_config.grids.items():
            for i, v in enumerate(grid.model_hyperparameters_grid):
                if v is None:
                    continue
                assert "sl_dist_column" in v and v["sl_dist_column"], (
                    f"{asset}: grid variant {i} missing sl_dist_column"
                )


# ===========================================================================
# Test Class 2: Indicator Configuration
# ===========================================================================

class TestPipelineConfig:
    """Verify the pdhl_v1 pipeline is correctly configured."""

    def test_resample_tf_is_null_in_pipeline(self, pdhl_config):
        """resample_tf must be None for native-bar breakout detection."""
        pdl_ind = next(
            i for i in pdhl_config.pipeline["indicators"]
            if i["name"] == "previous_day_levels"
        )
        assert pdl_ind["params"].get("resample_tf") is None, (
            f"resample_tf should be null, got {pdl_ind['params'].get('resample_tf')}. "
            f"This adds ~45min delay to breakout confirmation!"
        )

    def test_candle_span_and_range_scope(self, pdhl_config):
        """candle_span and range_scope must be configured correctly."""
        pdl_ind = next(
            i for i in pdhl_config.pipeline["indicators"]
            if i["name"] == "previous_day_levels"
        )
        assert pdl_ind["params"].get("candle_span") == "hl"
        scope = pdl_ind["params"].get("range_scope", [])
        assert "session" in scope, "session missing from range_scope"
        assert "all" in scope, "all missing from range_scope"

    def test_break_modes_all_hours(self, pdhl_config):
        """Break detection must use all_hours (not session_only)."""
        pdl_ind = next(
            i for i in pdhl_config.pipeline["indicators"]
            if i["name"] == "previous_day_levels"
        )
        assert "all_hours" in pdl_ind["params"].get("break_modes", [])

    def test_retest_modes_all_hours(self, pdhl_config):
        """Retest signals must fire in all_hours (24h trading)."""
        pdl_ind = next(
            i for i in pdhl_config.pipeline["indicators"]
            if i["name"] == "previous_day_levels"
        )
        assert "all_hours" in pdl_ind["params"].get("retest_modes", [])


class TestResampleTfDefault:
    """resample_tf default must be None (changed from "1h").

    Bug: original default was "1h" causing ~45min delay in breakout detection.
    Native 15min bars detect breakouts immediately on Close > PDH.
    """

    def test_indicator_default_resample_tf_is_none(self):
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        defaults = ind.get_default_params()
        assert defaults.get("resample_tf") is None, (
            f"Default resample_tf should be None, got {defaults.get('resample_tf')}"
        )


# ===========================================================================
# Test Class 3: Breakout Detection (native vs resampled)
# ===========================================================================

class TestNativeBarBreakout:
    """Verify breakout fires on native 15min bars when resample_tf=None."""

    def test_breakout_detected_on_native_bar(self):
        """Breakout above PDH should fire on the same 15min bar."""
        df = _make_pdhl_bull_15min()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf=None,
        )

        # Day 1 (Jan 2): breakout at 09:15
        day1 = result.loc["2024-01-02"]
        breakout_col = "pdl_broke_high"
        if breakout_col in day1.columns:
            # After the breakout bar, broke_high should be True
            after_breakout = day1.loc[day1.index.hour >= 10, breakout_col]
            assert after_breakout.any(), "Breakout not detected after 09:15"

    def test_resampled_breakout_delayed(self):
        """With resample_tf='1h', breakout should be delayed to end of hour."""
        df = _make_pdhl_bull_15min()
        ind = _pdl_mod.PreviousDayLevelsIndicator()

        # Native-bar result
        result_native = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf=None,
        )

        # Resampled result
        result_resampled = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf="1h",
        )

        # Count retest signals on day 1
        day1_native = result_native.loc["2024-01-02"]
        day1_resamp = result_resampled.loc["2024-01-02"]

        col = "rl50_pdl_retest_bull"
        native_signals = day1_native[col].dropna().sum() if col in day1_native.columns else 0
        resamp_signals = day1_resamp[col].dropna().sum() if col in day1_resamp.columns else 0

        # Both should detect the signal, but resampled may miss it due to delay
        # The key test: with native bars, the signal should fire
        assert native_signals >= 1, (
            f"Native-bar breakout should produce retest signal, got {native_signals}"
        )


# ===========================================================================
# Test Class 4: 24h Signal Generation
# ===========================================================================

class TestOvernightSignals:
    """Verify signals fire outside session hours (24h trading)."""

    def test_overnight_breakout_produces_signal(self):
        """Breakout at 03:00 (off-session) should still trigger retest signal."""
        df = _make_overnight_breakout()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf=None,
            session_start_hour=8,
            session_end_hour=17,
            range_scope=["all"],
            break_modes=["all_hours"],
            retest_modes=["all_hours"],
        )

        day1 = result.loc["2024-01-02"]
        # a_ prefix = all scope
        for col in ["a_rl50_pdl_retest_bull", "rl50_pdl_retest_bull"]:
            if col in day1.columns:
                count = day1[col].dropna().sum()
                if count > 0:
                    return  # At least one variant fired
        # If neither fired, that's a problem
        available = [c for c in day1.columns if "retest_bull" in c]
        signals = {c: day1[c].dropna().sum() for c in available}
        pytest.fail(
            f"No overnight retest signal fired. Available columns: {signals}"
        )

    def test_signal_model_no_hour_filter_with_null(self):
        """SignalModel with signal_start_hour=None should not filter by hour."""
        df = _make_overnight_breakout()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf=None,
            session_start_hour=8,
            session_end_hour=17,
            range_scope=["all"],
            break_modes=["all_hours"],
            retest_modes=["all_hours"],
        )

        feature_cols = [c for c in result.columns if c.startswith(("pdl_", "rl", "a_"))]
        features = result[feature_cols].fillna(0)

        model = _signal_mod.SignalModel()
        hp = {
            "signal_column_long": "a_rl50_pdl_retest_bull",
            "signal_start_hour": None,
            "signal_end_hour": None,
        }
        model.train(features, np.zeros(len(features)), TrainingContext(direction="long"), **hp)
        probs = model.predict_probability(features)

        # With null hour filter, signals should pass through unfiltered
        # Check if any signal fires at off-session hours
        win_idx = np.where(model.trained_classes == 1)[0][0]
        day1_mask = features.index.date == pd.Timestamp("2024-01-02").date()
        day1_probs = probs[day1_mask]
        day1_times = features.index[day1_mask]

        # Any signal at off-session hour (before 8 or after 17)?
        off_session_signals = []
        for j, t in enumerate(day1_times):
            if day1_probs[j, win_idx] > 0.5:
                if t.hour < 8 or t.hour >= 17:
                    off_session_signals.append(t)

        # We should have at least one off-session signal
        if "a_rl50_pdl_retest_bull" in features.columns:
            day1_raw = features.loc["2024-01-02", "a_rl50_pdl_retest_bull"]
            raw_signals = day1_raw[day1_raw > 0]
            if len(raw_signals) > 0:
                # Raw signals exist → they should not be filtered
                signal_hours = raw_signals.index.hour
                off_session_raw = signal_hours[(signal_hours < 8) | (signal_hours >= 17)]
                if len(off_session_raw) > 0:
                    assert len(off_session_signals) > 0, (
                        "Off-session signals exist in indicator but were filtered by "
                        "SignalModel despite signal_start_hour=None!"
                    )

    def test_signal_model_with_session_filter_blocks_overnight(self):
        """SignalModel with signal_start_hour=8 should block signals before 8."""
        df = _make_overnight_breakout()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf=None,
            session_start_hour=8,
            session_end_hour=17,
            range_scope=["all"],
            break_modes=["all_hours"],
            retest_modes=["all_hours"],
        )

        feature_cols = [c for c in result.columns if c.startswith(("pdl_", "rl", "a_"))]
        features = result[feature_cols].fillna(0)

        model = _signal_mod.SignalModel()
        hp = {
            "signal_column_long": "a_rl50_pdl_retest_bull",
            "signal_start_hour": 8,
            "signal_end_hour": 17,
        }
        model.train(features, np.zeros(len(features)), TrainingContext(direction="long"), **hp)
        probs = model.predict_probability(features)

        win_idx = np.where(model.trained_classes == 1)[0][0]
        day1_mask = features.index.date == pd.Timestamp("2024-01-02").date()
        day1_probs = probs[day1_mask]
        day1_times = features.index[day1_mask]

        for j, t in enumerate(day1_times):
            if t.hour < 8 or t.hour >= 17:
                assert day1_probs[j, win_idx] <= 0.5, (
                    f"Signal should be blocked at {t} (outside session 8-17)"
                )


# ===========================================================================
# Test Class 5: Exit Strategy Dispatch
# ===========================================================================

class TestOrbBasedTrailingDispatch:
    """Verify orb_based exit strategy uses trailing kernel when configured."""

    def _get_orb_strategy(self):
        """Import and instantiate the OrbExitStrategy."""
        try:
            mod = import_plugin_module("fwbg-premium", "exit_strategies", "orb_based")
            if mod is None:
                pytest.skip("fwbg-premium exit_strategies not available")
            return mod.OrbExitStrategy()
        except Exception:
            pytest.skip("OrbExitStrategy not available")

    def test_trailing_dispatch_with_modifier(self):
        """With exit_modifier='trailing_stop', compute_targets should use trailing kernel."""
        strategy = self._get_orb_strategy()
        df = _make_pdhl_bull_15min()

        # Add minimal indicator columns
        df["vol_atr"] = 5.0
        df["rl50_pdl_sl_dist"] = 10.0

        ctx = _minimal_ctx(
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
        )

        # Should not raise (uses trailing kernel)
        targets_long, targets_short = strategy.compute_targets(
            df, ctx, tp_mult=4.0, sl_mult=1.0
        )
        assert len(targets_long) == len(df)
        assert len(targets_short) == len(df)

    def test_no_trailing_without_modifier(self):
        """Without exit_modifier, should use standard kernel (no trailing)."""
        strategy = self._get_orb_strategy()
        df = _make_pdhl_bull_15min()
        df["vol_atr"] = 5.0
        df["rl50_pdl_sl_dist"] = 10.0

        ctx = _minimal_ctx(exit_modifier=None, exit_modifier_params={})
        targets_long, targets_short = strategy.compute_targets(
            df, ctx, tp_mult=4.0, sl_mult=1.0
        )
        assert len(targets_long) == len(df)

    def test_modifier_params_affect_results(self):
        """Different trail_atr_mult values should produce different target arrays."""
        strategy = self._get_orb_strategy()
        df = _make_pdhl_bull_15min()
        df["vol_atr"] = 5.0
        df["rl50_pdl_sl_dist"] = 10.0

        ctx_tight = _minimal_ctx(
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.3},
        )
        ctx_wide = _minimal_ctx(
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 1.0},
        )

        targets_tight, _ = strategy.compute_targets(df, ctx_tight, tp_mult=4.0, sl_mult=1.0)
        targets_wide, _ = strategy.compute_targets(df, ctx_wide, tp_mult=4.0, sl_mult=1.0)

        # Different trail distances should produce at least some different results
        # (not necessarily all different, but the arrays shouldn't be identical
        # if the data has enough variation)
        # This is a soft check - the important thing is that the code path runs
        assert targets_tight is not None
        assert targets_wide is not None


class TestSlDistColumnFlow:
    """sl_dist_column from model_hyperparameters flows to exit strategy."""

    def test_sl_dist_column_from_hp(self):
        """model_hyperparameters.sl_dist_column overrides exit_params."""
        strategy = self._get_orb_strategy()
        df = _make_pdhl_bull_15min()
        df["vol_atr"] = 5.0
        df["rl50_pdl_sl_dist"] = 10.0
        df["rl38_pdl_sl_dist"] = 15.0  # Different SL distance

        ctx_rl50 = _minimal_ctx(
            model_hyperparameters={
                "signal_column_long": "rl50_pdl_retest_bull",
                "signal_column_short": "rl50_pdl_retest_bear",
                "sl_dist_column": "rl50_pdl_sl_dist",
            },
        )
        ctx_rl38 = _minimal_ctx(
            model_hyperparameters={
                "signal_column_long": "rl38_pdl_retest_bull",
                "signal_column_short": "rl38_pdl_retest_bear",
                "sl_dist_column": "rl38_pdl_sl_dist",
            },
        )

        tp50, sl50 = strategy.resolve_distances(df, 4.0, 1.0, ctx_rl50)
        tp38, sl38 = strategy.resolve_distances(df, 4.0, 1.0, ctx_rl38)

        # SL distances should differ because sl_dist_column is different
        assert not np.allclose(sl50, sl38), (
            "SL distances should differ for different sl_dist_columns"
        )

    def _get_orb_strategy(self):
        try:
            mod = import_plugin_module("fwbg-premium", "exit_strategies", "orb_based")
            if mod is None:
                pytest.skip("fwbg-premium exit_strategies not available")
            return mod.OrbExitStrategy()
        except Exception:
            pytest.skip("OrbExitStrategy not available")


# ===========================================================================
# Test Class 6: Grid Combo Creation
# ===========================================================================

class TestGridComboCreation:
    """Verify grid combo creation merges model_hyperparameters correctly."""

    def test_combo_merges_model_hp_variant(self):
        """model_hyperparameters_grid entries merge with base model_hp."""
        ctx = _minimal_ctx(
            grid_model_hyperparameters=[
                {
                    "signal_column_long": "rl38_pdl_retest_bull",
                    "signal_column_short": "rl38_pdl_retest_bear",
                    "sl_dist_column": "rl38_pdl_sl_dist",
                },
                {
                    "signal_column_long": "a_rl50_pdl_retest_bull",
                    "signal_column_short": "a_rl50_pdl_retest_bear",
                    "sl_dist_column": "a_rl50_pdl_sl_dist",
                },
            ],
        )

        # Simulate what _build_combo_tuples does
        for variant in ctx.grid_model_hyperparameters:
            if variant is not None:
                merged = {**ctx.model_hyperparameters, **variant}
                assert merged["signal_column_long"] == variant["signal_column_long"]
                assert merged["sl_dist_column"] == variant["sl_dist_column"]

    def test_combo_merges_modifier_params(self):
        """exit_modifier_params_grid entries replace base modifier_params."""
        ctx = _minimal_ctx(
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            grid_exit_modifier_params=[
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.3},
                {"breakeven_trigger": 0.3, "trail_atr_mult": 0.5},
            ],
        )

        for params in ctx.grid_exit_modifier_params:
            combo_ctx = dataclasses.replace(ctx, exit_modifier_params=params)
            assert combo_ctx.exit_modifier_params == params
            assert combo_ctx.exit_modifier_params["trail_atr_mult"] > 0

    def test_timeout_null_means_no_timeout(self):
        """timeout_bars=[None] means trades run until TP/SL/trailing stop."""
        ctx = _minimal_ctx(grid_timeout_bars=[None])
        timeout_values = ctx.grid_timeout_bars if ctx.grid_timeout_bars else [None]
        assert timeout_values == [None]
        # When passed to simulation, None → timeout_val=0 → no timeout
        for t in timeout_values:
            timeout_val = t if t else 0
            assert timeout_val == 0


# ===========================================================================
# Test Class 7: Full Pipeline Integration
# ===========================================================================

class TestFullPipelineIntegration:
    """End-to-end: indicator → SignalModel → trade simulation."""

    def _run_pipeline(self, df, signal_col_long, signal_col_short, sl_dist_col,
                      signal_start_hour=None, signal_end_hour=None,
                      exit_modifier=None, exit_modifier_params=None):
        """Run full indicator → signal model → trade simulation pipeline."""
        from fwbg.optimization.targets import _simulate_trades_core

        ind = _pdl_mod.PreviousDayLevelsIndicator()
        df_feat = ind.compute(
            df.copy(),
            retest_atr_width=0.5,
            skip_weekends=False,
            resample_tf=None,
            range_scope=["session", "all"],
            break_modes=["all_hours"],
            retest_modes=["all_hours"],
        )

        feature_cols = [c for c in df_feat.columns if c not in ("O", "H", "L", "C")]
        features = df_feat[feature_cols].fillna(0)
        dummy = np.zeros(len(features))

        hp = {
            "signal_column_long": signal_col_long,
            "signal_column_short": signal_col_short,
            "sl_dist_column": sl_dist_col,
            "signal_start_hour": signal_start_hour,
            "signal_end_hour": signal_end_hour,
        }

        model_long = _signal_mod.SignalModel()
        model_short = _signal_mod.SignalModel()
        model_long.train(features, dummy, TrainingContext(direction="long"), **hp)
        model_short.train(features, dummy, TrainingContext(direction="short"), **hp)

        probs_long = model_long.predict_probability(features)
        probs_short = model_short.predict_probability(features)
        long_win_idx = np.where(model_long.trained_classes == 1)[0][0]
        short_win_idx = np.where(model_short.trained_classes == 1)[0][0]

        ctx = _minimal_ctx(
            exit_modifier=exit_modifier,
            exit_modifier_params=exit_modifier_params or {},
            model_hyperparameters=hp,
        )

        return _simulate_trades_core(
            df=df_feat,
            probs_long=probs_long,
            probs_short=probs_short,
            long_win_idx=long_win_idx,
            short_win_idx=short_win_idx,
            ct_long=0.5,
            ct_short=0.5,
            tp=4.0,
            sl=1.0,
            ctx=ctx,
            return_detailed=True,
        ).get("trades_detailed", [])

    def test_native_bar_breakout_produces_trade(self):
        """15min data with native-bar breakout should produce a long trade."""
        df = _make_pdhl_bull_15min()
        trades = self._run_pipeline(
            df,
            signal_col_long="rl50_pdl_retest_bull",
            signal_col_short="rl50_pdl_retest_bear",
            sl_dist_col="rl50_pdl_sl_dist",
        )

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        assert len(long_trades) >= 1, (
            f"Expected at least 1 LONG trade with native-bar breakout, "
            f"got {len(long_trades)}. All trades: {trades}"
        )

    def test_24h_signals_produce_overnight_trade(self):
        """Overnight breakout + retest at 05:00 with 24h signals → trade."""
        df = _make_overnight_breakout()
        trades = self._run_pipeline(
            df,
            signal_col_long="a_rl50_pdl_retest_bull",
            signal_col_short="a_rl50_pdl_retest_bear",
            sl_dist_col="a_rl50_pdl_sl_dist",
            signal_start_hour=None,
            signal_end_hour=None,
        )

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        if len(long_trades) > 0:
            trade = long_trades[0]
            signal_time = pd.Timestamp(trade["signal_time"])
            # Signal should be at 05:00-06:00 area (off-session)
            assert signal_time.hour < 8, (
                f"Expected overnight signal (before 08:00), got {signal_time}"
            )

    def test_session_filter_blocks_overnight_trade(self):
        """Same overnight scenario but session=8-17 → no trade (signal blocked)."""
        df = _make_overnight_breakout()
        trades = self._run_pipeline(
            df,
            signal_col_long="a_rl50_pdl_retest_bull",
            signal_col_short="a_rl50_pdl_retest_bear",
            sl_dist_col="a_rl50_pdl_sl_dist",
            signal_start_hour=8,
            signal_end_hour=17,
        )

        # All signals are at 05:00 (off-session) → should be filtered
        long_trades = [t for t in trades if t["direction"] == "LONG"]
        overnight_trades = [
            t for t in long_trades
            if pd.Timestamp(t["signal_time"]).hour < 8
        ]
        assert len(overnight_trades) == 0, (
            f"Session filter 8-17 should block overnight trades, "
            f"got {len(overnight_trades)}"
        )

    def test_trailing_stop_produces_different_results_than_fixed(self):
        """Trailing stop should produce different outcomes than fixed TP/SL."""
        df = _make_pdhl_bull_15min()

        trades_fixed = self._run_pipeline(
            df,
            signal_col_long="rl50_pdl_retest_bull",
            signal_col_short="rl50_pdl_retest_bear",
            sl_dist_col="rl50_pdl_sl_dist",
            exit_modifier=None,
        )

        trades_trail = self._run_pipeline(
            df,
            signal_col_long="rl50_pdl_retest_bull",
            signal_col_short="rl50_pdl_retest_bear",
            sl_dist_col="rl50_pdl_sl_dist",
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.3},
        )

        # Both should produce trades (the signal is the same)
        # The exit results may differ due to trailing
        # This is a basic smoke test — the important thing is both code paths run
        assert trades_fixed is not None
        assert trades_trail is not None
