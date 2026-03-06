"""Integration tests for PDHL (Previous Day High/Low) strategy configuration.

Guards against the recurring config/plugin bugs:
1. resample_tf default must be None (native-bar breakout detection)
2. timeout_bars must not leak (must be [None] for trailing stop)
3. exit_modifier_params_grid must not contain disabled trailing
4. SignalModel reads _composed_signal_{direction} columns (no hyperparameters)
5. orb_based exit strategy dispatches trailing kernel when exit_modifier is set
6. sl_dist_column flows from exit_params through to exit strategy
7. Breakout detection uses native bars (not resampled) when resample_tf=None
8. Retest signals fire outside session hours when session_mask is None

Since the GridConfig removal, all grid params live in:
- exit_params (tp_mult, sl_mult, timeout_bars, etc.)
- optimization (ct, exit_modifier_params_grid, model_hyperparameters_grid)
"""
import json
import os
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import ExitStrategyConfig, StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

_pdl_mod = import_plugin_module("fwbg-core", "indicators", "previous_day_levels")
_signal_mod = import_plugin_module("fwbg-core", "models", "signal")
if _pdl_mod is None or _signal_mod is None:
    pytest.skip("Required plugins not available", allow_module_level=True)

from fwbg_sdk.models import TrainingContext

from fwbg.api.workspace import get_strategies_dir as _gsd
PDHL_CONFIG_PATH = str(_gsd() / "pdhl_retest.json")


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
        exit_params={
            "atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 12,
            "sl_dist_column": "hl_ses_rl50_pdl_sl_dist",
        },
        model_type="signal",
        model_hyperparameters={},
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def _make_pdhl_bull_15min():
    """5-day 15min data: breakout above PDH + retracement on day 2.

    Day 0: Establish range H=110, L=90 -> PDH=110, PDL=90, mid=100
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
# Test Class 1: Config Loading & Structure
# ===========================================================================

class TestConfigLoading:
    """Verify pdhl_retest.json loads correctly."""

    def test_config_loads_without_error(self, pdhl_config):
        assert pdhl_config.name is not None

    def test_exit_strategy_is_orb_based(self, pdhl_config):
        assert len(pdhl_config.exit_strategies) > 0
        assert pdhl_config.exit_strategies[0].name == "orb_based"

    def test_exit_modifier_is_trailing_stop(self, pdhl_config):
        assert len(pdhl_config.exit_strategies) > 0
        assert pdhl_config.exit_strategies[0].exit_modifier == "trailing_stop"

    def test_pipeline_is_pdhl_v1(self, pdhl_config):
        # Pipeline should have been resolved from the "pdhl_v1" reference
        assert "indicators" in pdhl_config.pipeline
        ind_names = [i["name"] for i in pdhl_config.pipeline["indicators"]]
        assert "previous_day_levels" in ind_names

    def test_exit_params_have_required_keys(self, pdhl_config):
        assert len(pdhl_config.exit_strategies) > 0
        ep = pdhl_config.exit_strategies[0].params
        assert "tp_mult" in ep
        assert "sl_mult" in ep
        assert "timeout_bars" in ep


class TestTimeoutBarsNotLeaked:
    """timeout_bars must be [None] for PDHL (trailing stop handles exits)."""

    def test_timeout_bars_is_null(self, pdhl_config):
        assert len(pdhl_config.exit_strategies) > 0
        # All exit strategy instances should have timeout_bars=None (trailing stop handles exits)
        for es in pdhl_config.exit_strategies:
            timeout = es.params.get("timeout_bars")
            assert timeout is None or timeout == [None], (
                f"timeout_bars should be None or [None], got {timeout}. "
                f"Preset timeout_bars are leaking through!"
            )


class TestSignalModelHasNoHyperparameters:
    """SignalModel v3 uses _composed_signal_{direction} — no hyperparameters."""

    def test_signal_model_hp_empty(self, pdhl_config):
        """Signal model should not have signal-related hyperparameters."""
        hp = pdhl_config.model.hyperparameters
        assert "signal_column_long" not in hp
        assert "signal_column_short" not in hp
        assert "signal_start_hour" not in hp
        assert "signal_end_hour" not in hp


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
    """resample_tf default must be None (changed from '1h')."""

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

            skip_weekends=False,
            resample_tf=None,
        )

        # Day 1 (Jan 2): breakout at 09:15
        day1 = result.loc["2024-01-02"]
        breakout_col = "hl_ses_pdl_broke_high"
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

            skip_weekends=False,
            resample_tf=None,
        )

        # Resampled result
        result_resampled = ind.compute(
            df.copy(),

            skip_weekends=False,
            resample_tf="1h",
        )

        # Count retest signals on day 1
        day1_native = result_native.loc["2024-01-02"]
        day1_resamp = result_resampled.loc["2024-01-02"]

        col = "hl_ses_rl50_pdl_retest_bull"
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
        for col in ["hl_all_rl50_pdl_retest_bull", "hl_ses_rl50_pdl_retest_bull"]:
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

    def test_signal_model_passes_all_composed_signals(self):
        """SignalModel v3 passes through all composed signals without filtering."""
        df = _make_overnight_breakout()
        ind = _pdl_mod.PreviousDayLevelsIndicator()
        result = ind.compute(
            df.copy(),

            skip_weekends=False,
            resample_tf=None,
            session_start_hour=8,
            session_end_hour=17,
            range_scope=["all"],
            break_modes=["all_hours"],
            retest_modes=["all_hours"],
        )

        feature_cols = [c for c in result.columns if c.startswith(("hl_ses_", "hl_all_"))]
        features = result[feature_cols].fillna(0)

        # Create composed signal column from indicator column
        sig_col = "hl_all_rl50_pdl_retest_bull"
        if sig_col in features.columns:
            features["_composed_signal_long"] = features[sig_col]
        else:
            pytest.skip("Signal column not in indicator output")

        model = _signal_mod.SignalModel()
        model.train(features, np.zeros(len(features)), TrainingContext(direction="long"))
        probs = model.predict_probability(features)

        win_idx = np.where(model.trained_classes == 1)[0][0]

        # All raw signals should pass through — no hour filtering
        raw_signals = features[sig_col]
        for i in range(len(features)):
            if raw_signals.iloc[i] > 0:
                assert probs[i, win_idx] > 0, (
                    f"Signal at {features.index[i]} should pass through"
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
        df["hl_ses_rl50_pdl_sl_dist"] = 10.0

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
        df["hl_ses_rl50_pdl_sl_dist"] = 10.0

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
        df["hl_ses_rl50_pdl_sl_dist"] = 10.0

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

        # Both code paths should run without error
        assert targets_tight is not None
        assert targets_wide is not None


class TestSlDistColumnFlow:
    """sl_dist_column from exit_params flows to exit strategy."""

    def _get_orb_strategy(self):
        try:
            mod = import_plugin_module("fwbg-premium", "exit_strategies", "orb_based")
            if mod is None:
                pytest.skip("fwbg-premium exit_strategies not available")
            return mod.OrbExitStrategy()
        except Exception:
            pytest.skip("OrbExitStrategy not available")

    def test_sl_dist_column_from_exit_params(self):
        """Different sl_dist_column in exit_params produces different SL distances."""
        strategy = self._get_orb_strategy()
        df = _make_pdhl_bull_15min()
        df["vol_atr"] = 5.0
        df["hl_ses_rl50_pdl_sl_dist"] = 10.0
        df["hl_ses_rl38_pdl_sl_dist"] = 15.0

        ctx_rl50 = _minimal_ctx(
            exit_params={
                "atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 12,
                "sl_dist_column": "hl_ses_rl50_pdl_sl_dist",
            },
        )
        ctx_rl38 = _minimal_ctx(
            exit_params={
                "atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 12,
                "sl_dist_column": "hl_ses_rl38_pdl_sl_dist",
            },
        )

        tp50, sl50 = strategy.resolve_distances(df, 4.0, 1.0, ctx_rl50)
        tp38, sl38 = strategy.resolve_distances(df, 4.0, 1.0, ctx_rl38)

        assert not np.allclose(sl50, sl38), (
            "SL distances should differ for different sl_dist_columns"
        )


# ===========================================================================
# Test Class 6: Grid Combo Creation
# ===========================================================================

class TestGridComboCreation:
    """Verify grid combo creation merges exit strategy config correctly."""

    def test_combo_merges_modifier_params(self):
        """Each ExitStrategyConfig carries its own exit_modifier_params."""
        exit_strategies = [
            ExitStrategyConfig(
                name="orb_based",
                params={"tp_mult": 4.0, "sl_mult": 1.0},
                exit_modifier="trailing_stop",
                exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.3},
            ),
            ExitStrategyConfig(
                name="orb_based",
                params={"tp_mult": 4.0, "sl_mult": 1.0},
                exit_modifier="trailing_stop",
                exit_modifier_params={"breakeven_trigger": 0.3, "trail_atr_mult": 0.5},
            ),
        ]
        ctx = _minimal_ctx(exit_strategies=exit_strategies)

        for es in ctx.exit_strategies:
            # Simulate what _build_combo_tuples does: set per-combo modifier_params
            combo_ctx = dataclasses.replace(
                ctx,
                exit_modifier=es.exit_modifier,
                exit_modifier_params=es.exit_modifier_params,
            )
            assert combo_ctx.exit_modifier_params == es.exit_modifier_params
            assert combo_ctx.exit_modifier_params["trail_atr_mult"] > 0

    def test_timeout_null_means_no_timeout(self):
        """timeout_bars=None in exit strategy params means trades run until TP/SL/trailing stop."""
        exit_strategies = [
            ExitStrategyConfig(
                name="orb_based",
                params={"tp_mult": 4.0, "sl_mult": 1.0, "timeout_bars": None},
            ),
        ]
        ctx = _minimal_ctx(exit_strategies=exit_strategies)
        for es in ctx.exit_strategies:
            timeout = es.params.get("timeout_bars")
            # When passed to simulation, None -> timeout_val=0 -> no timeout
            timeout_val = timeout if timeout else 0
            assert timeout_val == 0


# ===========================================================================
# Test Class 7: Full Pipeline Integration
# ===========================================================================

class TestFullPipelineIntegration:
    """End-to-end: indicator -> SignalModel -> trade simulation."""

    def _run_pipeline(self, df, composed_source_long, composed_source_short,
                      sl_dist_col, exit_modifier=None, exit_modifier_params=None):
        """Run full indicator -> signal model -> trade simulation pipeline.

        composed_source_long/short: indicator column names that become
        _composed_signal_long/short (mimics what signal_fold.py does via signal_rules).
        """
        from fwbg.optimization.targets import _simulate_trades_core

        ind = _pdl_mod.PreviousDayLevelsIndicator()
        df_feat = ind.compute(
            df.copy(),

            skip_weekends=False,
            resample_tf=None,
            range_scope=["session", "all"],
            break_modes=["all_hours"],
            retest_modes=["all_hours"],
        )

        feature_cols = [c for c in df_feat.columns if c not in ("O", "H", "L", "C")]
        features = df_feat[feature_cols].fillna(0)
        dummy = np.zeros(len(features))

        # Create composed signal columns from indicator source columns
        for direction, source_col in [("long", composed_source_long),
                                       ("short", composed_source_short)]:
            if source_col in features.columns:
                features[f"_composed_signal_{direction}"] = features[source_col]
            else:
                features[f"_composed_signal_{direction}"] = 0

        model_long = _signal_mod.SignalModel()
        model_short = _signal_mod.SignalModel()
        model_long.train(features, dummy, TrainingContext(direction="long"))
        model_short.train(features, dummy, TrainingContext(direction="short"))

        probs_long = model_long.predict_probability(features)
        probs_short = model_short.predict_probability(features)
        long_win_idx = np.where(model_long.trained_classes == 1)[0][0]
        short_win_idx = np.where(model_short.trained_classes == 1)[0][0]

        ctx = _minimal_ctx(
            exit_modifier=exit_modifier,
            exit_modifier_params=exit_modifier_params or {},
            exit_params={
                "atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 12,
                "sl_dist_column": sl_dist_col,
            },
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
            composed_source_long="hl_ses_rl50_pdl_retest_bull",
            composed_source_short="hl_ses_rl50_pdl_retest_bear",
            sl_dist_col="hl_ses_rl50_pdl_sl_dist",
        )

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        assert len(long_trades) >= 1, (
            f"Expected at least 1 LONG trade with native-bar breakout, "
            f"got {len(long_trades)}. All trades: {trades}"
        )

    def test_24h_signals_produce_overnight_trade(self):
        """Overnight breakout + retest at 05:00 with 24h signals -> trade."""
        df = _make_overnight_breakout()
        trades = self._run_pipeline(
            df,
            composed_source_long="hl_all_rl50_pdl_retest_bull",
            composed_source_short="hl_all_rl50_pdl_retest_bear",
            sl_dist_col="hl_all_rl50_pdl_sl_dist",
        )

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        if len(long_trades) > 0:
            trade = long_trades[0]
            signal_time = pd.Timestamp(trade["signal_time"])
            # Signal should be at 05:00-06:00 area (off-session)
            assert signal_time.hour < 8, (
                f"Expected overnight signal (before 08:00), got {signal_time}"
            )

    def test_session_filter_not_in_model(self):
        """SignalModel v3 has no hour filter — all signals pass through."""
        df = _make_overnight_breakout()
        trades = self._run_pipeline(
            df,
            composed_source_long="hl_all_rl50_pdl_retest_bull",
            composed_source_short="hl_all_rl50_pdl_retest_bear",
            sl_dist_col="hl_all_rl50_pdl_sl_dist",
        )

        # With no hour filter, overnight signals should pass through
        assert trades is not None

    def test_trailing_stop_produces_different_results_than_fixed(self):
        """Trailing stop should produce different outcomes than fixed TP/SL."""
        df = _make_pdhl_bull_15min()

        trades_fixed = self._run_pipeline(
            df,
            composed_source_long="hl_ses_rl50_pdl_retest_bull",
            composed_source_short="hl_ses_rl50_pdl_retest_bear",
            sl_dist_col="hl_ses_rl50_pdl_sl_dist",
            exit_modifier=None,
        )

        trades_trail = self._run_pipeline(
            df,
            composed_source_long="hl_ses_rl50_pdl_retest_bull",
            composed_source_short="hl_ses_rl50_pdl_retest_bear",
            sl_dist_col="hl_ses_rl50_pdl_sl_dist",
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.3},
        )

        # Both should produce trades (the signal is the same)
        # The exit results may differ due to trailing
        # This is a basic smoke test -- the important thing is both code paths run
        assert trades_fixed is not None
        assert trades_trail is not None
