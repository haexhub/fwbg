"""
Tests für SimulationContext.

Fokus auf Edge Cases:
- Grid-Kombinationen Berechnung (exit_strategies × model_hp)
- Grenzwerte für Parameter
- Per-combo Felder
"""
import pytest

from fwbg.core.context import SimulationContext, TradeParams
from fwbg.core.config import ExitStrategyConfig, StrategyConfig
from fwbg.data.assets import AssetConfig


# --- TradeParams Tests ---


class TestTradeParams:
    """Tests für TradeParams Dataclass."""

    def test_creates_with_defaults(self):
        """Sollte mit Default-Werten erstellt werden."""
        params = TradeParams(tp=30, sl=20)

        assert params.tp == 30
        assert params.sl == 20
        assert params.ct == 0.5  # Default
        assert params.timeout_bars is None  # Default

    def test_rrr_calculation(self):
        """RRR sollte korrekt berechnet werden."""
        params = TradeParams(tp=30, sl=10)
        assert params.rrr == 3.0

        params = TradeParams(tp=20, sl=20)
        assert params.rrr == 1.0

        params = TradeParams(tp=10, sl=20)
        assert params.rrr == 0.5

    def test_rrr_zero_sl(self):
        """RRR bei SL=0 sollte 0 zurückgeben (nicht Division by Zero)."""
        params = TradeParams(tp=30, sl=0)
        assert params.rrr == 0

    def test_negative_values_allowed(self):
        """Negative Werte sollten erlaubt sein (Validierung ist extern)."""
        params = TradeParams(tp=-10, sl=-20)
        assert params.tp == -10
        assert params.sl == -20

    def test_float_values(self):
        """Float-Werte sollten funktionieren."""
        params = TradeParams(tp=1.5, sl=1.0, ct=0.55)
        assert params.tp == 1.5
        assert params.sl == 1.0
        assert params.ct == 0.55
        assert params.rrr == 1.5


# --- Plan 009 WP4: backtest window + cost multiplier ---


class TestBacktestWindowAndCostMultiplier:
    """start_date/end_date/cost_multiplier config fields + spread scaling."""

    def test_strategy_config_round_trips_new_fields(self):
        sc = StrategyConfig.from_dict(
            {
                "name": "t",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "cost_multiplier": 2.0,
            }
        )
        assert sc.start_date == "2024-01-01"
        assert sc.end_date == "2024-06-30"
        assert sc.cost_multiplier == 2.0

    def test_new_fields_default_to_full_series_and_unit_cost(self):
        sc = StrategyConfig.from_dict({"name": "t"})
        assert sc.start_date is None
        assert sc.end_date is None
        assert sc.cost_multiplier == 1.0

    def test_cost_multiplier_scales_spread_into_context(self):
        asset = AssetConfig(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0002,
            point=0.00001,
            currencies=["EUR", "USD"],
        )
        base = SimulationContext.create(asset, StrategyConfig.from_dict({"name": "t"}))
        assert base.spread == pytest.approx(0.0002)

        stressed = SimulationContext.create(
            asset, StrategyConfig.from_dict({"name": "t", "cost_multiplier": 2.0})
        )
        assert stressed.spread == pytest.approx(0.0004)


# --- SimulationContext Basic Tests ---


class TestSimulationContextBasic:
    """Grundlegende Tests für SimulationContext."""

    def test_creates_with_minimal_params(self):
        """Sollte mit minimalen Parametern erstellt werden."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
        )

        assert ctx.symbol == "EURUSD"
        assert ctx.asset_class == "FOREX"
        assert ctx.spread == 0.0001
        assert ctx.point == 0.00001

    def test_default_values(self):
        """Default-Werte sollten gesetzt sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
        )

        assert ctx.min_trades == 50
        assert ctx.min_rrr == 0.0
        assert ctx.long_enabled is True
        assert ctx.short_enabled is True
        assert ctx.exit_strategy == "fixed"
        assert ctx.min_fold_stability == 0.5
        assert ctx.early_termination is True
        assert ctx.exit_modifier is None
        assert ctx.exit_modifier_params == {}
        assert ctx.exit_strategies == []

    def test_exit_modifier_can_be_set(self):
        """exit_modifier und exit_modifier_params sollten setzbar sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
        )
        assert ctx.exit_modifier == "trailing_stop"
        assert ctx.exit_modifier_params == {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5}

    def test_custom_values(self):
        """Custom-Werte sollten übernommen werden."""
        ctx = SimulationContext(
            symbol="BTCUSD",
            asset_class="CRYPTO",
            spread=50.0,
            point=0.01,
            min_trades=100,
            min_rrr=1.5,
            long_enabled=True,
            short_enabled=False,
        )

        assert ctx.symbol == "BTCUSD"
        assert ctx.min_trades == 100
        assert ctx.min_rrr == 1.5
        assert ctx.short_enabled is False

    def test_exit_strategies_list(self):
        """exit_strategies Liste sollte setzbar sein."""
        es1 = ExitStrategyConfig(name="fixed", params={"tp_mult": 2.0, "sl_mult": 1.0})
        es2 = ExitStrategyConfig(name="atr_based", params={"atr_period": 14}, ct=[0.5, 0.55])
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategies=[es1, es2],
        )

        assert len(ctx.exit_strategies) == 2
        assert ctx.exit_strategies[0].name == "fixed"
        assert ctx.exit_strategies[1].name == "atr_based"


# --- Grid Combinations Tests ---


class TestSimulationContextCombinations:
    """Tests für Grid-Kombinationen Berechnung (exit_strategies × model_hp)."""

    def test_total_grid_combinations_single_exit_strategy(self):
        """Einzelne Exit-Strategie: CT values are inner CV, not grid combos."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategies=[
                ExitStrategyConfig(name="fixed", params={}, ct=[0.5, 0.55, 0.6]),
            ],
        )
        # 1 exit_strategy × 1 model HP = 1
        assert ctx.total_grid_combinations() == 1

    def test_total_grid_combinations_multiple_exit_strategies(self):
        """Mehrere Exit-Strategien: eine Combo pro Strategie."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategies=[
                ExitStrategyConfig(name="fixed", params={"tp_mult": 2.0}, ct=[0.5, 0.55]),
                ExitStrategyConfig(name="fixed", params={"tp_mult": 3.0}, ct=[0.5]),
                ExitStrategyConfig(name="atr_based", params={}, ct=[0.5, 0.6]),
            ],
        )
        # 3 exit_strategies × 1 model HP = 3
        assert ctx.total_grid_combinations() == 3

    def test_total_grid_combinations_with_model_hp(self):
        """Exit-Strategien × Model-Hyperparameters."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategies=[
                ExitStrategyConfig(name="fixed", params={}, ct=[0.5, 0.55]),
            ],
            grid_model_hyperparameters=[
                {"n_estimators": 100},
                {"n_estimators": 200},
                {"n_estimators": 300},
            ],
        )
        # 1 exit_strategy × 3 model HP = 3
        assert ctx.total_grid_combinations() == 3

    def test_total_grid_combinations_no_exit_strategies(self):
        """Ohne Exit-Strategien: n_exit = 1 (default)."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
        )
        # 1 (default) × 1 model HP = 1
        assert ctx.total_grid_combinations() == 1

    def test_total_grid_combinations_empty_exit_strategies(self):
        """Leere exit_strategies Liste: n_exit = 1 (default)."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategies=[],
        )
        assert ctx.total_grid_combinations() == 1

    def test_total_grid_combinations_complex(self):
        """Komplexes Grid mit vielen Strategien und Model-HPs."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategies=[
                ExitStrategyConfig(name="fixed", params={"tp_mult": 1.5}, ct=[0.5]),
                ExitStrategyConfig(name="fixed", params={"tp_mult": 2.0}, ct=[0.5, 0.55]),
                ExitStrategyConfig(name="fixed", params={"tp_mult": 2.5}, ct=[0.5]),
                ExitStrategyConfig(name="atr_based", params={"atr_period": 14}, ct=[0.5, 0.55, 0.6]),
            ],
            grid_model_hyperparameters=[
                {"n_estimators": 100},
                {"n_estimators": 200},
            ],
        )
        # 4 exit_strategies × 2 model HP = 8
        assert ctx.total_grid_combinations() == 8


# --- Edge Cases ---


class TestSimulationContextEdgeCases:
    """Edge Cases für SimulationContext."""

    def test_none_indicator_plugins(self):
        """None indicator_plugins sollte funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            indicator_plugins=None,
        )
        assert ctx.indicator_plugins is None

    def test_float_spread_point(self):
        """Float-Werte für spread und point sollten funktionieren."""
        ctx = SimulationContext(
            symbol="USDJPY",
            asset_class="FOREX",
            spread=0.01,
            point=0.001,
        )

        assert ctx.spread == 0.01
        assert ctx.point == 0.001

    def test_zero_spread(self):
        """Null-Spread sollte erlaubt sein."""
        ctx = SimulationContext(
            symbol="TEST",
            asset_class="TEST",
            spread=0.0,
            point=0.00001,
        )

        assert ctx.spread == 0.0

    def test_currencies_list(self):
        """Currencies-Liste sollte funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            currencies=["EUR", "USD"],
        )

        assert ctx.currencies == ["EUR", "USD"]

    def test_exit_params_dict(self):
        """exit_params dict sollte funktionieren (per-combo field)."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            exit_strategy="atr_based",
            exit_params={"atr_period": 14, "min_tp_pips": 10},
        )

        assert ctx.exit_params["atr_period"] == 14
        assert ctx.exit_params["min_tp_pips"] == 10


class TestSimulationContextTradeDirections:
    """Tests für Trade-Richtungen."""

    def test_both_directions_enabled_default(self):
        """Beide Richtungen sollten per Default aktiviert sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
        )

        assert ctx.long_enabled is True
        assert ctx.short_enabled is True

    def test_long_only(self):
        """Nur Long sollte konfigurierbar sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            long_enabled=True,
            short_enabled=False,
        )

        assert ctx.long_enabled is True
        assert ctx.short_enabled is False

    def test_short_only(self):
        """Nur Short sollte konfigurierbar sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            long_enabled=False,
            short_enabled=True,
        )

        assert ctx.long_enabled is False
        assert ctx.short_enabled is True

    def test_neither_direction(self):
        """Keine Richtung sollte konfigurierbar sein (auch wenn sinnlos)."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            long_enabled=False,
            short_enabled=False,
        )

        assert ctx.long_enabled is False
        assert ctx.short_enabled is False


class TestSimulationContextEarlyTermination:
    """Tests für Early Termination Parameter."""

    def test_early_termination_defaults(self):
        """Early Termination Defaults sollten gesetzt sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
        )

        assert ctx.early_termination is True
        assert ctx.min_fold_stability == 0.5
        assert ctx.first_fold_sanity_check is True
        assert ctx.first_fold_min_win_rate == 0.25
        assert ctx.first_fold_min_pnl == -10.0
        assert ctx.first_fold_min_trades == 5

    def test_early_termination_disabled(self):
        """Early Termination sollte deaktivierbar sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            early_termination=False,
            first_fold_sanity_check=False,
        )

        assert ctx.early_termination is False
        assert ctx.first_fold_sanity_check is False

    def test_custom_early_termination_params(self):
        """Custom Early Termination Parameter sollten funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            min_fold_stability=0.7,
            first_fold_min_win_rate=0.30,
            first_fold_min_pnl=-5.0,
            first_fold_min_trades=10,
        )

        assert ctx.min_fold_stability == 0.7
        assert ctx.first_fold_min_win_rate == 0.30
        assert ctx.first_fold_min_pnl == -5.0
        assert ctx.first_fold_min_trades == 10
