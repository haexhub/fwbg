"""
Tests für SimulationContext.

Fokus auf Edge Cases:
- Grid-Kombinationen Berechnung
- Separate Long/Short Grids
- Grenzwerte für Parameter
- Leere Grids
"""
import pytest

from fwbg.core.context import SimulationContext, TradeParams


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
        assert ctx.exit_strategy == "atr_based"
        assert ctx.min_fold_stability == 0.5
        assert ctx.early_termination is True

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


# --- Grid Functions Tests ---


class TestSimulationContextGrids:
    """Tests für Grid-Funktionen."""

    def test_get_long_grid_default(self):
        """get_long_grid sollte Standard-Grid zurückgeben."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20, 25],
            grid_sl=[10, 15, 20],
            grid_ct=[0.5, 0.55],
        )

        tp, sl, ct = ctx.get_long_grid()

        assert tp == [15, 20, 25]
        assert sl == [10, 15, 20]
        assert ct == [0.5, 0.55]

    def test_get_short_grid_default(self):
        """get_short_grid sollte Standard-Grid zurückgeben."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20, 25],
            grid_sl=[10, 15, 20],
            grid_ct=[0.5, 0.55],
        )

        tp, sl, ct = ctx.get_short_grid()

        assert tp == [15, 20, 25]
        assert sl == [10, 15, 20]
        assert ct == [0.5, 0.55]

    def test_separate_long_short_grids(self):
        """Separate Long/Short Grids sollten funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],
            grid_sl=[10, 15],
            grid_ct=[0.5],
            separate_long_short=True,
            long_grid_tp=[20, 25, 30],
            long_grid_sl=[15, 20],
            long_grid_ct=[0.52, 0.55],
            short_grid_tp=[10, 15],
            short_grid_sl=[20, 25],
            short_grid_ct=[0.5, 0.6],
        )

        long_tp, long_sl, long_ct = ctx.get_long_grid()
        short_tp, short_sl, short_ct = ctx.get_short_grid()

        # Long sollte separate Grids nutzen
        assert long_tp == [20, 25, 30]
        assert long_sl == [15, 20]
        assert long_ct == [0.52, 0.55]

        # Short sollte separate Grids nutzen
        assert short_tp == [10, 15]
        assert short_sl == [20, 25]
        assert short_ct == [0.5, 0.6]

    def test_separate_without_long_grid_fallback(self):
        """Ohne separate Grids sollte Fallback auf Standard erfolgen."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],
            grid_sl=[10, 15],
            grid_ct=[0.5],
            separate_long_short=True,
            # long_grid_* nicht gesetzt -> Fallback
        )

        long_tp, long_sl, long_ct = ctx.get_long_grid()

        # Sollte auf Standard fallen
        assert long_tp == [15, 20]
        assert long_sl == [10, 15]


class TestSimulationContextCombinations:
    """Tests für Grid-Kombinationen Berechnung."""

    def test_total_grid_combinations_simple(self):
        """Einfache Grid-Kombinationen sollten korrekt berechnet werden."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20, 25],  # 3
            grid_sl=[10, 15],  # 2
            grid_ct=[0.5],  # Nicht in Kombinationen
            feature_groups=["trend"],  # 1
        )

        # 3 TP × 2 SL × 1 Timeout × 1 Gruppe = 6
        assert ctx.total_grid_combinations() == 6

    def test_total_grid_combinations_with_timeout(self):
        """Timeout-Bars sollten in Kombinationen eingehen."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],  # 2
            grid_sl=[10, 15],  # 2
            grid_ct=[0.5],
            grid_timeout_bars=[None, 10, 20],  # 3
            feature_groups=["trend"],  # 1
        )

        # 2 × 2 × 3 × 1 = 12
        assert ctx.total_grid_combinations() == 12

    def test_total_grid_combinations_with_feature_groups(self):
        """Feature-Groups sollten in Kombinationen eingehen."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],  # 2
            grid_sl=[10],  # 1
            grid_ct=[0.5],
            feature_groups=["trend", "momentum", "volatility"],  # 3
        )

        # 2 × 1 × 1 × 3 = 6
        assert ctx.total_grid_combinations() == 6

    def test_total_grid_combinations_separate_long_short(self):
        """Separate Long/Short sollten korrekt addiert werden."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],
            grid_sl=[10, 15],
            grid_ct=[0.5],
            separate_long_short=True,
            long_grid_tp=[20, 25, 30],  # 3
            long_grid_sl=[15, 20],  # 2
            long_grid_ct=[0.5],
            short_grid_tp=[10, 15],  # 2
            short_grid_sl=[20],  # 1
            short_grid_ct=[0.5],
            feature_groups=["trend"],  # 1
        )

        # Long: 3 × 2 × 1 = 6
        # Short: 2 × 1 × 1 = 2
        # Total: (6 + 2) × 1 = 8
        assert ctx.total_grid_combinations() == 8

    def test_grid_combinations_per_feature_group(self):
        """Kombinationen pro Feature-Gruppe sollten korrekt sein."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20, 25],  # 3
            grid_sl=[10, 15],  # 2
            grid_ct=[0.5],
            feature_groups=["trend", "momentum"],  # Nicht relevant hier
        )

        # 3 × 2 × 1 (timeout) = 6
        assert ctx.grid_combinations_per_feature_group() == 6

    def test_grid_combinations_per_feature_group_separate(self):
        """Kombinationen pro Gruppe mit separate Long/Short."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],
            grid_sl=[10],
            grid_ct=[0.5],
            separate_long_short=True,
            long_grid_tp=[20, 25],  # 2
            long_grid_sl=[15],  # 1
            long_grid_ct=[0.5],
            short_grid_tp=[10, 15, 20],  # 3
            short_grid_sl=[20, 25],  # 2
            short_grid_ct=[0.5],
        )

        # Long: 2 × 1 = 2
        # Short: 3 × 2 = 6
        # Total: (2 + 6) × 1 (timeout) = 8
        assert ctx.grid_combinations_per_feature_group() == 8


class TestSimulationContextEdgeCases:
    """Edge Cases für SimulationContext."""

    def test_empty_grid_tp(self):
        """Leeres grid_tp sollte funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[],
            grid_sl=[10],
            grid_ct=[0.5],
        )

        assert ctx.total_grid_combinations() == 0

    def test_empty_grid_sl(self):
        """Leeres grid_sl sollte funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15, 20],
            grid_sl=[],
            grid_ct=[0.5],
        )

        assert ctx.total_grid_combinations() == 0

    def test_none_feature_groups(self):
        """None feature_groups sollte als 1 gezählt werden."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15],
            grid_sl=[10],
            grid_ct=[0.5],
            feature_groups=None,
        )

        # 1 × 1 × 1 × 1 = 1
        assert ctx.total_grid_combinations() == 1

    def test_none_timeout_bars(self):
        """None grid_timeout_bars sollte als 1 gezählt werden."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15],
            grid_sl=[10],
            grid_ct=[0.5],
            grid_timeout_bars=None,
        )

        assert ctx.total_grid_combinations() == 1

    def test_single_element_grids(self):
        """Grids mit einem Element sollten funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=[15],
            grid_sl=[10],
            grid_ct=[0.5],
            feature_groups=["trend"],
        )

        assert ctx.total_grid_combinations() == 1
        assert ctx.grid_combinations_per_feature_group() == 1

    def test_very_large_grids(self):
        """Sehr große Grids sollten korrekt berechnet werden."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            grid_tp=list(range(10, 60, 5)),  # 10 Werte
            grid_sl=list(range(10, 60, 5)),  # 10 Werte
            grid_ct=[0.5],
            grid_timeout_bars=[None, 10, 20, 30, 40],  # 5 Werte
            feature_groups=["trend", "momentum", "volatility"],  # 3 Gruppen
        )

        # 10 × 10 × 5 × 3 = 1500
        assert ctx.total_grid_combinations() == 1500

    def test_float_spread_point(self):
        """Float-Werte für spread und point sollten funktionieren."""
        ctx = SimulationContext(
            symbol="USDJPY",
            asset_class="FOREX",
            spread=0.01,  # 1 Pip für JPY
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
        """exit_params dict sollte funktionieren."""
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
