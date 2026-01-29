"""
SimulationContext - Kontext-Objekt für Trade-Simulationen.

Wird durch den gesamten Simulationsprozess gereicht und enthält alle
Parameter die für eine einzelne Asset-Optimierung benötigt werden.
"""
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .strategy_config import StrategyConfig
    from .asset_config import AssetConfig


@dataclass
class TradeParams:
    """
    Parameter für eine einzelne Trade-Konfiguration.

    Wird durch Grid-Search iteriert und an Simulationsfunktionen übergeben.
    Kapselt alle Parameter die pro Kombination variieren können.
    """
    tp: int  # Take-Profit Multiplikator
    sl: int  # Stop-Loss Multiplikator
    ct: float = 0.5  # Confidence Threshold (oder Tuple für separate L/S)
    timeout_bars: Optional[int] = None  # Time-based Exit (None = kein Timeout)

    @property
    def rrr(self) -> float:
        """Risk-Reward-Ratio."""
        return self.tp / self.sl if self.sl > 0 else 0


@dataclass
class SimulationContext:
    """
    Kontext für eine Asset-Optimierung.

    Dieses Objekt wird einmal pro Asset erstellt und durch alle
    Funktionen der Optimierungs-Pipeline gereicht.
    """
    # Asset-spezifisch
    symbol: str
    asset_class: str
    spread: float
    point: float
    currencies: List[str]

    # Aus StrategyConfig
    min_trades: int
    min_rrr: float

    # Gemeinsames Grid für dieses Asset (für combined Optimierung)
    grid_tp: List[int]
    grid_sl: List[int]
    grid_ct: List[float]
    grid_timeout_bars: List[int] = None  # Liste von Timeout-Werten zum Testen (None = kein Timeout)

    # Separate Grids für Long/Short (None = verwende gemeinsames Grid)
    long_grid_tp: List[int] = None
    long_grid_sl: List[int] = None
    long_grid_ct: List[float] = None
    short_grid_tp: List[int] = None
    short_grid_sl: List[int] = None
    short_grid_ct: List[float] = None

    # Flag für separate L/S Optimierung
    separate_long_short: bool = False

    # Trade-Richtungen: aktivierte Richtungen
    long_enabled: bool = True
    short_enabled: bool = True

    # Feature-Gruppen
    feature_groups: List[str] = None

    # Optional: Trade-Timeout (None = kein Timeout, Trade läuft bis TP/SL)
    max_trade_bars: int = None

    @classmethod
    def create(cls, asset: "AssetConfig", strategy: "StrategyConfig") -> "SimulationContext":
        """
        Erstellt SimulationContext aus Asset- und Strategy-Config.

        Args:
            asset: AssetConfig für das zu optimierende Asset
            strategy: StrategyConfig mit allen Strategie-Parametern
        """
        grid = strategy.get_grid_for_class(asset.asset_class)

        # Separate L/S Grids wenn aktiviert
        long_tp, long_sl, long_ct = None, None, None
        short_tp, short_sl, short_ct = None, None, None
        if grid.separate_long_short:
            long_tp, long_sl, long_ct = grid.get_long_grid()
            short_tp, short_sl, short_ct = grid.get_short_grid()

        return cls(
            symbol=asset.symbol,
            asset_class=asset.asset_class,
            spread=asset.spread,
            point=asset.point,
            currencies=asset.currencies,
            min_trades=strategy.filters.min_trades,
            min_rrr=strategy.filters.min_rrr,
            max_trade_bars=strategy.simulation.max_trade_bars,  # None = kein Timeout
            grid_tp=grid.tp,
            grid_sl=grid.sl,
            grid_ct=grid.ct,
            grid_timeout_bars=grid.timeout_bars,
            long_grid_tp=long_tp,
            long_grid_sl=long_sl,
            long_grid_ct=long_ct,
            short_grid_tp=short_tp,
            short_grid_sl=short_sl,
            short_grid_ct=short_ct,
            separate_long_short=grid.separate_long_short,
            long_enabled=strategy.model.long_enabled,
            short_enabled=strategy.model.short_enabled,
            feature_groups=strategy.features.get_groups(),
        )

    def get_long_grid(self) -> tuple:
        """Gibt (tp, sl, ct) Grid für Long Trades zurück."""
        if self.separate_long_short and self.long_grid_tp is not None:
            return (self.long_grid_tp, self.long_grid_sl, self.long_grid_ct)
        return (self.grid_tp, self.grid_sl, self.grid_ct)

    def get_short_grid(self) -> tuple:
        """Gibt (tp, sl, ct) Grid für Short Trades zurück."""
        if self.separate_long_short and self.short_grid_tp is not None:
            return (self.short_grid_tp, self.short_grid_sl, self.short_grid_ct)
        return (self.grid_tp, self.grid_sl, self.grid_ct)

    def total_grid_combinations(self) -> int:
        """Berechnet Gesamtzahl der Grid-Kombinationen (TP x SL x Timeout x Feature-Gruppen).

        Hinweis: CT wird innerhalb von run_inner_cv getestet, nicht in der äußeren Schleife.
        """
        n_timeout = len(self.grid_timeout_bars) if self.grid_timeout_bars else 1
        if self.separate_long_short:
            long_tp, long_sl, _ = self.get_long_grid()
            short_tp, short_sl, _ = self.get_short_grid()
            long_combos = len(long_tp) * len(long_sl) * n_timeout
            short_combos = len(short_tp) * len(short_sl) * n_timeout
            return (long_combos + short_combos) * len(self.feature_groups)
        return len(self.grid_tp) * len(self.grid_sl) * n_timeout * len(self.feature_groups)

    def grid_combinations_per_feature_group(self) -> int:
        """Berechnet Grid-Kombinationen pro Feature-Gruppe."""
        n_timeout = len(self.grid_timeout_bars) if self.grid_timeout_bars else 1
        if self.separate_long_short:
            long_tp, long_sl, _ = self.get_long_grid()
            short_tp, short_sl, _ = self.get_short_grid()
            return (len(long_tp) * len(long_sl) + len(short_tp) * len(short_sl)) * n_timeout
        return len(self.grid_tp) * len(self.grid_sl) * n_timeout
