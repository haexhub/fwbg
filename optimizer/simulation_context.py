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

    # Feature Selection Methode: "boruta", "boruta_plateau", "importance_based"
    feature_selection: str = "boruta"

    # Optional: Trade-Timeout (None = kein Timeout, Trade läuft bis TP/SL)
    max_trade_bars: int = None

    # Early Termination: Mindest-Fold-Stability für Kandidaten (0.5 = 50% der Folds profitabel)
    min_fold_stability: float = 0.5

    # Early Termination aktivieren: Kandidaten die min_fold_stability nicht erreichen können werden abgebrochen
    early_termination: bool = True

    # First-Fold Sanity Check: Bricht nach erstem Fold ab wenn Ergebnis katastrophal ist
    # Nur für extreme Fälle - normal schlechte Folds werden nicht abgebrochen
    first_fold_sanity_check: bool = True
    first_fold_min_win_rate: float = 0.25  # Minimum 25% Win-Rate
    first_fold_min_pnl: float = -10.0  # Minimum PnL (sehr großzügig)
    first_fold_min_trades: int = 5  # Minimum Trades im ersten Fold

    # Ressourcen-Limits für parallele Feature-Group-Verarbeitung
    # Defaults sind aggressiv (viele parallele Threads) - kann in Strategy-Config angepasst werden
    ram_per_feature_group_gb: float = 0.5
    cpu_per_feature_group: float = 0.5
    min_free_ram_percent: float = 0.15
    max_cpu_percent: float = 0.90

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
            feature_selection=strategy.features.feature_selection,
            # Ressourcen-Limits aus Strategy-Config
            ram_per_feature_group_gb=strategy.resources.ram_per_feature_group_gb,
            cpu_per_feature_group=strategy.resources.cpu_per_feature_group,
            min_free_ram_percent=strategy.resources.min_free_ram_percent,
            max_cpu_percent=strategy.resources.max_cpu_percent,
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
