"""
SimulationContext - Kontext-Objekt für Trade-Simulationen.

Wird durch den gesamten Simulationsprozess gereicht und enthält alle
Parameter die für eine einzelne Asset-Optimierung benötigt werden.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import StrategyConfig
    from ..plugins.exit_strategy import BaseExitStrategy


@dataclass
class TradeParams:
    """
    Parameter für eine einzelne Trade-Konfiguration.

    Wird durch Grid-Search iteriert und an Simulationsfunktionen übergeben.
    """
    tp: float  # Take-Profit (Multiplikator je nach Exit-Strategy)
    sl: float  # Stop-Loss (Multiplikator je nach Exit-Strategy)
    ct: float = 0.5  # Confidence Threshold
    timeout_bars: Optional[int] = None  # Time-based Exit

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
    currencies: List[str] = field(default_factory=list)

    # Filter-Limits
    min_trades: int = 50
    min_rrr: float = 0.0

    # Grid-Parameter
    grid_tp: List[float] = field(default_factory=list)
    grid_sl: List[float] = field(default_factory=list)
    grid_ct: List[float] = field(default_factory=list)
    grid_timeout_bars: List[Optional[int]] = None

    # Separate Long/Short Grids
    long_grid_tp: List[float] = None
    long_grid_sl: List[float] = None
    long_grid_ct: List[float] = None
    short_grid_tp: List[float] = None
    short_grid_sl: List[float] = None
    short_grid_ct: List[float] = None
    separate_long_short: bool = False

    # Trade-Richtungen
    long_enabled: bool = True
    short_enabled: bool = True

    # Features
    feature_groups: List[str] = None
    feature_selection: str = "boruta"
    max_features: int = 0

    # Trade-Simulation
    max_trade_bars: int = None

    # Early Termination
    min_fold_stability: float = 0.5
    early_termination: bool = True
    first_fold_sanity_check: bool = True
    first_fold_min_win_rate: float = 0.25
    first_fold_min_pnl: float = -10.0
    first_fold_min_trades: int = 5

    # Ressourcen-Limits
    ram_per_feature_group_gb: float = 0.5
    cpu_per_feature_group: float = 0.5
    min_free_ram_percent: float = 0.15
    max_cpu_percent: float = 0.90
    xgboost_n_jobs: int = 0

    # Exit-Strategy
    exit_strategy: str = "fixed"
    exit_params: dict = field(default_factory=dict)

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
        """Berechnet Gesamtzahl der Grid-Kombinationen."""
        n_timeout = len(self.grid_timeout_bars) if self.grid_timeout_bars else 1
        n_groups = len(self.feature_groups) if self.feature_groups else 1

        if self.separate_long_short:
            long_tp, long_sl, _ = self.get_long_grid()
            short_tp, short_sl, _ = self.get_short_grid()
            long_combos = len(long_tp) * len(long_sl) * n_timeout
            short_combos = len(short_tp) * len(short_sl) * n_timeout
            return (long_combos + short_combos) * n_groups

        return len(self.grid_tp) * len(self.grid_sl) * n_timeout * n_groups

    def grid_combinations_per_feature_group(self) -> int:
        """Berechnet Grid-Kombinationen pro Feature-Gruppe."""
        n_timeout = len(self.grid_timeout_bars) if self.grid_timeout_bars else 1

        if self.separate_long_short:
            long_tp, long_sl, _ = self.get_long_grid()
            short_tp, short_sl, _ = self.get_short_grid()
            return (len(long_tp) * len(long_sl) + len(short_tp) * len(short_sl)) * n_timeout

        return len(self.grid_tp) * len(self.grid_sl) * n_timeout
