"""
SimulationContext - Kontext-Objekt für Trade-Simulationen.

Wird durch den gesamten Simulationsprozess gereicht und enthält alle
Parameter die für eine einzelne Asset-Optimierung benötigt werden.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import StrategyConfig


@dataclass
class TradeParams:
    """
    Parameter für eine einzelne Trade-Konfiguration.

    Wird durch Grid-Search iteriert und an Simulationsfunktionen übergeben.
    """
    tp: float  # Take-Profit (ATR-Multiplikator)
    sl: float  # Stop-Loss (ATR-Multiplikator)
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

    # Grid-Parameter (ATR-Multiplikatoren)
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

    # Features (Indicator-Plugins)
    feature_groups: List[str] = None
    feature_selection: str = "boruta"
    max_features: int = 0
    min_z_score: float = 0.3  # Boruta Z-Score Threshold

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

    # Exit-Strategy (Plugin)
    exit_strategy: str = "atr_based"
    exit_params: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        asset: "AssetConfig",
        strategy: "StrategyConfig"
    ) -> "SimulationContext":
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
            max_trade_bars=None,  # Kein globales Limit, timeout_bars pro Kombination
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
            feature_groups=strategy.get_feature_groups(),
            feature_selection=strategy.feature_selector,
            max_features=strategy.feature_params.get("max_features", 0),
            min_z_score=strategy.feature_params.get("min_z_score", 0.3),
            # Ressourcen-Limits aus Strategy-Config
            ram_per_feature_group_gb=strategy.resources.ram_per_feature_group_gb,
            cpu_per_feature_group=strategy.resources.cpu_per_feature_group,
            min_free_ram_percent=strategy.resources.min_free_ram_percent,
            max_cpu_percent=strategy.resources.max_cpu_percent,
            xgboost_n_jobs=strategy.resources.xgboost_n_jobs,
            # Exit-Strategy Plugin
            exit_strategy=strategy.exit_strategy,
            exit_params=strategy.exit_params,
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

    def _effective_timeout_grid_size(self) -> int:
        """
        Berechnet effektive Timeout-Grid-Größe.

        Bei adaptive_timeout=True wird das Timeout pro Trade dynamisch berechnet,
        daher ist die Grid-Größe 1 (kein Timeout-Grid-Loop).
        """
        # Prüfe ob adaptive_timeout aktiviert ist
        adaptive_timeout = self.exit_params.get("adaptive_timeout", False)
        if adaptive_timeout:
            return 1  # Timeout wird dynamisch berechnet, kein Grid
        return len(self.grid_timeout_bars) if self.grid_timeout_bars else 1

    def total_grid_combinations(self) -> int:
        """Berechnet Gesamtzahl der Grid-Kombinationen."""
        n_timeout = self._effective_timeout_grid_size()
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
        n_timeout = self._effective_timeout_grid_size()

        if self.separate_long_short:
            long_tp, long_sl, _ = self.get_long_grid()
            short_tp, short_sl, _ = self.get_short_grid()
            return (
                (len(long_tp) * len(long_sl) + len(short_tp) * len(short_sl))
                * n_timeout
            )

        return len(self.grid_tp) * len(self.grid_sl) * n_timeout


# Type hint import für AssetConfig
if TYPE_CHECKING:
    from ...optimizer.asset_config import AssetConfig
