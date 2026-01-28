"""
SimulationContext - Kontext-Objekt für Trade-Simulationen.

Wird durch den gesamten Simulationsprozess gereicht und enthält alle
Parameter die für eine einzelne Asset-Optimierung benötigt werden.
"""
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .strategy_config import StrategyConfig
    from .asset_config import AssetConfig


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
    max_trade_bars: int
    min_trades: int
    min_rrr: float

    # Grid für dieses Asset
    grid_tp: List[int]
    grid_sl: List[int]
    grid_ct: List[float]

    # Feature-Gruppen
    feature_groups: List[str]

    @classmethod
    def create(cls, asset: "AssetConfig", strategy: "StrategyConfig") -> "SimulationContext":
        """
        Erstellt SimulationContext aus Asset- und Strategy-Config.

        Args:
            asset: AssetConfig für das zu optimierende Asset
            strategy: StrategyConfig mit allen Strategie-Parametern
        """
        grid = strategy.get_grid_for_class(asset.asset_class)

        return cls(
            symbol=asset.symbol,
            asset_class=asset.asset_class,
            spread=asset.spread,
            point=asset.point,
            currencies=asset.currencies,
            max_trade_bars=strategy.simulation.max_trade_bars,
            min_trades=strategy.filters.min_trades,
            min_rrr=strategy.filters.min_rrr,
            grid_tp=grid.tp,
            grid_sl=grid.sl,
            grid_ct=grid.ct,
            feature_groups=strategy.features.preferred_groups,
        )

    def total_grid_combinations(self) -> int:
        """Berechnet Gesamtzahl der Grid-Kombinationen (TP x SL x Feature-Gruppen).

        Hinweis: CT wird innerhalb von run_inner_cv getestet, nicht in der äußeren Schleife.
        """
        return len(self.grid_tp) * len(self.grid_sl) * len(self.feature_groups)

    def grid_combinations_per_feature_group(self) -> int:
        """Berechnet Grid-Kombinationen pro Feature-Gruppe."""
        return len(self.grid_tp) * len(self.grid_sl)
