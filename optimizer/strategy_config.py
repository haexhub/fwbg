"""
StrategyConfig - Zentrale Konfigurationsklasse für Trading-Strategien.

Diese Klasse kapselt alle strategie-spezifischen Parameter und wird
durch den gesamten Optimizer durchgereicht.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json

from .config import CLASS_GRIDS


@dataclass
class GridConfig:
    """Konfiguration für TP/SL/CT Grid-Search."""
    tp: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 40, 50, 60, 80])
    sl: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 40, 50, 60, 80])
    ct: List[float] = field(default_factory=lambda: [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridConfig":
        """Erstellt GridConfig aus Dictionary."""
        return cls(
            tp=data.get("tp", cls.tp),
            sl=data.get("sl", cls.sl),
            ct=data.get("ct", cls.ct),
        )

    def total_combinations(self) -> int:
        """Berechnet Gesamtzahl der Grid-Kombinationen."""
        return len(self.tp) * len(self.sl) * len(self.ct)


@dataclass
class SimulationParams:
    """Parameter für Trade-Simulation."""
    max_trade_bars: int = 48  # Maximale Dauer eines Trades in Bars
    tp_sl_basis: str = "spread_multiple"  # Basis für TP/SL Berechnung
    trailing_stop: bool = True
    slippage_model: str = "fixed"
    regime_filter: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationParams":
        """Erstellt SimulationParams aus Dictionary."""
        return cls(
            max_trade_bars=data.get("max_trade_bars", 48),
            tp_sl_basis=data.get("tp_sl_basis", "spread_multiple"),
            trailing_stop=data.get("trailing_stop", True),
            slippage_model=data.get("slippage_model", "fixed"),
            regime_filter=data.get("regime_filter", True),
        )


@dataclass
class ValidationParams:
    """Parameter für Cross-Validation."""
    method: str = "walk_forward"
    folds: int = 8
    oos_size: int = 4000
    min_trades: int = 50
    holdout_ratio: float = 0.20
    n_inner_folds: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationParams":
        """Erstellt ValidationParams aus Dictionary."""
        return cls(
            method=data.get("method", "walk_forward"),
            folds=data.get("folds", 8),
            oos_size=data.get("oos_size", 4000),
            min_trades=data.get("min_trades", 50),
            holdout_ratio=data.get("holdout_ratio", 0.20),
            n_inner_folds=data.get("n_inner_folds", 5),
        )


@dataclass
class FilterParams:
    """Filter-Parameter für Strategie-Auswahl."""
    min_rrr: float = 0.0  # Minimum Risk-Reward-Ratio
    min_trades: int = 50  # Minimum Trades für Validität
    min_annual_return: float = 10.0  # Minimum Jahresrendite in %

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterParams":
        """Erstellt FilterParams aus Dictionary."""
        return cls(
            min_rrr=data.get("min_rrr", 0.0),
            min_trades=data.get("min_trades", 50),
            min_annual_return=data.get("min_annual_return", 10.0),
        )


@dataclass
class FeatureParams:
    """Parameter für Feature-Auswahl."""
    technical_indicators: bool = True
    macro_indicators: bool = True
    time_features: bool = True
    multi_timeframe: bool = True
    feature_selection: str = "importance_based"
    preferred_groups: List[str] = field(default_factory=lambda: ["trend_momentum", "macro_vol", "full_technical"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureParams":
        """Erstellt FeatureParams aus Dictionary."""
        return cls(
            technical_indicators=data.get("technical_indicators", True),
            macro_indicators=data.get("macro_indicators", True),
            time_features=data.get("time_features", True),
            multi_timeframe=data.get("multi_timeframe", True),
            feature_selection=data.get("feature_selection", "importance_based"),
            preferred_groups=data.get("preferred_groups", ["trend_momentum", "macro_vol", "full_technical"]),
        )


@dataclass
class StrategyConfig:
    """
    Zentrale Konfigurationsklasse für eine Trading-Strategie.

    Diese Klasse wird durch den gesamten Optimizer durchgereicht und
    enthält alle Parameter die für die Optimierung relevant sind.
    """
    name: str = "Default Strategy"
    description: str = ""
    category: str = "default"
    tags: List[str] = field(default_factory=list)

    # Sub-Konfigurationen
    grids: Dict[str, GridConfig] = field(default_factory=dict)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    validation: ValidationParams = field(default_factory=ValidationParams)
    filters: FilterParams = field(default_factory=FilterParams)
    features: FeatureParams = field(default_factory=FeatureParams)

    # Metadata
    hypothesis: str = ""
    expected_outcome: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Erstellt StrategyConfig aus Dictionary (z.B. aus JSON-Datei)."""
        # Grids pro Asset-Klasse parsen
        grids = {}
        grids_data = data.get("grids", {})
        for asset_class, grid_data in grids_data.items():
            grids[asset_class] = GridConfig.from_dict(grid_data)

        return cls(
            name=data.get("name", "Default Strategy"),
            description=data.get("description", ""),
            category=data.get("category", "default"),
            tags=data.get("tags", []),
            grids=grids,
            simulation=SimulationParams.from_dict(data.get("simulation", {})),
            validation=ValidationParams.from_dict(data.get("validation", {})),
            filters=FilterParams.from_dict(data.get("filters", {})),
            features=FeatureParams.from_dict(data.get("features", {})),
            hypothesis=data.get("hypothesis", ""),
            expected_outcome=data.get("expected_outcome", ""),
            notes=data.get("notes", ""),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "StrategyConfig":
        """Lädt StrategyConfig aus JSON-Datei."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def default(cls) -> "StrategyConfig":
        """Erstellt Default-Konfiguration (Scalping-Stil)."""
        return cls(
            name="Default Scalping",
            grids={
                "FOREX": GridConfig(),
                "INDEX": GridConfig(
                    tp=[20, 30, 50, 70, 100, 150],
                    sl=[20, 30, 50, 70, 100, 150],
                    ct=[0.50, 0.52, 0.55, 0.60, 0.65, 0.70]
                ),
                "COMMODITY": GridConfig(
                    tp=[20, 30, 40, 60, 80, 100],
                    sl=[20, 30, 40, 60, 80, 100],
                    ct=[0.50, 0.52, 0.55, 0.58, 0.62, 0.65, 0.70]
                ),
                "CRYPTO": GridConfig(
                    tp=[20, 30, 50, 80, 120, 200],
                    sl=[20, 30, 50, 80, 120, 200],
                    ct=[0.50, 0.52, 0.55, 0.60, 0.65, 0.70]
                ),
            }
        )

    def get_grid_for_class(self, asset_class: str) -> GridConfig:
        """Gibt das Grid für eine Asset-Klasse zurück.

        Fallback-Reihenfolge:
        1. self.grids[asset_class] - aus Strategy-JSON
        2. self.grids["FOREX"] - als Fallback innerhalb der Strategy
        3. CLASS_GRIDS[asset_class] - aus config.py (wenn keine Grids in Strategy)
        4. CLASS_GRIDS["FOREX"] - letzter Fallback
        """
        if self.grids:
            return self.grids.get(asset_class, self.grids.get("FOREX", GridConfig()))

        # Keine Grids in Strategy definiert -> CLASS_GRIDS aus config.py verwenden
        class_grid = CLASS_GRIDS.get(asset_class, CLASS_GRIDS.get("FOREX", {}))
        return GridConfig(
            tp=class_grid.get("tp", [15, 20, 25, 30, 40, 50, 60, 80]),
            sl=class_grid.get("sl", [15, 20, 25, 30, 40, 50, 60, 80]),
            ct=class_grid.get("ct", [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Serialisierung."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "grids": {k: {"tp": v.tp, "sl": v.sl, "ct": v.ct} for k, v in self.grids.items()},
            "simulation": {
                "max_trade_bars": self.simulation.max_trade_bars,
                "tp_sl_basis": self.simulation.tp_sl_basis,
                "trailing_stop": self.simulation.trailing_stop,
                "slippage_model": self.simulation.slippage_model,
                "regime_filter": self.simulation.regime_filter,
            },
            "validation": {
                "method": self.validation.method,
                "folds": self.validation.folds,
                "oos_size": self.validation.oos_size,
                "min_trades": self.validation.min_trades,
            },
            "filters": {
                "min_rrr": self.filters.min_rrr,
                "min_trades": self.filters.min_trades,
            },
            "features": {
                "preferred_groups": self.features.preferred_groups,
            },
        }

    def log_summary(self, log_func=print):
        """Gibt eine Zusammenfassung der Konfiguration aus."""
        log_func(f"Strategy: {self.name}")
        log_func(f"  Max Trade Bars: {self.simulation.max_trade_bars} ({self.simulation.max_trade_bars / 24:.0f} Tage)")
        log_func(f"  Min RRR: {self.filters.min_rrr}")
        log_func(f"  Min Trades: {self.filters.min_trades}")
        log_func(f"  Feature Groups: {self.features.preferred_groups}")
