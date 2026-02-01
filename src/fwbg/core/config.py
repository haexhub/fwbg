"""
StrategyConfig - Zentrale Konfigurationsklasse für Trading-Strategien.

Neue Plugin-basierte Struktur ohne Legacy-Support.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class GridConfig:
    """Konfiguration für TP/SL/CT Grid-Search."""
    tp: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    sl: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    ct: List[float] = field(default_factory=lambda: [0.50, 0.52, 0.55, 0.60])
    timeout_bars: List[Optional[int]] = field(default_factory=lambda: [None])

    # Separate Long/Short Grids
    long_tp: List[float] = None
    long_sl: List[float] = None
    long_ct: List[float] = None
    short_tp: List[float] = None
    short_sl: List[float] = None
    short_ct: List[float] = None
    separate_long_short: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridConfig":
        """Erstellt GridConfig aus Dictionary."""
        timeout_raw = data.get("timeout_bars", [None])
        if timeout_raw is None:
            timeout_bars = [None]
        elif isinstance(timeout_raw, (int, float)):
            timeout_bars = [int(timeout_raw)]
        else:
            timeout_bars = [int(t) if t is not None else None for t in timeout_raw]

        has_separate = any(k in data for k in ["long_tp", "long_sl", "short_tp", "short_sl"])

        return cls(
            tp=data.get("tp", [1.0, 1.5, 2.0, 2.5]),
            sl=data.get("sl", [1.0, 1.5, 2.0]),
            ct=data.get("ct", [0.50, 0.52, 0.55, 0.60]),
            timeout_bars=timeout_bars,
            long_tp=data.get("long_tp"),
            long_sl=data.get("long_sl"),
            long_ct=data.get("long_ct"),
            short_tp=data.get("short_tp"),
            short_sl=data.get("short_sl"),
            short_ct=data.get("short_ct"),
            separate_long_short=data.get("separate_long_short", has_separate),
        )

    def get_long_grid(self) -> tuple:
        return (
            self.long_tp if self.long_tp is not None else self.tp,
            self.long_sl if self.long_sl is not None else self.sl,
            self.long_ct if self.long_ct is not None else self.ct,
        )

    def get_short_grid(self) -> tuple:
        return (
            self.short_tp if self.short_tp is not None else self.tp,
            self.short_sl if self.short_sl is not None else self.sl,
            self.short_ct if self.short_ct is not None else self.ct,
        )


@dataclass
class ModelConfig:
    """Konfiguration für das ML-Modell."""
    type: str = "xgboost"
    architecture: str = "unified"  # "unified" oder "long_short_separate"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    })
    trade_directions: List[str] = field(default_factory=lambda: ["long", "short"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        directions = data.get("trade_directions", ["long", "short"])
        if isinstance(directions, list):
            directions = [d.lower() for d in directions]

        return cls(
            type=data.get("type", "xgboost"),
            architecture=data.get("architecture", "unified"),
            hyperparameters=data.get("hyperparameters", {
                "n_estimators": 100, "max_depth": 5, "random_state": 42
            }),
            trade_directions=directions,
        )

    @property
    def is_long_short_separate(self) -> bool:
        return self.architecture == "long_short_separate"

    @property
    def long_enabled(self) -> bool:
        return "long" in self.trade_directions

    @property
    def short_enabled(self) -> bool:
        return "short" in self.trade_directions


@dataclass
class ValidationConfig:
    """Parameter für Cross-Validation."""
    method: str = "walk_forward"
    folds: int = 8
    oos_size: int = 4000
    min_trades: int = 50
    holdout_ratio: float = 0.20
    n_inner_folds: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        return cls(
            method=data.get("method", "walk_forward"),
            folds=data.get("folds", 8),
            oos_size=data.get("oos_size", 4000),
            min_trades=data.get("min_trades", 50),
            holdout_ratio=data.get("holdout_ratio", 0.20),
            n_inner_folds=data.get("n_inner_folds", 5),
        )


@dataclass
class FilterConfig:
    """Filter-Parameter für Strategie-Auswahl."""
    min_rrr: float = 0.0
    min_trades: int = 50
    min_annual_return: float = 10.0
    max_drawdown: float = 1.0
    min_sharpe: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        return cls(
            min_rrr=data.get("min_rrr", 0.0),
            min_trades=data.get("min_trades", 50),
            min_annual_return=data.get("min_annual_return", 10.0),
            max_drawdown=data.get("max_drawdown", 1.0),
            min_sharpe=data.get("min_sharpe", 0.0),
        )


@dataclass
class ResourceConfig:
    """Parameter für Ressourcen-Limits."""
    ram_per_feature_group_gb: float = 0.5
    cpu_per_feature_group: float = 0.5
    min_free_ram_percent: float = 0.15
    max_cpu_percent: float = 0.90
    xgboost_n_jobs: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceConfig":
        if data is None:
            return cls()

        min_free_ram = data.get("min_free_ram_percent", 0.15)
        if min_free_ram > 1:
            min_free_ram = min_free_ram / 100

        max_cpu = data.get("max_cpu_percent", 0.90)
        if max_cpu > 1:
            max_cpu = max_cpu / 100

        return cls(
            ram_per_feature_group_gb=data.get("ram_per_feature_group_gb", 0.5),
            cpu_per_feature_group=data.get("cpu_per_feature_group", 0.5),
            min_free_ram_percent=min_free_ram,
            max_cpu_percent=max_cpu,
            xgboost_n_jobs=data.get("xgboost_n_jobs", 0),
        )


@dataclass
class StrategyConfig:
    """
    Zentrale Konfigurationsklasse für eine Trading-Strategie.

    Neue Plugin-basierte Struktur:
    - indicators: Liste der zu verwendenden Indicator-Plugins
    - exit_strategy: Name des Exit-Strategy-Plugins
    - feature_selector: Name des Feature-Selector-Plugins
    - preprocessing: Liste der Preprocessor-Plugins
    """
    name: str = "Default Strategy"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Plugin-Konfiguration (neu)
    indicators: List[str] = field(default_factory=list)
    indicator_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    exit_strategy: str = "fixed"
    exit_params: Dict[str, Any] = field(default_factory=dict)

    feature_selector: str = "boruta"
    feature_params: Dict[str, Any] = field(default_factory=dict)

    preprocessing: List[str] = field(default_factory=list)
    preprocessing_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Grid-Konfiguration
    grids: Dict[str, GridConfig] = field(default_factory=dict)

    # Sub-Konfigurationen
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Metadata
    hypothesis: str = ""
    expected_outcome: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Erstellt StrategyConfig aus Dictionary (z.B. aus JSON-Datei)."""
        # Grids parsen
        grids = {}
        for asset_class, grid_data in data.get("grids", {}).items():
            grids[asset_class] = GridConfig.from_dict(grid_data)

        return cls(
            name=data.get("name", "Default Strategy"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            # Plugins
            indicators=data.get("indicators", []),
            indicator_params=data.get("indicator_params", {}),
            exit_strategy=data.get("exit_strategy", "fixed"),
            exit_params=data.get("exit_params", {}),
            feature_selector=data.get("feature_selector", "boruta"),
            feature_params=data.get("feature_params", {}),
            preprocessing=data.get("preprocessing", []),
            preprocessing_params=data.get("preprocessing_params", {}),
            # Grids
            grids=grids,
            # Sub-Configs
            model=ModelConfig.from_dict(data.get("model", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            filters=FilterConfig.from_dict(data.get("filters", {})),
            resources=ResourceConfig.from_dict(data.get("resources")),
            # Metadata
            hypothesis=data.get("hypothesis", ""),
            expected_outcome=data.get("expected_outcome", ""),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "StrategyConfig":
        """Lädt StrategyConfig aus JSON-Datei."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_grid_for_class(self, asset_class: str) -> GridConfig:
        """Gibt das Grid für eine Asset-Klasse zurück."""
        if self.grids:
            return self.grids.get(asset_class, self.grids.get("FOREX", GridConfig()))
        return GridConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Serialisierung."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "indicators": self.indicators,
            "indicator_params": self.indicator_params,
            "exit_strategy": self.exit_strategy,
            "exit_params": self.exit_params,
            "feature_selector": self.feature_selector,
            "feature_params": self.feature_params,
            "preprocessing": self.preprocessing,
            "preprocessing_params": self.preprocessing_params,
            "grids": {k: {"tp": v.tp, "sl": v.sl, "ct": v.ct} for k, v in self.grids.items()},
            "model": {
                "type": self.model.type,
                "architecture": self.model.architecture,
                "trade_directions": self.model.trade_directions,
            },
            "filters": {
                "min_rrr": self.filters.min_rrr,
                "min_trades": self.filters.min_trades,
            },
        }
