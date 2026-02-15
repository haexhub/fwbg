"""
StrategyConfig - Zentrale Konfigurationsklasse für Trading-Strategien.

Plugin-basierte Struktur mit Pipeline-Format.
Alle Config-Klassen sind hier definiert - keine Duplikate in anderen Modulen.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class RegimeCondition:
    """A single regime filter condition on a DataFrame column."""
    column: str      # e.g. "trend_adx_14", "macro_vix"
    operator: str    # ">=", "<=", ">", "<"
    value: float


@dataclass
class RegimeFilterGridConfig:
    """Konfiguration für Regime-Filter als Grid-Parameter."""
    condition_grids: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeFilterGridConfig":
        if data is None:
            return cls()
        grids = data.get("condition_grids", [])
        return cls(condition_grids=grids)

    def get_combinations(self) -> List[Dict[str, Any]]:
        """Generiert alle Regime-Filter-Kombinationen als Kartesisches Produkt."""
        import itertools

        if not self.condition_grids:
            return [{"conditions": []}]

        # Each grid has values list — build cartesian product
        value_lists = [grid["values"] for grid in self.condition_grids]
        combinations = []

        for values in itertools.product(*value_lists):
            conditions = []
            for grid, val in zip(self.condition_grids, values):
                if val is not None:
                    conditions.append({
                        "column": grid["column"],
                        "operator": grid["operator"],
                        "value": val,
                    })
            combinations.append({"conditions": conditions})

        return combinations

    def total_combinations(self) -> int:
        if not self.condition_grids:
            return 1
        result = 1
        for grid in self.condition_grids:
            result *= len(grid["values"])
        return result


@dataclass
class GridConfig:
    """Konfiguration für TP/SL/CT Grid-Search (ATR-Multiplikatoren)."""
    tp: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    sl: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    ct: List[float] = field(default_factory=lambda: [0.50, 0.52, 0.55, 0.60])
    timeout_bars: List[Optional[int]] = field(default_factory=lambda: [None])

    # Regime-Filter Grid
    regime_filter_grid: RegimeFilterGridConfig = field(
        default_factory=RegimeFilterGridConfig
    )

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

        has_separate = any(
            k in data for k in ["long_tp", "long_sl", "short_tp", "short_sl"]
        )
        regime_grid = RegimeFilterGridConfig.from_dict(
            data.get("regime_filter_grid")
        )

        return cls(
            tp=data.get("tp", [1.0, 1.5, 2.0, 2.5]),
            sl=data.get("sl", [1.0, 1.5, 2.0]),
            ct=data.get("ct", [0.50, 0.52, 0.55, 0.60]),
            timeout_bars=timeout_bars,
            regime_filter_grid=regime_grid,
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

    def total_combinations(self) -> int:
        """Berechnet Gesamtzahl der Grid-Kombinationen."""
        if self.separate_long_short:
            long_tp, long_sl, long_ct = self.get_long_grid()
            short_tp, short_sl, short_ct = self.get_short_grid()
            return (
                len(long_tp) * len(long_sl) * len(long_ct) +
                len(short_tp) * len(short_sl) * len(short_ct)
            )
        return len(self.tp) * len(self.sl) * len(self.ct)


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
    embargo_bars: int = 0
    sample_weights: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        return cls(
            method=data.get("method", "walk_forward"),
            folds=data.get("folds", 8),
            oos_size=data.get("oos_size", 4000),
            min_trades=data.get("min_trades", 50),
            holdout_ratio=data.get("holdout_ratio", 0.20),
            n_inner_folds=data.get("n_inner_folds", 5),
            embargo_bars=data.get("embargo_bars", 0),
            sample_weights=data.get("sample_weights", False),
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
class RegimeFilterConfig:
    """Parameter für Regime-Filter (generic conditions)."""
    conditions: List[RegimeCondition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeFilterConfig":
        if data is None:
            return cls()
        raw_conditions = data.get("conditions", [])
        conditions = [
            RegimeCondition(
                column=c["column"],
                operator=c["operator"],
                value=c["value"],
            )
            for c in raw_conditions
        ]
        return cls(conditions=conditions)


@dataclass
class ResourceConfig:
    """
    Ressourcen-Limits für Optimizer.

    Globale Limits - System optimiert dynamisch während des Runs:
    - ram_per_worker_gb: Geschätzter RAM pro Asset-Worker
    - min_free_ram_percent: Mindest-freier RAM (System pausiert wenn unterschritten)
    - max_cpu_percent: Maximale CPU-Auslastung (System pausiert wenn überschritten)
    - xgboost_n_jobs: XGBoost-Threading (0=auto, 1=single, -1=alle Kerne)
    - threads_per_asset: CPU-Threads pro Asset (0=auto basierend auf GPU-Verfügbarkeit)
    """
    ram_per_worker_gb: float = 4.0
    min_free_ram_percent: float = 0.15
    max_cpu_percent: float = 0.80
    xgboost_n_jobs: int = 0
    threads_per_asset: int = 0
    max_concurrent_assets: int = 0  # 0 = auto (CPU/RAM-basiert)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceConfig":
        if data is None:
            return cls()

        # Normalisiere Prozent-Werte
        min_free_ram = data.get("min_free_ram_percent", 0.15)
        if min_free_ram > 1:
            min_free_ram = min_free_ram / 100

        max_cpu = data.get("max_cpu_percent", 0.80)
        if max_cpu > 1:
            max_cpu = max_cpu / 100

        return cls(
            ram_per_worker_gb=data.get("ram_per_worker_gb", 4.0),
            min_free_ram_percent=min_free_ram,
            max_cpu_percent=max_cpu,
            xgboost_n_jobs=data.get("xgboost_n_jobs", 0),
            threads_per_asset=data.get("threads_per_asset", 0),
            max_concurrent_assets=data.get("max_concurrent_assets", 0),
        )


@dataclass
class StrategyConfig:
    """
    Zentrale Konfigurationsklasse für eine Trading-Strategie.

    Pipeline Format:
       "pipeline": {
         "preprocessing": [{"name": "...", "params": {...}}],
         "indicators": [{"name": "...", "params": {...}}],
         "feature_selection": [{"name": "...", "params": {...}}]
       }
    """
    name: str = "Default Strategy"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Pipeline configuration
    pipeline: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Exit strategy
    exit_strategy: str = "fixed"
    exit_params: Dict[str, Any] = field(default_factory=dict)

    # Risk management
    risk_management: str = "kelly"
    risk_params: Dict[str, Any] = field(default_factory=dict)

    # Grid-Konfiguration
    grids: Dict[str, GridConfig] = field(default_factory=dict)

    # Assets-Filter
    assets: Dict[str, Any] = field(default_factory=dict)

    # Sub-Konfigurationen
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    regime_filter: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)

    # Metadata
    hypothesis: str = ""
    expected_outcome: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Erstellt StrategyConfig aus Dictionary (z.B. aus JSON-Datei)."""
        grids = {}
        for asset_class, grid_data in data.get("grids", {}).items():
            grids[asset_class] = GridConfig.from_dict(grid_data)

        pipeline = data.get("pipeline", {})

        return cls(
            name=data.get("name", "Default Strategy"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            pipeline=pipeline,
            exit_strategy=data.get("exit_strategy", "fixed"),
            exit_params=data.get("exit_params", {}),
            risk_management=data.get("risk_management", "kelly"),
            risk_params=data.get("risk_params", {}),
            grids=grids,
            assets=data.get("assets", {}),
            model=ModelConfig.from_dict(data.get("model", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            filters=FilterConfig.from_dict(data.get("filters", {})),
            resources=ResourceConfig.from_dict(data.get("resources")),
            regime_filter=RegimeFilterConfig.from_dict(data.get("regime_filter")),
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
            return self.grids.get(
                asset_class, self.grids.get("FOREX", GridConfig())
            )
        return GridConfig()

    def get_data_loading(self) -> List[Dict[str, Any]]:
        """Returns configured data loading plugins from pipeline."""
        return self.pipeline.get("data_loading", [])

    def get_indicators(self) -> List[Dict[str, Any]]:
        """Returns configured indicator plugins from pipeline."""
        return self.pipeline.get("indicators", [])

    def get_preprocessing(self) -> List[Dict[str, Any]]:
        """Returns configured preprocessing plugins from pipeline."""
        return self.pipeline.get("preprocessing", [])

    def get_feature_selection(self) -> List[Dict[str, Any]]:
        """Returns configured feature selection plugins from pipeline."""
        return self.pipeline.get("feature_selection", [])

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Serialisierung."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "pipeline": self.pipeline,
            "exit_strategy": self.exit_strategy,
            "exit_params": self.exit_params,
            "risk_management": self.risk_management,
            "risk_params": self.risk_params,
            "grids": {
                k: {"tp": v.tp, "sl": v.sl, "ct": v.ct, "timeout_bars": v.timeout_bars}
                for k, v in self.grids.items()
            },
            "assets": self.assets,
            "model": {
                "type": self.model.type,
                "architecture": self.model.architecture,
                "trade_directions": self.model.trade_directions,
                "hyperparameters": self.model.hyperparameters,
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
                "min_annual_return": self.filters.min_annual_return,
                "max_drawdown": self.filters.max_drawdown,
            },
            "hypothesis": self.hypothesis,
            "expected_outcome": self.expected_outcome,
        }

    def log_summary(self, log_func=print):
        """Gibt eine Zusammenfassung der Konfiguration aus."""
        log_func(f"Strategy: {self.name}")
        log_func(f"  Exit Strategy: {self.exit_strategy}")
        log_func(f"  Min RRR: {self.filters.min_rrr}")
        log_func(f"  Min Trades: {self.filters.min_trades}")
        indicators = self.get_indicators()
        log_func(
            f"  Indicators: {len(indicators)} plugins"
        )
