"""
StrategyConfig - Zentrale Konfigurationsklasse für Trading-Strategien.

Plugin-basierte Struktur mit Pipeline-Format.
Alle Config-Klassen sind hier definiert - keine Duplikate in anderen Modulen.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import glob
import json
import os


@dataclass
class RegimeCondition:
    """A single regime filter condition on a DataFrame column.

    Bitmask encoding (like Linux file permissions):
        Bit 2 (4) = Long allowed
        Bit 1 (2) = Short allowed
        Bit 0 (1) = Sideways allowed (future use)
        7 = all allowed, 6 = Long+Short, 4 = Long only, 2 = Short only, 0 = blocked
    """
    column: str           # e.g. "trend_adx_14", "macro_vix"
    operator: str         # ">=", "<=", ">", "<"
    value: float
    directions: int = 6       # Bitmask when condition is TRUE (default: Long+Short)
    else_directions: int = 0  # Bitmask when condition is FALSE (default: blocked)


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
                        "directions": grid.get("directions", 6),
                        "else_directions": grid.get("else_directions", 0),
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
class ExitStrategyConfig:
    """Configuration for a single exit strategy instance.

    Each instance is an independent grid element with fixed params.
    The optimizer iterates over the list of instances.
    """
    name: str = "fixed"
    params: Dict[str, Any] = field(default_factory=dict)
    ct: List[float] = field(default_factory=lambda: [0.5])
    long_ct: List[float] | None = None
    short_ct: List[float] | None = None
    min_rrr: float = 0
    exit_modifier: Optional[str] = None
    exit_modifier_params: Dict[str, Any] = field(default_factory=dict)
    entry_modifier: Optional[str] = None
    entry_modifier_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExitStrategyConfig":
        ct = data.get("ct", [0.5])
        if isinstance(ct, (int, float)):
            ct = [ct]
        long_ct = data.get("long_ct")
        if isinstance(long_ct, (int, float)):
            long_ct = [long_ct]
        short_ct = data.get("short_ct")
        if isinstance(short_ct, (int, float)):
            short_ct = [short_ct]
        return cls(
            name=data.get("name", "fixed"),
            params=data.get("params", {}),
            ct=ct,
            long_ct=long_ct,
            short_ct=short_ct,
            min_rrr=data.get("min_rrr", 0),
            exit_modifier=data.get("exit_modifier"),
            exit_modifier_params=data.get("exit_modifier_params", {}),
            entry_modifier=data.get("entry_modifier"),
            entry_modifier_params=data.get("entry_modifier_params", {}),
        )


@dataclass
class OptimizationConfig:
    """Global optimization parameters for grid search."""

    regime_filter_grid: RegimeFilterGridConfig = field(
        default_factory=RegimeFilterGridConfig
    )
    model_hyperparameters_grid: list[dict] | None = None
    indicator_grid: dict[str, dict[str, list]] | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "OptimizationConfig":
        if not data:
            return cls()
        rfg = data.get("regime_filter_grid")
        regime = (
            RegimeFilterGridConfig.from_dict(rfg)
            if rfg
            else RegimeFilterGridConfig()
        )

        mhg = data.get("model_hyperparameters_grid")
        if isinstance(mhg, dict):
            mhg = [mhg]

        return cls(
            regime_filter_grid=regime,
            model_hyperparameters_grid=mhg,
            indicator_grid=data.get("indicator_grid"),
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
    required_features: List[str] = field(default_factory=list)

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
            required_features=data.get("required_features", []),
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
class EarlyPruningConfig:
    """Early Pruning: Zweiphasiger Grid-Search zur Laufzeitreduktion."""
    enabled: bool = False
    keep_ratio: float = 0.5
    min_survivors: int = 10
    min_folds_before_pruning_ratio: float = 0.3

    @classmethod
    def from_dict(cls, data) -> "EarlyPruningConfig":
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", False),
            keep_ratio=data.get("keep_ratio", 0.5),
            min_survivors=data.get("min_survivors", 10),
            min_folds_before_pruning_ratio=data.get("min_folds_before_pruning_ratio", 0.3),
        )


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
    early_pruning: EarlyPruningConfig = field(default_factory=EarlyPruningConfig)
    probability_calibration: bool = False
    calibration_method: str = "isotonic"  # "isotonic" or "sigmoid" (Platt Scaling)
    meta_labeling: bool = False

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
            early_pruning=EarlyPruningConfig.from_dict(data.get("early_pruning")),
            probability_calibration=data.get("probability_calibration", False),
            calibration_method=data.get("calibration_method", "isotonic"),
            meta_labeling=data.get("meta_labeling", False),
        )


@dataclass
class FilterConfig:
    """Filter-Parameter für Strategie-Auswahl."""
    min_rrr: float = 0.0
    min_trades: int = 50
    min_eval_trades: int = 10  # Min trades per CT in inner CV evaluation
    min_annual_return: float = 10.0
    max_drawdown: float = 1.0
    min_sharpe: float = 0.0
    min_fold_stability: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        return cls(
            min_rrr=data.get("min_rrr", 0.0),
            min_trades=data.get("min_trades", 50),
            min_eval_trades=data.get("min_eval_trades", 10),
            min_annual_return=data.get("min_annual_return", 10.0),
            max_drawdown=data.get("max_drawdown", 1.0),
            min_sharpe=data.get("min_sharpe", 0.0),
            min_fold_stability=data.get("min_fold_stability", 0.0),
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
                directions=c.get("directions", 6),
                else_directions=c.get("else_directions", 0),
            )
            for c in raw_conditions
        ]
        return cls(conditions=conditions)


@dataclass
class ResourceConfig:
    """Resource limits for optimization runs.

    KISS: max_concurrent_assets is the primary and only reliable control.
    Each model plugin manages its own thread count internally.
    """
    max_concurrent_assets: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceConfig":
        if data is None:
            return cls()
        return cls(
            max_concurrent_assets=data.get("max_concurrent_assets", 1),
        )


def _load_json_preset(name: str, presets_dir: str) -> dict:
    """Load a JSON preset file from the given directory.

    Tries ``{name}.json`` first; if not found, falls back to the highest-version
    ``{name}_v*.json`` file (versioned naming scheme).
    """
    real_dir = os.path.realpath(presets_dir)

    path = os.path.join(presets_dir, f"{name}.json")
    resolved = os.path.realpath(path)
    if not resolved.startswith(real_dir):
        raise ValueError(f"Preset name '{name}' resolves outside allowed directory")

    if not os.path.isfile(resolved):
        # Fall back to versioned filename (e.g. name_v1.json, name_v2.json …)
        matches = sorted(glob.glob(os.path.join(presets_dir, f"{name}_v*.json")))
        if not matches:
            raise FileNotFoundError(f"Preset '{name}' not found at {path}")
        resolved = os.path.realpath(matches[-1])  # highest lexicographic = highest version
        if not resolved.startswith(real_dir):
            raise ValueError(f"Preset name '{name}' resolves outside allowed directory")

    with open(resolved, "r") as f:
        data = json.load(f)
    data.pop("_meta", None)  # strip embedded metadata before returning content
    return data


def _resolve_section(
    value: "Optional[Union[str, Dict[str, Any]]]",
    section_dir: str,
    strategy_dir: Optional[str],
) -> "Optional[Dict[str, Any]]":
    """Resolve a config section: string loads preset file, dict/None passes through."""
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        base = os.path.dirname(strategy_dir) if strategy_dir else os.getcwd()
        return _load_json_preset(value, os.path.join(base, section_dir))
    return value


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

    # Exit strategies (each instance is an independent grid element)
    exit_strategies: List[ExitStrategyConfig] = field(default_factory=list)

    # Risk management
    risk_management: str = "kelly"
    risk_params: Dict[str, Any] = field(default_factory=dict)

    # Optimization (grid search parameters)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Assets-Filter
    assets: Dict[str, Any] = field(default_factory=dict)

    # Sub-Konfigurationen
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    regime_filter: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)

    # Default data source name for data_loading pipeline entries without explicit source
    datasource: Optional[str] = None

    # Timeframe override (None = use TIMEFRAME env var)
    timeframe: Optional[str] = None

    # Signal rules for composed entry signals (visual rule builder)
    signal_rules: Optional[Dict[str, Any]] = None

    # Metadata
    hypothesis: str = ""
    expected_outcome: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Erstellt StrategyConfig aus Dictionary (z.B. aus JSON-Datei)."""
        strategy_dir = data.get("_strategy_dir")

        # Resolve sections: string loads preset file, dict/None passes through
        pipeline = _resolve_section(data.get("pipeline", {}), "pipelines", strategy_dir)
        model_data = _resolve_section(data.get("model", {}), "models", strategy_dir)
        validation_data = _resolve_section(data.get("validation", {}), "validations", strategy_dir)
        filters_data = _resolve_section(data.get("filters", {}), "filters", strategy_dir)
        resources_data = _resolve_section(data.get("resources"), "resources", strategy_dir)
        risk_params = _resolve_section(data.get("risk_params", {}), "risk_params", strategy_dir)
        optimization = OptimizationConfig.from_dict(data.get("optimization"))

        # Resolve regime_filter: string → preset, dict → inline, None → empty
        regime_data = _resolve_section(data.get("regime_filter"), "regime_filters", strategy_dir)
        regime_filter = RegimeFilterConfig()
        if isinstance(regime_data, dict):
            if "condition_grids" in regime_data:
                # Grid preset → merge into optimization.regime_filter_grid
                optimization = OptimizationConfig(
                    regime_filter_grid=RegimeFilterGridConfig.from_dict(regime_data),
                    model_hyperparameters_grid=optimization.model_hyperparameters_grid,
                    indicator_grid=optimization.indicator_grid,
                )
            else:
                regime_filter = RegimeFilterConfig.from_dict(regime_data)

        # Parse exit_strategies array
        raw_exits = data.get("exit_strategies", [])
        exit_strategies = [ExitStrategyConfig.from_dict(e) for e in raw_exits]

        return cls(
            name=data.get("name", "Default Strategy"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            pipeline=pipeline,
            exit_strategies=exit_strategies,
            risk_management=data.get("risk_management", "kelly"),
            risk_params=risk_params,
            optimization=optimization,
            assets=data.get("assets", {}),
            model=ModelConfig.from_dict(model_data),
            validation=ValidationConfig.from_dict(validation_data),
            filters=FilterConfig.from_dict(filters_data),
            resources=ResourceConfig.from_dict(resources_data),
            regime_filter=regime_filter,
            datasource=data.get("datasource"),
            timeframe=data.get("timeframe"),
            hypothesis=data.get("hypothesis", ""),
            expected_outcome=data.get("expected_outcome", ""),
            signal_rules=data.get("signal_rules"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "StrategyConfig":
        """Lädt StrategyConfig aus JSON-Datei."""
        with open(path, "r") as f:
            data = json.load(f)
        data["_strategy_dir"] = os.path.dirname(os.path.abspath(path))
        return cls.from_dict(data)

    def get_data_loading(self) -> List[Dict[str, Any]]:
        """Returns configured data loading plugins from pipeline.

        Entries without an explicit 'source' inherit self.datasource.
        """
        entries = self.pipeline.get("data_loading", [])
        if not self.datasource:
            return entries
        result = []
        for entry in entries:
            if not entry.get("source"):
                entry = {**entry, "source": self.datasource}
            result.append(entry)
        return result

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
        exit_strats = []
        for es in self.exit_strategies:
            d: Dict[str, Any] = {"name": es.name, "params": es.params, "ct": es.ct}
            if es.long_ct:
                d["long_ct"] = es.long_ct
            if es.short_ct:
                d["short_ct"] = es.short_ct
            if es.min_rrr:
                d["min_rrr"] = es.min_rrr
            if es.exit_modifier:
                d["exit_modifier"] = es.exit_modifier
                d["exit_modifier_params"] = es.exit_modifier_params
            if es.entry_modifier:
                d["entry_modifier"] = es.entry_modifier
                d["entry_modifier_params"] = es.entry_modifier_params
            exit_strats.append(d)

        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "pipeline": self.pipeline,
            "exit_strategies": exit_strats,
            "risk_management": self.risk_management,
            "risk_params": self.risk_params,
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
            **({"signal_rules": self.signal_rules} if self.signal_rules else {}),
        }

    def log_summary(self, log_func=print):
        """Gibt eine Zusammenfassung der Konfiguration aus."""
        log_func(f"Strategy: {self.name}")
        log_func(f"  Exit Strategies: {len(self.exit_strategies)} instances")
        for es in self.exit_strategies:
            log_func(f"    - {es.name}: {es.params}")
        log_func(f"  Min Trades: {self.filters.min_trades}")
        indicators = self.get_indicators()
        log_func(
            f"  Indicators: {len(indicators)} plugins"
        )
