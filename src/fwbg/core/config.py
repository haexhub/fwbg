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

    # Exit-Modifier-Params Grid: Liste von Modifier-Param-Dicts zum Vergleichen
    # [None] = nur ctx-Default verwenden (kein Grid), [dict1, dict2] = Grid über Modifier-Params
    exit_modifier_params_grid: List[Optional[dict]] = field(default_factory=lambda: [None])

    # Per-asset model hyperparameters override (merged into base model config)
    model_hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Per-asset required features (always included in feature selection)
    required_features: List[str] = field(default_factory=list)

    # Model-Hyperparameters Grid: Liste von HP-Dicts zum Vergleichen
    # [None] = nur ctx-Default verwenden (kein Grid), [dict1, dict2] = Grid über Model-HPs
    model_hyperparameters_grid: List[Optional[Dict[str, Any]]] = field(default_factory=lambda: [None])

    # Per-asset indicator param overrides (merged into pipeline indicator params).
    # Keys are indicator names, values are param dicts.
    # Example: {"previous_day_levels": {"session_start_hour": 8, "session_end_hour": 17}}
    indicator_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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

        emp_raw = data.get("exit_modifier_params_grid")
        if emp_raw is None:
            exit_modifier_params_grid = [None]
        elif isinstance(emp_raw, dict):
            exit_modifier_params_grid = [emp_raw]
        else:
            exit_modifier_params_grid = list(emp_raw)

        mhp_raw = data.get("model_hyperparameters_grid")
        if mhp_raw is None:
            model_hyperparameters_grid = [None]
        elif isinstance(mhp_raw, dict):
            model_hyperparameters_grid = [mhp_raw]
        else:
            model_hyperparameters_grid = list(mhp_raw)

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
            exit_modifier_params_grid=exit_modifier_params_grid,
            model_hyperparameters=data.get("model_hyperparameters", {}),
            required_features=data.get("required_features", []),
            model_hyperparameters_grid=model_hyperparameters_grid,
            indicator_overrides=data.get("indicator_overrides", {}),
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


def _resolve_regime_filter(
    value: "Optional[Union[str, Dict[str, Any]]]", regime_filters_dir: str
):
    """Resolve a regime filter reference: string loads file, dict passes through, None stays None."""
    if value is None:
        return None
    if isinstance(value, str):
        return _load_json_preset(value, regime_filters_dir)
    return value


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


def _parse_grids(grids_data: dict, strategy_dir: Optional[str] = None) -> Dict[str, GridConfig]:
    """Parse grids from strategy data, supporting both legacy inline and preset formats."""
    if not grids_data:
        return {}

    # Legacy format: no "assignments" key → all values are inline grid dicts
    if "assignments" not in grids_data:
        return {
            asset_class: GridConfig.from_dict(grid_data)
            for asset_class, grid_data in grids_data.items()
        }

    # New preset format
    base_dir = strategy_dir if strategy_dir else os.getcwd()
    presets_dir = os.path.join(base_dir, grids_data.get("presets_dir", "grids"))
    regime_filters_dir = os.path.join(
        base_dir, grids_data.get("regime_filters_dir", "regime_filters")
    )

    # Shared regime_filter_grid (strategy-level)
    shared_regime = _resolve_regime_filter(
        grids_data.get("regime_filter_grid"), regime_filters_dir
    )

    # Cache for loaded preset files
    preset_cache: Dict[str, dict] = {}
    assignments = grids_data["assignments"]
    result: Dict[str, GridConfig] = {}

    for asset_class, assignment in assignments.items():
        if isinstance(assignment, str):
            # String → load preset by name
            preset_name = assignment
            if preset_name not in preset_cache:
                preset_cache[preset_name] = _load_json_preset(preset_name, presets_dir)
            resolved_data = dict(preset_cache[preset_name])
        elif isinstance(assignment, dict) and "preset" in assignment:
            # Dict with "preset" → load + override with all assignment keys
            preset_name = assignment["preset"]
            if preset_name not in preset_cache:
                preset_cache[preset_name] = _load_json_preset(preset_name, presets_dir)
            resolved_data = dict(preset_cache[preset_name])
            for key, value in assignment.items():
                if key != "preset":
                    resolved_data[key] = value
        elif isinstance(assignment, dict):
            # Inline legacy dict (no "preset" key)
            resolved_data = assignment
        else:
            raise ValueError(f"Invalid assignment for '{asset_class}': {assignment}")

        # Resolve regime_filter_grid: assignment-level > shared > none
        asset_regime = assignment.get("regime_filter_grid") if isinstance(assignment, dict) else None
        if asset_regime is not None:
            resolved_data["regime_filter_grid"] = _resolve_regime_filter(
                asset_regime, regime_filters_dir
            )
        elif shared_regime is not None and "regime_filter_grid" not in resolved_data:
            resolved_data["regime_filter_grid"] = shared_regime

        result[asset_class] = GridConfig.from_dict(resolved_data)

    return result


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

    # Exit modifier (optional add-on replacing the simulation kernel)
    exit_modifier: Optional[str] = None
    exit_modifier_params: Dict[str, Any] = field(default_factory=dict)

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

    # Default data source name for data_loading pipeline entries without explicit source
    datasource: Optional[str] = None

    # Timeframe override (None = use TIMEFRAME env var)
    timeframe: Optional[str] = None

    # Preview mode: limit data to first N calendar days (None = no limit)
    days_limit: Optional[int] = None

    # Metadata
    hypothesis: str = ""
    expected_outcome: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Erstellt StrategyConfig aus Dictionary (z.B. aus JSON-Datei)."""
        strategy_dir = data.get("_strategy_dir")
        grids = _parse_grids(data.get("grids", {}), strategy_dir)

        # Resolve sections: string loads preset file, dict/None passes through
        pipeline = _resolve_section(data.get("pipeline", {}), "pipelines", strategy_dir)
        exit_params = _resolve_section(data.get("exit_params", {}), "exit_params", strategy_dir)
        exit_modifier_params = _resolve_section(
            data.get("exit_modifier_params", {}), "exit_modifier_params", strategy_dir
        )
        model_data = _resolve_section(data.get("model", {}), "models", strategy_dir)
        validation_data = _resolve_section(data.get("validation", {}), "validations", strategy_dir)
        filters_data = _resolve_section(data.get("filters", {}), "filters", strategy_dir)
        resources_data = _resolve_section(data.get("resources"), "resources", strategy_dir)
        risk_params = _resolve_section(data.get("risk_params", {}), "risk_params", strategy_dir)

        return cls(
            name=data.get("name", "Default Strategy"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            pipeline=pipeline,
            exit_strategy=data.get("exit_strategy", "fixed"),
            exit_params=exit_params,
            exit_modifier=data.get("exit_modifier"),
            exit_modifier_params=exit_modifier_params or {},
            risk_management=data.get("risk_management", "kelly"),
            risk_params=risk_params,
            grids=grids,
            assets=data.get("assets", {}),
            model=ModelConfig.from_dict(model_data),
            validation=ValidationConfig.from_dict(validation_data),
            filters=FilterConfig.from_dict(filters_data),
            resources=ResourceConfig.from_dict(resources_data),
            regime_filter=RegimeFilterConfig.from_dict(data.get("regime_filter")),
            datasource=data.get("datasource"),
            timeframe=data.get("timeframe"),
            days_limit=data.get("days_limit"),
            hypothesis=data.get("hypothesis", ""),
            expected_outcome=data.get("expected_outcome", ""),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "StrategyConfig":
        """Lädt StrategyConfig aus JSON-Datei."""
        with open(path, "r") as f:
            data = json.load(f)
        data["_strategy_dir"] = os.path.dirname(os.path.abspath(path))
        return cls.from_dict(data)

    def get_grid(self, symbol: str, asset_class: str) -> GridConfig:
        """Gibt das Grid für ein Symbol oder eine Asset-Klasse zurück.

        Resolution order: symbol → asset_class → FOREX → default.
        """
        if self.grids:
            if symbol in self.grids:
                return self.grids[symbol]
            return self.grids.get(
                asset_class, self.grids.get("FOREX", GridConfig())
            )
        return GridConfig()

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

    def get_indicators(self, indicator_overrides: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Returns configured indicator plugins from pipeline.

        If indicator_overrides is provided (from GridConfig per asset class),
        merge them into the indicator params. This allows per-asset session
        hours, etc.
        """
        indicators = self.pipeline.get("indicators", [])
        if not indicator_overrides:
            return indicators

        import copy
        result = []
        for ind in indicators:
            name = ind.get("name", "")
            overrides = indicator_overrides.get(name)
            if overrides:
                ind = copy.deepcopy(ind)
                params = ind.get("params", {})
                params.update(overrides)
                ind["params"] = params
            result.append(ind)
        return result

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
                k: {
                    "tp": v.tp, "sl": v.sl, "ct": v.ct,
                    "timeout_bars": v.timeout_bars,
                    **({"regime_filter_grid": {"condition_grids": v.regime_filter_grid.condition_grids}}
                       if v.regime_filter_grid.condition_grids else {}),
                }
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
