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
class RegimeFilterGridConfig:
    """Konfiguration für Regime-Filter als Grid-Parameter.

    Ermöglicht das Testen verschiedener Regime-Filter-Kombinationen im Grid-Search.
    Jede Kombination wird als separate Optimierung durchgeführt.

    HINWEIS: Default ist [0] für adx_min = kein Filter aktiv.
    ML soll selbst lernen, welche Marktbedingungen relevant sind.
    """
    # ADX Grid: Liste von min_values - 0 bedeutet deaktiviert (Default)
    adx_min: List[float] = field(default_factory=lambda: [0.0])

    # VIX Grid: Liste von max_values (None = deaktiviert, Default)
    vix_max: List[float] = field(default_factory=lambda: [None])

    # Hurst Grid: Liste von (min, max) Dicts (None = deaktiviert, Default)
    hurst: List[Dict[str, float]] = field(default_factory=lambda: [None])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeFilterGridConfig":
        """Erstellt RegimeFilterGridConfig aus Dictionary."""
        if data is None:
            return cls()

        # ADX: Kann Liste von Werten sein oder einzelner Wert (Default: 0 = kein Filter)
        adx_raw = data.get("adx_min", [0.0])
        if not isinstance(adx_raw, list):
            adx_raw = [adx_raw]

        # VIX: Kann Liste von Werten sein oder einzelner Wert
        vix_raw = data.get("vix_max", [None])
        if not isinstance(vix_raw, list):
            vix_raw = [vix_raw]

        # Hurst: Kann Liste von Dicts sein, Presets, oder None
        hurst_raw = data.get("hurst", [None])
        if not isinstance(hurst_raw, list):
            hurst_raw = [hurst_raw]

        return cls(
            adx_min=adx_raw,
            vix_max=vix_raw,
            hurst=hurst_raw,
        )

    def get_combinations(self) -> List[Dict[str, Any]]:
        """Generiert alle Regime-Filter-Kombinationen.

        Returns:
            Liste von Dicts mit Regime-Filter-Parametern
        """
        import itertools
        combinations = []

        for adx, vix, hurst in itertools.product(self.adx_min, self.vix_max, self.hurst):
            config = {
                "adx_enabled": adx is not None and adx > 0,
                "adx_min": adx if adx and adx > 0 else 0,
                "vix_enabled": vix is not None,
                "vix_max": vix if vix else 25.0,
                "hurst_enabled": hurst is not None,
                "hurst_min": hurst.get("min") if isinstance(hurst, dict) else None,
                "hurst_max": hurst.get("max") if isinstance(hurst, dict) else None,
            }
            combinations.append(config)

        return combinations

    def total_combinations(self) -> int:
        """Anzahl der Regime-Filter-Kombinationen."""
        return len(self.adx_min) * len(self.vix_max) * len(self.hurst)


@dataclass
class GridConfig:
    """Konfiguration für TP/SL/CT Grid-Search.

    Unterstützt separate Grids für Long und Short Trades.
    Wenn long_*/short_* nicht gesetzt sind, werden die gemeinsamen Werte verwendet.
    """
    # Gemeinsame Grids (werden verwendet wenn keine separaten L/S Grids definiert)
    tp: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 40, 50, 60, 80])
    sl: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 40, 50, 60, 80])
    ct: List[float] = field(default_factory=lambda: [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70])

    # Time-based Exit: Nach X Bars ohne TP/SL zum Close schließen
    # None = kein Timeout, der Trade läuft bis TP/SL
    # Liste von Werten zum Testen, z.B. [None, 12, 24, 48]
    timeout_bars: List[int] = field(default_factory=lambda: [None])

    # Regime-Filter Grid (verschiedene Filter-Kombinationen testen)
    regime_filter_grid: RegimeFilterGridConfig = field(default_factory=RegimeFilterGridConfig)

    # Separate Grids für Long (None = verwende gemeinsame Werte)
    long_tp: List[int] = None
    long_sl: List[int] = None
    long_ct: List[float] = None

    # Separate Grids für Short (None = verwende gemeinsame Werte)
    short_tp: List[int] = None
    short_sl: List[int] = None
    short_ct: List[float] = None

    # Flag ob separate L/S Optimierung aktiv ist
    separate_long_short: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridConfig":
        """Erstellt GridConfig aus Dictionary."""
        # Default-Werte explizit, da dataclass default_factory keine Klassenattribute erstellt
        default_tp = [15, 20, 25, 30, 40, 50, 60, 80]
        default_sl = [15, 20, 25, 30, 40, 50, 60, 80]
        default_ct = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]

        # Prüfe ob separate L/S Grids definiert sind
        has_separate = any(k in data for k in ["long_tp", "long_sl", "short_tp", "short_sl"])

        # timeout_bars: kann None, int, oder Liste sein
        timeout_raw = data.get("timeout_bars", [None])
        if timeout_raw is None:
            timeout_bars = [None]
        elif isinstance(timeout_raw, (int, float)):
            timeout_bars = [int(timeout_raw)]
        else:
            # Liste - None-Werte beibehalten
            timeout_bars = [int(t) if t is not None else None for t in timeout_raw]

        # Regime-Filter Grid parsen
        regime_grid = RegimeFilterGridConfig.from_dict(data.get("regime_filter_grid"))

        return cls(
            tp=data.get("tp", default_tp),
            sl=data.get("sl", default_sl),
            ct=data.get("ct", default_ct),
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
        """Gibt (tp, sl, ct) Grid für Long Trades zurück."""
        return (
            self.long_tp if self.long_tp is not None else self.tp,
            self.long_sl if self.long_sl is not None else self.sl,
            self.long_ct if self.long_ct is not None else self.ct,
        )

    def get_short_grid(self) -> tuple:
        """Gibt (tp, sl, ct) Grid für Short Trades zurück."""
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
            # Separate Optimierung: Long + Short Kombinationen
            return (len(long_tp) * len(long_sl) * len(long_ct) +
                    len(short_tp) * len(short_sl) * len(short_ct))
        return len(self.tp) * len(self.sl) * len(self.ct)


@dataclass
class ModelConfig:
    """Konfiguration für das ML-Modell."""
    type: str = "xgboost"
    architecture: str = "unified"  # "unified", "long_short_separate", "ensemble"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    })
    # Trade-Richtungen: ["long", "short"] oder nur ["long"] / ["short"]
    trade_directions: List[str] = field(default_factory=lambda: ["long", "short"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Erstellt ModelConfig aus Dictionary."""
        # Normalisiere trade_directions zu lowercase
        directions = data.get("trade_directions", ["long", "short"])
        if isinstance(directions, list):
            directions = [d.lower() for d in directions]
        return cls(
            type=data.get("type", "xgboost"),
            architecture=data.get("architecture", "unified"),
            hyperparameters=data.get("hyperparameters", {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            }),
            trade_directions=directions,
        )

    @property
    def is_long_short_separate(self) -> bool:
        """Prüft ob separate Long/Short Modelle trainiert werden sollen."""
        return self.architecture == "long_short_separate"

    @property
    def long_enabled(self) -> bool:
        """Prüft ob Long-Trades aktiviert sind."""
        return "long" in self.trade_directions

    @property
    def short_enabled(self) -> bool:
        """Prüft ob Short-Trades aktiviert sind."""
        return "short" in self.trade_directions


@dataclass
class SimulationParams:
    """Parameter für Trade-Simulation."""
    max_trade_bars: int = None  # None = kein Timeout, Trade läuft bis TP/SL
    tp_sl_basis: str = "spread_multiple"  # Basis für TP/SL Berechnung
    trailing_stop: bool = True
    slippage_model: str = "fixed"
    regime_filter: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationParams":
        """Erstellt SimulationParams aus Dictionary."""
        return cls(
            max_trade_bars=data.get("max_trade_bars"),  # None wenn nicht definiert
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
    max_drawdown: float = 1.0  # Maximum Drawdown (1.0 = kein Filter, 0.6 = max 60%)
    min_sharpe: float = 0.0  # Minimum Sharpe Ratio (0 = kein Filter)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterParams":
        """Erstellt FilterParams aus Dictionary."""
        return cls(
            min_rrr=data.get("min_rrr", 0.0),
            min_trades=data.get("min_trades", 50),
            min_annual_return=data.get("min_annual_return", 10.0),
            max_drawdown=data.get("max_drawdown", 1.0),
            min_sharpe=data.get("min_sharpe", 0.0),
        )


@dataclass
class RegimeFilterParams:
    """Parameter für Regime-Filter.

    Regime-Filter bestimmen, wann Trading erlaubt ist basierend auf
    Marktbedingungen wie Trend-Stärke, Volatilität und Markt-Charakter.

    HINWEIS: Defaults sind jetzt alle deaktiviert - ML soll selbst lernen,
    welche Marktbedingungen relevant sind (via Features wie trend_adx_*,
    macro_vix, regime_hurst_*).

    Hurst-Exponent Interpretation:
    - H > 0.5: Trending/Persistent (gut für Trend-Following)
    - H = 0.5: Random Walk (schwierig zu traden)
    - H < 0.5: Mean-Reverting (gut für Mean-Reversion Strategien)
    """
    # ADX Filter (Trend-Stärke) - Default: deaktiviert
    adx_enabled: bool = False
    adx_min: float = 0.0  # 0 = kein Filter

    # VIX Filter (Markt-Volatilität) - Default: deaktiviert
    vix_enabled: bool = False
    vix_max: float = None  # None = kein Filter

    # Hurst Exponent Filter (Markt-Charakter) - Default: deaktiviert
    hurst_enabled: bool = False
    hurst_min: float = None  # Minimum Hurst (None = kein Minimum)
    hurst_max: float = None  # Maximum Hurst (None = kein Maximum)
    hurst_window: int = 100  # Fenster für Rolling Hurst

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeFilterParams":
        """Erstellt RegimeFilterParams aus Dictionary."""
        return cls(
            adx_enabled=data.get("adx_enabled", False),
            adx_min=data.get("adx_min", 0.0),
            vix_enabled=data.get("vix_enabled", False),
            vix_max=data.get("vix_max"),
            hurst_enabled=data.get("hurst_enabled", False),
            hurst_min=data.get("hurst_min"),
            hurst_max=data.get("hurst_max"),
            hurst_window=data.get("hurst_window", 100),
        )


@dataclass
class PreprocessingParams:
    """Parameter für Daten-Preprocessing.

    Unterstützte Transformationen:
    - fractional_differentiation: Stationäre Preise mit Memory (López de Prado)
    - log_returns: Log-Returns statt absoluter Preise
    - normalize: Z-Score Normalisierung der OHLC-Daten
    """
    # Fractional Differentiation
    fractional_differentiation: bool = False
    frac_diff_auto_d: bool = True  # Automatische d-Optimierung via ADF-Test
    frac_diff_default_d: float = 0.4  # Default d-Wert wenn auto_d=False

    # Log-Returns Transformation
    log_returns: bool = False  # Verwendet log(P_t/P_{t-1}) statt absoluter Preise

    # Normalisierung
    normalize: bool = False  # Z-Score Normalisierung
    normalize_window: int = 100  # Rolling-Window für Z-Score

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingParams":
        """Erstellt PreprocessingParams aus Dictionary."""
        return cls(
            fractional_differentiation=data.get("fractional_differentiation", False),
            frac_diff_auto_d=data.get("frac_diff_auto_d", True),
            frac_diff_default_d=data.get("frac_diff_default_d", 0.4),
            log_returns=data.get("log_returns", False),
            normalize=data.get("normalize", False),
            normalize_window=data.get("normalize_window", 100),
        )

    @property
    def has_any(self) -> bool:
        """Prüft ob irgendein Preprocessing aktiviert ist."""
        return self.fractional_differentiation or self.log_returns or self.normalize


@dataclass
class ResourceParams:
    """Parameter für Ressourcen-Limits bei paralleler Feature-Group-Verarbeitung.

    Diese Werte steuern, wie viele Feature-Groups parallel verarbeitet werden
    können, basierend auf verfügbarem RAM und CPU.

    Die Defaults sind eher aggressiv, da Feature-Group-Threads sich viele Daten
    teilen (Copy-on-Write) und einzelne Threads nicht sehr RAM-intensiv sind.
    """
    # RAM pro Feature-Group-Thread in GB (typisch: 0.5-1.5GB)
    # Niedrigerer Wert = mehr parallele Threads
    ram_per_feature_group_gb: float = 0.5

    # CPU-Kerne pro Feature-Group-Thread (typisch: 0.5-1.0)
    # Niedrigerer Wert = mehr parallele Threads (gut für I/O-bound tasks)
    cpu_per_feature_group: float = 0.5

    # Mindestens X% RAM frei halten (0.0-1.0)
    # Niedrigerer Wert = mehr RAM für Threads verfügbar
    min_free_ram_percent: float = 0.15

    # Maximal X% der CPU-Kerne nutzen (0.0-1.0)
    # Höherer Wert = mehr CPU-Kerne verfügbar
    max_cpu_percent: float = 0.90

    # XGBoost n_jobs: Anzahl Kerne pro XGBoost-Modell
    # 0 = automatisch (Kerne / parallele Feature-Gruppen)
    # 1 = single-threaded (für VPS/Production die nur traden)
    # -1 = alle Kerne (WARNUNG: Überparallelisierung bei vielen Feature-Gruppen!)
    xgboost_n_jobs: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceParams":
        """Erstellt ResourceParams aus Dictionary."""
        if data is None:
            return cls()

        # Normalisiere Prozent-Werte: 80 -> 0.80, 0.80 -> 0.80
        min_free_ram = data.get("min_free_ram_percent", 0.20)
        if min_free_ram > 1:
            min_free_ram = min_free_ram / 100

        max_cpu = data.get("max_cpu_percent", 0.80)
        if max_cpu > 1:
            max_cpu = max_cpu / 100

        return cls(
            ram_per_feature_group_gb=data.get("ram_per_feature_group_gb", 1.0),
            cpu_per_feature_group=data.get("cpu_per_feature_group", 1.0),
            min_free_ram_percent=min_free_ram,
            max_cpu_percent=max_cpu,
            xgboost_n_jobs=data.get("xgboost_n_jobs", 0),
        )


@dataclass
class FeatureParams:
    """Parameter für Feature-Auswahl.

    feature_selection Optionen:
    - "boruta": Boruta findet alle statistisch relevanten Features, KEIN hartes Limit
    - "boruta_plateau": Boruta + Plateau-Validierung (filtert instabile Lookback-Perioden)
    - "importance_based": Altes Verhalten mit top_n=5 Features pro Gruppe
    """
    technical_indicators: bool = True
    macro_indicators: bool = True
    time_features: bool = True
    multi_timeframe: bool = True
    feature_selection: str = "boruta"  # Default: Boruta ohne Feature-Limit
    preferred_groups: List[str] = None  # None = alle Feature-Gruppen testen

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureParams":
        """Erstellt FeatureParams aus Dictionary."""
        return cls(
            technical_indicators=data.get("technical_indicators", True),
            macro_indicators=data.get("macro_indicators", True),
            time_features=data.get("time_features", True),
            multi_timeframe=data.get("multi_timeframe", True),
            feature_selection=data.get("feature_selection", "boruta"),
            preferred_groups=data.get("preferred_groups"),  # None = alle Gruppen
        )

    def get_groups(self) -> List[str]:
        """Gibt die zu testenden Feature-Gruppen zurück. None = alle."""
        if self.preferred_groups is None:
            from .config import DEFAULT_FEATURE_GROUPS
            return DEFAULT_FEATURE_GROUPS
        return self.preferred_groups


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
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    validation: ValidationParams = field(default_factory=ValidationParams)
    filters: FilterParams = field(default_factory=FilterParams)
    features: FeatureParams = field(default_factory=FeatureParams)
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    regime_filter: RegimeFilterParams = field(default_factory=RegimeFilterParams)
    resources: ResourceParams = field(default_factory=ResourceParams)

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
        model_config = ModelConfig.from_dict(data.get("model", {}))

        for asset_class, grid_data in grids_data.items():
            grid = GridConfig.from_dict(grid_data)
            # Separate L/S CT-Optimierung NUR wenn explizit long_ct/short_ct im Grid definiert
            # (architecture: long_short_separate betrifft nur die ML-Models, nicht das CT-Grid)
            grids[asset_class] = grid

        return cls(
            name=data.get("name", "Default Strategy"),
            description=data.get("description", ""),
            category=data.get("category", "default"),
            tags=data.get("tags", []),
            grids=grids,
            model=model_config,
            simulation=SimulationParams.from_dict(data.get("simulation", {})),
            validation=ValidationParams.from_dict(data.get("validation", {})),
            filters=FilterParams.from_dict(data.get("filters", {})),
            features=FeatureParams.from_dict(data.get("features", {})),
            preprocessing=PreprocessingParams.from_dict(data.get("preprocessing", {})),
            regime_filter=RegimeFilterParams.from_dict(data.get("regime_filter", {})),
            resources=ResourceParams.from_dict(data.get("resources")),
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
            "model": {
                "type": self.model.type,
                "architecture": self.model.architecture,
                "hyperparameters": self.model.hyperparameters,
            },
            "simulation": {
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
        log_func(f"  Min RRR: {self.filters.min_rrr}")
        log_func(f"  Min Trades: {self.filters.min_trades}")
        groups = self.features.get_groups()
        log_func(f"  Feature Groups: {len(groups)} ({', '.join(groups[:3])}{'...' if len(groups) > 3 else ''})")
