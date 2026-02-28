"""
SimulationContext - Kontext-Objekt für Trade-Simulationen.

Wird durch den gesamten Simulationsprozess gereicht und enthält alle
Parameter die für eine einzelne Asset-Optimierung benötigt werden.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import StrategyConfig, RegimeFilterGridConfig



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

    # Features (from Pipeline)
    indicator_plugins: List[dict] = None  # List of {"name": ..., "params": ...}
    feature_selection_plugins: List[dict] = None  # List of {"name": ..., "params": ...}

    # Trade-Simulation
    max_trade_bars: int = None

    # Early Termination
    min_fold_stability: float = 0.5
    early_termination: bool = True
    first_fold_sanity_check: bool = True
    first_fold_min_win_rate: float = 0.25
    first_fold_min_pnl: float = -10.0
    first_fold_min_trades: int = 5

    # Minimum trades per CT in inner CV evaluation (evaluate_on_validation).
    # Default 10 ensures statistical significance.  Lower for rare-event signal
    # models where each inner fold may have few entry signals.
    min_eval_trades: int = 10

    # Model
    model_type: str = "xgboost"

    # Validation
    n_inner_folds: int = 5  # Anzahl Inner-Folds für Nested CV
    embargo_bars: int = 0  # Embargo-Gap zwischen Train/Test Folds (AFML Ch. 7)
    sample_weights: bool = False  # Uniqueness-basierte Sample Weights (AFML Ch. 4)
    early_pruning_enabled: bool = False
    early_pruning_keep_ratio: float = 0.5
    early_pruning_min_survivors: int = 10
    early_pruning_min_folds_before_pruning_ratio: float = 0.3
    probability_calibration: bool = False
    calibration_method: str = "isotonic"
    meta_labeling: bool = False

    # Exit-Strategy (Plugin)
    exit_strategy: str = "fixed"
    exit_params: dict = field(default_factory=dict)

    # Exit-Modifier: optionales Plugin das die Simulation der Exit-Strategie erweitert
    # (z.B. trailing_stop, breakeven_stop) — kann unabhängig von der Exit-Strategie getestet werden
    exit_modifier: Optional[str] = None
    exit_modifier_params: dict = field(default_factory=dict)

    # Grid über Exit-Modifier-Params: Liste von Dicts zum Durchsuchen
    # [None] = nur ctx-Default verwenden; [dict1, dict2] = Grid über Modifier-Params
    grid_exit_modifier_params: List[Optional[dict]] = field(default_factory=lambda: [None])

    # Model Hyperparameters (from StrategyConfig, merged with per-asset grid overrides)
    model_hyperparameters: dict = field(default_factory=dict)

    # Session boundaries for indicator calculation (Opening Range, PDH/PDL).
    # Also used by SignalModel to filter entries (via model_hyperparameters).
    session_start_hour: Optional[int] = None
    session_end_hour: Optional[int] = None

    # Separate exit session: when trades may be closed (TP/SL/timeout).
    # Defaults to session_start_hour/session_end_hour when not set.
    # Allows wider exit window (e.g., full CFD hours) while keeping
    # Opening Range / PDH/PDL calculation on exchange hours.
    exit_session_start_hour: Optional[int] = None
    exit_session_end_hour: Optional[int] = None

    # Required features: always included in feature selection, bypass selection plugins
    required_features: List[str] = field(default_factory=list)

    # Grid über Model-Hyperparameters: Liste von HP-Dicts zum Durchsuchen
    # [None] = nur ctx-Default verwenden; [dict1, dict2] = Grid über Model-HPs
    grid_model_hyperparameters: List[Optional[dict]] = field(default_factory=lambda: [None])

    # Regime-Filter Grid (from optimization config)
    regime_filter_grid: "RegimeFilterGridConfig" = None

    # Preprocessing (from Pipeline)
    preprocessing_plugins: List[dict] = None  # List of {"name": ..., "params": ...}

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
        # Grid values from exit_params (all values are arrays after normalization)
        ep = strategy.exit_params

        # TP/SL: use tp_mult/sl_mult keys, fall back to tp/sl for backward compat
        grid_tp = ep.get("tp_mult", ep.get("tp", [1.0, 1.5, 2.0, 2.5]))
        grid_sl = ep.get("sl_mult", ep.get("sl", [1.0, 1.5, 2.0]))
        grid_timeout = ep.get("timeout_bars", [None])

        # Long/Short overrides from exit_params prefixes
        long_tp = ep.get("long_tp_mult", ep.get("long_tp"))
        long_sl = ep.get("long_sl_mult", ep.get("long_sl"))
        short_tp = ep.get("short_tp_mult", ep.get("short_tp"))
        short_sl = ep.get("short_sl_mult", ep.get("short_sl"))
        separate = any([long_tp, long_sl, short_tp, short_sl])

        # CT, regime from optimization
        opt = strategy.optimization
        grid_ct = opt.ct
        long_ct = opt.long_ct
        short_ct = opt.short_ct
        regime = opt.regime_filter_grid

        # Exit modifier params grid and model hyperparameters grid from optimization
        grid_exit_modifier_params = opt.exit_modifier_params_grid or [None]
        grid_model_hyperparameters = opt.model_hyperparameters_grid

        # Model hyperparameters from base model config
        model_hp = dict(strategy.model.hyperparameters)

        # Session hours from model hyperparameters (if configured)
        session_start = model_hp.get("signal_start_hour")
        session_end = model_hp.get("signal_end_hour")
        exit_session_start = model_hp.get("exit_session_start_hour")
        exit_session_end = model_hp.get("exit_session_end_hour")

        # Required features from base model config
        req_feats = list(strategy.model.required_features)

        # Auto-add signal columns + sl_dist_column from base model_hyperparameters
        for key in ('signal_column_long', 'signal_column_short', 'sl_dist_column'):
            val = model_hp.get(key)
            if val and val not in req_feats:
                req_feats.append(val)

        # Auto-add signal columns + sl_dist_column from model_hyperparameters_grid
        for hp_variant in (grid_model_hyperparameters or [None]):
            if hp_variant and isinstance(hp_variant, dict):
                for key in ('signal_column_long', 'signal_column_short', 'sl_dist_column'):
                    val = hp_variant.get(key)
                    if val and val not in req_feats:
                        req_feats.append(val)

        return cls(
            symbol=asset.symbol,
            asset_class=asset.asset_class,
            spread=asset.spread,
            point=asset.point,
            currencies=asset.currencies,
            min_trades=strategy.filters.min_trades,
            min_eval_trades=strategy.filters.min_eval_trades,
            min_rrr=strategy.filters.min_rrr,
            max_trade_bars=None,  # Kein globales Limit, timeout_bars pro Kombination
            grid_tp=grid_tp,
            grid_sl=grid_sl,
            grid_ct=grid_ct,
            grid_timeout_bars=grid_timeout,
            long_grid_tp=long_tp,
            long_grid_sl=long_sl,
            long_grid_ct=long_ct,
            short_grid_tp=short_tp,
            short_grid_sl=short_sl,
            short_grid_ct=short_ct,
            separate_long_short=separate,
            long_enabled=strategy.model.long_enabled,
            short_enabled=strategy.model.short_enabled,
            # Pipeline: Indicators
            indicator_plugins=strategy.get_indicators(),
            # Pipeline: Feature Selection (list of plugins, chained)
            feature_selection_plugins=strategy.get_feature_selection() or None,
            # Model type
            model_type=strategy.model.type,
            # Exit-Strategy Plugin
            exit_strategy=strategy.exit_strategy,
            exit_params=strategy.exit_params,
            # Exit-Modifier Plugin (optional)
            exit_modifier=strategy.exit_modifier,
            exit_modifier_params=strategy.exit_modifier_params,
            # Grid über Exit-Modifier-Params (from optimization)
            grid_exit_modifier_params=grid_exit_modifier_params,
            # Grid über Model-Hyperparameters (from optimization)
            grid_model_hyperparameters=grid_model_hyperparameters or [None],
            # Model Hyperparameters
            model_hyperparameters=model_hp,
            # Session boundaries for indicators + entry filtering
            session_start_hour=session_start,
            session_end_hour=session_end,
            # Separate exit session (wider window for CFD assets)
            exit_session_start_hour=exit_session_start,
            exit_session_end_hour=exit_session_end,
            required_features=req_feats,
            # Regime-Filter Grid (from optimization)
            regime_filter_grid=regime,
            # Pipeline: Preprocessing
            preprocessing_plugins=strategy.get_preprocessing(),
            # Validation
            n_inner_folds=strategy.validation.n_inner_folds,
            embargo_bars=strategy.validation.embargo_bars,
            sample_weights=strategy.validation.sample_weights,
            early_pruning_enabled=strategy.validation.early_pruning.enabled,
            early_pruning_keep_ratio=strategy.validation.early_pruning.keep_ratio,
            early_pruning_min_survivors=strategy.validation.early_pruning.min_survivors,
            early_pruning_min_folds_before_pruning_ratio=strategy.validation.early_pruning.min_folds_before_pruning_ratio,
            probability_calibration=strategy.validation.probability_calibration,
            calibration_method=strategy.validation.calibration_method,
            meta_labeling=strategy.validation.meta_labeling,
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
        n_modifier = len(self.grid_exit_modifier_params) if self.grid_exit_modifier_params else 1
        n_model_hp = len(self.grid_model_hyperparameters) if self.grid_model_hyperparameters else 1
        # With pipeline, we don't iterate over feature groups anymore
        # All indicator plugins are applied together

        if self.separate_long_short:
            long_tp, long_sl, _ = self.get_long_grid()
            short_tp, short_sl, _ = self.get_short_grid()
            long_combos = len(long_tp) * len(long_sl) * n_timeout
            short_combos = len(short_tp) * len(short_sl) * n_timeout
            return (long_combos + short_combos) * n_modifier * n_model_hp

        return len(self.grid_tp) * len(self.grid_sl) * n_timeout * n_modifier * n_model_hp


# Type hint import für AssetConfig
if TYPE_CHECKING:
    from ...optimizer.asset_config import AssetConfig
