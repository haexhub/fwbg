"""
Nested Cross-Validation für unbiased Optimizer-Evaluation.

Struktur:
[=============== INNER (80%) ===============][== HOLDOUT (20%) ==]
                    ↓                                     ↓
            Grid-Search hier                    Finale Evaluation
            (Walk-Forward Folds)                (NIE während Optimierung gesehen!)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from xgboost import XGBClassifier

from fwbg.core.context import SimulationContext
from fwbg.simulation.trade import simulate_pro_trade, compute_targets_numba
from fwbg.builtins.feature_selection.plateau import select_plateau_features
from fwbg.builtins.feature_selection.boruta import select_features_boruta
from fwbg.utils.progress import report_progress
from fwbg.utils.xgb_config import get_xgboost_n_jobs
from fwbg.builtins.exit_strategies import get_strategy
from fwbg.builtins.exit_strategies.base import GridParams


def _validate_targets(
    targets_long: np.ndarray,
    targets_short: np.ndarray,
    ctx: SimulationContext
) -> Tuple[bool, bool]:
    """
    Prüft ob genug Targets für Long/Short vorhanden sind.

    Args:
        targets_long: Long-Targets Array
        targets_short: Short-Targets Array
        ctx: SimulationContext

    Returns:
        (has_long, has_short) - Boolean Tuple
    """
    min_per_direction = ctx.min_trades // 2
    n_long = np.count_nonzero(targets_long)
    n_short = np.count_nonzero(targets_short)
    has_long = ctx.long_enabled and n_long >= min_per_direction
    has_short = ctx.short_enabled and n_short >= min_per_direction
    return has_long, has_short


def _simulate_trades_core(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct_long: float,
    ct_short: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
    direction_filter: int = None,
) -> Dict[str, Any]:
    """
    Kern-Funktion für Trade-Simulation (konsolidiert aus 3 ähnlichen Funktionen).

    Args:
        df: DataFrame mit OHLC-Daten und _regime_ok
        probs_long: Wahrscheinlichkeiten für Long-Trades (oder None)
        probs_short: Wahrscheinlichkeiten für Short-Trades (oder None)
        long_win_idx: Index der Win-Klasse im Long-Modell
        short_win_idx: Index der Win-Klasse im Short-Modell
        ct_long: Confidence Threshold für Long
        ct_short: Confidence Threshold für Short
        tp: Take-Profit Multiplikator
        sl: Stop-Loss Multiplikator
        ctx: SimulationContext mit allen Parametern
        return_detailed: Wenn True, auch volle Trade-Details zurückgeben
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        direction_filter: None=beide, 1=nur Long, -1=nur Short

    Returns:
        dict mit trades und optional trades_detailed
    """
    opn = df["O"].values
    cls = df["C"].values
    hgh = df["H"].values
    low = df["L"].values
    # ATR ist optional - wenn nicht vorhanden (volatility indicator nicht konfiguriert), Dummy-Array
    atr = df["_atr"].values if "_atr" in df.columns else np.zeros(len(df))
    regime = df["_regime_ok"].values
    timestamps = df.index.values

    trades = []
    trades_detailed = [] if return_detailed else None
    next_allowed_entry = 0

    # Simuliere bis zum vorletzten Bar (letzter Bar kann kein Entry sein)
    for i in range(len(df) - 1):
        if i < next_allowed_entry:
            continue

        if not regime[i]:
            continue

        direction = None
        # Long-Check (wenn nicht gefiltert)
        if direction_filter in (None, 1):
            if ctx.long_enabled and probs_long is not None and probs_long[i, long_win_idx] >= ct_long:
                direction = 1
        # Short-Check (wenn nicht gefiltert und Long nicht getroffen)
        if direction is None and direction_filter in (None, -1):
            if ctx.short_enabled and probs_short is not None and probs_short[i, short_win_idx] >= ct_short:
                direction = -1

        if direction:
            trade = simulate_pro_trade(
                cls, hgh, low, atr, i, direction, tp, sl, ctx.spread,
                timestamps=timestamps, symbol=ctx.symbol, opens=opn,
                max_bars=ctx.max_trade_bars,
                timeout_bars=timeout_bars
            )
            if trade:
                trades.append(trade["result"])
                next_allowed_entry = trade["exit_idx"] + 1

                if return_detailed:
                    trade["ct"] = ct_long if direction == 1 else ct_short
                    trade["hour"] = df.index[i].hour
                    trades_detailed.append(trade)

    result = {"trades": trades}
    if return_detailed:
        result["trades_detailed"] = trades_detailed
    return result


def simulate_trades_sequential(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
) -> Dict[str, Any]:
    """Simuliert Trades sequentiell mit gleichem CT für Long/Short."""
    return _simulate_trades_core(
        df, probs_long, probs_short, long_win_idx, short_win_idx,
        ct, ct, tp, sl, ctx, return_detailed, timeout_bars
    )


def nested_cv_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.20,
    n_inner_folds: int = 5,
    oos_size: int = 4000
) -> Dict[str, Any]:
    """
    Nested Cross-Validation Split für unbiased Evaluation.

    OPTIMIERUNG: Verwendet Views statt Copies wo möglich.
    .copy() nur für inner_df und holdout_df (die modifiziert werden können).
    Fold-DataFrames sind Views (read-only).

    Returns:
        dict mit inner_folds, holdout_df, inner_df
    """
    total_len = len(df)
    holdout_size = int(total_len * holdout_ratio)
    inner_size = total_len - holdout_size

    # Copy nur für die Haupt-DataFrames (werden später modifiziert: _regime_ok)
    inner_df = df.iloc[:inner_size].copy()
    holdout_df = df.iloc[inner_size:].copy()

    inner_folds = []
    val_size = min(oos_size, inner_size // (n_inner_folds + 2))

    for i in range(n_inner_folds):
        val_end = inner_size - (i * val_size)
        val_start = val_end - val_size
        train_end = val_start

        if train_end < val_size * 2:
            continue

        # Views statt Copies - DataFrame-Slices sind read-only in der Simulation
        # HINWEIS: Falls später Modifikationen nötig sind, muss .copy() hinzugefügt werden
        train_df = inner_df.iloc[:train_end]
        val_df = inner_df.iloc[val_start:val_end]
        inner_folds.append((train_df, val_df))

    return {
        "inner_folds": list(reversed(inner_folds)),
        "holdout_df": holdout_df,
        "inner_df": inner_df,
    }


def compute_targets(
    df: pd.DataFrame,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Berechnet Long/Short Targets für einen DataFrame.

    Args:
        df: DataFrame mit OHLC-Daten
        tp: Take-Profit Multiplikator
        sl: Stop-Loss Multiplikator
        ctx: SimulationContext
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        (targets_long, targets_short, has_long, has_short)
    """
    targets_long = np.zeros(len(df))
    targets_short = np.zeros(len(df))

    opn_v = df["O"].values
    cls_v = df["C"].values
    hgh_v = df["H"].values
    low_v = df["L"].values
    # ATR ist optional - wenn nicht vorhanden (volatility indicator nicht konfiguriert), Dummy-Array
    atr_v = df["_atr"].values if "_atr" in df.columns else np.zeros(len(df))
    timestamps = df.index.values

    # Simuliere bis zum vorletzten Bar (letzter Bar kann kein Entry sein)
    for i in range(len(df) - 1):
        trade_long = simulate_pro_trade(
            cls_v, hgh_v, low_v, atr_v, i, 1, tp, sl, ctx.spread,
            timestamps=timestamps, symbol=ctx.symbol, opens=opn_v,
            max_bars=ctx.max_trade_bars,  # None = kein Limit
            timeout_bars=timeout_bars  # Time-based Exit
        )
        trade_short = simulate_pro_trade(
            cls_v, hgh_v, low_v, atr_v, i, -1, tp, sl, ctx.spread,
            timestamps=timestamps, symbol=ctx.symbol, opens=opn_v,
            max_bars=ctx.max_trade_bars,  # None = kein Limit
            timeout_bars=timeout_bars  # Time-based Exit
        )
        if trade_long and trade_long["result"] == 1.0:
            targets_long[i] = 1
        if trade_short and trade_short["result"] == 1.0:
            targets_short[i] = 1

    has_long, has_short = _validate_targets(targets_long, targets_short, ctx)

    return targets_long, targets_short, has_long, has_short


def compute_targets_cached(
    full_df: pd.DataFrame,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
    exit_strategy_mode: str = "fixed",
    grid_params: GridParams = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Targets einmal auf dem gesamten DataFrame (für Caching).

    Diese Funktion berechnet Targets nur einmal pro TP/SL/Timeout-Kombination.
    Die Ergebnisse können dann für jeden Fold per Index-Slice wiederverwendet werden.

    Unterstützt verschiedene Exit-Strategien:
    - "fixed": Fixe TP/SL-Werte (spread-basiert)
    - "atr_based": ATR-basierte dynamische TP/SL

    Args:
        full_df: Gesamter Inner-DataFrame (nicht nur ein Fold!)
        tp: Take-Profit Wert (Spread-Multiplikator bei fixed, ATR-Multiplikator bei atr_based)
        sl: Stop-Loss Wert (Spread-Multiplikator bei fixed, ATR-Multiplikator bei atr_based)
        ctx: SimulationContext
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        exit_strategy_mode: "fixed" oder "atr_based"
        grid_params: GridParams-Objekt mit allen Parametern (wenn vorhanden, werden tp/sl ignoriert)

    Returns:
        (targets_long, targets_short) - Arrays mit gleicher Länge wie full_df
    """
    # Dispatch zu Exit-Strategie
    if exit_strategy_mode == "atr_based":
        # ATR-basierte Exit-Strategie verwenden
        strategy_cls = get_strategy("atr_based")
        strategy = strategy_cls()

        # Extra-Parameter aus Context (atr_period, min_tp_pips, etc.)
        extra = {}
        if hasattr(ctx, 'exit_params') and ctx.exit_params:
            extra = ctx.exit_params.copy()

        # GridParams-Objekt erstellen
        grid_params = GridParams(
            tp_value=float(tp),
            sl_value=float(sl),
            timeout_bars=timeout_bars,
            extra=extra,
        )

        return strategy.compute_targets(full_df, ctx, params=grid_params)

    # Default: Fixed Exit Strategy (Numba-optimiert)
    opn_v = full_df["O"].values.astype(np.float64)
    cls_v = full_df["C"].values.astype(np.float64)
    hgh_v = full_df["H"].values.astype(np.float64)
    low_v = full_df["L"].values.astype(np.float64)

    # Distanzen berechnen (gleiche Logik wie simulate_pro_trade)
    tp_distance = ctx.spread * tp
    sl_distance = ctx.spread * sl
    slippage = ctx.spread * 0.5

    # max_bars: Wie weit maximal simuliert wird (None = bis zum Ende)
    max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(full_df)

    # timeout_bars: Wann Trade geschlossen wird (0 = kein Timeout)
    timeout_val = timeout_bars if timeout_bars else 0

    return compute_targets_numba(
        opn_v, cls_v, hgh_v, low_v,
        tp_distance, sl_distance, ctx.spread, slippage,
        max_bars, timeout_val
    )


def slice_targets_for_fold(
    full_targets_long: np.ndarray,
    full_targets_short: np.ndarray,
    full_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    ctx: SimulationContext
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Extrahiert Targets für einen bestimmten Fold aus gecachten Gesamt-Targets.

    Args:
        full_targets_long: Gecachte Long-Targets für gesamten DataFrame
        full_targets_short: Gecachte Short-Targets für gesamten DataFrame
        full_df: Der gesamte DataFrame (für Index-Mapping)
        fold_df: Der Fold-DataFrame (Train oder Val)
        ctx: SimulationContext

    Returns:
        (targets_long, targets_short, has_long, has_short)
    """
    # Finde die Positionen des Fold im Gesamt-DataFrame
    # Verwende searchsorted für robuste Index-Suche (auch bei nicht-eindeutigen Indices)
    try:
        # Versuche get_loc (schnell bei eindeutigen Indices)
        start_loc = full_df.index.get_loc(fold_df.index[0])
        end_loc = full_df.index.get_loc(fold_df.index[-1])

        # get_loc kann slice, int oder array zurückgeben
        if isinstance(start_loc, slice):
            start_idx = start_loc.start if start_loc.start is not None else 0
        elif isinstance(start_loc, np.ndarray):
            start_idx = np.where(start_loc)[0][0]
        else:
            start_idx = start_loc

        if isinstance(end_loc, slice):
            end_idx = end_loc.stop if end_loc.stop is not None else len(full_df)
        elif isinstance(end_loc, np.ndarray):
            end_idx = np.where(end_loc)[0][-1] + 1
        else:
            end_idx = end_loc + 1

    except (KeyError, IndexError):
        # Fallback: Nutze get_indexer (vectorisiert, O(log n) bei sortiertem Index)
        fold_start = fold_df.index[0]
        fold_end = fold_df.index[-1]

        indices = full_df.index.get_indexer([fold_start, fold_end])
        if indices[0] >= 0 and indices[1] >= 0:
            start_idx = indices[0]
            end_idx = indices[1] + 1
        else:
            # Index nicht gefunden - verwende Länge des fold_df
            # Dies sollte nur passieren wenn fold_df nicht aus full_df stammt
            start_idx = 0
            end_idx = len(fold_df)

    # Slice die gecachten Targets
    fold_targets_long = full_targets_long[start_idx:end_idx]
    fold_targets_short = full_targets_short[start_idx:end_idx]

    # Prüfe ob genug Targets vorhanden (nutzt konsolidierte Funktion)
    has_long, has_short = _validate_targets(fold_targets_long, fold_targets_short, ctx)

    return fold_targets_long, fold_targets_short, has_long, has_short


def select_features_from_fold(
    train_df: pd.DataFrame,
    targets: np.ndarray,
    group_features: List[str],
    min_trades: int,
    feature_selection: str = "boruta",
    max_features: int = 0,
    min_z_score: float = 0.3,
) -> Tuple[Optional[List[str]], Dict[str, float]]:
    """
    Wählt Features basierend auf einem Training-Fold.

    Args:
        train_df: Training DataFrame
        targets: Target Array
        group_features: Features der aktuellen Gruppe
        min_trades: Minimum Trades
        feature_selection:
            - "boruta" (default): Boruta findet alle relevanten Features
            - "boruta_plateau": Boruta + Plateau-Validierung (kombiniert)
            - "importance_based": Altes Verhalten mit top_n=5
        max_features: Maximum Features pro Modell (0 = Default 15)
        min_z_score: Minimum Z-Score für Boruta Feature-Akzeptanz (Default 0.3)

    Returns:
        (selected_features, importances_dict) oder (None, {})
    """
    if np.count_nonzero(targets) < min_trades // 2:
        return None, {}

    # Nur verfügbare Features nutzen
    available_features = [f for f in group_features if f in train_df.columns]
    if not available_features:
        return None, {}

    if feature_selection == "boruta":
        # Boruta: Findet relevante Features, begrenzt durch max_features
        return select_features_boruta(
            train_df, targets, available_features,
            min_trades=min_trades,
            min_z_score=min_z_score,
            max_features=max_features,
        )

    elif feature_selection == "boruta_plateau":
        # Kombination: Boruta findet relevante Features, Plateau filtert danach
        # Bei boruta_plateau etwas lockerer (min_z_score * 0.8), Plateau filtert dann weiter
        boruta_features, importances = select_features_boruta(
            train_df, targets, available_features,
            min_trades=min_trades,
            min_z_score=min_z_score * 0.8,
            max_features=max_features,
        )

        if boruta_features and len(boruta_features) >= 2:
            # Plateau-Validierung auf Boruta-Ergebnis
            # Features mit instabilen Nachbarn werden abgewertet
            from .plateau import calculate_feature_plateau_score
            plateau_results = calculate_feature_plateau_score(importances, boruta_features)

            # Behalte Features die Plateau-Check bestehen (is_plateau=True)
            # ODER die keine Nachbarn haben (können nicht validiert werden)
            stable_features = [
                f for f in boruta_features
                if f in plateau_results and (
                    plateau_results[f]["is_plateau"] or
                    len(plateau_results[f]["neighbors"]) == 0
                )
            ]

            # Fallback: Wenn zu wenige stabil, nutze alle Boruta-Features
            if len(stable_features) >= 2:
                return stable_features, importances
            return boruta_features, importances

        return boruta_features, importances

    else:
        # Altes Verhalten: Importance + Plateau mit top_n=5
        # Verwende Default-Hyperparameter (kein ctx verfügbar in dieser Funktion)
        params = {
            "n_estimators": 50,  # Reduziert für Feature Selection
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": get_xgboost_n_jobs(),
        }

        model = XGBClassifier(**params)
        model.fit(train_df[available_features], targets)
        importances = pd.Series(model.feature_importances_, index=available_features)

        plateau_features = select_plateau_features(
            importances.to_dict(), available_features,
            top_n=5, min_importance=0
        )

        if len(plateau_features) >= 2:
            return plateau_features, importances.to_dict()

        return None, importances.to_dict()


def train_model(
    train_df: pd.DataFrame,
    targets: np.ndarray,
    features: Optional[List[str]],
    min_trades: int,
    ctx: SimulationContext,
    use_reduced_params: bool = False
) -> Optional[XGBClassifier]:
    """
    Trainiert ein XGBoost-Modell.

    Args:
        use_reduced_params: Wenn True, werden Hyperparameter halbiert (für Inner CV)
    """
    if features is None or np.count_nonzero(targets) < min_trades // 2:
        return None

    # Hole Hyperparameter aus Context (aus StrategyConfig)
    params = ctx.model_hyperparameters.copy()

    # Für Inner CV: Halbiere n_estimators für schnelleres Ranking
    if use_reduced_params:
        params["n_estimators"] = max(10, params.get("n_estimators", 100) // 2)
        # max_depth bleibt gleich (wichtiger für Modellqualität)

    # Standard-Parameter falls nicht gesetzt
    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)
    params["n_jobs"] = get_xgboost_n_jobs()

    model = XGBClassifier(**params)
    model.fit(train_df[features], targets)
    return model


def _get_probs(
    model: Optional[XGBClassifier],
    df: pd.DataFrame,
    features: Optional[List[str]]
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Berechnet Wahrscheinlichkeiten für ein Modell."""
    if not features or model is None:
        return None, None
    probs = model.predict_proba(df[features])
    if 1 in model.classes_:
        win_idx = np.where(model.classes_ == 1)[0][0]
        return probs, win_idx
    return None, None


def evaluate_on_validation(
    val_df: pd.DataFrame,
    mod_long: Optional[XGBClassifier],
    mod_short: Optional[XGBClassifier],
    features_long: Optional[List[str]],
    features_short: Optional[List[str]],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
) -> Tuple[Optional[float], float, Dict[float, List[float]]]:
    """
    Evaluiert Modelle auf Validation-Set und findet besten CT.

    Bei separate_long_short=True werden separate CTs für Long und Short optimiert.

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        (best_ct, best_pnl, trades_by_ct)
        Bei separate_long_short: best_ct ist ein Tuple (ct_long, ct_short)
    """
    probs_long, long_win_idx = _get_probs(mod_long, val_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, val_df, features_short)

    # Separate CT-Optimierung wenn aktiviert
    if ctx.separate_long_short:
        return _evaluate_separate_ct(
            val_df, probs_long, probs_short, long_win_idx, short_win_idx,
            tp, sl, ctx, timeout_bars
        )

    # Standard: Gemeinsamer CT für Long und Short
    # SEQUENTIELLE Evaluierung - kein nested Threading
    # (Feature-Gruppen sind bereits parallelisiert)
    trades_by_ct = {}
    for ct in ctx.grid_ct:
        result = simulate_trades_sequential(
            val_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct, tp, sl, ctx, return_detailed=False, timeout_bars=timeout_bars
        )
        trades_by_ct[ct] = result["trades"]

    # Besten CT finden
    best_ct = None
    best_pnl = float("-inf")
    for ct, ct_trades in trades_by_ct.items():
        if len(ct_trades) >= 10:
            ct_pnl = sum(ct_trades)
            if ct_pnl > best_pnl:
                best_pnl = ct_pnl
                best_ct = ct

    return best_ct, best_pnl, trades_by_ct


def _optimize_ct_for_direction(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    win_idx: int,
    ct_values: List[float],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    direction: int,
    timeout_bars: int = None,
    min_trades: int = 5,
) -> Tuple[Optional[float], float, Dict[float, List[float]]]:
    """
    Optimiert CT für eine einzelne Richtung (Long oder Short).

    Args:
        val_df: Validation DataFrame
        probs: Wahrscheinlichkeiten
        win_idx: Win-Index
        ct_values: Liste der CT-Werte zum Testen
        tp/sl: Take-Profit/Stop-Loss
        ctx: SimulationContext
        direction: 1=Long, -1=Short
        timeout_bars: Optional Timeout
        min_trades: Minimum Trades für gültigen CT

    Returns:
        (best_ct, best_pnl, trades_by_ct)
    """
    # SEQUENTIELLE Evaluierung - kein nested Threading
    trades_by_ct = {}
    for ct in ct_values:
        result = _simulate_single_direction(
            val_df, probs, win_idx, ct, tp, sl, ctx,
            direction=direction, timeout_bars=timeout_bars
        )
        trades_by_ct[ct] = result["trades"]

    # Besten CT finden
    best_ct = None
    best_pnl = float("-inf")
    for ct, trades in trades_by_ct.items():
        if len(trades) >= min_trades:
            pnl = sum(trades)
            if pnl > best_pnl:
                best_pnl = pnl
                best_ct = ct

    return best_ct, best_pnl, trades_by_ct


def _evaluate_separate_ct(
    val_df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
) -> Tuple[Optional[tuple], float, Dict]:
    """
    Optimiert CT separat für Long und Short Trades.

    Long und Short werden UNABHÄNGIG voneinander optimiert:
    - Finde besten CT für Long (nur Long-Trades simulieren)
    - Finde besten CT für Short (nur Short-Trades simulieren)
    - Kombiniere die besten CTs

    Das reduziert die Komplexität von O(n²) auf O(2n).
    Bei 6 CT-Werten: 12 Simulationen statt 36.

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        ((ct_long, ct_short), best_combined_pnl, trades_info)
    """
    trades_info = {"long": {}, "short": {}, "combined": {}}

    # Grid für Long und Short CTs
    long_cts = ctx.long_grid_ct if ctx.long_grid_ct else ctx.grid_ct
    short_cts = ctx.short_grid_ct if ctx.short_grid_ct else ctx.grid_ct

    # === LONG CT OPTIMIERUNG (unabhängig) ===
    best_ct_long, best_pnl_long, long_trades_by_ct = None, float("-inf"), {}
    if ctx.long_enabled and probs_long is not None:
        best_ct_long, best_pnl_long, long_trades_by_ct = _optimize_ct_for_direction(
            val_df, probs_long, long_win_idx, long_cts, tp, sl, ctx,
            direction=1, timeout_bars=timeout_bars
        )
        trades_info["long"] = long_trades_by_ct

    # === SHORT CT OPTIMIERUNG (unabhängig) ===
    best_ct_short, best_pnl_short, short_trades_by_ct = None, float("-inf"), {}
    if ctx.short_enabled and probs_short is not None:
        best_ct_short, best_pnl_short, short_trades_by_ct = _optimize_ct_for_direction(
            val_df, probs_short, short_win_idx, short_cts, tp, sl, ctx,
            direction=-1, timeout_bars=timeout_bars
        )
        trades_info["short"] = short_trades_by_ct

    # === KOMBINATION ===
    # Wenn nur eine Richtung aktiviert/erfolgreich ist, verwende Default-CT für die andere
    if best_ct_long is None and best_ct_short is None:
        return None, float("-inf"), trades_info

    # Fallback auf mittleren CT-Wert wenn eine Richtung keine Trades hat
    if best_ct_long is None:
        best_ct_long = long_cts[len(long_cts) // 2] if long_cts else 0.5
        best_pnl_long = 0.0

    if best_ct_short is None:
        best_ct_short = short_cts[len(short_cts) // 2] if short_cts else 0.5
        best_pnl_short = 0.0

    # Kombinierter PnL (für Vergleich mit anderen Grid-Kombinationen)
    # Simuliere einmal mit den optimalen CTs um echten kombinierten PnL zu bekommen
    combined_result = simulate_trades_sequential_separate_ct(
        val_df, probs_long, probs_short, long_win_idx, short_win_idx,
        best_ct_long, best_ct_short, tp, sl, ctx, return_detailed=False,
        timeout_bars=timeout_bars
    )
    combined_trades = combined_result["trades"]
    combined_pnl = sum(combined_trades) if len(combined_trades) >= 10 else float("-inf")

    trades_info["combined"] = {
        "ct_long": best_ct_long,
        "ct_short": best_ct_short,
        "trades": combined_trades,
    }

    return (best_ct_long, best_ct_short), combined_pnl, trades_info


def _simulate_single_direction(
    df: pd.DataFrame,
    probs: np.ndarray,
    win_idx: int,
    ct: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    direction: int,
    timeout_bars: int = None,
) -> Dict[str, Any]:
    """Simuliert Trades für eine einzelne Richtung (Long oder Short)."""
    if direction == 1:
        return _simulate_trades_core(
            df, probs, None, win_idx, None,
            ct, 0.0, tp, sl, ctx, False, timeout_bars, direction_filter=1
        )
    else:
        return _simulate_trades_core(
            df, None, probs, None, win_idx,
            0.0, ct, tp, sl, ctx, False, timeout_bars, direction_filter=-1
        )


def simulate_trades_sequential_separate_ct(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct_long: float,
    ct_short: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
) -> Dict[str, Any]:
    """Simuliert Trades mit separaten CT-Thresholds für Long und Short."""
    return _simulate_trades_core(
        df, probs_long, probs_short, long_win_idx, short_win_idx,
        ct_long, ct_short, tp, sl, ctx, return_detailed, timeout_bars
    )


def run_inner_cv(
    inner_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    group_features: List[str],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    global_grid_pos: int,
    total_grid_combos: int,
    timeout_bars: int = None,
    cached_targets: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Führt Inner Cross-Validation für eine Grid-Kombination durch.

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        cached_targets: Optional - vorberechnete Targets {fold_idx: (targets_long, targets_short)}
                        Wenn übergeben, werden diese statt Neuberechnung verwendet.

    Returns:
        dict mit success, avg_val_pnl, best_ct, selected_features etc.
        Bei separate_long_short: best_ct ist ein Tuple (ct_long, ct_short)

    Early Termination:
        Wenn early_termination aktiviert ist (ctx.early_termination=True) und
        der Kandidat mathematisch nicht mehr min_fold_stability erreichen kann,
        wird die Evaluation vorzeitig abgebrochen.

        Beispiel: Bei 5 Folds und min_fold_stability=0.5 müssen mindestens 3
        Folds profitabel sein. Nach 3 Verlusten in Folge kann dies nicht mehr
        erreicht werden → Abbruch.
    """
    inner_val_pnls = []
    selected_features_long = None
    selected_features_short = None
    best_ct_votes = {}

    # Early Termination Setup
    total_folds = len(inner_folds)
    min_fold_stability = getattr(ctx, 'min_fold_stability', 0.5)
    early_termination_enabled = getattr(ctx, 'early_termination', True)
    min_profitable = int(np.ceil(total_folds * min_fold_stability))
    profitable_count = 0
    failed_count = 0
    early_terminated = False
    first_fold_failed = False

    # First-Fold Sanity Check Setup
    first_fold_sanity_check = getattr(ctx, 'first_fold_sanity_check', True)
    first_fold_min_win_rate = getattr(ctx, 'first_fold_min_win_rate', 0.25)
    first_fold_min_pnl = getattr(ctx, 'first_fold_min_pnl', -10.0)
    first_fold_min_trades = getattr(ctx, 'first_fold_min_trades', 5)

    # === PREPROCESSING SETUP ===
    # Preprocessing-Objekte erstellen (werden pro Fold neu gefittet)
    preprocessors = []
    if ctx.preprocessing:
        from fwbg.core import get_preprocessor
        for pp_name in ctx.preprocessing:
            try:
                pp_cls = get_preprocessor(pp_name)
                preprocessors.append((pp_name, pp_cls))
            except Exception:
                # Fehler beim Laden des Preprocessors - skippen
                pass

    for fold_idx, (train_df, val_df) in enumerate(inner_folds):
        # === PREPROCESSING: FIT/TRANSFORM ===
        # WICHTIG: Preprocessing MUSS in der CV-Schleife passieren um Lookahead Bias zu verhindern!
        # 1. fit() auf Train-Daten (lernt Parameter NUR von Train)
        # 2. transform() auf Train und Val-Daten (mit gelernten Parametern)
        if preprocessors:
            for pp_name, pp_cls in preprocessors:
                try:
                    params = ctx.preprocessing_params.get(pp_name, {})
                    pp = pp_cls()

                    # Fit auf Train-Daten (KEIN Lookahead!)
                    pp.fit(train_df, **params)

                    # Transform beide Datasets mit gelernten Parametern
                    train_df = pp.transform(train_df, **params)
                    val_df = pp.transform(val_df, **params)
                except Exception as e:
                    # Preprocessing fehlgeschlagen - Log und überspringe
                    import traceback
                    print(f"[DEBUG] Preprocessing {pp_name} failed: {e}")
                    traceback.print_exc()
                    # Fold wird trotzdem weiterverarbeitet mit Original-Daten
                    pass
        # Early Termination Check: Kann min_fold_stability noch erreicht werden?
        if early_termination_enabled and min_profitable > 0:
            remaining_folds = total_folds - fold_idx
            max_possible_profitable = profitable_count + remaining_folds
            if max_possible_profitable < min_profitable:
                # Nicht mehr möglich, genug profitable Folds zu sammeln
                early_terminated = True
                break

        # Fold-Progress (wird von progress_callback überschrieben, aber hilft beim Debugging)
        # report_progress(ctx.symbol, fold_idx + 1, len(inner_folds), "inner_cv", global_grid_pos, total_grid_combos)

        # Targets: MUSS nach Preprocessing berechnet werden (Indices müssen matchen!)
        # Wenn Preprocessing die DataFrames verändert (z.B. Zeilen entfernt), müssen
        # Targets auf den transformierten DataFrames neu berechnet werden.
        # Cache kann NICHT verwendet werden, da er auf den Original-DataFrames basiert!
        targets_long, targets_short, has_long, has_short = compute_targets(train_df, tp, sl, ctx, timeout_bars)

        if not has_long and not has_short:
            failed_count += 1
            continue

        # Feature-Auswahl auf erstem Fold (mit Fallback auf spätere Folds)
        # Wenn Fold 0 fehlschlägt, versuche es auf dem nächsten Fold
        if selected_features_long is None and selected_features_short is None:
            if has_long:
                selected_features_long, _ = select_features_from_fold(
                    train_df, targets_long, group_features, ctx.min_trades,
                    feature_selection=ctx.feature_selection,
                    max_features=ctx.max_features,
                    min_z_score=ctx.min_z_score,
                )
            if has_short:
                selected_features_short, _ = select_features_from_fold(
                    train_df, targets_short, group_features, ctx.min_trades,
                    feature_selection=ctx.feature_selection,
                    max_features=ctx.max_features,
                    min_z_score=ctx.min_z_score,
                )

        if not selected_features_long and not selected_features_short:
            failed_count += 1
            continue

        # Inner CV: Verwende reduzierte Parameter für schnelleres Ranking
        # KEINE Parallelisierung hier: XGBoost nutzt intern bereits n_jobs für Threading.
        # Paralleles Training würde zu Thread-Kontention führen.
        mod_long = train_model(train_df, targets_long, selected_features_long, ctx.min_trades, ctx, use_reduced_params=True) if has_long else None
        mod_short = train_model(train_df, targets_short, selected_features_short, ctx.min_trades, ctx, use_reduced_params=True) if has_short else None

        best_fold_ct, best_fold_pnl, trades_by_ct = evaluate_on_validation(
            val_df, mod_long, mod_short,
            selected_features_long, selected_features_short,
            tp, sl, ctx, timeout_bars
        )

        if best_fold_ct:
            inner_val_pnls.append(best_fold_pnl)
            best_ct_votes[best_fold_ct] = best_ct_votes.get(best_fold_ct, 0) + 1

            # Zähle profitable/unprofitable Folds für Early Termination
            if best_fold_pnl > 0:
                profitable_count += 1
            else:
                failed_count += 1

            # First-Fold Sanity Check: Nur nach erstem Fold, nur für extreme Fälle
            if fold_idx == 0 and first_fold_sanity_check:
                # Berechne Win-Rate aus dem trades_by_ct für den besten CT
                # Wir haben die trades nicht direkt, aber wir können sie aus evaluate_on_validation holen
                # Vereinfachte Version: Nutze PnL als Proxy
                # Bei RRR=1 entspricht PnL ungefähr (wins - losses)
                # Annahme: Bei n Trades ist Win-Rate = (n + PnL) / (2*n)
                # Aber wir brauchen die tatsächlichen Trades für die Win-Rate
                # trades_by_ct enthält Trades pro CT-Wert
                fold_trades = trades_by_ct.get(best_fold_ct, [])
                n_fold_trades = len(fold_trades)

                if n_fold_trades > 0:
                    fold_win_rate = fold_trades.count(1.0) / n_fold_trades

                    # Sanity Check: Nur bei extremen Fällen abbrechen
                    # Katastrophal = Win-Rate < 25% UND PnL stark negativ UND genug Trades
                    is_catastrophic = (
                        fold_win_rate < first_fold_min_win_rate and
                        best_fold_pnl < first_fold_min_pnl and
                        n_fold_trades >= first_fold_min_trades
                    )

                    if is_catastrophic:
                        first_fold_failed = True
                        break
                elif n_fold_trades < first_fold_min_trades:
                    # Zu wenige Trades im ersten Fold - auch abbrechen
                    first_fold_failed = True
                    break
        else:
            failed_count += 1

    if early_terminated:
        return {"success": False, "early_terminated": True, "failed_folds": failed_count}

    if first_fold_failed:
        return {"success": False, "first_fold_failed": True, "reason": "catastrophic_first_fold"}

    if not inner_val_pnls or not best_ct_votes:
        return {"success": False}

    selected_features = []
    if selected_features_long:
        selected_features.extend(selected_features_long)
    if selected_features_short:
        selected_features.extend([f for f in selected_features_short if f not in selected_features])

    if not selected_features:
        return {"success": False}

    profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)
    fold_stability = profitable_folds / len(inner_val_pnls) if inner_val_pnls else 0

    # Bester CT (kann Tuple sein bei separate_long_short)
    # Bei Tuples: Finde den CT mit den meisten Votes
    # Counter.most_common() funktioniert nicht direkt mit Tuples als Keys
    if best_ct_votes:
        best_ct = max(best_ct_votes.keys(), key=lambda x: best_ct_votes[x])
    else:
        # Kein CT gefunden - sollte nicht passieren, aber Fallback
        best_ct = ctx.grid_ct[len(ctx.grid_ct) // 2] if ctx.grid_ct else 0.5

    result = {
        "success": True,
        "avg_val_pnl": np.mean(inner_val_pnls),
        "best_ct": best_ct,
        "selected_features_long": selected_features_long,
        "selected_features_short": selected_features_short,
        "selected_features": selected_features,
        "fold_stability": fold_stability,
        "fold_pnls": inner_val_pnls,
    }

    # Bei separater Optimierung: Extrahiere ct_long und ct_short
    if ctx.separate_long_short and isinstance(best_ct, tuple):
        result["ct_long"] = best_ct[0]
        result["ct_short"] = best_ct[1]

    return result


def evaluate_on_holdout(
    holdout_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    candidate: Dict[str, Any],
    ctx: SimulationContext
) -> Dict[str, Any]:
    """
    Finale Evaluation auf dem Holdout-Set.
    Trainiert Modell auf GESAMTEM Inner-Set und testet auf Holdout.

    Returns:
        dict mit trades, trades_detailed, pnl, win_rate, n_trades
        Bei separate_long_short: zusätzlich long_stats und short_stats
    """
    # === PREPROCESSING: FIT/TRANSFORM ===
    # Preprocessing auf dem gesamten Inner-Set fitten und auf Holdout anwenden
    if ctx.preprocessing:
        from fwbg.core import get_preprocessor
        for pp_name in ctx.preprocessing:
            try:
                pp_cls = get_preprocessor(pp_name)
                params = ctx.preprocessing_params.get(pp_name, {})
                pp = pp_cls()

                # Fit auf GESAMTEM Inner-Set (alle Folds zusammen)
                pp.fit(inner_df, **params)

                # Transform beide Datasets
                inner_df = pp.transform(inner_df, **params)
                holdout_df = pp.transform(holdout_df, **params)
            except Exception:
                # Preprocessing fehlgeschlagen - trotzdem weitermachen
                pass

    tp, sl, ct = candidate["params"]
    timeout_bars = candidate.get("timeout_bars")  # Time-based Exit
    features_long = candidate.get("selected_features_long")
    features_short = candidate.get("selected_features_short")

    targets_long, targets_short, has_long, has_short = compute_targets(inner_df, tp, sl, ctx, timeout_bars)

    # Holdout: Verwende volle Parameter für finale Evaluation
    mod_long = train_model(inner_df, targets_long, features_long, ctx.min_trades, ctx, use_reduced_params=False) if has_long and features_long else None
    mod_short = train_model(inner_df, targets_short, features_short, ctx.min_trades, ctx, use_reduced_params=False) if has_short and features_short else None

    if not mod_long and not mod_short:
        return {"trades": [], "trades_detailed": [], "pnl": 0, "win_rate": 0, "n_trades": 0}

    probs_long, long_win_idx = _get_probs(mod_long, holdout_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, holdout_df, features_short)

    # Prüfe ob CT ein Tuple ist (separate Long/Short CTs)
    if isinstance(ct, tuple):
        ct_long, ct_short = ct
        result = simulate_trades_sequential_separate_ct(
            holdout_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct_long, ct_short, tp, sl, ctx, return_detailed=True,
            timeout_bars=timeout_bars
        )
    else:
        result = simulate_trades_sequential(
            holdout_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct, tp, sl, ctx, return_detailed=True, timeout_bars=timeout_bars
        )

    trades = result["trades"]
    trades_detailed = result["trades_detailed"]
    pnl = sum(trades) if trades else 0
    win_rate = trades.count(1.0) / len(trades) if trades else 0

    output = {
        "trades": trades,
        "trades_detailed": trades_detailed,
        "pnl": pnl,
        "win_rate": win_rate,
        "n_trades": len(trades),
    }

    # Bei separater Optimierung: Statistiken pro Richtung hinzufügen
    if ctx.separate_long_short and trades_detailed:
        long_trades = [t for t in trades_detailed if t.get("direction") == "LONG"]
        short_trades = [t for t in trades_detailed if t.get("direction") == "SHORT"]

        output["long_stats"] = {
            "n_trades": len(long_trades),
            "wins": sum(1 for t in long_trades if t.get("result") == 1.0),
            "pnl": sum(t.get("result", 0) for t in long_trades),
            "win_rate": sum(1 for t in long_trades if t.get("result") == 1.0) / len(long_trades) if long_trades else 0,
        }
        output["short_stats"] = {
            "n_trades": len(short_trades),
            "wins": sum(1 for t in short_trades if t.get("result") == 1.0),
            "pnl": sum(t.get("result", 0) for t in short_trades),
            "win_rate": sum(1 for t in short_trades if t.get("result") == 1.0) / len(short_trades) if short_trades else 0,
        }

    return output
