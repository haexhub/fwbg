"""
Walk-Forward Optimierung und Symbol-Verarbeitung
"""
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import psutil

from fwbg.data.config import (
    DATA_PATH, MACRO_INDICATORS, LOOKBACKS_HOURS, LOOKBACKS_DAYS,
    OOS_SIZE, tf_cfg, MIN_TRADES, WALK_FORWARD_FOLDS
)
from fwbg.core.config import StrategyConfig, RegimeFilterConfig
from fwbg.data.assets import get_asset
from fwbg.core.context import SimulationContext
from fwbg.data.loader import load_data_aligned, load_macro_csv
from fwbg.builtins.indicators import (
    compute_indicator_pool, get_feature_columns, compute_regime_filter,
    filter_features_by_group
)
from fwbg.simulation.trade import (
    calculate_sharpe_ratio, calculate_calmar_ratio,
    monte_carlo_permutation_test, monte_carlo_equity_simulation,
    adjust_kelly_for_target_dd, find_optimal_circuit_breaker, calculate_equity_smoothness
)
from fwbg.builtins.feature_selection.plateau import (
    calculate_param_plateau_score, select_best_plateau_candidate
)
from fwbg.utils.progress import (
    report_done, report_phase, report_progress, set_parallel_mode
)
from fwbg.utils.logging import log
from .nested_cv import nested_cv_split, run_inner_cv, evaluate_on_holdout
from fwbg.utils.xgb_config import set_xgboost_n_jobs


# Globale Throttling-Variablen
_last_throttle_check = 0
_throttle_wait_count = 0


def _wait_for_resources(
    max_cpu_percent: float = 0.80,
    min_free_ram_percent: float = 0.15,
    check_interval: float = 1.0,
    max_wait: float = 120.0,
    sym: str = None
):
    """
    Wartet, bis genug CPU/RAM-Ressourcen verfügbar sind.

    Diese Funktion pausiert die Grid-Search-Iteration, wenn:
    - CPU-Auslastung > max_cpu_percent
    - Freier RAM < min_free_ram_percent

    Args:
        max_cpu_percent: Maximale CPU-Auslastung (0.0-1.0 oder Prozent)
        min_free_ram_percent: Minimaler freier RAM (0.0-1.0 oder Prozent)
        check_interval: Sekunden zwischen Checks während des Wartens
        max_wait: Maximale Wartezeit in Sekunden
        sym: Asset-Symbol für Log-Ausgaben
    """
    global _last_throttle_check, _throttle_wait_count

    # Normalisiere Prozent-Werte
    max_cpu = max_cpu_percent / 100 if max_cpu_percent > 1 else max_cpu_percent
    min_ram = min_free_ram_percent / 100 if min_free_ram_percent > 1 else min_free_ram_percent

    # Nicht zu häufig checken (Performance) - aber mind. alle 5 Sekunden
    now = time.time()
    if now - _last_throttle_check < 5.0:
        return

    _last_throttle_check = now

    wait_start = time.time()
    waited = False

    while True:
        # CPU-Check: Mehrere Samples für stabilen Wert
        cpu_readings = []
        for _ in range(3):
            cpu_readings.append(psutil.cpu_percent(interval=0.15))
        cpu_percent = sum(cpu_readings) / len(cpu_readings) / 100.0

        # RAM-Check
        mem = psutil.virtual_memory()
        free_ram_percent = mem.available / mem.total

        # Prüfe ob Ressourcen OK
        cpu_ok = cpu_percent < max_cpu
        ram_ok = free_ram_percent > min_ram

        if cpu_ok and ram_ok:
            if waited:
                _throttle_wait_count += 1
                elapsed = time.time() - wait_start
                log(1, f"RESUME nach {elapsed:.1f}s (CPU: {cpu_percent*100:.0f}%, RAM: {free_ram_percent*100:.0f}% frei)", sym)
            break

        # Max Wartezeit erreicht?
        if time.time() - wait_start > max_wait:
            log(1, f"TIMEOUT nach {max_wait}s - fahre fort (CPU: {cpu_percent*100:.0f}%, RAM: {free_ram_percent*100:.0f}% frei)", sym)
            break

        # Erste Warnung loggen
        if not waited:
            reasons = []
            if not cpu_ok:
                reasons.append(f"CPU {cpu_percent*100:.0f}%")
            if not ram_ok:
                reasons.append(f"RAM {free_ram_percent*100:.0f}%")
            log(1, f"PAUSE ({', '.join(reasons)})", sym)
            waited = True

        # Warten bevor nächster Check
        time.sleep(check_interval)

    return


def _process_single_grid_combo(
    tp: int,
    sl: int,
    timeout_bars,
    group_features: list,
    inner_folds: list,
    ctx,
    regime_config: dict,
    feature_group: str,
    global_grid_pos: int,
    total_grid_combos: int,
    cached_targets: dict,
) -> tuple:
    """
    Verarbeitet eine einzelne Grid-Kombination (TP/SL/Timeout).

    Thread-safe und kann parallel für verschiedene Kombinationen aufgerufen werden.

    Returns:
        Tuple von (candidate_or_none, grid_result_or_none)
    """
    rrr = tp / sl

    # === INNER CV: Grid-Search auf Inner Folds ===
    inner_result = run_inner_cv(
        inner_folds, group_features, tp, sl, ctx,
        global_grid_pos, total_grid_combos,
        timeout_bars=timeout_bars,
        cached_targets=cached_targets,
    )

    if not inner_result["success"]:
        return None, None

    # Kandidat speichern (noch OHNE Holdout-Evaluation!)
    candidate = {
        "inner_val_pnl": inner_result["avg_val_pnl"],
        "params": (tp, sl, inner_result["best_ct"]),
        "timeout_bars": timeout_bars,
        "feats": inner_result["selected_features"],
        "feature_group": feature_group,
        "rrr": rrr,
        "selected_features_long": inner_result["selected_features_long"],
        "selected_features_short": inner_result["selected_features_short"],
        "fold_stability": inner_result.get("fold_stability", 0),
        "regime_filter": regime_config,
    }

    # Bei separater L/S Optimierung: CT-Tuple aufschlüsseln
    if ctx.separate_long_short and "ct_long" in inner_result:
        candidate["ct_long"] = inner_result["ct_long"]
        candidate["ct_short"] = inner_result["ct_short"]

    # Grid-Result mit CT (kann Tuple sein)
    conf_thresh = inner_result["best_ct"]
    grid_result = {
        "feature_group": feature_group,
        "tp_mult": tp,
        "sl_mult": sl,
        "timeout_bars": timeout_bars,
        "conf_thresh": conf_thresh,
        "rrr": rrr,
        "inner_val_pnl": inner_result["avg_val_pnl"],
        "fold_stability": inner_result.get("fold_stability", 0),
        "features": inner_result["selected_features"],
        "regime_filter": regime_config,
    }
    if isinstance(conf_thresh, tuple):
        grid_result["ct_long"] = conf_thresh[0]
        grid_result["ct_short"] = conf_thresh[1]

    return candidate, grid_result


def _process_tp_sl_combo_wrapper(args):
    """
    Wrapper-Funktion für parallele Verarbeitung einer TP/SL+timeout Kombination.

    Args:
        args: Tuple mit allen benötigten Parametern

    Returns:
        Tuple von (candidate_or_none, grid_result_or_none, combo_idx)
    """
    (tp, sl, timeout_bars, combo_idx, group_features, inner_folds, ctx, regime_config,
     feature_group, grid_offset, total_grid_combos, inner_df) = args

    from .nested_cv import compute_targets_cached, slice_targets_for_fold

    global_grid_pos = grid_offset + combo_idx + 1

    # Berechne Targets für diese Kombination
    cached_targets = None
    if inner_df is not None:
        full_targets_long, full_targets_short = compute_targets_cached(
            inner_df, tp, sl, ctx, timeout_bars,
            exit_strategy_mode=ctx.exit_strategy,
        )
        cached_targets = {}
        for fold_idx, (train_df, _) in enumerate(inner_folds):
            fold_targets_long, fold_targets_short, _, _ = slice_targets_for_fold(
                full_targets_long, full_targets_short, inner_df, train_df, ctx
            )
            cached_targets[fold_idx] = (fold_targets_long, fold_targets_short)

    # Inner CV ausführen
    candidate, grid_result = _process_single_grid_combo(
        tp, sl, timeout_bars,
        group_features, inner_folds, ctx, regime_config,
        feature_group, global_grid_pos, total_grid_combos,
        cached_targets
    )

    return candidate, grid_result, combo_idx


def _process_feature_group(
    fg_idx: int,
    feature_group: str,
    full_pool: list,
    inner_folds: list,
    grid,
    ctx,
    regime_config: dict,
    sym: str,
    n_feature_groups: int,
    parallel_mode: bool = False,
    progress_callback=None,
    inner_df=None,
) -> tuple:
    """
    Verarbeitet eine Feature-Gruppe (Grid-Search über TP/SL/Timeout).

    Diese Funktion ist thread-safe und kann parallel für verschiedene
    Feature-Gruppen aufgerufen werden.

    HINWEIS: TP/SL-Kombinationen werden SEQUENTIELL verarbeitet.
    Parallelisierung erfolgt auf Feature-Gruppen-Ebene (mit Ressourcen-Check),
    nicht innerhalb einer Feature-Gruppe. Das verhindert unkontrollierte
    Ressourcen-Überlastung durch verschachtelte Thread-Pools.

    Args:
        parallel_mode: Wenn True, wird Progress-Reporting für diesen Thread
                       unterdrückt um Chaos in der Progressbar zu vermeiden.

    Returns:
        Tuple von (candidates_list, grid_results_list)
    """
    # Setze Parallel-Modus für diesen Thread (unterdrückt Progress-Updates)
    if parallel_mode:
        set_parallel_mode(True)

    group_features = filter_features_by_group(full_pool, feature_group)

    if len(group_features) < 3:
        log(2, f"  Feature-Gruppe '{feature_group}': nur {len(group_features)} Features - übersprungen", sym)
        return [], []

    log(1, f"Feature-Gruppe {fg_idx+1}/{n_feature_groups}: {feature_group} ({len(group_features)} Features)", sym)

    grid_per_fg = ctx.grid_combinations_per_feature_group()
    total_grid_combos = ctx.total_grid_combinations()
    grid_offset = fg_idx * grid_per_fg

    # Timeout-Werte: Bei adaptive_timeout nur [None], sonst Grid-Werte
    adaptive_timeout = ctx.exit_params.get("adaptive_timeout", False)
    if adaptive_timeout:
        timeout_values = [None]  # Timeout wird in compute_targets dynamisch berechnet
    else:
        timeout_values = grid.timeout_bars if grid.timeout_bars else [None]

    # Erstelle alle Kombinationen
    combos = []
    combo_idx = 0

    skipped_combos = 0
    for tp in grid.tp:
        for sl in grid.sl:
            rrr = tp / sl
            if ctx.min_rrr > 0 and rrr < ctx.min_rrr:
                skipped_combos += len(timeout_values)
                log(2, f"  Grid (TP={tp}, SL={sl}) - SKIP (RRR {rrr:.2f} < {ctx.min_rrr})", sym)
                continue

            for timeout_bars in timeout_values:
                combos.append((
                    tp, sl, timeout_bars, combo_idx,
                    group_features, inner_folds, ctx, regime_config,
                    feature_group, grid_offset, total_grid_combos, inner_df
                ))
                combo_idx += 1

    # Progress-Update für übersprungene Combos (alle auf einmal)
    if skipped_combos > 0 and progress_callback:
        for _ in range(skipped_combos):
            progress_callback(0, grid_per_fg)

    candidates = []
    grid_results = []

    # Sequentielle Verarbeitung der TP/SL-Kombinationen
    # (Parallelisierung erfolgt auf Feature-Gruppen-Ebene mit Ressourcen-Check)
    for combo in combos:
        # Ressourcen-Check: Pausiere wenn CPU/RAM zu hoch
        # check_interval=1.0 während Warten, max_wait=300s (5 min)
        _wait_for_resources(
            max_cpu_percent=ctx.max_cpu_percent,
            min_free_ram_percent=ctx.min_free_ram_percent,
            check_interval=1.0,
            max_wait=300.0,
            sym=sym
        )

        candidate, grid_result, idx = _process_tp_sl_combo_wrapper(combo)

        if progress_callback:
            progress_callback(idx + 1, grid_per_fg)

        if candidate:
            candidates.append(candidate)
        if grid_result:
            grid_results.append(grid_result)

    return candidates, grid_results


def _process_feature_groups_parallel(
    feature_groups: list,
    full_pool: list,
    inner_folds: list,
    grid,
    ctx,
    regime_config: dict,
    sym: str,
    inner_df=None,
) -> tuple:
    """
    Verarbeitet Feature-Gruppen parallel mit RAM/CPU-Kontrolle pro Thread.

    RAM und CPU werden pro Feature-Group-Thread berechnet und begrenzt.
    Das ermöglicht maximale Parallelisierung innerhalb eines Assets,
    während das System nicht überlastet wird.

    Die Ressourcen-Limits werden aus ctx gelesen (konfigurierbar in Strategy-Config).

    Returns:
        Tuple von (all_candidates, all_grid_results)
    """
    import psutil
    import threading

    n_feature_groups = len(feature_groups)
    all_candidates = []
    all_grid_results = []

    # Gesamte Grid-Kombinationen für Progress-Tracking
    total_grid_combos = ctx.total_grid_combinations()

    # Thread-safe Zähler für aggregierten Grid-Fortschritt
    progress_lock = threading.Lock()
    completed_grid_combos = [0]  # Liste für Mutability in Closure

    def progress_callback(grid_count, grid_per_fg, fg_idx=None):
        """
        Callback für Grid-Fortschritt aus Feature-Group-Threads.

        Args:
            grid_count: Aktuelle Anzahl abgeschlossener Kombinationen in dieser Feature-Gruppe
            grid_per_fg: Gesamtzahl Kombinationen pro Feature-Gruppe
            fg_idx: Index der Feature-Gruppe (optional, für besseres Tracking)
        """
        with progress_lock:
            # Inkrementiere globalen Zähler
            completed_grid_combos[0] += 1
            current = completed_grid_combos[0]

            # Progress-Update senden (Phase-Text wird in progress.py generiert)
            # Parallel-Modus kurz deaktivieren für Update
            set_parallel_mode(False)
            report_progress(sym, 0, 0, "grid_search", current, total_grid_combos)
            set_parallel_mode(True)

    # RAM/CPU-Limits aus Strategy-Config (via ctx)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    free_ram_gb = psutil.virtual_memory().available / (1024**3)
    total_cores = mp.cpu_count()

    # Konfigurierbare globale Limits
    min_free_ram_percent = ctx.min_free_ram_percent
    max_cpu_percent = ctx.max_cpu_percent

    # Dynamische RAM-Schätzung pro Feature-Group
    # Schätze ~25% von ram_per_worker_gb pro Feature-Group Thread
    # (Feature-Groups sind weniger RAM-intensiv als vollständige Asset-Worker)
    ram_per_worker_gb = ctx.ram_per_worker_gb
    estimated_ram_per_fg = ram_per_worker_gb * 0.25

    # Berechne minimalen freien RAM (ABSOLUT)
    min_free_ram_gb = total_ram_gb * min_free_ram_percent

    # Aktuelle RAM-Auslastung berücksichtigen
    current_used_percent = psutil.virtual_memory().percent / 100.0
    target_max_used_percent = 1.0 - min_free_ram_percent

    if current_used_percent > target_max_used_percent:
        # System ist bereits überlastet
        available_ram_for_threads = max(0, free_ram_gb - min_free_ram_gb)
        log(2, f"  WARNUNG: RAM bereits bei {current_used_percent*100:.0f}% (Ziel: max {target_max_used_percent*100:.0f}%)", sym)
    else:
        # Berechne wie viel RAM wir noch nutzen dürfen
        max_usable_ram_gb = total_ram_gb * target_max_used_percent
        currently_used_gb = total_ram_gb * current_used_percent
        available_ram_for_threads = max(0, max_usable_ram_gb - currently_used_gb)

    # RAM-basiertes Limit
    if available_ram_for_threads < estimated_ram_per_fg:
        ram_based_limit = 1  # Mindestens 1 Thread sequentiell
    else:
        ram_based_limit = max(1, int(available_ram_for_threads / estimated_ram_per_fg))

    # CPU-Limit: Maximal max_cpu_percent der Kerne nutzen
    # Jede Feature-Group bekommt dynamisch CPU zugeteilt basierend auf Verfügbarkeit
    cpu_based_limit = max(1, int(total_cores * max_cpu_percent))

    # Effektives Limit: Das Minimum aus RAM, CPU und Anzahl Feature-Groups
    max_workers = min(ram_based_limit, cpu_based_limit, n_feature_groups)

    # XGBoost n_jobs: aus Config oder automatisch berechnen
    # 0 = auto (Kerne / parallele Worker, min 2)
    # 1 = single-threaded (für VPS/Production)
    # -1 = alle Kerne (Vorsicht: Überparallelisierung!)
    if ctx.xgboost_n_jobs == 0:
        # Automatisch: Kerne gleichmäßig auf Worker verteilen
        # Minimum 2 Kerne pro XGBoost für sinnvolle Parallelisierung
        # Nicht alle Worker sind gleichzeitig CPU-intensiv (I/O, Warten etc.)
        xgb_n_jobs = max(2, total_cores // max_workers)
    else:
        xgb_n_jobs = ctx.xgboost_n_jobs
    set_xgboost_n_jobs(xgb_n_jobs)

    # Detailliertes Logging (Level 2 = nur bei Verbose)
    log(2, "=== Ressourcen-Konfiguration für Feature-Gruppen ===", sym)
    log(2, f"  System: {total_ram_gb:.1f}GB RAM total, {free_ram_gb:.1f}GB frei, {total_cores} CPU-Kerne", sym)
    log(2, f"  Config: {ram_per_thread_gb}GB RAM/Thread, {cpu_per_thread} CPU/Thread", sym)
    log(2, f"  Limits: min_free_ram={min_free_ram_percent*100:.0f}%, max_cpu={max_cpu_percent*100:.0f}%", sym)
    if ctx.xgboost_n_jobs == 0:
        log(2, f"  XGBoost n_jobs: {xgb_n_jobs} (auto: {total_cores} Kerne / {max_workers} Worker)", sym)
    else:
        log(2, f"  XGBoost n_jobs: {xgb_n_jobs} (konfiguriert)", sym)
    log(3, "  Berechnung:", sym)
    log(3, f"    - RAM-Reserve: {min_free_ram_gb:.1f}GB (={min_free_ram_percent*100:.0f}% von {total_ram_gb:.1f}GB)", sym)
    log(3, f"    - Verfügbar für Threads: {available_ram_for_threads:.1f}GB", sym)
    log(3, f"    - RAM-basiertes Limit: {ram_based_limit} Threads", sym)
    log(3, f"    - CPU-basiertes Limit: {cpu_based_limit} Threads ({usable_cores} nutzbare Kerne / {cpu_per_thread} pro Thread)", sym)
    log(2, f"  => Effektives Limit: {max_workers} parallele Feature-Gruppen", sym)

    completed = 0
    start_time = time.time()

    # Initialen Progress reporten (grid_pos=0, wird durch progress_callback aktualisiert)
    report_progress(sym, 0, n_feature_groups, "feature_groups", 0, total_grid_combos)

    # HINWEIS: ThreadPoolExecutor statt ProcessPoolExecutor ist hier BEABSICHTIGT:
    # 1. XGBoost parallelisiert intern bereits mit n_jobs (libgomp/OpenMP)
    # 2. ProcessPoolExecutor würde Pickle-Overhead für DataFrames verursachen
    # 3. Feature-Gruppen sind I/O-bound (DataFrame-Slicing) + CPU-bound (XGBoost)
    # 4. XGBoost releases GIL während der Berechnung
    # Der Hauptteil der CPU-Arbeit geschieht in XGBoost (C++), nicht in Python.

    def can_start_feature_group(active_workers: int) -> bool:
        """
        Prüft ob eine weitere Feature-Gruppe gestartet werden kann.

        WICHTIG: Feature-Gruppen-Parallelisierung ist INNERHALB eines Assets.
        Die Asset-Ebene (AdaptivePoolManager) kontrolliert bereits die Gesamtlast.
        Hier prüfen wir nur RAM, um den lokalen Thread-Pool nicht zu überladen.
        CPU-Check ist auf Asset-Ebene bereits gemacht worden.
        """
        if active_workers >= max_workers:
            return False

        # RAM-Check: Genug freier RAM für eine weitere Feature-Gruppe?
        current_free_ram = psutil.virtual_memory().available / (1024**3)
        if current_free_ram < min_free_ram_gb + ram_per_thread_gb:
            return False

        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        fg_iter = iter(enumerate(feature_groups))
        active_count = 0

        # Starte konservativ mit 1 Feature-Gruppe
        try:
            fg_idx, feature_group = next(fg_iter)
            future = executor.submit(
                _process_feature_group,
                fg_idx, feature_group, full_pool, inner_folds,
                grid, ctx, regime_config, sym, n_feature_groups,
                parallel_mode=True,
                progress_callback=progress_callback,
                inner_df=inner_df,
            )
            futures[future] = feature_group
            active_count += 1
            log(2, f"Gestartet: Feature-Gruppe '{feature_group}' (1/{n_feature_groups})", sym)
        except StopIteration:
            pass

        # Adaptive Verarbeitung: Starte neue Tasks wenn Ressourcen frei
        fg_remaining = True
        last_scale_check = time.time()

        while futures or fg_remaining:
            # Fertige Tasks einsammeln
            done_futures = [f for f in list(futures.keys()) if f.done()]

            for future in done_futures:
                feature_group = futures.pop(future)
                active_count -= 1
                completed += 1

                try:
                    fg_candidates, fg_grid_results = future.result()
                    all_candidates.extend(fg_candidates)
                    all_grid_results.extend(fg_grid_results)

                    current_mem = psutil.virtual_memory()
                    current_cpu = psutil.cpu_percent(interval=0.1)
                    elapsed = time.time() - start_time
                    log(2, f"Feature-Gruppe '{feature_group}' fertig ({completed}/{n_feature_groups}) "
                           f"- {len(fg_candidates)} Kandidaten, "
                           f"RAM: {current_mem.percent:.1f}% ({current_mem.available/(1024**3):.1f}GB frei), "
                           f"CPU: {current_cpu:.1f}%, Zeit: {elapsed:.1f}s", sym)
                except Exception as e:
                    log(1, f"Fehler bei Feature-Gruppe '{feature_group}': {e}", sym)

            # Periodisch prüfen ob neue Feature-Gruppen gestartet werden können
            now = time.time()
            if fg_remaining and now - last_scale_check >= 1.0:
                last_scale_check = now

                while fg_remaining and can_start_feature_group(active_count):
                    try:
                        fg_idx, feature_group = next(fg_iter)
                        future = executor.submit(
                            _process_feature_group,
                            fg_idx, feature_group, full_pool, inner_folds,
                            grid, ctx, regime_config, sym, n_feature_groups,
                            parallel_mode=True,
                            progress_callback=progress_callback,
                            inner_df=inner_df,
                        )
                        futures[future] = feature_group
                        active_count += 1
                        log(2, f"Gestartet: Feature-Gruppe '{feature_group}' ({active_count} aktiv)", sym)
                    except StopIteration:
                        fg_remaining = False
                        break

            # Kurz warten
            if futures:
                time.sleep(0.2)

    total_elapsed = time.time() - start_time
    final_mem = psutil.virtual_memory()

    # Phase-Update: Feature-Gruppen fertig
    report_phase(sym, f"Kandidaten: {len(all_candidates)}")

    log(2, "=== Feature-Gruppen abgeschlossen ===", sym)
    log(2, f"  Verarbeitet: {completed}/{n_feature_groups} Gruppen in {total_elapsed:.1f}s", sym)
    log(2, f"  Kandidaten gefunden: {len(all_candidates)}", sym)
    log(2, f"  Finale RAM-Auslastung: {final_mem.percent:.1f}% ({final_mem.available/(1024**3):.1f}GB frei)", sym)
    return all_candidates, all_grid_results


def walk_forward_split(df, n_folds=WALK_FORWARD_FOLDS, oos_size=OOS_SIZE):
    """
    Generiert Walk-Forward Fenster: [(train_df, val_df, test_df), ...]

    DEPRECATED: Nutze nested_cv_split() für unbiased Evaluation.
    Diese Funktion wird nur noch für Abwärtskompatibilität beibehalten.

    Jeder Fold hat (60/20/20 Split relativ zum verfügbaren Bereich):
    - train_df: Für Modell-Training (~60%)
    - val_df: Für Hyperparameter-Optimierung (~20%, z.B. Confidence Threshold)
    - test_df: Für finale Out-of-Sample Evaluation (~20%, KEINE Parameterauswahl!)

    Args:
        df: DataFrame mit allen Daten
        n_folds: Anzahl der Walk-Forward Folds
        oos_size: Größe des OOS-Fensters pro Fold (definiert die 20% OOS)
    """
    total_len = len(df)
    min_train = total_len - (n_folds * oos_size)

    # Validation-Größe = gleich wie OOS für 60/20/20 Split
    val_size = oos_size

    if min_train < oos_size * 3:
        # Nicht genug Daten für Walk-Forward mit 60/20/20
        # Fallback: 60/20/20 auf gesamten Datensatz
        test_size = int(total_len * 0.2)
        val_size = int(total_len * 0.2)
        train_size = total_len - test_size - val_size

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        return [(train_df, val_df, test_df)]

    folds = []
    for i in range(n_folds):
        # OOS (Test) Bereich
        test_end = total_len - (i * oos_size)
        test_start = test_end - oos_size

        # Validation Bereich (gleich groß wie OOS, direkt davor)
        val_end = test_start
        val_start = val_end - val_size

        # Training Bereich (alles davor)
        train_end = val_start

        if train_end < oos_size:
            # Nicht genug Daten für diesen Fold
            continue

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        folds.append((train_df, val_df, test_df))

    return list(reversed(folds))  # Chronologisch sortieren


def process_symbol(csv_path: str, strategy: StrategyConfig) -> dict:
    """
    Verarbeitet ein einzelnes Symbol mit Walk-Forward Optimierung.

    Args:
        csv_path: Pfad zur CSV-Datei
        strategy: StrategyConfig mit allen Strategie-Parametern
    """
    sym = os.path.basename(csv_path).split("_")[0]
    t_start = time.time()

    if sym in ["VIX", "DXY"]:
        log(2, "Übersprungen (Makro-Asset)", sym)
        return {"symbol": sym, "status": "macro_asset"}

    log(1, "START", sym)
    report_phase(sym, "Lade Daten...")

    try:
        t0 = time.time()
        df = load_data_aligned(csv_path)
        if df is None:
            log(1, "SKIP - Keine Daten", sym)
            return {"symbol": sym, "status": "no_data"}
        log(2, f"Daten geladen: {len(df)} Zeilen ({time.time()-t0:.1f}s)", sym)
        report_phase(sym, "Makro-Indikatoren...")

        # === ALLE MAKRO-INDIKATOREN LADEN ===
        t0 = time.time()
        df["_date"] = df.index.date
        macro_count = 0

        for filename, prefix in MACRO_INDICATORS.items():
            macro_path = f"{DATA_PATH}/{filename}.csv"
            macro_df = load_macro_csv(macro_path)
            if macro_df is not None:
                try:
                    macro_lookup = macro_df["Close"].to_dict()

                    col_name = f"macro_{prefix}"
                    df[col_name] = df["_date"].map(lambda d: macro_lookup.get(pd.Timestamp(d), np.nan))
                    df[col_name] = df[col_name].ffill()

                    # Stunden-basierte Lookbacks
                    for lb_h in LOOKBACKS_HOURS:
                        df[f"{col_name}_chg_{lb_h}h"] = df[col_name].pct_change(lb_h) * 100

                    # Tages-basierte Lookbacks
                    for lb_d in LOOKBACKS_DAYS:
                        df[f"{col_name}_chg_{lb_d}d"] = df[col_name].pct_change(24 * lb_d) * 100

                    macro_count += 1
                except Exception:
                    pass

        df = df.drop(columns=["_date"], errors="ignore")
        log(2, f"Makro-Indikatoren: {macro_count} geladen ({time.time()-t0:.1f}s)", sym)

        # === ABGELEITETE FEATURES (Spreads & Ratios) ===
        t0 = time.time()
        if "macro_tnx" in df.columns and "macro_irx" in df.columns:
            df["macro_yield_curve_10y_3m"] = df["macro_tnx"] - df["macro_irx"]
        if "macro_tnx" in df.columns and "macro_fvx" in df.columns:
            df["macro_yield_curve_10y_5y"] = df["macro_tnx"] - df["macro_fvx"]

        if "macro_vix" in df.columns and "macro_vvix" in df.columns:
            df["macro_vix_vvix_ratio"] = df["macro_vix"] / (df["macro_vvix"] + 1e-10)

        if "macro_spx" in df.columns and "macro_tlt" in df.columns:
            df["macro_risk_ratio_spx_tlt"] = df["macro_spx"] / (df["macro_tlt"] + 1e-10)
        if "macro_hyg" in df.columns and "macro_lqd" in df.columns:
            df["macro_credit_spread_proxy"] = df["macro_hyg"] / (df["macro_lqd"] + 1e-10)

        if "macro_russell" in df.columns and "macro_spx" in df.columns:
            df["macro_smallcap_ratio"] = df["macro_russell"] / (df["macro_spx"] + 1e-10)

        if "macro_xlk" in df.columns and "macro_xlu" in df.columns:
            df["macro_tech_defensive_ratio"] = df["macro_xlk"] / (df["macro_xlu"] + 1e-10)

        # Zinsdaten laden
        for rate_name, rate_file in [("fed", "FED_RATE.csv"), ("ecb", "ECB_RATE.csv")]:
            rate_path = f"{DATA_PATH}/{rate_file}"
            if os.path.exists(rate_path):
                try:
                    rate_df = pd.read_csv(rate_path, parse_dates=["Date"], index_col="Date")
                    rate_series = rate_df["Rate"].reindex(df.index, method="ffill")
                    df[f"macro_{rate_name}_rate"] = rate_series
                    for lb in [30, 90, 180]:
                        df[f"macro_{rate_name}_chg_{lb}d"] = df[f"macro_{rate_name}_rate"].diff(24 * lb)
                except Exception:
                    pass

        if "macro_fed_rate" in df.columns and "macro_ecb_rate" in df.columns:
            df["macro_rate_diff_usd_eur"] = df["macro_fed_rate"] - df["macro_ecb_rate"]

        log(3, f"Abgeleitete Features berechnet ({time.time()-t0:.1f}s)", sym)

        # === PREPROCESSING (optional) ===
        preprocessing_info = None
        if strategy.preprocessing:  # Liste von Preprocessor-Namen
            report_phase(sym, "Preprocessing...")
            t0 = time.time()
            rows_before = len(df)

            # Neues Plugin-Format: preprocessing ist Liste von Strings
            from fwbg.core import get_preprocessor

            applied = []
            for pp_name in strategy.preprocessing:
                try:
                    pp_cls = get_preprocessor(pp_name)
                    params = strategy.preprocessing_params.get(pp_name, {})
                    pp = pp_cls()
                    df = pp.transform(df, **params)
                    applied.append(pp_name)
                except Exception as e:
                    log(2, f"Preprocessor {pp_name} fehlgeschlagen: {e}", sym)

            preprocessing_info = {"applied": applied}
            log(2, f"Preprocessing: {applied} ({rows_before} -> {len(df)} Zeilen, {time.time()-t0:.1f}s)", sym)

        t0 = time.time()

        def indicator_progress(name, idx, total):
            report_phase(sym, f"Indikatoren: {name} ({idx}/{total})")

        report_phase(sym, "Berechne Indikatoren...")
        df = compute_indicator_pool(df, progress_callback=indicator_progress).dropna()
        log(2, f"Indikatoren berechnet: {len(df)} Zeilen nach dropna ({time.time()-t0:.1f}s)", sym)

        if len(df) < MIN_TRADES * 2:
            log(1, f"SKIP - Zu wenig Daten nach dropna ({len(df)} < {MIN_TRADES * 2})", sym)
            return {"symbol": sym, "status": "insufficient_data", "rows": len(df)}

        # Feature-Pool vorbereiten (unabhängig von Regime-Filter)
        full_pool = get_feature_columns(df)
        log(2, f"Feature-Pool: {len(full_pool)} Features", sym)

        # Entferne Features mit inf/nan (XGBoost verträgt keine inf)
        t0 = time.time()
        clean_pool = []
        excluded_inf = 0
        excluded_nan = 0
        for col in full_pool:
            if col in df.columns:
                has_inf = np.isinf(df[col]).any()
                nan_ratio = df[col].isna().sum() / len(df)
                if has_inf:
                    excluded_inf += 1
                elif nan_ratio >= 0.1:
                    excluded_nan += 1
                else:
                    clean_pool.append(col)
        full_pool = clean_pool
        log(2, f"Clean Pool: {len(full_pool)} Features (excl: {excluded_inf} inf, {excluded_nan} nan) ({time.time()-t0:.1f}s)", sym)

        if len(full_pool) < 5:
            log(1, f"SKIP - Zu wenig saubere Features ({len(full_pool)} < 5)", sym)
            return {"symbol": sym, "status": "insufficient_features", "features": len(full_pool)}

        # Asset-Konfiguration laden
        asset = get_asset(sym)

        # SimulationContext erstellen (wird durch alle Funktionen gereicht)
        ctx = SimulationContext.create(asset, strategy)

        # Kurzreferenzen für lokale Verwendung
        grid = strategy.get_grid_for_class(asset.asset_class)

        candidates = []
        all_grid_results = []  # Alle Kombinationen tracken

        # Feature-Gruppen aus Strategy-Config
        feature_groups_to_test = ctx.feature_groups

        # Regime-Filter Kombinationen aus Grid (falls definiert)
        regime_filter_combinations = grid.regime_filter_grid.get_combinations()
        n_regime_combos = len(regime_filter_combinations)

        # Berechne Gesamtzahl der Kombinationen inkl. Regime-Filter
        base_combos = ctx.total_grid_combinations()
        total_combos = base_combos * n_regime_combos
        log(1, f"Grid-Search: {len(feature_groups_to_test)} FG x {len(grid.tp)}x{len(grid.sl)}x{len(grid.ct)} x {n_regime_combos} Regime = {total_combos} Kombinationen", sym)
        if ctx.min_rrr > 0:
            log(1, f"Min RRR Filter: {ctx.min_rrr} (Scalping-Strategien mit RRR < {ctx.min_rrr} werden gefiltert)", sym)

        # === NESTED CV: Holdout Split ===
        # Die letzten 20% werden KOMPLETT zurückgehalten für finale Evaluation
        cv_split = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=5)
        inner_folds = cv_split["inner_folds"]
        holdout_df = cv_split["holdout_df"]
        inner_df = cv_split["inner_df"]

        log(1, f"Nested CV: {len(inner_df)} Inner / {len(holdout_df)} Holdout (nie gesehen während Grid-Search)", sym)

        # === ÄUSSERSTE SCHLEIFE: Regime-Filter Kombinationen ===
        for rf_idx, regime_config in enumerate(regime_filter_combinations):
            # Erstelle RegimeFilterConfig aus Kombination
            regime_params = RegimeFilterConfig.from_dict(regime_config)

            # Berechne _regime_ok für diese Kombination
            df["_regime_ok"] = compute_regime_filter(df, regime_params)

            # Update inner_folds mit neuem regime_ok
            # inner_folds ist eine Liste von (train_df, val_df) Tupeln
            for train_df_fold, val_df_fold in inner_folds:
                train_df_fold["_regime_ok"] = df.loc[train_df_fold.index, "_regime_ok"]
                val_df_fold["_regime_ok"] = df.loc[val_df_fold.index, "_regime_ok"]
            holdout_df["_regime_ok"] = df.loc[holdout_df.index, "_regime_ok"]
            inner_df["_regime_ok"] = df.loc[inner_df.index, "_regime_ok"]

            # Log Regime-Filter Info
            regime_desc = []
            if regime_params.adx_enabled:
                regime_desc.append(f"ADX>={regime_params.adx_min}")
            if regime_params.vix_enabled:
                regime_desc.append(f"VIX<={regime_params.vix_max}")
            if regime_params.hurst_enabled:
                hurst_parts = []
                if regime_params.hurst_min is not None:
                    hurst_parts.append(f"H>={regime_params.hurst_min}")
                if regime_params.hurst_max is not None:
                    hurst_parts.append(f"H<={regime_params.hurst_max}")
                regime_desc.append(" & ".join(hurst_parts) if hurst_parts else "Hurst")
            regime_str = " + ".join(regime_desc) if regime_desc else "No Filter"

            if n_regime_combos > 1:
                log(1, f"Regime {rf_idx+1}/{n_regime_combos}: {regime_str}", sym)

            # Feature-Gruppen verarbeiten mit adaptivem Threading
            n_feature_groups = len(feature_groups_to_test)

            if n_feature_groups <= 1:
                # Nur 1 Feature-Gruppe: sequentiell
                for fg_idx, feature_group in enumerate(feature_groups_to_test):
                    fg_candidates, fg_grid_results = _process_feature_group(
                        fg_idx, feature_group, full_pool, inner_folds,
                        grid, ctx, regime_config, sym, n_feature_groups,
                        inner_df=inner_df
                    )
                    candidates.extend(fg_candidates)
                    all_grid_results.extend(fg_grid_results)
            else:
                # Mehrere Feature-Gruppen: parallel ohne zusätzliche RAM-Prüfung
                # RAM-Kontrolle erfolgt auf Asset-Ebene im AdaptivePoolManager
                fg_candidates, fg_grid_results = _process_feature_groups_parallel(
                    feature_groups_to_test, full_pool, inner_folds,
                    grid, ctx, regime_config, sym, inner_df=inner_df
                )
                candidates.extend(fg_candidates)
                all_grid_results.extend(fg_grid_results)

        log(2, f"Inner CV fertig: {len(candidates)} Kandidaten ({time.time()-t_start:.1f}s)", sym)

        grid_results = all_grid_results

        if not candidates:
            log(1, "SKIP - Keine profitablen Kandidaten", sym)
            report_done(sym, "no_candidates")
            return {"symbol": sym, "status": "no_candidates", "grid_results": grid_results}

        # === PLATEAU-BASIERTE AUSWAHL (basierend auf Inner CV PnL) ===
        # Sortiere nach Inner Validation PnL für Plateau-Berechnung
        for c in candidates:
            c["score"] = c["inner_val_pnl"]

        candidates = calculate_param_plateau_score(
            candidates,
            grid.tp,
            grid.sl,
            grid.ct
        )

        # Wähle besten Plateau-Kandidaten
        b = select_best_plateau_candidate(
            candidates,
            grid.tp,
            grid.sl,
            grid.ct,
            min_neighbors=2
        )

        if not b:
            # Fallback: Bester nach Inner Val PnL
            candidates.sort(key=lambda x: x["inner_val_pnl"], reverse=True)
            b = candidates[0] if candidates else None

        if not b:
            report_done(sym, "no_plateau")
            return {"symbol": sym, "status": "no_plateau", "grid_results": grid_results}

        # === TOP-N KANDIDATEN SAMMELN (für Vergleich) ===
        # Sortiere alle Kandidaten nach Inner Val PnL und behalte Top 5
        # mit unterschiedlichen RRR-Werten für Diversität
        TOP_N = 5
        MIN_ANNUAL_RETURN = strategy.filters.min_annual_return  # Aus Strategie-Config
        candidates.sort(key=lambda x: x.get("inner_val_pnl", 0), reverse=True)

        # Berechne Inner CV Zeitraum für Jahresrendite-Schätzung
        n_inner_folds = len(inner_folds)
        inner_val_size = len(inner_folds[0][1]) if inner_folds else OOS_SIZE
        inner_total_bars = n_inner_folds * inner_val_size
        bars_per_year = tf_cfg["bars_per_hour"] * 24 * 250  # 250 Trading-Tage pro Jahr
        inner_years = inner_total_bars / bars_per_year if bars_per_year > 0 else 1

        def estimate_annual_return(inner_val_pnl):
            """Schätzt Jahresrendite aus Inner CV PnL."""
            if inner_val_pnl <= 0 or inner_years <= 0:
                return -100
            final_equity = 100 + inner_val_pnl
            return ((final_equity / 100.0) ** (1 / inner_years) - 1) * 100

        # Sammle Top-N mit RRR-Diversität und Mindest-Rendite
        top_candidates_for_export = []
        seen_rrr = set()
        skipped_low_return = 0
        for c in candidates:
            # Prüfe Mindest-Rendite
            est_return = estimate_annual_return(c.get("inner_val_pnl", 0))
            if est_return < MIN_ANNUAL_RETURN:
                skipped_low_return += 1
                continue

            rrr_bucket = round(c["rrr"], 1)  # Bucket auf 0.1 gerundet
            if rrr_bucket not in seen_rrr or len(top_candidates_for_export) < 3:
                top_candidates_for_export.append({
                    "rank": len(top_candidates_for_export) + 1,
                    "params": c["params"],
                    "rrr": c["rrr"],
                    "inner_val_pnl": c.get("inner_val_pnl", 0),
                    "est_annual_return": est_return,
                    "feature_group": c.get("feature_group", "unknown"),
                    "feats": c.get("feats", []),
                    "plateau_score": c.get("plateau_score", 0),
                    "selected_features_long": c.get("selected_features_long", []),
                    "selected_features_short": c.get("selected_features_short", []),
                })
                seen_rrr.add(rrr_bucket)
            if len(top_candidates_for_export) >= TOP_N:
                break

        if skipped_low_return > 0:
            log(2, f"  {skipped_low_return} Kandidaten übersprungen (< {MIN_ANNUAL_RETURN}%/Jahr)", sym)
        log(2, f"Top-{len(top_candidates_for_export)} Kandidaten gesammelt (RRR: {[c['rrr'] for c in top_candidates_for_export]})", sym)

        # === HOLDOUT EVALUATION ===
        # JETZT erst evaluieren wir auf dem Holdout-Set (nie vorher gesehen!)
        ct_param = b['params'][2]
        if isinstance(ct_param, tuple):
            ct_str = f"CT_L={ct_param[0]:.2f}/CT_S={ct_param[1]:.2f}"
        else:
            ct_str = f"CT={ct_param:.2f}"
        log(1, f"Holdout-Evaluation für besten Kandidaten (TP={b['params'][0]}, SL={b['params'][1]}, {ct_str})", sym)

        holdout_result = evaluate_on_holdout(holdout_df, inner_df, b, ctx)

        if holdout_result["n_trades"] < ctx.min_trades:
            log(1, f"SKIP - Zu wenig Holdout-Trades ({holdout_result['n_trades']} < {ctx.min_trades})", sym)
            report_done(sym, "insufficient_holdout_trades")
            return {"symbol": sym, "status": "insufficient_holdout_trades", "grid_results": grid_results}

        # Füge Holdout-Ergebnisse zum Kandidaten hinzu
        b["tr"] = holdout_result["trades"]
        b["trades_detailed"] = holdout_result["trades_detailed"]
        b["pnl"] = holdout_result["pnl"]
        b["holdout_win_rate"] = holdout_result["win_rate"]

        # Berechne Metriken auf Holdout-Trades
        wr = holdout_result["win_rate"]
        tr = holdout_result["trades"]

        preliminary_kelly = max(0, min(0.05, (
            (wr * b["rrr"] - (1 - wr)) / b["rrr"]
        ) / 4)) if wr > 0 else 0.01

        trade_returns = [preliminary_kelly * b["rrr"] if r > 0 else -preliminary_kelly for r in tr]

        # Trades pro Jahr = (Anzahl Trades / Anzahl Bars) * Bars pro Jahr
        holdout_bars = len(holdout_df)
        trades_per_year = (len(tr) / holdout_bars) * bars_per_year if holdout_bars > 0 else len(tr)
        b["sharpe"] = calculate_sharpe_ratio(trade_returns, trades_per_year=trades_per_year)
        b["calmar"] = calculate_calmar_ratio(tr, preliminary_kelly, b["rrr"])
        b["smoothness"] = calculate_equity_smoothness(tr, preliminary_kelly, b["rrr"])

        # Good hours aus Holdout
        hour_pnl = {}
        for t in holdout_result["trades_detailed"]:
            h = t["hour"]
            hour_pnl[h] = hour_pnl.get(h, 0) + t["result"]
        b["good_hours"] = sorted([h for h, pnl in hour_pnl.items() if pnl > 0]) or list(range(24))

        # 1/4 Kelly
        p = wr
        q = 1 - p
        rrr = b["rrr"]
        full_kelly = (p * rrr - q) / rrr if rrr > 0 else 0
        fk = max(0, min(0.05, full_kelly / 4))

        if fk <= 0:
            report_done(sym, "no_kelly")
            # Speichere trotzdem Holdout-Info für Analyse
            return {
                "symbol": sym,
                "status": "no_kelly",
                "grid_results": grid_results,
                "best_candidate": {
                    "params": {"tp": b["params"][0], "sl": b["params"][1], "ct": b["params"][2]},
                    "rrr": b["rrr"],
                    "inner_val_pnl": b.get("inner_val_pnl", 0),
                },
                "holdout_result": {
                    "win_rate": wr,
                    "n_trades": holdout_result["n_trades"],
                    "pnl": holdout_result["pnl"],
                    "full_kelly": full_kelly,
                    "reason": f"Kelly <= 0 (WR={wr*100:.1f}%, RRR={rrr:.2f}, benötigt WR >= {1/(rrr+1)*100:.1f}%)"
                }
            }

        # === MONTE CARLO TESTS ===
        # Prüfe ob Ergebnisse statistisch signifikant sind
        report_phase(sym, "Monte Carlo Validierung...")
        log(2, "=== Monte Carlo Validierung ===", sym)
        log(2, "  Starte Permutations-Test (1000 Samples)...", sym)
        t_mc = time.time()
        mc_perm = monte_carlo_permutation_test(b["tr"], n_permutations=1000)
        log(2, "  Starte Equity-Simulation (500 Samples)...", sym)
        mc_equity = monte_carlo_equity_simulation(b["tr"], fk, rrr, n_simulations=500)

        log(2, f"  Monte Carlo fertig: p={mc_perm['p_value']:.3f}, "
               f"Equity median={mc_equity['median_equity']:.1f}, "
               f"bankruptcy={mc_equity['bankruptcy_rate']:.1%} ({time.time()-t_mc:.1f}s)", sym)

        # Filtere nicht-signifikante Ergebnisse
        if not mc_perm["is_significant"]:
            log(1, f"SKIP - Nicht signifikant (p={mc_perm['p_value']:.3f} >= 0.05)", sym)
            report_done(sym, "not_significant")
            return {
                "symbol": sym,
                "status": "not_significant",
                "p_value": mc_perm["p_value"],
                "grid_results": grid_results
            }

        # Warne bei hoher Bankruptcy-Rate
        if mc_equity["bankruptcy_rate"] > 0.1:
            log(1, f"WARNUNG: {mc_equity['bankruptcy_rate']:.1%} Bankruptcy-Rate in MC-Simulation", sym)

        # === KELLY-ANPASSUNG FÜR ZIEL-DRAWDOWN ===
        # Passe Kelly an, um Max DD auf ~30% zu begrenzen
        kelly_adjustment = adjust_kelly_for_target_dd(b["tr"], fk, rrr, target_max_dd=0.30)
        if kelly_adjustment["scale_factor"] < 1.0:
            log(2, f"Kelly angepasst: {fk*100:.2f}% -> {kelly_adjustment['adjusted_kelly']*100:.2f}% "
                   f"(DD: {kelly_adjustment['original_dd']*100:.0f}% -> {kelly_adjustment['adjusted_dd']*100:.0f}%)", sym)
            fk = kelly_adjustment["adjusted_kelly"]

        # === CIRCUIT BREAKER OPTIMIERUNG ===
        # Finde optimale Pause-Parameter
        circuit_breaker = find_optimal_circuit_breaker(
            b["tr"], fk, rrr,
            loss_range=(3, 8),      # Pausiere nach 3-8 Verlusten
            pause_range=(5, 30)     # Pause für 5-30 Trades
        )

        if circuit_breaker["optimal_pause_after_losses"] > 0:
            log(2, f"Circuit Breaker: Pause nach {circuit_breaker['optimal_pause_after_losses']} Verlusten "
                   f"für {circuit_breaker['optimal_pause_bars']} Trades "
                   f"(DD: {circuit_breaker['baseline_dd']*100:.0f}% -> {circuit_breaker['optimized_dd']*100:.0f}%)", sym)

        # Ensemble-Gewichte (Top-3 nach Inner Val PnL, ohne den Besten)
        ensemble_weights = []
        candidates.sort(key=lambda x: x.get("inner_val_pnl", 0), reverse=True)
        top_candidates = [c for c in candidates[:3] if c != b]
        total_inner_pnl = sum(c.get("inner_val_pnl", 0) for c in top_candidates) + b.get("inner_val_pnl", 0)

        if total_inner_pnl > 0:
            for c in top_candidates:
                c_inner_pnl = c.get("inner_val_pnl", 0)
                if c_inner_pnl > 0:
                    ensemble_weights.append({
                        "tp_mult": c["params"][0],
                        "sl_mult": c["params"][1],
                        "conf_thresh": c["params"][2],
                        "weight": c_inner_pnl / total_inner_pnl,
                    })

        # CT-Werte extrahieren (kann Tuple sein bei separate_long_short)
        ct_value = b["params"][2]
        if isinstance(ct_value, tuple):
            ct_long, ct_short = ct_value
            ct_display = ct_long  # Für Kompatibilität mit altem Code
        else:
            ct_long = ct_short = ct_value
            ct_display = ct_value

        result = {
            "symbol": sym,
            "status": "ok",
            "pnl": b["pnl"],
            "config": {
                "kelly_risk": fk,
                "point_value": asset.point,
                "spread": ctx.spread,
                "tp_mult": b["params"][0],
                "sl_mult": b["params"][1],
                "conf_thresh": ct_display,
                # Separate CTs bei long_short_separate
                "ct_long": ct_long,
                "ct_short": ct_short,
                "separate_long_short": ctx.separate_long_short,
                "feature_group": b.get("feature_group", "unknown"),
                "features": b["feats"],
                "good_hours": b.get("good_hours", list(range(24))),
                "ensemble": ensemble_weights if ensemble_weights else None,
                "dd_scaling": {"10": 0.5, "20": 0.25},
                # Circuit Breaker Parameter (von KI optimiert)
                "circuit_breaker": {
                    "pause_after_losses": circuit_breaker["optimal_pause_after_losses"],
                    "pause_bars": circuit_breaker["optimal_pause_bars"],
                    "enabled": circuit_breaker["optimal_pause_after_losses"] > 0,
                },
                # Kelly-Anpassung Info
                "kelly_adjustment": {
                    "original_kelly": kelly_adjustment["adjusted_kelly"] / kelly_adjustment["scale_factor"] if kelly_adjustment["scale_factor"] > 0 else fk,
                    "scale_factor": kelly_adjustment["scale_factor"],
                    "target_dd": 0.30,
                },
                # Regime-Filter (optimaler aus Grid-Search)
                "regime_filter": b.get("regime_filter", {}),
            },
            "tr_trace": b["tr"],
            "trades_detailed": b.get("trades_detailed", []),  # Volle Trade-Details mit Zeiten, Preisen etc.
            "rrr": b["rrr"],
            "win_rate": wr,
            "sharpe": b["sharpe"],
            "calmar": b["calmar"],
            "currencies": asset.currencies,
            "grid_results": grid_results,  # Alle getesteten Kombinationen
            # Top-N Kandidaten für Export (mit RRR-Diversität)
            "top_candidates": top_candidates_for_export,
            # Fold-Stabilität (aus Inner CV)
            "fold_stability": b.get("fold_stability", 0),
            # Monte Carlo Statistiken
            "monte_carlo": {
                "p_value": mc_perm["p_value"],
                "is_significant": mc_perm["is_significant"],
                "percentile": mc_perm["percentile"],
                "equity_median": mc_equity["median_equity"],
                "equity_p5": mc_equity["p5_equity"],
                "equity_p95": mc_equity["p95_equity"],
                "bankruptcy_rate": mc_equity["bankruptcy_rate"],
            },
            # Equity-Smoothness
            "smoothness": b.get("smoothness", {}),
            # Nested CV Info
            "nested_cv": {
                "inner_samples": len(inner_df),
                "holdout_samples": len(holdout_df),
                "inner_val_pnl": b.get("inner_val_pnl", 0),
                "holdout_pnl": holdout_result["pnl"],
                "fold_stability": b.get("fold_stability", 0),
            },
            # Optimaler Regime-Filter (vom Grid-Search gefunden)
            "regime_filter": b.get("regime_filter", {}),
        }

        # Bei separater L/S Optimierung: Statistiken pro Richtung hinzufügen
        if ctx.separate_long_short:
            long_stats = holdout_result.get("long_stats", {})
            short_stats = holdout_result.get("short_stats", {})
            result["long_short_stats"] = {
                "long": {
                    "n_trades": long_stats.get("n_trades", 0),
                    "win_rate": long_stats.get("win_rate", 0),
                    "pnl": long_stats.get("pnl", 0),
                    "ct": ct_long,
                },
                "short": {
                    "n_trades": short_stats.get("n_trades", 0),
                    "win_rate": short_stats.get("win_rate", 0),
                    "pnl": short_stats.get("pnl", 0),
                    "ct": ct_short,
                },
            }

        smoothness_info = b.get("smoothness", {})
        smoothness_score = smoothness_info.get("smoothness_score", 0)
        log(1, f"OK (Holdout) - WR={wr:.1%} Sharpe={b['sharpe']:.2f} Smooth={smoothness_score:.2f} "
               f"p={mc_perm['p_value']:.3f} Trades={len(b['tr'])} ({time.time()-t_start:.1f}s)", sym)
        report_done(sym, "ok")
        return result

    except Exception as e:
        log(1, f"FEHLER: {e}", sym)
        import traceback
        traceback.print_exc()
        report_done(sym, "error")
        return {"symbol": sym, "status": "error", "error": str(e)}
