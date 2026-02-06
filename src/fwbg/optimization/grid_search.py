"""
Grid-Search Funktionen für Walk-Forward Optimierung.

Enthält:
- _process_single_grid_combo: Einzelne Grid-Kombination verarbeiten
- _process_tp_sl_combo_wrapper: Wrapper für parallele Verarbeitung
- _process_feature_group: Feature-Gruppe verarbeiten
- _process_feature_groups_parallel: Parallele Feature-Gruppen-Verarbeitung
"""
import time
import numpy as np
import multiprocessing as mp
import psutil
from concurrent.futures import ThreadPoolExecutor

from fwbg.builtins.indicators import filter_features_by_group
from fwbg.utils.progress import report_progress, report_phase, set_parallel_mode
from fwbg.utils.logging import log
from fwbg.utils.xgb_config import set_xgboost_n_jobs
from .nested_cv import run_inner_cv


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
    # WICHTIG: NUR cachen wenn KEIN Preprocessing aktiv ist!
    # Bei Preprocessing werden die DataFrames transformiert und Targets müssen
    # auf den transformierten Daten neu berechnet werden (in run_inner_cv).
    cached_targets = None
    has_preprocessing = ctx.preprocessing and len(ctx.preprocessing) > 0

    if inner_df is not None and not has_preprocessing:
        # Ohne Preprocessing können wir Targets vorab berechnen und cachen
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

    # Parallele Verarbeitung der TP/SL-Kombinationen innerhalb einer Feature-Gruppe
    # XGBoost parallelisiert intern (n_jobs), daher begrenzen wir die Anzahl
    # paralleler Combos um Überparallelisierung zu vermeiden
    n_combos = len(combos)
    max_combo_workers = min(4, n_combos)  # Max 4 parallele TP/SL-Combos

    if max_combo_workers > 1 and n_combos > 1:
        # Parallele Verarbeitung
        with ThreadPoolExecutor(max_workers=max_combo_workers) as combo_executor:
            futures = {combo_executor.submit(_process_tp_sl_combo_wrapper, combo): combo[3]
                       for combo in combos}

            for future in futures:
                try:
                    candidate, grid_result, idx = future.result()

                    if progress_callback:
                        progress_callback(idx + 1, grid_per_fg)

                    if candidate:
                        candidates.append(candidate)
                    if grid_result:
                        grid_results.append(grid_result)
                except Exception as e:
                    log(1, f"Fehler bei Grid-Combo: {e}", sym)
    else:
        # Sequentielle Verarbeitung (nur 1 Combo oder Worker)
        for combo in combos:
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
            # Parallel-Modus NICHT unterdrücken - wir wollen die Updates sehen!
            report_progress(sym, 0, 0, "grid_search", current, total_grid_combos)

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

    # Variablen für Logging und can_start_feature_group
    ram_per_thread_gb = estimated_ram_per_fg
    cpu_per_thread = 1.0  # Jede Feature-Group nutzt ~1 CPU-Kern

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
    usable_cores = int(total_cores * max_cpu_percent)
    cpu_based_limit = max(1, usable_cores)

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
    #
    # RAM-Check wurde entfernt: max_workers berücksichtigt bereits RAM-Kapazität.
    # Laufzeit-RAM-Checks waren kontraproduktiv und verlangsamten den Start.

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        fg_iter = iter(enumerate(feature_groups))
        active_count = 0

        # Starte sofort alle Feature-Gruppen bis zum max_workers Limit
        # RAM-Check entfernt: max_workers berücksichtigt bereits RAM-Kapazität
        for fg_idx, feature_group in fg_iter:
            if active_count >= max_workers:
                # Iterator zurücksetzen für restliche Feature-Gruppen
                fg_iter = iter([(i, fg) for i, fg in enumerate(feature_groups) if i > fg_idx])
                break
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

        log(2, f"Gestartet: {active_count} von {n_feature_groups} Feature-Gruppen (max: {max_workers})", sym)

        # Restliche Feature-Gruppen als Liste für einfacheren Zugriff
        remaining_fgs = [(i, fg) for i, fg in enumerate(feature_groups) if i >= active_count]
        fg_idx_offset = active_count

        while futures or remaining_fgs:
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

                    elapsed = time.time() - start_time
                    log(2, f"Feature-Gruppe '{feature_group}' fertig ({completed}/{n_feature_groups}) "
                           f"- {len(fg_candidates)} Kandidaten, Zeit: {elapsed:.1f}s", sym)
                except Exception as e:
                    log(1, f"Fehler bei Feature-Gruppe '{feature_group}': {e}", sym)

                # Sofort nächste Feature-Gruppe starten wenn noch welche warten
                if remaining_fgs and active_count < max_workers:
                    fg_idx, next_fg = remaining_fgs.pop(0)
                    future = executor.submit(
                        _process_feature_group,
                        fg_idx, next_fg, full_pool, inner_folds,
                        grid, ctx, regime_config, sym, n_feature_groups,
                        parallel_mode=True,
                        progress_callback=progress_callback,
                        inner_df=inner_df,
                    )
                    futures[future] = next_fg
                    active_count += 1
                    log(2, f"Gestartet: Feature-Gruppe '{next_fg}' ({active_count} aktiv)", sym)

            # Kurz warten wenn noch Tasks laufen
            if futures:
                time.sleep(0.1)

    total_elapsed = time.time() - start_time
    final_mem = psutil.virtual_memory()

    # Phase-Update: Feature-Gruppen fertig
    report_phase(sym, f"Kandidaten: {len(all_candidates)}")

    log(2, "=== Feature-Gruppen abgeschlossen ===", sym)
    log(2, f"  Verarbeitet: {completed}/{n_feature_groups} Gruppen in {total_elapsed:.1f}s", sym)
    log(2, f"  Kandidaten gefunden: {len(all_candidates)}", sym)
    log(2, f"  Finale RAM-Auslastung: {final_mem.percent:.1f}% ({final_mem.available/(1024**3):.1f}GB frei)", sym)
    return all_candidates, all_grid_results
