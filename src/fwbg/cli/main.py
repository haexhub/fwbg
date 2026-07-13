"""
Hauptprogramm für den Walk-Forward Optimizer
"""

import os
import glob
import json
import math
import warnings
import argparse
from functools import partial

import matplotlib
matplotlib.use("Agg")
from tabulate import tabulate

class _SafeJsonEncoder(json.JSONEncoder):
    """JSON encoder that handles inf/nan as null."""

    def default(self, obj):
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return None
        return str(obj)

    def encode(self, obj):
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj):
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj


from fwbg.data import config as data_config  # noqa: E402
from fwbg.core.config import StrategyConfig  # noqa: E402
from fwbg.optimization.process import process_symbol  # noqa: E402
from fwbg.utils.progress import ProgressTracker, init_progress_queue, report_result  # noqa: E402
from fwbg.results.storage import (  # noqa: E402
    generate_run_id,
    create_run_directory,
    save_run_results,
    load_run,
)
from fwbg.optimization.resource_manager import SimplePoolManager, get_resource_info  # noqa: E402
from fwbg.simulation.equity import simulate_equity_from_pnl, filter_correlated_assets  # noqa: E402
from fwbg.results.plotting import create_asset_plot  # noqa: E402
from .commands import (  # noqa: E402
    show_runs,
    show_comparison,
    prompt_strategy_metadata,
    load_strategy_from_file,
    analyze_reversed_strategies,
)

warnings.filterwarnings("ignore")


def run_optimizer(
    description=None,
    save_results=True,
    strategy_metadata=None,
    asset_filter=None,
    run_id=None,
):
    """
    Führt die Walk-Forward Optimierung aus.

    Args:
        description: Optionale Beschreibung für diesen Run
        save_results: Wenn True, werden Ergebnisse in test_results/ gespeichert
        strategy_metadata: Strukturierte Strategie-Metadaten (dict oder via create_strategy_metadata())
        asset_filter: Liste von Assets die getestet werden sollen (None = alle)
    """
    # Strategy-Config für Worker erstellen
    if strategy_metadata:
        strategy = StrategyConfig.from_dict(strategy_metadata)
    else:
        strategy = StrategyConfig()

    # Datasource aus Strategy übernehmen (via Data-Source-Registry)
    if strategy.datasource:
        from fwbg.core.data_sources import get_data_source, CSVSourceConfig
        try:
            ds = get_data_source(strategy.datasource)
            if isinstance(ds, CSVSourceConfig) and ds.exists():
                data_config.DATA_PATH = str(ds.path)
        except ValueError:
            print(f"Warnung: Datasource '{strategy.datasource}' nicht gefunden")

    # CLI --data-path hat höchste Priorität
    cli_data_path = (strategy_metadata or {}).get("_cli_data_path")
    if cli_data_path:
        data_config.DATA_PATH = cli_data_path

    # Timeframe aus Strategy übernehmen (überschreibt Modul-Globals)
    if strategy.timeframe:
        tf = strategy.timeframe
        # CRITICAL: also set the env var, not just the module globals. Optimizer
        # workers run in a ProcessPoolExecutor; on Python 3.14 the default start
        # method is "forkserver" (not "fork"), so workers re-import data_config
        # fresh and would otherwise fall back to the HOUR defaults — inflating
        # test_period_years ~4x and halving the annualized Sharpe on M15. The
        # re-import reads TIMEFRAME from the environment, so propagate it there.
        os.environ["TIMEFRAME"] = tf
        data_config.TIMEFRAME = tf
        tf_cfg = data_config.TIMEFRAME_CONFIG.get(tf, data_config.TIMEFRAME_CONFIG["HOUR"])
        data_config.tf_cfg = tf_cfg
        data_config.OOS_SIZE = tf_cfg["oos_size"]
        data_config.WINDOW_SIZE = tf_cfg["window_size"]

    # Prüfe ob DATA_PATH gesetzt ist
    if not data_config.DATA_PATH:
        print("Fehler: Kein Datenpfad konfiguriert!")
        print("Bitte 'datasource' in der Strategy setzen oder --data-path übergeben.")
        return None

    # Lade nur Dateien für das gewählte Timeframe
    files = sorted(glob.glob(f"{data_config.DATA_PATH}/*_{data_config.TIMEFRAME}.csv"))

    # Fallback: resample from a lower timeframe if no files found
    if not files:
        from fwbg.data.resample import find_fallback_files
        files, source_tf = find_fallback_files(data_config.DATA_PATH, data_config.TIMEFRAME)
        if files:
            data_config.RESAMPLE_FROM = source_tf
            print(f"No {data_config.TIMEFRAME} files found — will resample from {source_tf}")

    # Strategy-Metadaten auswerten für Filter
    if strategy_metadata:
        # Assets aus Strategy
        strat_assets = strategy_metadata.get("assets", {})
        if strat_assets:
            # Filter: nur diese Assets
            if strat_assets.get("filter"):
                asset_filter = asset_filter or []
                asset_filter.extend(strat_assets["filter"])
            # Exclude: diese Assets ausschließen
            if strat_assets.get("exclude"):
                exclude_assets = strat_assets["exclude"]
                files = [f for f in files if not any(a in f for a in exclude_assets)]
            # Classes: nur bestimmte Asset-Klassen
            if strat_assets.get("classes"):
                from fwbg.data.assets import AssetRegistry
                ASSET_CONFIG = AssetRegistry.DEFAULT_ASSETS

                allowed_classes = strat_assets["classes"]
                files = [
                    f
                    for f in files
                    if any(
                        ASSET_CONFIG.get(os.path.basename(f).split("_")[0], {}).get(
                            "class"
                        )
                        in allowed_classes
                        for _ in [1]
                    )
                ]

    max_concurrent_assets = strategy.resources.max_concurrent_assets

    # Filter nach bestimmten Assets wenn angegeben
    if asset_filter:
        files = [f for f in files if any(a in f for a in asset_filter)]
    if not files:
        print(f"Keine Dateien für Timeframe {data_config.TIMEFRAME} gefunden!")
        print(f"Verfügbare Dateien: {glob.glob(f'{data_config.DATA_PATH}/*.csv')[:5]}...")
        return None

    # Run-ID und Verzeichnis erstellen
    if not run_id:
        run_id = generate_run_id(description)
    if save_results:
        run_path = create_run_directory(run_id, description, strategy_metadata)
        print(f"\nRun ID: {run_id}")
        print(f"Results: {run_path}/")
        if strategy_metadata:
            print(f"Strategy: {strategy_metadata.get('name', '-')}")
    else:
        run_path = None

    print(f"\nFWBG Strategy Backtester 2.0 | Timeframe: {data_config.TIMEFRAME}")
    if description:
        print(f"Description: {description}")
    print(f"Walk-Forward Folds: {data_config.WALK_FORWARD_FOLDS}, OOS Size: {data_config.OOS_SIZE}")
    print(f"Korrelations-Threshold: {data_config.CORR_THRESHOLD}")
    print("-" * 60)

    # Ressourcen-Info anzeigen
    res_info = get_resource_info()
    print(f"System: {res_info['cpu_cores']} Cores, {res_info['ram_total_gb']:.1f} GB RAM")
    print(f"Verfügbar: {res_info['ram_available_gb']:.1f} GB ({res_info['ram_free_percent']:.0f}% frei)")
    print("-" * 60)

    # Log-Level anzeigen
    log_level = int(os.environ.get("OPTIMIZER_LOG", "1"))
    print(f"Log-Level: {log_level} (OPTIMIZER_LOG=0..3 für mehr/weniger Details)")
    print("-" * 60)

    # Dateien auflisten
    print(f"\nVerarbeite {len(files)} Assets:")
    for i, f in enumerate(files[:10]):
        print(f"  {i + 1}. {os.path.basename(f)}")
    if len(files) > 10:
        print(f"  ... und {len(files) - 10} weitere")
    print()

    # Asset-Namen für Progress-Tracking extrahieren
    asset_names = [os.path.basename(f).split("_")[0] for f in files]

    # Progress-Queue für Worker-Kommunikation (deadlock-frei)
    progress_queue = init_progress_queue()

    # Simple pool with fixed worker count
    pool_manager = SimplePoolManager(
        max_concurrent_assets=max_concurrent_assets,
        progress_queue=progress_queue,
    )

    # Progress-Tracker im Hauptprozess mit Queue für Worker-Updates
    from pathlib import Path
    run_dir_path = Path(run_path) if run_path else None
    progress_tracker = ProgressTracker(
        len(files), asset_names, queue=progress_queue,
        run_directory=run_dir_path, run_id=run_id,
        strategy_name=strategy.name,
    )

    def update_progress(completed, total):
        progress_tracker.update_completed(completed)

    # Buffer für Ergebnis-Ausgaben (werden nach Progress-UI-Stop ausgegeben)
    result_output_buffer = []

    # Callback für inkrementelle Ergebnis-Speicherung
    def on_result_ready(result):
        """Wird nach jedem fertigen Asset aufgerufen."""
        if not result:
            return

        sym = result.get("symbol", "?")
        status = result.get("status", "unknown")

        # Asset als fertig markieren (für UI "Fertig" statt "Wartend")
        progress_tracker.update_completed(progress_tracker.completed_assets + 1, sym)

        # Sofortige Kurz-Zusammenfassung anzeigen
        cfg = result.get("config") or result.get("best_config", {})
        wr = result.get("win_rate", 0)
        pnl = result.get("pnl", 0)
        rrr_val = result.get("rrr", 0)
        tp = cfg.get("tp_mult", "?")
        sl = cfg.get("sl_mult", "?")
        n_trades = result.get("walk_forward", {}).get("total_trades", len(result.get("tr_trace", [])))

        wf = result.get("walk_forward", {})
        fold_tag = f" [Fold {wf.get('best_fold_id', '?')}]" if wf.get("config_inconsistent") else ""

        if status == "ok":
            sharpe = result.get("sharpe", 0)
            summary = f"WR={wr:.1%} PnL={pnl:.1f} Sharpe={sharpe:.2f} TP={tp} SL={sl} ({n_trades}T){fold_tag}"
        elif status == "no_edge":
            summary = f"No edge{fold_tag} WR={wr:.1%} PnL={pnl:.1f} RRR={rrr_val:.2f} TP={tp} SL={sl} ({n_trades}T)"
        elif status == "not_significant":
            p_val = result.get("monte_carlo", {}).get("p_value", 0)
            sharpe = result.get("sharpe", 0)
            summary = f"p={p_val:.3f} WR={wr:.1%} PnL={pnl:.1f} Sharpe={sharpe:.2f} TP={tp} SL={sl} ({n_trades}T){fold_tag}"
        elif status == "no_successful_folds":
            summary = "Keine profitablen Konfigurationen"
        else:
            summary = status
        report_result(sym, status, summary)

        if not save_results or not run_path:
            return

        # Grid-Details pro Symbol in Unterverzeichnis speichern
        sym_dir = os.path.join(run_path, "grid_details", sym)
        os.makedirs(sym_dir, exist_ok=True)

        # --- config.json: Konfiguration, Metriken, Status ---
        config_data = {
            "symbol": sym,
            "status": status,
            "total_combinations": len(result.get("grid_results", [])),
            "model_hyperparameters": cfg.get("model_hyperparameters"),
            "signal_meta": cfg.get("signal_meta"),
        }

        if result.get("error"):
            config_data["error"] = result["error"]

        if status == "ok":
            config_data["selected_config"] = result.get("config", {})
            config_data["nested_cv"] = result.get("nested_cv", {})
            config_data["monte_carlo"] = result.get("monte_carlo", {})
            config_data["smoothness"] = result.get("smoothness", {})

        if status in ["no_edge", "not_significant"]:
            if result.get("best_config"):
                config_data["best_config"] = result["best_config"]
            if result.get("pnl") is not None:
                config_data["metrics"] = {
                    "pnl": result.get("pnl", 0),
                    "win_rate": result.get("win_rate", 0),
                    "rrr": result.get("rrr", 0),
                    "sharpe": result.get("sharpe", 0),
                    "calmar": result.get("calmar", 0),
                }
            if result.get("monte_carlo"):
                config_data["monte_carlo"] = result["monte_carlo"]
            if result.get("reason"):
                config_data["reason"] = result["reason"]

        if result.get("holdout_result"):
            config_data["holdout_result"] = result["holdout_result"]
        if result.get("best_candidate"):
            config_data["best_candidate"] = result["best_candidate"]

        with open(os.path.join(sym_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2, cls=_SafeJsonEncoder)

        # --- fold_results.json: Walk-Forward Folds + Bias-Check ---
        fold_data = {}
        if result.get("walk_forward"):
            fold_data["walk_forward"] = result["walk_forward"]
        if result.get("bias_check"):
            fold_data["bias_check"] = result["bias_check"]
        if result.get("trade_analytics"):
            fold_data["trade_analytics"] = result["trade_analytics"]
        if fold_data:
            with open(os.path.join(sym_dir, "fold_results.json"), "w") as f:
                json.dump(fold_data, f, indent=2, cls=_SafeJsonEncoder)

        # --- trades.json: Unified-Simulation Trades ---
        if result.get("tr_trace"):
            trades_data = {
                "tr_trace": result["tr_trace"],
                "trades_detailed": result.get("trades_detailed", []),
            }
            with open(os.path.join(sym_dir, "trades.json"), "w") as f:
                json.dump(trades_data, f, indent=2, cls=_SafeJsonEncoder)

        # --- unified_metrics.json: Metriken für alle abgeschlossenen Runs ---
        # Quelle: tr_trace (Unified Simulation) oder Fold-Trades als Fallback
        _trades = result.get("tr_trace")
        if not _trades:
            # Fallback: PnL-Werte aus Fold test_trades_trace extrahieren
            wf = result.get("walk_forward", {})
            fold_pnls = []
            for fold in wf.get("fold_details", []):
                for entry in fold.get("test_trades_trace", []):
                    if isinstance(entry, dict):
                        fold_pnls.append(entry.get("pnl_raw", 0))
                    elif isinstance(entry, (int, float)):
                        fold_pnls.append(float(entry))
            if fold_pnls:
                _trades = fold_pnls

        if _trades:
            _risk = result.get("config", {}).get("risk_per_trade", 0.01)
            _years = result.get("test_period_years", 1)
            _eq_result = simulate_equity_from_pnl(_trades, fk=_risk)
            _final_eq = _eq_result["final_equity"]
            _annual_return = ((_final_eq / 100.0) ** (1 / _years) - 1) * 100 if _final_eq > 0 and _years > 0 else -100

            _wins = [p for p in _trades if p > 0]
            _losses = [abs(p) for p in _trades if p < 0]
            _gross_profit = sum(_wins)
            _gross_loss = sum(_losses)
            _profit_factor = round(_gross_profit / _gross_loss, 2) if _gross_loss > 0 else 0
            # FX pnl_raw is in price units (~1e-3); rounding to 2 dp collapsed
            # every avg to 0.0. Use 6 dp so the values survive.
            _avg_win = round(sum(_wins) / len(_wins), 6) if _wins else 0
            _avg_loss = round(sum(_losses) / len(_losses), 6) if _losses else 0

            # Direction counts: result["trades_detailed"] is never populated at
            # the top level, so the old read was always 0. The real per-direction
            # totals live in trade_analytics (built from the fold trades).
            _ta = result.get("trade_analytics") or {}
            _n_long = (_ta.get("long_stats") or {}).get("total", 0)
            _n_short = (_ta.get("short_stats") or {}).get("total", 0)

            unified_metrics = {
                "pnl": round(sum(_trades), 4),
                "win_rate": round(len(_wins) / len(_trades), 4) if _trades else 0,
                "rrr": result.get("rrr", 0),
                "sharpe": result.get("sharpe", 0),
                "calmar": result.get("calmar", 0),
                "trades": len(_trades),
                "annual_return": round(_annual_return, 1),
                "test_period_years": round(_years, 2),
                "max_drawdown": round(_eq_result["max_drawdown"], 4),
                "final_equity": round(_final_eq, 2),
                "risk_per_trade": _risk,
                "profit_factor": _profit_factor,
                "avg_win": _avg_win,
                "avg_loss": _avg_loss,
                "n_wins": len(_wins),
                "n_losses": len(_losses),
                "n_long": _n_long,
                "n_short": _n_short,
            }
            with open(os.path.join(sym_dir, "unified_metrics.json"), "w") as f:
                json.dump(unified_metrics, f, indent=2)

        # --- grid_results.json: Alle Grid-Kombinationen ---
        if result.get("grid_results"):
            grid_results_data = {
                "total_combinations": len(result["grid_results"]),
                "grid_results": result["grid_results"],
            }
            with open(os.path.join(sym_dir, "grid_results.json"), "w") as f:
                json.dump(grid_results_data, f, indent=2, cls=_SafeJsonEncoder)

        # --- Memory cleanup: strip heavy data now that it's on disk ---
        result.pop("grid_results", None)
        wf = result.get("walk_forward", {})
        for fold in wf.get("fold_details", []):
            # Replace full trade dicts with PnL-only values
            trace = fold.get("test_trades_trace", [])
            if trace and isinstance(trace[0], dict):
                fold["test_trades_trace"] = [t.get("pnl_raw", 0) for t in trace]
            fold.pop("test_trades_detail", None)

        # Zusammenfassung für später sammeln (wird nach Progress-UI ausgegeben)
        output_lines = []
        if status in ["ok", "no_edge", "not_significant"]:
            # Config extrahieren (entweder aus "config" oder "best_config")
            cfg = result.get("config") or result.get("best_config", {})
            tp = cfg.get("tp_mult", "?")
            sl = cfg.get("sl_mult", "?")
            ct = cfg.get("conf_thresh", "?")

            # Metriken
            wr = result.get("win_rate", 0)
            rrr = result.get("rrr", 0)
            pnl = result.get("pnl", 0)
            sharpe = result.get("sharpe", 0)
            calmar = result.get("calmar", 0)

            # Walk-Forward
            wf = result.get("walk_forward", {})
            std_wr = wf.get("std_win_rate", 0)
            std_pnl = wf.get("std_pnl", 0)
            n_folds = wf.get("n_folds", 0)
            mean_bias = wf.get("mean_bias_ratio", 0)
            bias_ratios = wf.get("bias_ratios", [])
            total_trades = wf.get("total_trades", len(result.get("tr_trace", [])))

            # Monte Carlo
            mc = result.get("monte_carlo", {})
            p_value = mc.get("p_value", 0)

            risk_per_trade = cfg.get("risk_per_trade", 0)

            # Header
            status_symbol = "✓" if status == "ok" else "✗"
            status_text = "PROFITABLE" if status == "ok" else status.upper()
            output_lines.append(f"\n{'='*60}")
            output_lines.append(f"{status_symbol} {sym} - {status_text}")
            output_lines.append(f"{'='*60}")

            # Details
            output_lines.append(f"  Best Config: TP={tp}, SL={sl}, CT={ct:.2f}")
            output_lines.append(f"  Walk-Forward: WR={wr:.1%}±{std_wr:.1%}, RRR={rrr:.2f}, PnL={pnl:.1f}±{std_pnl:.1f}")
            output_lines.append(f"  Performance: Sharpe={sharpe:.2f}, Calmar={calmar:.2f}, Trades={total_trades} ({n_folds} folds)")
            output_lines.append(f"  Bias: Mean={mean_bias:.2f}x, Ratios={[f'{r:.2f}' for r in bias_ratios]}")

            # Status-spezifische Zeilen
            if status == "ok":
                output_lines.append(f"  Risk/Trade={risk_per_trade:.4f}, p={p_value:.3f}")
            elif status == "no_edge":
                output_lines.append("  Reason: No profitable edge")
            elif status == "not_significant":
                output_lines.append(f"  Reason: p-value={p_value:.3f} (not significant)")

            if wf.get("config_inconsistent"):
                fold_id = wf.get("best_fold_id", "?")
                output_lines.append(f"  ⚠ Fold configs INCONSISTENT - using best fold {fold_id} only")

            # Plot im Asset-Verzeichnis speichern
            if result.get("tr_trace") and run_path:
                try:
                    create_asset_plot(result, sym_dir)
                    output_lines.append(f"  Plot: {sym_dir}/equity.png")
                except Exception as e:
                    output_lines.append(f"  Plot-Fehler: {e}")

        elif status == "no_successful_folds":
            grid_count = len(result.get("grid_results", []))
            output_lines.append(f"\n{'='*60}")
            output_lines.append(f"✗ {sym} - NO_SUCCESSFUL_FOLDS")
            output_lines.append(f"{'='*60}")
            output_lines.append(f"  {grid_count} Kombinationen getestet, kein Fold mit genug Test-Trades")

            best_grid = result.get("best_grid_result")
            if best_grid:
                output_lines.append(f"  Best Grid: TP={best_grid.get('tp_mult', '?')}, "
                                    f"SL={best_grid.get('sl_mult', '?')}, "
                                    f"CT={best_grid.get('conf_thresh', '?')}, "
                                    f"Inner PnL={best_grid.get('inner_val_pnl', 0):.1f}, "
                                    f"Stability={best_grid.get('fold_stability', 0):.0%}")

        else:
            # Andere Status (insufficient_data, etc.)
            grid_count = len(result.get("grid_results", []))
            output_lines.append(f"\n✗ {sym} - {status} ({grid_count} Kombinationen getestet)")

        # In Buffer speichern für Ausgabe nach Progress-UI-Stop
        result_output_buffer.append("\n".join(output_lines))

    print(f"\nStarte Verarbeitung von {len(files)} Assets...\n")
    progress_tracker.start()

    # Worker-Funktion mit Strategy-Config via partial wrappen
    worker_func = partial(process_symbol, strategy=strategy)

    try:
        raw_results = pool_manager.map_adaptive(
            func=worker_func,
            items=files,
            progress_callback=update_progress,
            result_callback=on_result_ready
        )
    finally:
        progress_tracker.stop()

    # Gepufferte Ergebnis-Ausgaben jetzt anzeigen
    for output in result_output_buffer:
        if output:
            print(output)

    # Stats ausgeben
    stats = pool_manager.get_status()
    print(f"\nPeak Workers: {stats['peak_workers']}")

    # Trenne erfolgreiche von fehlgeschlagenen Ergebnissen
    all_results = raw_results
    successful_results = [r for r in raw_results if r and r.get("status") == "ok"]
    failed_results = [r for r in raw_results if r and r.get("status") != "ok"]
    none_results = sum(1 for r in raw_results if r is None)

    print(f"{len(successful_results)} Assets haben die Optimierung bestanden.")

    # Zeige übersprungene Assets immer an
    if failed_results or none_results:
        print(f"\nÜbersprungene Assets ({len(failed_results) + none_results}):")
        for fr in failed_results:
            status = fr.get("status", "unknown")
            grid_count = len(fr.get("grid_results", []))
            print(
                f"  - {fr['symbol']}: {status}"
                + (f" ({grid_count} Kombinationen getestet)" if grid_count else "")
            )
        if none_results:
            print(f"  - {none_results}x Fehler (None zurückgegeben)")

    # Korrelationsfilter anwenden (nur auf erfolgreiche)
    filtered = filter_correlated_assets(successful_results, data_config.CORR_THRESHOLD)
    print(f"{len(filtered)} Assets nach Korrelationsfilter.")

    # Top 10 auswählen
    elite = filtered[:10]

    final_assets = {}
    table_data = []

    for e in elite:
        sym = e["symbol"]

        # Metriken aus unified_metrics.json lesen
        sym_dir = os.path.join(run_path, "grid_details", sym) if run_path else None
        um = {}
        if sym_dir:
            um_path = os.path.join(sym_dir, "unified_metrics.json")
            if os.path.isfile(um_path):
                with open(um_path) as f:
                    um = json.load(f)

        wr = um.get("win_rate", 0)
        rrr = um.get("rrr", 0)
        sharpe = um.get("sharpe", 0)
        max_dd = um.get("max_drawdown", 0)
        max_dd_pct = max_dd * 100
        risk = um.get("risk_per_trade", e["config"].get("risk_per_trade", 0.01))

        # Jahresrendite aus unified_metrics berechnen
        final_equity = um.get("final_equity", 100.0)
        years = um.get("test_period_years", 1)
        if final_equity > 0 and years > 0:
            annual_return = ((final_equity / 100.0) ** (1 / years) - 1) * 100
        else:
            annual_return = -100

        # Trade-Richtungen extrahieren
        trades_detailed = e.get("trades_detailed", [])
        trade_directions = [td.get("direction", "LONG") for td in trades_detailed]

        # Elite-Plot im Asset-Verzeichnis speichern
        if sym_dir and os.path.isdir(sym_dir):
            plot_stats = create_asset_plot(e, sym_dir, trade_directions=trade_directions, unified_metrics=um)
            if plot_stats:
                n_long, n_short, _, _ = plot_stats
            else:
                n_long = n_short = 0
        else:
            n_long = sum(1 for d in trade_directions if d == "LONG")
            n_short = sum(1 for d in trade_directions if d == "SHORT")

        # Monte Carlo Statistiken
        mc_stats = e.get("monte_carlo", {})
        p_value = mc_stats.get("p_value", 1.0)
        fold_stability = e.get("fold_stability", 0)

        # Filter aus Strategy-Config (bereits durch StrategyConfig.from_dict aufgelöst)
        f = strategy.filters
        MIN_ANNUAL_RETURN = f.min_annual_return
        MIN_SHARPE = f.min_sharpe
        MAX_DRAWDOWN = f.max_drawdown
        MIN_FOLD_STABILITY = f.min_fold_stability
        is_profitable = (
            sharpe >= MIN_SHARPE
            and annual_return >= MIN_ANNUAL_RETURN
            and max_dd < MAX_DRAWDOWN
            and mc_stats.get("is_significant", False)
            and fold_stability >= MIN_FOLD_STABILITY
        )

        if is_profitable:
            export_config = {k: v for k, v in e["config"].items()}
            final_assets[sym] = export_config
            status = "OK"
        else:
            status = "SKIP"

        # Long/Short Info für Tabelle
        long_short_str = f"{n_long}L/{n_short}S"

        table_data.append(
            [
                sym,
                f"{risk * 100:.2f}%",
                f"{wr:.1%}",
                f"{rrr:.2f}",
                f"{sharpe:.2f}",
                f"{um.get('calmar', 0):.2f}",
                um.get("trades", len(e["tr_trace"])),
                long_short_str,
                f"{annual_return:+.0f}%/y",
                f"{max_dd_pct:.0f}%",
                f"{p_value:.3f}",
                f"{fold_stability:.0%}",
                status,
            ]
        )

    print(
        "\n"
        + tabulate(
            table_data,
            headers=[
                "Asset",
                "Risk",
                "WinRate",
                "RRR",
                "Sharpe",
                "Calmar",
                "Trades",
                "L/S",
                "Return",
                "MaxDD",
                "p-val",
                "Folds",
                "Status",
            ],
            tablefmt="psql",
        )
    )

    # Währungs-Exposure anzeigen
    all_currencies = {}
    for e in elite:
        sharpe = e.get("sharpe", 0)
        wr = e["win_rate"]
        rrr = e["rrr"]
        expectancy = wr * rrr - (1 - wr)
        if sharpe >= 1.0 and expectancy > 0.05:
            for c in e.get("currencies", []):
                all_currencies[c] = all_currencies.get(c, 0) + 1
    if all_currencies:
        print(
            f"\nWährungs-Exposure (nur profitable): {dict(sorted(all_currencies.items(), key=lambda x: -x[1]))}"
        )

    # Ergebnisse speichern
    if save_results and run_path:
        save_run_results(
            run_path=run_path,
            raw_results=successful_results,
            filtered_results=filtered,
            elite_results=elite,
            final_assets=final_assets,
            table_data=table_data,
            description=description,
            strategy_metadata=strategy_metadata,
            all_results=all_results,
        )
        print(f"\nErgebnisse gespeichert in: {run_path}/")
        print(f"Assets-Config:           {run_path}/assets.json")

    n_profitable = len(final_assets)

    return {
        "run_id": run_id,
        "run_path": run_path,
        "profitable_count": n_profitable,
        "final_assets": final_assets,
    }


def _run_data_command(argv):
    """Führt Daten-ETL-Kommandos aus (fwbg data <subcommand>)."""
    data_parser = argparse.ArgumentParser(
        prog="fwbg data",
        description="Datenquellen-Verwaltung",
    )
    subparsers = data_parser.add_subparsers(dest="subcmd")

    prep_parser = subparsers.add_parser("prepare", help="Rohdaten in Standard-Format konvertieren")
    prep_parser.add_argument("--source", required=True, help="Name der Datenquelle (z.B. dukascopy)")

    _list_parser = subparsers.add_parser("list", help="Registrierte Datenquellen anzeigen")

    args = data_parser.parse_args(argv)

    from fwbg.core.data_sources import get_data_source, discover_sources, _DATA_SOURCES, CSVSourceConfig

    if not _DATA_SOURCES:
        discover_sources()

    if args.subcmd == "prepare":
        try:
            source = get_data_source(args.source)
        except ValueError as e:
            print(f"Fehler: {e}")
            return

        if not isinstance(source, CSVSourceConfig):
            print(f"Fehler: '{args.source}' ist keine CSV-Datenquelle")
            return

        if source.raw_path is None:
            print(f"Fehler: Datenquelle '{args.source}' hat keinen raw_path konfiguriert")
            return

        print(f"Konvertiere {args.source}: {source.raw_path} → {source.path}")
        converted = source.prepare()
        print(f"Fertig: {len(converted)} Symbole konvertiert: {converted}")

    elif args.subcmd == "list":
        if not _DATA_SOURCES:
            print("Keine Datenquellen registriert.")
            return
        print(f"Registrierte Datenquellen ({len(_DATA_SOURCES)}):")
        for name, src in _DATA_SOURCES.items():
            print(f"  {name}: {src.source_type.value} → {src.path}")

    else:
        data_parser.print_help()


def main():
    """CLI-Einstiegspunkt mit Argument-Parsing."""
    import sys
    import faulthandler
    faulthandler.enable()  # Dump Python traceback on SIGSEGV/SIGABRT (C-level crashes)

    # Handle 'data' subcommand (ETL: raw → datasource)
    if len(sys.argv) > 1 and sys.argv[1] == "data":
        _run_data_command(sys.argv[2:])
        return

    # Handle 'api' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        api_parser = argparse.ArgumentParser(description="FWBG API Server")
        api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
        api_parser.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")
        api_args = api_parser.parse_args(sys.argv[2:])
        from fwbg.api import run_server
        run_server(host=api_args.host, port=api_args.port)
        return

    # Handle 'analyze' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        from fwbg.cli._analyze import run_analyze
        run_analyze(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="FWBG Strategy Backtester - Walk-Forward Validation für Trading-Strategien",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m fwbg.cli                                    # Standard-Run
  python -m fwbg.cli -d "Test mit neuen Makros"        # Mit Beschreibung
  python -m fwbg.cli --strategy                        # Interaktive Strategie-Eingabe
  python -m fwbg.cli --strategy-file strategies/configs/x.json # Strategie aus Datei laden
  python -m fwbg.cli --list                            # Alle Runs anzeigen
  python -m fwbg.cli --list --tags baseline            # Nach Tags filtern
  python -m fwbg.cli --compare RUN1 RUN2               # Runs vergleichen
  python -m fwbg.cli --no-save                         # Ohne Speichern
  python -m fwbg.cli --assets EURUSD,GBPUSD            # Nur bestimmte Assets
  python -m fwbg.cli --reverse-worst RUN_ID            # Schlechteste Strategien umkehren

Kategorien: baseline, feature_test, model_test, hyperparameter, production, experiment
        """,
    )

    parser.add_argument("-d", "--description", type=str, help="Beschreibung für diesen Run")
    parser.add_argument("--strategy", action="store_true", help="Interaktive Strategie-Metadaten-Eingabe")
    parser.add_argument("--strategy-file", type=str, metavar="FILE", help="Strategie-Metadaten aus JSON-Datei laden")
    parser.add_argument("--list", action="store_true", help="Alle vorhandenen Runs anzeigen")
    parser.add_argument("--tags", type=str, help="Runs nach Tags filtern und auflisten (komma-getrennt)")
    parser.add_argument("--compare", nargs="+", metavar="RUN_ID", help="Runs vergleichen")
    parser.add_argument("--no-save", action="store_true", help="Ergebnisse nicht in test_results speichern")
    parser.add_argument("--load", type=str, metavar="RUN_ID", help="Details eines Runs anzeigen")
    parser.add_argument("--assets", type=str, help="Nur bestimmte Assets testen (komma-getrennt)")
    parser.add_argument("--asset-classes", type=str, help="Nur bestimmte Asset-Klassen testen (komma-getrennt)")
    parser.add_argument("--reverse-worst", type=str, metavar="RUN_ID", help="Analysiere schlechteste Strategien umgekehrt")
    parser.add_argument("--reverse-n", type=int, default=10, help="Anzahl der schlechtesten Strategien (default: 10)")
    parser.add_argument("--timeframe", type=str, help="Timeframe (überschreibt TIMEFRAME env)")
    parser.add_argument("--data-path", type=str, metavar="DIR",
                        help="Datenpfad (überschreibt DATA_PATH, z.B. data/dukascopy/datasource)")
    parser.add_argument("--run-id", type=str, metavar="ID",
                        help="Feste Run-ID verwenden (statt automatisch generierter ID)")
    parser.add_argument("--start-date", type=str, metavar="ISO",
                        help="Backtest-Fenster Start (ISO, überschreibt Strategy-start_date)")
    parser.add_argument("--end-date", type=str, metavar="ISO",
                        help="Backtest-Fenster Ende (ISO, überschreibt Strategy-end_date)")
    parser.add_argument("--cost-multiplier", type=float, metavar="X",
                        help="Spread/Slippage-Multiplikator (z.B. 2.0 für Kosten-Stresstest)")

    args = parser.parse_args()

    # --tags impliziert --list
    if args.tags or args.list:
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        show_runs(tags=tags)

    elif args.compare:
        show_comparison(args.compare)

    elif args.load:
        run_data = load_run(args.load)
        if run_data:
            print(json.dumps(run_data, indent=2, cls=_SafeJsonEncoder))
        else:
            print(f"Run {args.load} nicht gefunden.")

    elif args.reverse_worst:
        analyze_reversed_strategies(args.reverse_worst, top_n=args.reverse_n)

    else:
        # Strategie-Metadaten
        strategy_metadata = None
        if args.strategy_file:
            strategy_metadata = load_strategy_from_file(args.strategy_file)
        elif args.strategy:
            strategy_metadata = prompt_strategy_metadata()

        # CLI --timeframe überschreibt Strategy-Timeframe
        if strategy_metadata is None:
            strategy_metadata = {}
        if args.timeframe:
            strategy_metadata["timeframe"] = args.timeframe

        # CLI backtest-window + cost-stress overrides (Plan 009 WP4).
        if args.start_date:
            strategy_metadata["start_date"] = args.start_date
        if args.end_date:
            strategy_metadata["end_date"] = args.end_date
        if args.cost_multiplier is not None:
            strategy_metadata["cost_multiplier"] = args.cost_multiplier

        # CLI --data-path hat höchste Priorität → wird in strategy_metadata injiziert
        # damit run_optimizer es nach der Strategy-Datasource nochmal überschreibt
        if args.data_path:
            if strategy_metadata is None:
                strategy_metadata = {}
            strategy_metadata["_cli_data_path"] = args.data_path

        # Parse asset filter
        asset_filter = None
        if args.assets:
            asset_filter = [a.strip().upper() for a in args.assets.split(",")]

        # Expand asset classes to individual assets
        if args.asset_classes:
            from fwbg.data.assets import AssetRegistry
            registry = AssetRegistry()
            classes = [c.strip().upper() for c in args.asset_classes.split(",")]
            class_assets = []
            for cls in classes:
                class_assets.extend(registry.symbols_by_class(cls))
            if asset_filter:
                asset_filter = list(set(asset_filter + class_assets))
            else:
                asset_filter = class_assets
            if not asset_filter:
                print(f"Keine Assets für Klassen {classes} gefunden!")
                return

        result = run_optimizer(
            description=args.description,
            save_results=not args.no_save,
            strategy_metadata=strategy_metadata if strategy_metadata else None,
            asset_filter=asset_filter,
            run_id=args.run_id,
        )
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
