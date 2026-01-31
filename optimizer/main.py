"""
Hauptprogramm für den Walk-Forward Optimizer
"""

import os
import glob
import json
import warnings
import argparse
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tabulate import tabulate

from .config import (
    ACCOUNT_NAME,
    DATA_PATH,
    EXPORT_FILE,
    PLOT_PATH,
    TIMEFRAME,
    WALK_FORWARD_FOLDS,
    OOS_SIZE,
    CORR_THRESHOLD,
)
from .strategy_config import StrategyConfig
from .process import process_symbol
from .progress import ProgressTracker, init_progress_queue
from .results import (
    generate_run_id,
    create_run_directory,
    save_run_results,
    load_run,
)
from .resource_manager import AdaptivePoolManager, get_resource_info
from .equity import simulate_equity, filter_correlated_assets
from .plotting import create_incremental_plot, create_elite_plot
from .cli import (
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
    feature_groups=None,
):
    """
    Führt die Walk-Forward Optimierung aus.

    Args:
        description: Optionale Beschreibung für diesen Run
        save_results: Wenn True, werden Ergebnisse in test_results/ gespeichert
        strategy_metadata: Strukturierte Strategie-Metadaten (dict oder via create_strategy_metadata())
        asset_filter: Liste von Assets die getestet werden sollen (None = alle)
        feature_groups: Liste von Feature-Gruppen die getestet werden sollen (None = default)
    """
    # Lade nur Dateien für das gewählte Timeframe
    files = sorted(glob.glob(f"{DATA_PATH}/*_{TIMEFRAME}.csv"))

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
                from .config import ASSET_CONFIG

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

    # Strategy-Config für Worker erstellen
    if strategy_metadata:
        # CLI feature_groups überschreiben Strategy-Config
        if feature_groups:
            if "features" not in strategy_metadata:
                strategy_metadata["features"] = {}
            strategy_metadata["features"]["preferred_groups"] = feature_groups

        strategy = StrategyConfig.from_dict(strategy_metadata)
    elif feature_groups:
        strategy = StrategyConfig()
        strategy.features.preferred_groups = feature_groups
    else:
        strategy = StrategyConfig()

    # Ressourcen-Einstellungen aus Strategy (oder Defaults)
    resource_settings = {
        "max_cpu_percent": 0.80,
        "min_free_ram_percent": 0.20,
        "ram_per_worker_gb": 3.0,
    }
    if strategy_metadata:
        strat_resources = strategy_metadata.get("resources", {})
        if strat_resources:
            if strat_resources.get("max_cpu_percent") is not None:
                resource_settings["max_cpu_percent"] = strat_resources["max_cpu_percent"]
            if strat_resources.get("min_free_ram_percent") is not None:
                resource_settings["min_free_ram_percent"] = strat_resources["min_free_ram_percent"]
            if strat_resources.get("ram_per_worker_gb") is not None:
                resource_settings["ram_per_worker_gb"] = strat_resources["ram_per_worker_gb"]

    # Filter nach bestimmten Assets wenn angegeben
    if asset_filter:
        files = [f for f in files if any(a in f for a in asset_filter)]
    if not files:
        print(f"Keine Dateien für Timeframe {TIMEFRAME} gefunden!")
        print(f"Verfügbare Dateien: {glob.glob(f'{DATA_PATH}/*.csv')[:5]}...")
        return None

    # Run-ID und Verzeichnis erstellen
    run_id = generate_run_id(description)
    if save_results:
        run_path, plots_path = create_run_directory(run_id, description, strategy_metadata)
        print(f"\nRun ID: {run_id}")
        print(f"Results: {run_path}/")
        if strategy_metadata:
            print(f"Strategy: {strategy_metadata.get('name', '-')}")
    else:
        run_path = None
        plots_path = PLOT_PATH
        os.makedirs(plots_path, exist_ok=True)

    print(f"\nMASTER-OPTIMIZER 2.0 | Account: {ACCOUNT_NAME} | Timeframe: {TIMEFRAME}")
    if description:
        print(f"Description: {description}")
    print(f"Walk-Forward Folds: {WALK_FORWARD_FOLDS}, OOS Size: {OOS_SIZE}")
    print(f"Korrelations-Threshold: {CORR_THRESHOLD}")
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

    # Adaptive Pool mit Einstellungen aus Strategy oder Defaults
    pool_manager = AdaptivePoolManager(
        max_cpu_percent=resource_settings["max_cpu_percent"],
        min_free_ram_percent=resource_settings["min_free_ram_percent"],
        ram_per_worker_gb=resource_settings["ram_per_worker_gb"],
        verbose=True,
        progress_queue=progress_queue,
    )

    # Progress-Tracker im Hauptprozess mit Queue für Worker-Updates
    progress_tracker = ProgressTracker(len(files), asset_names, queue=progress_queue)

    def update_progress(completed, total):
        progress_tracker.update_completed(completed)

    # Callback für inkrementelle Ergebnis-Speicherung
    def on_result_ready(result):
        """Wird nach jedem fertigen Asset aufgerufen."""
        if not result or not save_results or not run_path:
            return

        sym = result.get("symbol", "?")
        status = result.get("status", "unknown")

        # Grid-Details sofort speichern
        grid_details_path = os.path.join(run_path, "grid_details")
        os.makedirs(grid_details_path, exist_ok=True)
        grid_file = os.path.join(grid_details_path, f"{sym}.json")

        grid_data = {
            "symbol": sym,
            "status": status,
            "total_combinations": len(result.get("grid_results", [])),
            "grid_results": result.get("grid_results", []),
        }
        if result.get("holdout_result"):
            grid_data["holdout_result"] = result["holdout_result"]
        if result.get("best_candidate"):
            grid_data["best_candidate"] = result["best_candidate"]

        with open(grid_file, "w") as f:
            json.dump(grid_data, f, indent=2)

        # Zusammenfassung ausgeben
        if status == "ok":
            config = result.get("config", {})
            tp = config.get("tp_mult", "?")
            sl = config.get("sl_mult", "?")
            ct = config.get("conf_thresh", "?")
            wr = result.get("win_rate", 0)
            rrr = result.get("rrr", 0)
            trades = len(result.get("tr_trace", []))
            pnl = result.get("pnl", 0)

            kelly_raw = (wr * rrr - (1 - wr)) / rrr if rrr > 0 else 0

            print(f"\n{'='*60}")
            print(f"✓ {sym} - PROFITABLE")
            print(f"{'='*60}")
            print(f"  Parameter: TP={tp}, SL={sl}, CT={ct:.2f}")
            print(f"  Performance: WR={wr:.1%}, RRR={rrr:.2f}, Trades={trades}")
            print(f"  PnL={pnl:.1f}, Kelly={kelly_raw:.4f}")

            # Plot erstellen
            if result.get("tr_trace"):
                try:
                    create_incremental_plot(result, plots_path)
                    print(f"  Plot: {plots_path}/{sym}.png")
                except Exception as e:
                    print(f"  Plot-Fehler: {e}")
        else:
            grid_count = len(result.get("grid_results", []))
            print(f"\n✗ {sym} - {status} ({grid_count} Kombinationen getestet)")
            if status == "no_kelly" and result.get("holdout_result"):
                hr = result["holdout_result"]
                bc = result.get("best_candidate", {})
                params = bc.get("params", {})
                print(f"  Best Candidate: TP={params.get('tp')}, SL={params.get('sl')}, CT={params.get('ct')}")
                print(f"  Holdout: WR={hr['win_rate']:.1%}, Trades={hr['n_trades']}, PnL={hr['pnl']:.1f}")
                print(f"  {hr['reason']}")

    print(f"\nStarte Verarbeitung von {len(files)} Assets...\n")
    progress_tracker.start()

    # Worker-Funktion mit Strategy-Config via partial wrappen
    worker_func = partial(process_symbol, strategy=strategy)

    raw_results = pool_manager.map_adaptive(
        func=worker_func,
        items=files,
        progress_callback=update_progress,
        result_callback=on_result_ready
    )

    progress_tracker.stop()

    # Stats ausgeben
    stats = pool_manager.get_status()
    print(f"\nPeak Workers: {stats['peak_workers']}, RAM-Throttles: {stats['ram_throttle_count']}")

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
    filtered = filter_correlated_assets(successful_results, CORR_THRESHOLD)
    print(f"{len(filtered)} Assets nach Korrelationsfilter.")

    # Top 10 auswählen
    elite = filtered[:10]

    # Portfolio-Shield
    shield = 1.0 / (len(elite) ** 0.5) if elite else 1.0
    final_assets = {}
    table_data = []

    for e in elite:
        e["config"]["kelly_risk"] *= shield

        # Equity-Simulation
        kelly = e["config"]["kelly_risk"]
        rrr = e["rrr"]
        sim = simulate_equity(e["tr_trace"], kelly, rrr)

        eq = sim["equity_curve"]
        final_equity = sim["final_equity"]
        max_dd = sim["max_drawdown"]
        max_dd_pct = max_dd * 100
        drawdowns = sim["drawdowns"]

        # Gewinn pro Trade berechnen
        profit_per_trade = []
        for i in range(1, len(eq)):
            profit_per_trade.append(eq[i] - eq[i - 1])

        # Trade-Richtungen extrahieren
        trades_detailed = e.get("trades_detailed", [])
        trade_directions = [td.get("direction", "LONG") for td in trades_detailed]

        # Elite-Plot erstellen
        n_long, n_short, long_wr, short_wr = create_elite_plot(
            e, plots_path, trade_directions, profit_per_trade,
            eq, drawdowns, max_dd, rrr
        )

        # Profitabilitätsprüfung
        sharpe = e.get("sharpe", 0)
        wr = e["win_rate"]

        # Jahresrendite berechnen
        bars_per_year = 24 * 250 if TIMEFRAME == "HOUR" else 96 * 250
        total_oos_bars = WALK_FORWARD_FOLDS * OOS_SIZE
        years = total_oos_bars / bars_per_year if bars_per_year > 0 else 1

        if final_equity > 0 and years > 0:
            annual_return = ((final_equity / 100.0) ** (1 / years) - 1) * 100
        else:
            annual_return = -100

        # Monte Carlo Statistiken
        mc_stats = e.get("monte_carlo", {})
        p_value = mc_stats.get("p_value", 1.0)
        fold_stability = e.get("fold_stability", 0)

        # Filter aus Strategy-Config oder Defaults
        strat_filters = strategy_metadata.get("filters", {}) if strategy_metadata else {}
        MIN_ANNUAL_RETURN = strat_filters.get("min_annual_return", 10.0)
        MIN_SHARPE = strat_filters.get("min_sharpe", 0.0)
        MAX_DRAWDOWN = strat_filters.get("max_drawdown", 1.0)
        is_profitable = (
            sharpe >= MIN_SHARPE
            and annual_return >= MIN_ANNUAL_RETURN
            and max_dd < MAX_DRAWDOWN
            and mc_stats.get("is_significant", False)
            and fold_stability >= 0.5
        )

        if is_profitable:
            export_config = {k: v for k, v in e["config"].items()}
            final_assets[e["symbol"]] = export_config
            status = "OK"
        else:
            status = "SKIP"

        # Long/Short Info für Tabelle
        long_short_str = f"{n_long}L/{n_short}S"

        table_data.append(
            [
                e["symbol"],
                f"{e['config']['kelly_risk'] * 100:.2f}%",
                f"{wr:.1%}",
                f"{rrr:.2f}",
                f"{sharpe:.2f}",
                f"{e.get('calmar', 0):.2f}",
                len(e["tr_trace"]),
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
                "Kelly",
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
        print("\nUm die Ergebnisse zu aktivieren:")
        print(f"  cp {run_path}/assets.json {EXPORT_FILE}")

    n_profitable = len(final_assets)

    return {
        "run_id": run_id,
        "run_path": run_path,
        "profitable_count": n_profitable,
        "final_assets": final_assets,
    }


def main():
    """CLI-Einstiegspunkt mit Argument-Parsing."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimizer für Trading-Strategien",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m optimizer                              # Standard-Run
  python -m optimizer -d "Test mit neuen Makros"   # Mit Beschreibung
  python -m optimizer --strategy                   # Interaktive Strategie-Eingabe
  python -m optimizer --strategy-file strat.json   # Strategie aus Datei laden
  python -m optimizer --list                       # Alle Runs anzeigen
  python -m optimizer --list --tags baseline       # Nach Tags filtern
  python -m optimizer --compare RUN1 RUN2          # Runs vergleichen
  python -m optimizer --no-save                    # Ohne Speichern
  python -m optimizer --assets EURUSD,GBPUSD       # Nur bestimmte Assets
  python -m optimizer --features trend,momentum    # Nur bestimmte Feature-Gruppen
  python -m optimizer --reverse-worst RUN_ID       # Schlechteste Strategien umkehren

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
    parser.add_argument("--features", type=str, help="Feature-Gruppen testen (komma-getrennt)")
    parser.add_argument("--list-features", action="store_true", help="Verfügbare Feature-Gruppen anzeigen")
    parser.add_argument("--reverse-worst", type=str, metavar="RUN_ID", help="Analysiere schlechteste Strategien umgekehrt")
    parser.add_argument("--reverse-n", type=int, default=10, help="Anzahl der schlechtesten Strategien (default: 10)")
    parser.add_argument("--cpu", type=float, help="Max CPU-Auslastung (0.0-1.0)")
    parser.add_argument("--ram-reserve", type=float, help="Min freier RAM-Anteil (0.0-1.0)")
    parser.add_argument("--ram-per-worker", type=float, help="RAM pro Worker in GB")

    args = parser.parse_args()

    if args.list_features:
        from .config import FEATURE_GROUPS, DEFAULT_FEATURE_GROUPS

        print("\n" + "=" * 70)
        print("VERFÜGBARE FEATURE-GRUPPEN")
        print("=" * 70 + "\n")
        for name, group in FEATURE_GROUPS.items():
            default_mark = " [DEFAULT]" if name in DEFAULT_FEATURE_GROUPS else ""
            print(f"{name}{default_mark}")
            print(f"  Name: {group['name']}")
            print(f"  Prefixes: {', '.join(group['prefixes'])}")
            print(f"  Beschreibung: {group['description']}")
            print()
        print("Verwendung: python -m optimizer --features trend,momentum,macro")
        return

    # --tags impliziert --list
    if args.tags or args.list:
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        show_runs(tags=tags)

    elif args.compare:
        show_comparison(args.compare)

    elif args.load:
        run_data = load_run(args.load)
        if run_data:
            print(json.dumps(run_data, indent=2, default=str))
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

        # CLI-Ressourcen-Einstellungen in Strategy überschreiben
        if strategy_metadata is None:
            strategy_metadata = {}
        if "resources" not in strategy_metadata:
            strategy_metadata["resources"] = {}
        if args.cpu is not None:
            strategy_metadata["resources"]["max_cpu_percent"] = args.cpu
        if args.ram_reserve is not None:
            strategy_metadata["resources"]["min_free_ram_percent"] = args.ram_reserve
        if args.ram_per_worker is not None:
            strategy_metadata["resources"]["ram_per_worker_gb"] = args.ram_per_worker

        # Parse asset filter
        asset_filter = None
        if args.assets:
            asset_filter = [a.strip().upper() for a in args.assets.split(",")]

        # Expand asset classes to individual assets
        if args.asset_classes:
            from .asset_config import AssetRegistry
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

        # Parse feature groups filter
        feature_groups = None
        if args.features:
            feature_groups = [g.strip() for g in args.features.split(",") if g.strip()]

        run_optimizer(
            description=args.description,
            save_results=not args.no_save,
            strategy_metadata=strategy_metadata if strategy_metadata else None,
            asset_filter=asset_filter,
            feature_groups=feature_groups,
        )


if __name__ == "__main__":
    main()
