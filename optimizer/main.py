"""
Hauptprogramm für den Walk-Forward Optimizer
"""
import os
import glob
import json
import random
import warnings
import argparse
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

from .config import (
    ACCOUNT_NAME, DATA_PATH, BASE_PATH, EXPORT_FILE, PLOT_PATH,
    TIMEFRAME, WALK_FORWARD_FOLDS, OOS_SIZE, CORR_THRESHOLD,
    convert_numpy
)
from .process import process_symbol
from .results import (
    generate_run_id, create_run_directory, save_run_results,
    list_runs, load_run, compare_runs, create_strategy_metadata
)
from .resource_manager import AdaptivePoolManager, get_resource_info

warnings.filterwarnings("ignore")


def simulate_equity(trades, kelly_risk, rrr, start_equity=100.0, compound_cap=1e6):
    """
    Simuliert die Equity-Kurve basierend auf Trade-Ergebnissen.

    Args:
        trades: Liste von Trade-Ergebnissen (>0 = Gewinn, <=0 = Verlust)
        kelly_risk: Risk pro Trade als Anteil des Kapitals (z.B. 0.02 = 2%)
        rrr: Risk-Reward-Ratio (z.B. 2.0 = TP ist 2x SL)
        start_equity: Startkapital (default: 100.0)
        compound_cap: Ab diesem Equity-Wert wird nicht mehr kompoundiert,
                     sondern mit fixer Positionsgröße weitergehandelt (default: 1e6)

    Returns:
        dict mit:
            - equity_curve: Liste der Equity-Werte
            - final_equity: Endkapital
            - max_drawdown: Maximaler Drawdown (0.0-1.0)
            - drawdowns: Liste der Drawdown-Werte in Prozent
    """
    equity = start_equity
    equity_curve = [equity]
    peak = equity
    max_dd = 0
    drawdowns = [0.0]

    for trade_result in trades:
        # Effektive Equity für Positionsberechnung (gecappt)
        effective_equity = min(equity, compound_cap)

        if trade_result > 0:
            # Gewinn: Kelly * RRR (basierend auf effektiver Equity)
            equity += effective_equity * kelly_risk * rrr
        else:
            # Verlust: Kelly (basierend auf effektiver Equity)
            equity -= effective_equity * kelly_risk

        equity_curve.append(equity)

        # Drawdown berechnen
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        drawdowns.append(dd * 100)

        # Bankrott-Check
        if equity <= 0:
            equity = 0
            max_dd = 1.0
            break

    return {
        "equity_curve": equity_curve,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "drawdowns": drawdowns,
    }


def filter_correlated_assets(results, threshold=CORR_THRESHOLD):
    """
    Filtert Assets mit zu hoher Währungskorrelation.
    Verhindert z.B. 5x USD-Long gleichzeitig.
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda x: x["pnl"], reverse=True)
    selected = []
    currency_exposure = {}

    for r in sorted_results:
        currencies = r.get("currencies", [])
        if not currencies:
            selected.append(r)
            continue

        max_exposure = max(
            (currency_exposure.get(c, 0) for c in currencies),
            default=0
        )

        max_allowed = int(1 / (1 - threshold)) if threshold < 1 else 10

        if max_exposure < max_allowed:
            selected.append(r)
            for c in currencies:
                currency_exposure[c] = currency_exposure.get(c, 0) + 1

    return selected


def run_optimizer(description=None, save_results=True, strategy_metadata=None, asset_filter=None, feature_groups=None):
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
                files = [f for f in files if any(
                    ASSET_CONFIG.get(os.path.basename(f).split("_")[0], {}).get("class") in allowed_classes
                    for _ in [1]  # Workaround für list comprehension
                )]

        # Feature-Gruppen aus Strategy
        strat_features = strategy_metadata.get("feature_groups", {})
        if strat_features and strat_features.get("groups"):
            feature_groups = feature_groups or []
            feature_groups.extend(strat_features["groups"])

    # Feature-Gruppen setzen (für Worker-Prozesse via Environment)
    if feature_groups:
        os.environ["OPTIMIZER_FEATURE_GROUPS"] = ",".join(feature_groups)

    # Ressourcen-Einstellungen aus Strategy (oder Defaults)
    resource_settings = {
        "max_cpu_percent": 0.80,
        "min_free_ram_percent": 0.25,
        "ram_per_worker_gb": 4.0,
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
        print(f"  {i+1}. {os.path.basename(f)}")
    if len(files) > 10:
        print(f"  ... und {len(files) - 10} weitere")
    print()

    # Asset-Namen für Progress-Tracking extrahieren
    asset_names = [os.path.basename(f).split("_")[0] for f in files]

    # Adaptive Pool mit Einstellungen aus Strategy oder Defaults
    # HINWEIS: Progress-Dict deaktiviert wegen Deadlock mit ProcessPoolExecutor
    pool_manager = AdaptivePoolManager(
        max_cpu_percent=resource_settings["max_cpu_percent"],
        min_free_ram_percent=resource_settings["min_free_ram_percent"],
        ram_per_worker_gb=resource_settings["ram_per_worker_gb"],
        verbose=True,
        progress_dict=None  # Kein shared dict - verhindert Deadlock
    )

    # Einfacher Progress-Callback
    def update_progress(completed, total):
        pct = completed / total * 100 if total > 0 else 0
        print(f"\rFortschritt: {completed}/{total} ({pct:.0f}%)   ", end="", flush=True)

    print(f"\nStarte Verarbeitung von {len(files)} Assets...\n")

    raw_results = pool_manager.map_adaptive(
        func=process_symbol,
        items=files,
        progress_callback=update_progress
    )

    print()  # Newline nach Progress

    # Stats ausgeben
    stats = pool_manager.get_status()
    print(f"\nPeak Workers: {stats['peak_workers']}, RAM-Throttles: {stats['ram_throttle_count']}")

    # Trenne erfolgreiche von fehlgeschlagenen Ergebnissen
    all_results = raw_results  # Alle Ergebnisse (inkl. grid_results)
    successful_results = [r for r in raw_results if r and r.get("status") == "ok"]
    failed_results = [r for r in raw_results if r and r.get("status") != "ok"]
    none_results = sum(1 for r in raw_results if r is None)

    print(f"{len(successful_results)} Assets haben die Optimierung bestanden.")

    # Zeige übersprungene Assets immer an
    if failed_results or none_results:
        print(f"\nÜbersprungene Assets ({len(failed_results) + none_results}):")
        for fr in failed_results:
            status = fr.get('status', 'unknown')
            grid_count = len(fr.get('grid_results', []))
            print(f"  - {fr['symbol']}: {status}" + (f" ({grid_count} Kombinationen getestet)" if grid_count else ""))
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

        # Equity-Simulation via Hilfsfunktion
        kelly = e["config"]["kelly_risk"]
        rrr = e["rrr"]
        sim = simulate_equity(e["tr_trace"], kelly, rrr)

        eq = sim["equity_curve"]
        final_equity = sim["final_equity"]
        max_dd = sim["max_drawdown"]
        max_dd_pct = max_dd * 100
        drawdowns = sim["drawdowns"]

        # Gewinn pro Trade berechnen (aus Equity-Kurve)
        profit_per_trade = []
        for i in range(1, len(eq)):
            profit_per_trade.append(eq[i] - eq[i - 1])

        # Equity Plot mit Drawdown und Profit per Trade (logarithmisch)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), height_ratios=[3, 1, 1])

        ax1.plot(eq, color="blue", linewidth=1.5)
        ax1.fill_between(range(len(eq)), eq, alpha=0.3)
        ax1.set_yscale("log")  # Logarithmische Y-Achse
        ax1.set_title(
            f"{e['symbol']} | WR: {e['win_rate']:.1%} | RRR: {rrr:.2f} | "
            f"Sharpe: {e.get('sharpe', 0):.2f} | MaxDD: {max_dd_pct:.0f}%"
        )
        ax1.set_ylabel("Kapital (log, Start=100)")
        ax1.set_xlabel("")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(max(drawdowns) * 1.1 if drawdowns else 1, 0)
        ax2.set_xlabel("")
        ax2.grid(True, alpha=0.3)

        # Profit per Trade als Bar-Chart (grün = Gewinn, rot = Verlust)
        # Symmetrisch-logarithmische Skala für bessere Lesbarkeit
        colors = ["green" if p > 0 else "red" for p in profit_per_trade]
        ax3.bar(range(len(profit_per_trade)), profit_per_trade, color=colors, alpha=0.7, width=1.0)
        ax3.axhline(y=0, color="black", linewidth=0.5)
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("Gewinn/Trade (symlog)")
        # symlog: logarithmisch für große Werte, linear nahe 0
        ax3.set_yscale("symlog", linthresh=0.1)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{plots_path}/{e['symbol']}.png", dpi=100)
        plt.close()

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

        # Erweiterte Profitabilitätsprüfung inkl. Monte Carlo
        MIN_ANNUAL_RETURN = 10  # Mindestens 10%/Jahr
        is_profitable = (
            sharpe >= 1.0 and
            annual_return >= MIN_ANNUAL_RETURN and
            max_dd < 0.6 and
            mc_stats.get("is_significant", False) and
            fold_stability >= 0.5  # Mindestens 50% der Folds profitabel
        )

        if is_profitable:
            export_config = {k: v for k, v in e["config"].items()}
            final_assets[e["symbol"]] = export_config
            status = "OK"
        else:
            status = "SKIP"

        table_data.append([
            e["symbol"],
            f"{e['config']['kelly_risk'] * 100:.2f}%",
            f"{wr:.1%}",
            f"{rrr:.2f}",
            f"{sharpe:.2f}",
            f"{e.get('calmar', 0):.2f}",
            len(e["tr_trace"]),
            f"{annual_return:+.0f}%/y",
            f"{max_dd_pct:.0f}%",
            f"{p_value:.3f}",
            f"{fold_stability:.0%}",
            status,
        ])

    print(
        "\n"
        + tabulate(
            table_data,
            headers=["Asset", "Kelly", "WinRate", "RRR", "Sharpe", "Calmar", "Trades", "Return", "MaxDD", "p-val", "Folds", "Status"],
            tablefmt="psql"
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
        print(f"\nWährungs-Exposure (nur profitable): {dict(sorted(all_currencies.items(), key=lambda x: -x[1]))}")

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
            all_results=all_results,  # Alle Ergebnisse inkl. grid_results
        )
        print(f"\nErgebnisse gespeichert in: {run_path}/")
        print(f"Assets-Config:           {run_path}/assets.json")
        print(f"\nUm die Ergebnisse zu aktivieren:")
        print(f"  cp {run_path}/assets.json {EXPORT_FILE}")

    n_profitable = len(final_assets)

    return {
        "run_id": run_id,
        "run_path": run_path,
        "profitable_count": n_profitable,
        "final_assets": final_assets,
    }


def show_runs():
    """Zeigt alle vorhandenen Runs an."""
    runs = list_runs()

    if not runs:
        print("Keine Test-Runs gefunden.")
        return

    print(f"\n{'='*80}")
    print("VORHANDENE TEST-RUNS")
    print(f"{'='*80}\n")

    table_data = []
    for r in runs:
        table_data.append([
            r["run_id"],
            r.get("timeframe", "?"),
            r.get("profitable_count", "?"),
            r.get("description", "-")[:40] if r.get("description") else "-",
        ])

    print(tabulate(
        table_data,
        headers=["Run ID", "Timeframe", "Profitable", "Description"],
        tablefmt="psql"
    ))


def show_comparison(run_ids):
    """Vergleicht mehrere Runs."""
    comparison = compare_runs(run_ids)

    if not comparison:
        print("Keine Runs zum Vergleichen gefunden.")
        return

    print(f"\n{'='*80}")
    print("RUN-VERGLEICH")
    print(f"{'='*80}\n")

    table_data = []
    for r in comparison["runs"]:
        stats = r.get("elite_stats", {})
        table_data.append([
            r["run_id"],
            r.get("timeframe", "?"),
            r.get("profitable_count", 0),
            f"{stats.get('avg_sharpe', 0):.2f}",
            f"{stats.get('avg_win_rate', 0):.1%}",
            stats.get("total_trades", 0),
            r.get("description", "-")[:30] if r.get("description") else "-",
        ])

    print(tabulate(
        table_data,
        headers=["Run ID", "TF", "Profitable", "Avg Sharpe", "Avg WR", "Trades", "Description"],
        tablefmt="psql"
    ))

    print(f"\nAlle profitablen Symbole: {', '.join(comparison['all_symbols'])}")


def prompt_strategy_metadata():
    """Interaktive Eingabe von Strategie-Metadaten."""
    print("\n" + "="*60)
    print("STRATEGIE-METADATEN")
    print("="*60)
    print("(Enter drücken um Feld zu überspringen)\n")

    name = input("Strategie-Name: ").strip() or None
    if not name:
        return None

    category = input("Kategorie [baseline/feature_test/model_test/hyperparameter/production]: ").strip() or "experiment"
    tags_input = input("Tags (komma-getrennt): ").strip()
    tags = [t.strip() for t in tags_input.split(",")] if tags_input else []

    hypothesis = input("Hypothese (was wird getestet?): ").strip() or None
    expected = input("Erwartetes Ergebnis: ").strip() or None
    baseline = input("Baseline Run-ID (für Vergleich): ").strip() or None

    # Änderungen
    changes = []
    print("\nÄnderungen vs Baseline (leer lassen zum Beenden):")
    while True:
        component = input("  Komponente (z.B. model, features, simulation): ").strip()
        if not component:
            break
        before = input("  Vorher: ").strip()
        after = input("  Nachher: ").strip()
        changes.append({"component": component, "before": before, "after": after})

    # Modell
    print("\nModell-Konfiguration:")
    model_arch = input("  Architektur [unified/long_short_separate/ensemble]: ").strip() or "unified"

    notes = input("\nNotizen: ").strip() or None

    return create_strategy_metadata(
        name=name,
        category=category,
        tags=tags,
        hypothesis=hypothesis,
        expected_outcome=expected,
        baseline_run=baseline,
        changes=changes,
        model_architecture=model_arch,
        notes=notes,
    )


def load_strategy_from_file(filepath):
    """Lädt Strategie-Metadaten aus einer JSON-Datei."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden von {filepath}: {e}")
        return None


def analyze_reversed_strategies(run_id, top_n=10):
    """
    Analysiert die schlechtesten Strategien und zeigt, wie sie umgekehrt performen würden.

    Bei einer Strategie mit sehr niedriger Win-Rate sollte theoretisch das Umkehren
    (Long->Short, Short->Long) zu einer hohen Win-Rate führen.
    In der Praxis funktioniert das wegen Spread-Kosten meist nicht.
    """
    from .results import load_run

    run_data = load_run(run_id)
    if not run_data:
        print(f"Run {run_id} nicht gefunden.")
        return

    print(f"\n{'='*80}")
    print(f"UMGEKEHRTE STRATEGIEN ANALYSE - Run: {run_id}")
    print(f"{'='*80}\n")

    # Sammle alle Grid-Ergebnisse
    all_grids = []
    grid_details_path = f"test_results/{run_id}/grid_details"
    results_path = f"test_results/{run_id}/results.json"

    if os.path.exists(grid_details_path):
        for filename in os.listdir(grid_details_path):
            if filename.endswith(".json"):
                with open(os.path.join(grid_details_path, filename)) as f:
                    data = json.load(f)
                    sym = data.get("symbol", filename.replace(".json", ""))
                    for gr in data.get("grid_results", []):
                        gr["symbol"] = sym
                        all_grids.append(gr)
    elif os.path.exists(results_path):
        # Fallback: Lade aus results.json (ältere Runs)
        # HINWEIS: elite_results enthält nur die BESTEN Strategien, nicht die schlechtesten!
        with open(results_path) as f:
            results = json.load(f)
            for r in results.get("elite_results", []):
                # Extrahiere Grid-ähnliche Daten aus elite_results
                sym = r.get("symbol", "?")
                config = r.get("config", {})
                all_grids.append({
                    "symbol": sym,
                    "feature_group": config.get("feature_group", "unknown"),
                    "tp_mult": config.get("tp_mult", 0),
                    "sl_mult": config.get("sl_mult", 0),
                    "conf_thresh": config.get("conf_thresh", 0),
                    "win_rate": r.get("win_rate", 0),
                    "pnl": r.get("pnl", 0),
                    "trades": r.get("trades", 0),
                    "sharpe": r.get("sharpe", 0),
                    "rrr": r.get("rrr", 1),
                })
        print("HINWEIS: Dieser Run hat keine grid_details.")
        print("         Es werden nur die Elite-Ergebnisse (BESTE Strategien) analysiert.")
        print("         Für echte Reverse-Analyse einen neuen Run durchführen.\n")

    if not all_grids:
        print("Keine Grid-Ergebnisse gefunden.")
        print("Hinweis: Grid-Details werden erst bei neueren Runs gespeichert.")
        print("Führe einen neuen Optimizer-Run durch, um --reverse-worst nutzen zu können.")
        return

    # Sortiere nach PnL (schlechteste zuerst)
    all_grids.sort(key=lambda x: x.get("pnl", 0))

    worst = all_grids[:top_n]

    print(f"Top {len(worst)} SCHLECHTESTE Strategien (nach PnL):\n")

    table_data = []
    reversed_table = []

    for w in worst:
        sym = w.get("symbol", "?")
        fg = w.get("feature_group", "?")
        tp = w.get("tp_mult", 0)
        sl = w.get("sl_mult", 0)
        ct = w.get("conf_thresh", 0)
        wr = w.get("win_rate", 0)
        pnl = w.get("pnl", 0)
        trades = w.get("trades", 0)
        sharpe = w.get("sharpe", 0)
        rrr = w.get("rrr", 1)

        # Original
        table_data.append([
            sym, fg[:15], f"{tp}/{sl}", ct, f"{wr:.1%}", trades, f"{pnl:+.1f}", f"{sharpe:.2f}"
        ])

        # Umgekehrte Berechnung:
        # Wenn WR = 30%, dann hat die umgekehrte Strategie WR = 70%
        # Aber: RRR kehrt sich auch um! TP=20/SL=40 -> TP=40/SL=20
        reversed_wr = 1 - wr
        reversed_rrr = 1 / rrr if rrr > 0 else 1

        # PnL umkehren: Jeder Win wird Loss, jeder Loss wird Win
        # Aber mit umgekehrtem RRR!
        # Original: Win gibt +RRR, Loss gibt -1
        # Reversed: Original-Win (jetzt Loss) gibt -1/RRR, Original-Loss (jetzt Win) gibt +1
        # Erwartungswert: reversed_wr * 1 - (1 - reversed_wr) * (1/rrr)
        #                = reversed_wr - (1-reversed_wr)/rrr

        if trades > 0:
            # Simuliere umgekehrte Trades
            # Original hatte: wins = wr * trades, losses = (1-wr) * trades
            # Reversed: wins = (1-wr) * trades mit RRR=1, losses = wr * trades mit loss=1/rrr
            wins_count = int((1 - wr) * trades)
            losses_count = trades - wins_count

            # Bei Umkehrung: wir gewinnen was vorher verloren hat (mit neuem RRR)
            # und verlieren was vorher gewonnen hat
            reversed_pnl = wins_count * 1.0 - losses_count * (1 / rrr if rrr > 0 else 1)

            # Sharpe approximieren
            if trades > 1:
                # Grobe Approximation
                avg_return = reversed_pnl / trades
                # Volatilität bleibt ähnlich
                reversed_sharpe = avg_return * (trades ** 0.5) * 10  # Grobe Skalierung
            else:
                reversed_sharpe = 0
        else:
            reversed_pnl = 0
            reversed_sharpe = 0

        reversed_table.append([
            sym, fg[:15], f"{sl}/{tp}", ct,  # TP/SL getauscht
            f"{reversed_wr:.1%}", trades, f"{reversed_pnl:+.1f}", f"{reversed_sharpe:.2f}"
        ])

    print("ORIGINAL (schlechteste):")
    print(tabulate(
        table_data,
        headers=["Symbol", "Feature-Gruppe", "TP/SL", "CT", "WinRate", "Trades", "PnL", "Sharpe"],
        tablefmt="psql"
    ))

    print("\n" + "-"*80)
    print("\nUMGEKEHRT (Long<->Short, TP<->SL):")
    print(tabulate(
        reversed_table,
        headers=["Symbol", "Feature-Gruppe", "TP/SL", "CT", "WinRate", "Trades", "PnL", "Sharpe"],
        tablefmt="psql"
    ))

    print("\n" + "="*80)
    print("FAZIT:")
    print("="*80)

    # Vergleiche PnL-Summen
    orig_pnl = sum(w.get("pnl", 0) for w in worst)
    # Berechne reversed PnL nochmal sauber
    rev_pnl = 0
    for w in worst:
        wr = w.get("win_rate", 0)
        trades = w.get("trades", 0)
        rrr = w.get("rrr", 1)
        if trades > 0 and rrr > 0:
            wins_count = int((1 - wr) * trades)
            losses_count = trades - wins_count
            rev_pnl += wins_count * 1.0 - losses_count * (1 / rrr)

    print(f"\nOriginal PnL (Summe):   {orig_pnl:+.1f}")
    print(f"Reversed PnL (Summe):   {rev_pnl:+.1f}")

    if rev_pnl > 0 and orig_pnl < 0:
        improvement = abs(rev_pnl - orig_pnl)
        print(f"\n-> Umkehrung verbessert um {improvement:.1f} Punkte!")
        print("   ABER: Diese Berechnung ignoriert Spread-Kosten beim Umkehren.")
        print("   In der Realität frisst der Spread oft den Gewinn auf.")
    elif rev_pnl < orig_pnl:
        print(f"\n-> Umkehrung macht es SCHLECHTER!")
        print("   Das liegt an den asymmetrischen Spread-Kosten und RRR-Umkehrung.")
    else:
        print(f"\n-> Kein signifikanter Unterschied.")

    print("\nWICHTIG: Umkehrung funktioniert in der Praxis selten, weil:")
    print("  1. Spread-Kosten bei jedem Trade anfallen (egal ob Long oder Short)")
    print("  2. RRR sich umkehrt (TP/SL tauschen)")
    print("  3. Markt-Mikrostruktur asymmetrisch ist")


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
  python -m optimizer --list --category model_test # Nach Kategorie filtern
  python -m optimizer --compare RUN1 RUN2          # Runs vergleichen
  python -m optimizer --no-save                    # Ohne Speichern
  python -m optimizer --assets EURUSD,GBPUSD       # Nur bestimmte Assets
  python -m optimizer --features trend,momentum    # Nur bestimmte Feature-Gruppen
  python -m optimizer --reverse-worst RUN_ID       # Schlechteste Strategien umkehren

Kategorien: baseline, feature_test, model_test, hyperparameter, production, experiment
        """
    )

    parser.add_argument("-d", "--description", type=str, help="Beschreibung für diesen Run")
    parser.add_argument("--strategy", action="store_true", help="Interaktive Strategie-Metadaten-Eingabe")
    parser.add_argument("--strategy-file", type=str, metavar="FILE", help="Strategie-Metadaten aus JSON-Datei laden")
    parser.add_argument("--list", action="store_true", help="Alle vorhandenen Runs anzeigen")
    parser.add_argument("--category", type=str, help="Filtert --list nach Kategorie")
    parser.add_argument("--tags", type=str, help="Filtert --list nach Tags (komma-getrennt)")
    parser.add_argument("--compare", nargs="+", metavar="RUN_ID", help="Runs vergleichen")
    parser.add_argument("--no-save", action="store_true", help="Ergebnisse nicht in test_results speichern")
    parser.add_argument("--load", type=str, metavar="RUN_ID", help="Details eines Runs anzeigen")
    parser.add_argument("--assets", type=str, help="Nur bestimmte Assets testen (komma-getrennt, z.B. BTCUSD,ETHUSD)")
    parser.add_argument("--features", type=str, help="Feature-Gruppen testen (komma-getrennt, z.B. trend,momentum,macro)")
    parser.add_argument("--list-features", action="store_true", help="Verfügbare Feature-Gruppen anzeigen")
    parser.add_argument("--reverse-worst", type=str, metavar="RUN_ID", help="Analysiere schlechteste Strategien eines Runs umgekehrt")
    parser.add_argument("--reverse-n", type=int, default=10, help="Anzahl der schlechtesten Strategien für --reverse-worst (default: 10)")
    # Ressourcen-Einstellungen
    parser.add_argument("--cpu", type=float, help="Max CPU-Auslastung (0.0-1.0, z.B. 0.80 für 80%%)")
    parser.add_argument("--ram-reserve", type=float, help="Min freier RAM-Anteil (0.0-1.0, z.B. 0.25 für 25%%)")
    parser.add_argument("--ram-per-worker", type=float, help="RAM pro Worker in GB (z.B. 4.0)")

    args = parser.parse_args()

    if args.list_features:
        from .config import FEATURE_GROUPS, DEFAULT_FEATURE_GROUPS
        print("\n" + "="*70)
        print("VERFÜGBARE FEATURE-GRUPPEN")
        print("="*70 + "\n")
        for name, group in FEATURE_GROUPS.items():
            default_mark = " [DEFAULT]" if name in DEFAULT_FEATURE_GROUPS else ""
            print(f"{name}{default_mark}")
            print(f"  Name: {group['name']}")
            print(f"  Prefixes: {', '.join(group['prefixes'])}")
            print(f"  Beschreibung: {group['description']}")
            print()
        print(f"Verwendung: python -m optimizer --features trend,momentum,macro")
        return

    if args.list:
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        runs = list_runs(category=args.category, tags=tags)

        if not runs:
            print("Keine Test-Runs gefunden.")
            return

        print(f"\n{'='*100}")
        print("VORHANDENE TEST-RUNS")
        if args.category:
            print(f"(Gefiltert nach Kategorie: {args.category})")
        if tags:
            print(f"(Gefiltert nach Tags: {tags})")
        print(f"{'='*100}\n")

        table_data = []
        for r in runs:
            table_data.append([
                r["run_id"][:20],
                r.get("timeframe", "?"),
                r.get("category", "-")[:12] if r.get("category") else "-",
                r.get("strategy_name", "-")[:20] if r.get("strategy_name") else "-",
                r.get("profitable_count", "?"),
                r.get("model_architecture", "-")[:10] if r.get("model_architecture") else "-",
            ])

        print(tabulate(
            table_data,
            headers=["Run ID", "TF", "Kategorie", "Strategie", "OK", "Architektur"],
            tablefmt="psql"
        ))

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

        # Parse feature groups filter
        if args.features:
            os.environ["OPTIMIZER_FEATURE_GROUPS"] = args.features

        run_optimizer(
            description=args.description,
            save_results=not args.no_save,
            strategy_metadata=strategy_metadata if strategy_metadata else None,
            asset_filter=asset_filter
        )


if __name__ == "__main__":
    main()
