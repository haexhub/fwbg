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


def run_optimizer(description=None, save_results=True, strategy_metadata=None):
    """
    Führt die Walk-Forward Optimierung aus.

    Args:
        description: Optionale Beschreibung für diesen Run
        save_results: Wenn True, werden Ergebnisse in test_results/ gespeichert
        strategy_metadata: Strukturierte Strategie-Metadaten (dict oder via create_strategy_metadata())
    """
    # Lade nur Dateien für das gewählte Timeframe
    files = sorted(glob.glob(f"{DATA_PATH}/*_{TIMEFRAME}.csv"))
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
    import os
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

    # Adaptive Pool: 80% CPU, aber immer 20% RAM freihalten
    pool_manager = AdaptivePoolManager(
        max_cpu_percent=0.80,
        min_free_ram_percent=0.20,
        verbose=True  # Zeige Pool-Status
    )

    # Progress-Tracking mit tqdm
    pbar = tqdm(total=len(files), desc="Assets", dynamic_ncols=True)

    def update_progress(completed, total):
        pbar.n = completed
        pbar.refresh()

    print("\nStarte Verarbeitung...\n")
    raw_results = pool_manager.map_adaptive(
        func=process_symbol,
        items=files,
        progress_callback=update_progress
    )

    pbar.close()

    # Stats ausgeben
    stats = pool_manager.get_status()
    print(f"\nPeak Workers: {stats['peak_workers']}, RAM-Throttles: {stats['ram_throttle_count']}")
    print(f"{len(raw_results)} Assets haben die Optimierung bestanden.")

    # Korrelationsfilter anwenden
    filtered = filter_correlated_assets(raw_results, CORR_THRESHOLD)
    print(f"{len(filtered)} Assets nach Korrelationsfilter.")

    # Top 10 auswählen
    elite = filtered[:10]

    # Portfolio-Shield
    shield = 1.0 / (len(elite) ** 0.5) if elite else 1.0
    final_assets = {}
    table_data = []

    for e in elite:
        e["config"]["kelly_risk"] *= shield

        # Equity Plot mit Drawdown
        cap, eq = 100.0, [100.0]
        peak = 100.0
        drawdowns = [0.0]
        for r in e["tr_trace"]:
            cap *= 1 + (
                e["config"]["kelly_risk"] * e["rrr"]
                if r > 0
                else -e["config"]["kelly_risk"]
            )
            eq.append(cap)
            if cap > peak:
                peak = cap
            drawdowns.append((peak - cap) / peak * 100)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1])

        ax1.plot(eq, color="blue", linewidth=1.5)
        ax1.fill_between(range(len(eq)), eq, alpha=0.3)
        ax1.set_title(
            f"{e['symbol']} | WR: {e['win_rate']:.1%} | RRR: {e['rrr']:.2f} | "
            f"Sharpe: {e.get('sharpe', 0):.2f} | Calmar: {e.get('calmar', 0):.2f}"
        )
        ax1.set_ylabel("Equity")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.5)
        ax2.set_xlabel("Trade #")
        ax2.set_ylabel("Drawdown %")
        ax2.set_ylim(max(drawdowns) * 1.1 if drawdowns else 1, 0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{plots_path}/{e['symbol']}.png", dpi=100)
        plt.close()

        # Profitabilitätsprüfung
        sharpe = e.get("sharpe", 0)
        wr = e["win_rate"]
        rrr = e["rrr"]

        # Equity-Simulation
        START_EQUITY = 1000.0
        SAFETY_MARGIN = 0.10

        equity = START_EQUITY
        kelly = e["config"]["kelly_risk"]
        max_dd = 0
        peak = equity

        random.seed(42)
        conservative_trades = []
        for res in e["tr_trace"]:
            if res > 0 and random.random() < SAFETY_MARGIN:
                conservative_trades.append(-1.0)
            else:
                conservative_trades.append(res)

        for res in conservative_trades:
            risk_amount = equity * kelly
            if res > 0:
                equity += risk_amount * rrr
            else:
                equity -= risk_amount

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            if equity <= 0:
                equity = 0
                max_dd = 1.0
                break

        final_equity = equity
        max_dd_pct = max_dd * 100

        # Jahresrendite berechnen
        bars_per_year = 24 * 250 if TIMEFRAME == "HOUR" else 96 * 250
        total_oos_bars = WALK_FORWARD_FOLDS * OOS_SIZE
        years = total_oos_bars / bars_per_year if bars_per_year > 0 else 1

        if final_equity > 0 and years > 0:
            annual_return = ((final_equity / START_EQUITY) ** (1 / years) - 1) * 100
        else:
            annual_return = -100

        is_profitable = sharpe >= 1.0 and annual_return > 0 and max_dd < 0.6

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
            status,
        ])

    print(
        "\n"
        + tabulate(
            table_data,
            headers=["Asset", "Kelly", "WinRate", "RRR", "Sharpe", "Calmar", "Trades", "Return", "MaxDD", "Status"],
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
            raw_results=raw_results,
            filtered_results=filtered,
            elite_results=elite,
            final_assets=final_assets,
            table_data=table_data,
            description=description,
            strategy_metadata=strategy_metadata
        )
        print(f"\nErgebnisse gespeichert in: {run_path}/")

    # Auch ins Account-Verzeichnis exportieren (für den Bot)
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(EXPORT_FILE, "w") as f:
        json.dump(convert_numpy(final_assets), f, indent=4)

    n_profitable = len(final_assets)
    print(f"\n{n_profitable} profitable Assets in {EXPORT_FILE} exportiert.")

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

    args = parser.parse_args()

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

    else:
        # Strategie-Metadaten
        strategy_metadata = None
        if args.strategy_file:
            strategy_metadata = load_strategy_from_file(args.strategy_file)
        elif args.strategy:
            strategy_metadata = prompt_strategy_metadata()

        run_optimizer(
            description=args.description,
            save_results=not args.no_save,
            strategy_metadata=strategy_metadata
        )


if __name__ == "__main__":
    main()
