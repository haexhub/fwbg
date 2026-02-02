"""
CLI-Hilfsfunktionen für den Optimizer.

Enthält:
- Interaktive Strategie-Eingabe
- Run-Anzeige und -Vergleich
- Reverse-Analyse
"""
import os
import json

from tabulate import tabulate

from fwbg.results.storage import (
    list_runs,
    load_run,
    compare_runs,
    create_strategy_metadata,
)


def show_runs(tags=None):
    """Zeigt alle vorhandenen Runs an."""
    runs = list_runs(tags=tags)

    if not runs:
        print("Keine Test-Runs gefunden.")
        return

    print(f"\n{'=' * 100}")
    print("VORHANDENE TEST-RUNS")
    if tags:
        print(f"(Gefiltert nach Tags: {tags})")
    print(f"{'=' * 100}\n")

    table_data = []
    for r in runs:
        run_tags = r.get("tags", [])
        tags_str = ", ".join(run_tags[:3]) if run_tags else "-"
        table_data.append(
            [
                r["run_id"][:20],
                r.get("timeframe", "?"),
                r.get("strategy_name", "-")[:20] if r.get("strategy_name") else "-",
                r.get("profitable_count", "?"),
                r.get("model_architecture", "-")[:10]
                if r.get("model_architecture")
                else "-",
                tags_str[:25],
            ]
        )

    print(
        tabulate(
            table_data,
            headers=["Run ID", "TF", "Strategie", "OK", "Architektur", "Tags"],
            tablefmt="psql",
        )
    )


def show_comparison(run_ids):
    """Vergleicht mehrere Runs."""
    comparison = compare_runs(run_ids)

    if not comparison:
        print("Keine Runs zum Vergleichen gefunden.")
        return

    print(f"\n{'=' * 80}")
    print("RUN-VERGLEICH")
    print(f"{'=' * 80}\n")

    table_data = []
    for r in comparison["runs"]:
        stats = r.get("elite_stats", {})
        table_data.append(
            [
                r["run_id"],
                r.get("timeframe", "?"),
                r.get("profitable_count", 0),
                f"{stats.get('avg_sharpe', 0):.2f}",
                f"{stats.get('avg_win_rate', 0):.1%}",
                stats.get("total_trades", 0),
                r.get("description", "-")[:30] if r.get("description") else "-",
            ]
        )

    print(
        tabulate(
            table_data,
            headers=[
                "Run ID",
                "TF",
                "Profitable",
                "Avg Sharpe",
                "Avg WR",
                "Trades",
                "Description",
            ],
            tablefmt="psql",
        )
    )

    print(f"\nAlle profitablen Symbole: {', '.join(comparison['all_symbols'])}")


def prompt_strategy_metadata():
    """Interaktive Eingabe von Strategie-Metadaten."""
    print("\n" + "=" * 60)
    print("STRATEGIE-METADATEN")
    print("=" * 60)
    print("(Enter drücken um Feld zu überspringen)\n")

    name = input("Strategie-Name: ").strip() or None
    if not name:
        return None

    category = (
        input(
            "Kategorie [baseline/feature_test/model_test/hyperparameter/production]: "
        ).strip()
        or "experiment"
    )
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
    model_arch = (
        input("  Architektur [unified/long_short_separate/ensemble]: ").strip()
        or "unified"
    )

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
    run_data = load_run(run_id)
    if not run_data:
        print(f"Run {run_id} nicht gefunden.")
        return

    print(f"\n{'=' * 80}")
    print(f"UMGEKEHRTE STRATEGIEN ANALYSE - Run: {run_id}")
    print(f"{'=' * 80}\n")

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
        with open(results_path) as f:
            results = json.load(f)
            for r in results.get("elite_results", []):
                sym = r.get("symbol", "?")
                config = r.get("config", {})
                all_grids.append(
                    {
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
                    }
                )
        print("HINWEIS: Dieser Run hat keine grid_details.")
        print(
            "         Es werden nur die Elite-Ergebnisse (BESTE Strategien) analysiert."
        )
        print("         Für echte Reverse-Analyse einen neuen Run durchführen.\n")

    if not all_grids:
        print("Keine Grid-Ergebnisse gefunden.")
        print("Hinweis: Grid-Details werden erst bei neueren Runs gespeichert.")
        print(
            "Führe einen neuen Optimizer-Run durch, um --reverse-worst nutzen zu können."
        )
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
        table_data.append(
            [
                sym,
                fg[:15],
                f"{tp}/{sl}",
                ct,
                f"{wr:.1%}",
                trades,
                f"{pnl:+.1f}",
                f"{sharpe:.2f}",
            ]
        )

        # Umgekehrte Berechnung
        reversed_wr = 1 - wr

        if trades > 0:
            wins_count = int((1 - wr) * trades)
            losses_count = trades - wins_count
            reversed_pnl = wins_count * 1.0 - losses_count * (1 / rrr if rrr > 0 else 1)

            if trades > 1:
                avg_return = reversed_pnl / trades
                reversed_sharpe = avg_return * (trades**0.5) * 10
            else:
                reversed_sharpe = 0
        else:
            reversed_pnl = 0
            reversed_sharpe = 0

        reversed_table.append(
            [
                sym,
                fg[:15],
                f"{sl}/{tp}",
                ct,
                f"{reversed_wr:.1%}",
                trades,
                f"{reversed_pnl:+.1f}",
                f"{reversed_sharpe:.2f}",
            ]
        )

    print("ORIGINAL (schlechteste):")
    print(
        tabulate(
            table_data,
            headers=[
                "Symbol",
                "Feature-Gruppe",
                "TP/SL",
                "CT",
                "WinRate",
                "Trades",
                "PnL",
                "Sharpe",
            ],
            tablefmt="psql",
        )
    )

    print("\n" + "-" * 80)
    print("\nUMGEKEHRT (Long<->Short, TP<->SL):")
    print(
        tabulate(
            reversed_table,
            headers=[
                "Symbol",
                "Feature-Gruppe",
                "TP/SL",
                "CT",
                "WinRate",
                "Trades",
                "PnL",
                "Sharpe",
            ],
            tablefmt="psql",
        )
    )

    print("\n" + "=" * 80)
    print("FAZIT:")
    print("=" * 80)

    orig_pnl = sum(w.get("pnl", 0) for w in worst)
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
        print("\n-> Umkehrung macht es SCHLECHTER!")
        print("   Das liegt an den asymmetrischen Spread-Kosten und RRR-Umkehrung.")
    else:
        print("\n-> Kein signifikanter Unterschied.")

    print("\nWICHTIG: Umkehrung funktioniert in der Praxis selten, weil:")
    print("  1. Spread-Kosten bei jedem Trade anfallen (egal ob Long oder Short)")
    print("  2. RRR sich umkehrt (TP/SL tauschen)")
    print("  3. Markt-Mikrostruktur asymmetrisch ist")
