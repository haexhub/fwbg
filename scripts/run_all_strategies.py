#!/usr/bin/env python3
"""
Batch-Skript zum sequentiellen Testen aller Strategien.

Startet jeden Strategy-Run nacheinander und speichert die Ergebnisse.
Ideal für Overnight-Runs.

Verwendung:
    python scripts/run_all_strategies.py                    # Alle Strategien
    python scripts/run_all_strategies.py --list             # Zeigt alle Strategien
    python scripts/run_all_strategies.py --only scalping swing_trading  # Nur bestimmte
    python scripts/run_all_strategies.py --exclude default baseline_unified  # Ausschließen
    python scripts/run_all_strategies.py --dry-run          # Zeigt was laufen würde
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Strategien die standardmäßig übersprungen werden (Test/Legacy-Dateien)
DEFAULT_EXCLUDE = [
    "default",
    "baseline_unified",
    "long_short_separate",
    "test_termination",
    "all_styles",
    "symmetric_grid",  # Das ist die alte Default-Config
]

# Strategien in empfohlener Reihenfolge (schnellste zuerst für schnelles Feedback)
STRATEGY_ORDER = [
    # Session-basiert (kleinere Grids, schneller)
    "asian_session",
    "london_newyork",
    # Hauptstrategien
    "scalping",
    "swing_trading",
    # Technische Strategien
    "mean_reversion",
    "breakout",
    "trend_following",
    # Spezielle Strategien
    "volatility_breakout",
    "macro_driven",
    "high_confidence",
]


def get_strategy_files(strategies_dir: Path) -> dict[str, Path]:
    """Findet alle Strategy-Dateien und gibt ein Dict zurück."""
    strategies = {}
    for f in strategies_dir.glob("*.json"):
        name = f.stem
        strategies[name] = f
    return strategies


def load_strategy_info(filepath: Path) -> dict:
    """Lädt Basis-Infos aus einer Strategy-Datei."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        return {
            "name": data.get("name", filepath.stem),
            "category": data.get("category", "unknown"),
            "description": data.get("description", "-"),
            "grids": data.get("grids", {}),
        }
    except Exception as e:
        return {"name": filepath.stem, "error": str(e)}


def estimate_duration(strategy_info: dict) -> str:
    """Schätzt die ungefähre Laufzeit basierend auf Grid-Größe."""
    grids = strategy_info.get("grids", {})
    if not grids:
        return "~30-60 min (default grid)"

    # Berechne Kombinationen für FOREX als Referenz
    forex_grid = grids.get("FOREX", {})
    tp_count = len(forex_grid.get("tp", []))
    sl_count = len(forex_grid.get("sl", []))

    combos = tp_count * sl_count if tp_count and sl_count else 64

    if combos <= 20:
        return "~15-30 min"
    elif combos <= 40:
        return "~30-60 min"
    elif combos <= 60:
        return "~60-90 min"
    else:
        return "~90-120 min"


def find_latest_run_dir(project_root: Path, strategy_name: str, after_time: datetime) -> Path | None:
    """Findet das neueste Run-Verzeichnis für eine Strategie."""
    test_results = project_root / "test_results"
    if not test_results.exists():
        return None

    # Suche nach Verzeichnissen die nach after_time erstellt wurden
    candidates = []
    for d in test_results.iterdir():
        if not d.is_dir():
            continue
        # Format: YYYYMMDD_HHMMSS_hash oder YYYYMMDD_HHMMSS_description
        if not re.match(r"\d{8}_\d{6}_", d.name):
            continue

        # Prüfe ob es die richtige Strategie ist (über strategy.json oder Beschreibung)
        strategy_file = d / "strategy.json"
        if strategy_file.exists():
            try:
                with open(strategy_file) as f:
                    strat_data = json.load(f)
                # Prüfe ob Strategie-Name passt
                strat_name = strat_data.get("name", "").lower()
                if strategy_name.lower().replace("_", " ") in strat_name.lower() or \
                   strategy_name.lower() in d.name.lower():
                    # Prüfe Zeitstempel
                    try:
                        dir_time = datetime.strptime(d.name[:15], "%Y%m%d_%H%M%S")
                        if dir_time >= after_time:
                            candidates.append((dir_time, d))
                    except ValueError:
                        pass
            except (json.JSONDecodeError, IOError):
                pass

    if not candidates:
        return None

    # Neuestes zurückgeben
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_elite_table(run_dir: Path) -> str | None:
    """Extrahiert die Elite-Assets-Tabelle aus summary.txt."""
    summary_file = run_dir / "summary.txt"
    if not summary_file.exists():
        return None

    try:
        with open(summary_file) as f:
            content = f.read()

        # Finde den ELITE ASSETS Abschnitt
        match = re.search(
            r"ELITE ASSETS\n-+\n(.*?)(?=\n\nPROFITABLE ASSETS|\n======|\Z)",
            content,
            re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None
    except IOError:
        return None


def extract_results_summary(run_dir: Path) -> dict | None:
    """Extrahiert Zusammenfassung aus results.json."""
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file) as f:
            data = json.load(f)

        elite = data.get("elite_results", [])
        return {
            "profitable_count": len([e for e in elite if e.get("sharpe", 0) >= 1.0]),
            "total_elite": len(elite),
            "avg_sharpe": sum(e.get("sharpe", 0) for e in elite) / len(elite) if elite else 0,
            "avg_winrate": sum(e.get("win_rate", 0) for e in elite) / len(elite) if elite else 0,
            "total_trades": sum(len(e.get("tr_trace", [])) for e in elite),
        }
    except (json.JSONDecodeError, IOError):
        return None


def run_strategy(strategy_name: str, strategy_file: Path, project_root: Path, dry_run: bool = False) -> dict:
    """Führt einen einzelnen Strategy-Run aus."""
    start_time = datetime.now()

    cmd = [
        sys.executable, "-m", "optimizer",
        "--strategy-file", str(strategy_file),
        "-d", f"Batch: {strategy_name}"
    ]

    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy_name}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    if dry_run:
        print("[DRY RUN] Würde jetzt laufen...")
        return {"status": "dry_run", "strategy": strategy_name, "duration": 0}

    try:
        # Run mit Live-Output
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=False
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success" if result.returncode == 0 else f"error (code {result.returncode})"

        # Finde Run-Verzeichnis und extrahiere Ergebnisse
        run_dir = find_latest_run_dir(project_root, strategy_name, start_time)
        elite_table = None
        results_summary = None
        run_id = None

        if run_dir:
            run_id = run_dir.name
            elite_table = extract_elite_table(run_dir)
            results_summary = extract_results_summary(run_dir)

        print(f"\n{'='*70}")
        print(f"FINISHED: {strategy_name}")
        print(f"Status: {status}")
        print(f"Duration: {timedelta(seconds=int(duration))}")
        if run_id:
            print(f"Run ID: {run_id}")
        if results_summary:
            print(f"Profitable: {results_summary['profitable_count']}/{results_summary['total_elite']}")
            print(f"Avg Sharpe: {results_summary['avg_sharpe']:.2f}")
            print(f"Avg WinRate: {results_summary['avg_winrate']:.1%}")
        print(f"{'='*70}\n")

        return {
            "status": status,
            "strategy": strategy_name,
            "duration": duration,
            "returncode": result.returncode,
            "run_id": run_id,
            "elite_table": elite_table,
            "results_summary": results_summary,
        }

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n[ERROR] {strategy_name}: {e}")
        return {
            "status": f"exception: {e}",
            "strategy": strategy_name,
            "duration": duration
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch-Runner für alle Trading-Strategien",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
    python scripts/run_all_strategies.py                    # Alle empfohlenen Strategien
    python scripts/run_all_strategies.py --list             # Zeigt alle Strategien
    python scripts/run_all_strategies.py --only scalping    # Nur eine Strategie
    python scripts/run_all_strategies.py --all              # Alle inkl. Test-Dateien
    python scripts/run_all_strategies.py --dry-run          # Zeigt nur was laufen würde
        """
    )

    parser.add_argument("--list", action="store_true", help="Zeigt alle verfügbaren Strategien")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Nur bestimmte Strategien testen")
    parser.add_argument("--exclude", nargs="+", metavar="NAME", help="Strategien ausschließen")
    parser.add_argument("--all", action="store_true", help="Alle Strategien inkl. Test/Legacy-Dateien")
    parser.add_argument("--dry-run", action="store_true", help="Zeigt was laufen würde, ohne auszuführen")

    args = parser.parse_args()

    # Strategies-Verzeichnis finden
    project_root = Path(__file__).parent.parent
    strategies_dir = project_root / "strategies"

    if not strategies_dir.exists():
        print(f"Error: Strategies-Verzeichnis nicht gefunden: {strategies_dir}")
        sys.exit(1)

    # Alle Strategien laden
    all_strategies = get_strategy_files(strategies_dir)

    # --list: Zeige alle Strategien
    if args.list:
        print(f"\n{'='*80}")
        print("VERFÜGBARE STRATEGIEN")
        print(f"{'='*80}\n")

        for name in sorted(all_strategies.keys()):
            info = load_strategy_info(all_strategies[name])
            excluded = name in DEFAULT_EXCLUDE
            marker = " [EXCLUDED by default]" if excluded else ""
            in_order = " [QUEUED]" if name in STRATEGY_ORDER else ""

            print(f"{name}{marker}{in_order}")
            print(f"  Name: {info.get('name', '-')}")
            print(f"  Category: {info.get('category', '-')}")
            print(f"  Est. Duration: {estimate_duration(info)}")
            print()

        print(f"Total: {len(all_strategies)} Strategien")
        print(f"Default Queue: {len(STRATEGY_ORDER)} Strategien")
        return

    # Strategien zum Ausführen bestimmen
    if args.only:
        # Nur angegebene Strategien
        to_run = []
        for name in args.only:
            if name in all_strategies:
                to_run.append(name)
            else:
                print(f"Warning: Strategie '{name}' nicht gefunden, überspringe...")
    elif args.all:
        # Alle Strategien
        to_run = list(all_strategies.keys())
    else:
        # Standard: STRATEGY_ORDER, ohne DEFAULT_EXCLUDE
        to_run = [s for s in STRATEGY_ORDER if s in all_strategies]
        # Füge andere gefundene Strategien hinzu (nicht in exclude)
        for name in all_strategies:
            if name not in to_run and name not in DEFAULT_EXCLUDE:
                to_run.append(name)

    # Zusätzliche Exclusions anwenden
    if args.exclude:
        to_run = [s for s in to_run if s not in args.exclude]

    if not to_run:
        print("Keine Strategien zum Ausführen gefunden.")
        sys.exit(1)

    # Zusammenfassung anzeigen
    print(f"\n{'='*80}")
    print("BATCH STRATEGY RUNNER")
    print(f"{'='*80}\n")

    total_estimated = 0
    print("Strategien in der Queue:")
    for i, name in enumerate(to_run, 1):
        info = load_strategy_info(all_strategies[name])
        est = estimate_duration(info)
        print(f"  {i}. {name} ({est})")

    print(f"\nTotal: {len(to_run)} Strategien")
    print(f"Geschätzte Gesamtzeit: {len(to_run) * 45}-{len(to_run) * 90} Minuten")

    if args.dry_run:
        print("\n[DRY RUN MODE] - Keine tatsächliche Ausführung")

    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    # Strategien nacheinander ausführen
    results = []
    batch_start = datetime.now()

    for i, name in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] Starte {name}...")
        result = run_strategy(name, all_strategies[name], project_root, dry_run=args.dry_run)
        results.append(result)

        # Kurze Pause zwischen Runs (RAM freigeben)
        if not args.dry_run and i < len(to_run):
            print("Warte 10 Sekunden vor dem nächsten Run...")
            time.sleep(10)

    batch_end = datetime.now()
    total_duration = (batch_end - batch_start).total_seconds()

    # Zusammenfassung
    print(f"\n{'='*80}")
    print("BATCH ABGESCHLOSSEN")
    print(f"{'='*80}\n")

    successful = [r for r in results if r.get("returncode") == 0 or r.get("status") == "dry_run"]
    failed = [r for r in results if r.get("returncode", 0) != 0 and r.get("status") != "dry_run"]

    print(f"Erfolgreich: {len(successful)}/{len(results)}")
    if failed:
        print(f"Fehlgeschlagen: {len(failed)}")
        for r in failed:
            print(f"  - {r['strategy']}: {r['status']}")

    print(f"\nGesamtdauer: {timedelta(seconds=int(total_duration))}")
    print(f"Ende: {batch_end.strftime('%Y-%m-%d %H:%M:%S')}")

    # Detaillierte Ergebnisse pro Strategie
    print(f"\n{'='*80}")
    print("ERGEBNISSE PRO STRATEGIE")
    print(f"{'='*80}")

    for r in results:
        if r.get("status") == "dry_run":
            continue

        print(f"\n{'─'*80}")
        print(f"STRATEGIE: {r['strategy']}")
        print(f"{'─'*80}")
        print(f"Status: {r['status']}")
        print(f"Dauer: {timedelta(seconds=int(r.get('duration', 0)))}")

        if r.get("run_id"):
            print(f"Run ID: {r['run_id']}")

        summary = r.get("results_summary")
        if summary:
            print(f"\nKennzahlen:")
            print(f"  Profitable Assets: {summary['profitable_count']}/{summary['total_elite']}")
            print(f"  Durchschn. Sharpe: {summary['avg_sharpe']:.2f}")
            print(f"  Durchschn. WinRate: {summary['avg_winrate']:.1%}")
            print(f"  Gesamt Trades: {summary['total_trades']}")

        elite_table = r.get("elite_table")
        if elite_table:
            print(f"\nElite Assets:")
            print(elite_table)

    # Log-Dateien schreiben
    if not args.dry_run:
        batch_id = batch_start.strftime('%Y%m%d_%H%M%S')
        log_file = project_root / "test_results" / f"batch_{batch_id}.json"
        summary_file = project_root / "test_results" / f"batch_{batch_id}_summary.txt"
        log_file.parent.mkdir(exist_ok=True)

        # JSON Log
        with open(log_file, "w") as f:
            json.dump({
                "start": batch_start.isoformat(),
                "end": batch_end.isoformat(),
                "duration_seconds": total_duration,
                "strategies_run": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "results": results
            }, f, indent=2)

        # Text Summary mit allen Tabellen
        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BATCH RUN SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Start: {batch_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ende: {batch_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dauer: {timedelta(seconds=int(total_duration))}\n")
            f.write(f"Erfolgreich: {len(successful)}/{len(results)}\n\n")

            for r in results:
                if r.get("status") == "dry_run":
                    continue

                f.write("─" * 80 + "\n")
                f.write(f"STRATEGIE: {r['strategy']}\n")
                f.write("─" * 80 + "\n")
                f.write(f"Status: {r['status']}\n")
                f.write(f"Dauer: {timedelta(seconds=int(r.get('duration', 0)))}\n")

                if r.get("run_id"):
                    f.write(f"Run ID: {r['run_id']}\n")

                summary = r.get("results_summary")
                if summary:
                    f.write(f"\nKennzahlen:\n")
                    f.write(f"  Profitable Assets: {summary['profitable_count']}/{summary['total_elite']}\n")
                    f.write(f"  Durchschn. Sharpe: {summary['avg_sharpe']:.2f}\n")
                    f.write(f"  Durchschn. WinRate: {summary['avg_winrate']:.1%}\n")
                    f.write(f"  Gesamt Trades: {summary['total_trades']}\n")

                elite_table = r.get("elite_table")
                if elite_table:
                    f.write(f"\nElite Assets:\n")
                    f.write(elite_table + "\n")

                f.write("\n")

        print(f"\nLogs gespeichert:")
        print(f"  JSON: {log_file}")
        print(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
