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
import json
import os
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


def run_strategy(strategy_name: str, strategy_file: Path, dry_run: bool = False) -> dict:
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
            cwd=Path(__file__).parent.parent,
            check=False
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success" if result.returncode == 0 else f"error (code {result.returncode})"

        print(f"\n{'='*70}")
        print(f"FINISHED: {strategy_name}")
        print(f"Status: {status}")
        print(f"Duration: {timedelta(seconds=int(duration))}")
        print(f"{'='*70}\n")

        return {
            "status": status,
            "strategy": strategy_name,
            "duration": duration,
            "returncode": result.returncode
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
        result = run_strategy(name, all_strategies[name], dry_run=args.dry_run)
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

    # Log-Datei schreiben
    if not args.dry_run:
        log_file = project_root / "test_results" / f"batch_{batch_start.strftime('%Y%m%d_%H%M%S')}.json"
        log_file.parent.mkdir(exist_ok=True)

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

        print(f"\nLog gespeichert: {log_file}")


if __name__ == "__main__":
    main()
