"""
Test Results Management - Persistierung von Optimierungs-Ergebnissen
"""
import os
import json
import hashlib
import subprocess
from datetime import datetime

from fwbg.data.config import (
    TIMEFRAME, WALK_FORWARD_FOLDS, OOS_SIZE,
    CORR_THRESHOLD, MIN_TRADES,
    convert_numpy
)

RESULTS_BASE_PATH = "test_results"


# =============================================================================
# STRATEGIE-METADATEN SCHEMA
# =============================================================================

STRATEGY_SCHEMA = {
    # Kurze Beschreibung
    "name": str,           # z.B. "Separate Long/Short Modelle"
    "description": str,    # Ausführliche Beschreibung

    # Kategorisierung
    "category": str,       # "baseline", "feature_test", "model_test", "hyperparameter", "production"
    "tags": list,          # z.B. ["long_short", "macro", "xgboost"]

    # Hypothese & Erwartungen
    "hypothesis": str,     # Was wird getestet? z.B. "Separate Modelle für Long/Short sollten besser performen"
    "expected_outcome": str,  # Was erwarten wir? z.B. "Höhere Sharpe Ratio durch spezialisierte Modelle"
    "baseline_run": str,   # Run-ID des Baselines für Vergleich (optional)

    # Änderungen gegenüber Baseline
    "changes": list,       # Liste der Änderungen: [{"component": "model", "before": "unified", "after": "long/short"}]

    # Modell-Konfiguration
    "model": {
        "type": str,       # "xgboost", "lightgbm", "random_forest", etc.
        "architecture": str,  # "unified", "long_short_separate", "ensemble"
        "hyperparameters": dict,  # {"n_estimators": 100, "max_depth": 5, ...}
    },

    # Feature-Konfiguration
    "features": {
        "technical_indicators": bool,
        "macro_indicators": bool,
        "time_features": bool,
        "multi_timeframe": bool,
        "custom_features": list,  # Liste zusätzlicher Feature-Namen
        "feature_selection": str,  # "boruta", "boruta_plateau"
    },

    # Trade-Simulation
    "simulation": {
        "tp_sl_basis": str,  # "spread_multiple", "atr_multiple", "fixed_pips"
        "trailing_stop": bool,
        "slippage_model": str,  # "fixed", "variable", "none"
        "regime_filter": bool,
    },

    # Validierung
    "validation": {
        "method": str,     # "walk_forward", "k_fold", "train_test_split"
        "folds": int,
        "oos_ratio": float,
    },

    # Asset-Filter
    "assets": {
        "filter": list,    # z.B. ["EURUSD", "GBPUSD"] - None = alle Assets
        "exclude": list,   # z.B. ["BTCUSD"] - Assets die ausgeschlossen werden
        "classes": list,   # z.B. ["FOREX", "INDEX"] - Asset-Klassen filtern
    },

    # Ressourcen-Einstellungen
    "resources": {
        "max_concurrent_assets": int,  # z.B. 2 = 2 Assets parallel
    },

    # Notizen
    "notes": str,          # Freitext für zusätzliche Beobachtungen
}


def get_git_info():
    """Holt Git-Informationen für Reproduzierbarkeit."""
    git_info = {}
    try:
        # Aktueller Commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()

        # Branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Dirty status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0

        # Letzte Commit-Message
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["last_commit_message"] = result.stdout.strip()[:100]

    except Exception:
        pass

    return git_info


def create_strategy_metadata(
    name,
    description=None,
    category="experiment",
    tags=None,
    hypothesis=None,
    expected_outcome=None,
    baseline_run=None,
    changes=None,
    model_type="xgboost",
    model_architecture="unified",
    model_hyperparameters=None,
    use_technical=True,
    use_macro=True,
    use_time=True,
    use_multi_timeframe=True,
    custom_features=None,
    feature_selection="boruta",
    tp_sl_basis="spread_multiple",
    trailing_stop=True,
    slippage_model="fixed",
    regime_filter=True,
    validation_method="walk_forward",
    notes=None,
    # Neue Parameter für Asset- und Feature-Filter
    assets_filter=None,
    assets_exclude=None,
    assets_classes=None,
    # Ressourcen-Einstellungen
    max_concurrent_assets=None,
):
    """
    Erstellt strukturierte Strategie-Metadaten.

    Args:
        name: Kurzname der Strategie
        description: Ausführliche Beschreibung
        category: Kategorie (baseline, feature_test, model_test, hyperparameter, production)
        tags: Liste von Tags für Filterung
        hypothesis: Was wird getestet?
        expected_outcome: Was wird erwartet?
        baseline_run: Run-ID für Vergleich
        changes: Liste von Änderungen [{component, before, after}]
        ... (weitere Parameter)

    Returns:
        dict: Strukturierte Metadaten
    """
    return {
        "name": name,
        "description": description or "",
        "category": category,
        "tags": tags or [],
        "hypothesis": hypothesis or "",
        "expected_outcome": expected_outcome or "",
        "baseline_run": baseline_run,
        "changes": changes or [],
        "model": {
            "type": model_type,
            "architecture": model_architecture,
            "hyperparameters": model_hyperparameters or {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42,
            },
        },
        "features": {
            "technical_indicators": use_technical,
            "macro_indicators": use_macro,
            "time_features": use_time,
            "multi_timeframe": use_multi_timeframe,
            "custom_features": custom_features or [],
            "feature_selection": feature_selection,
        },
        "simulation": {
            "tp_sl_basis": tp_sl_basis,
            "trailing_stop": trailing_stop,
            "slippage_model": slippage_model,
            "regime_filter": regime_filter,
        },
        "validation": {
            "method": validation_method,
            "folds": WALK_FORWARD_FOLDS,
            "oos_size": OOS_SIZE,
        },
        "assets": {
            "filter": assets_filter,      # z.B. ["EURUSD", "GBPUSD"]
            "exclude": assets_exclude,    # z.B. ["BTCUSD"]
            "classes": assets_classes,    # z.B. ["FOREX", "INDEX"]
        },
        "resources": {
            "max_concurrent_assets": max_concurrent_assets,  # z.B. 2
        },
        "notes": notes or "",
    }


def generate_run_id(description=None):
    """
    Generiert eine eindeutige Run-ID.
    Format: YYYYMMDD_HHMMSS_[short_hash]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"{TIMEFRAME}_{WALK_FORWARD_FOLDS}"
    short_hash = hashlib.md5(f"{timestamp}_{config_str}".encode()).hexdigest()[:6]
    return f"{timestamp}_{short_hash}"


def create_run_directory(run_id, description=None, strategy_metadata=None):
    """
    Erstellt das Verzeichnis für einen Test-Run.

    Struktur:
    test_results/
    └── [run_id]/
        ├── config.json      # Technische Konfigurationsparameter
        ├── strategy.json    # Strategie-Metadaten (was wurde getestet?)
        ├── results.json     # Optimierungs-Ergebnisse
        ├── summary.txt      # Menschenlesbare Zusammenfassung
        ├── assets.json      # Export für den Bot
        └── plots/           # Equity-Kurven
    """
    run_path = os.path.join(RESULTS_BASE_PATH, run_id)
    plots_path = os.path.join(run_path, "plots")

    os.makedirs(run_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    # Git-Info für Reproduzierbarkeit
    git_info = get_git_info()

    # Technische Konfiguration
    config = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "timeframe": TIMEFRAME,
        "walk_forward_folds": WALK_FORWARD_FOLDS,
        "oos_size": OOS_SIZE,
        "corr_threshold": CORR_THRESHOLD,
        "min_trades": MIN_TRADES,
        "git": git_info,
    }

    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Strategie-Metadaten speichern
    if strategy_metadata:
        strategy_path = os.path.join(run_path, "strategy.json")
        with open(strategy_path, "w") as f:
            json.dump(strategy_metadata, f, indent=2)

    return run_path, plots_path


def save_run_results(run_path, raw_results, filtered_results, elite_results,
                     final_assets, table_data, description=None, strategy_metadata=None,
                     all_results=None):
    """
    Speichert alle Ergebnisse eines Runs.
    """
    # Vollständige Ergebnisse
    results_path = os.path.join(run_path, "results.json")
    results_data = {
        "total_processed": len(all_results) if all_results else len(raw_results),
        "significant_count": len(raw_results),
        "filtered_results_count": len(filtered_results),
        "elite_count": len(elite_results),
        "profitable_count": len(final_assets),
        "elite_results": convert_numpy([{
            "symbol": r["symbol"],
            "pnl": r["pnl"],
            "win_rate": r["win_rate"],
            "rrr": r["rrr"],
            "sharpe": r.get("sharpe", 0),
            "calmar": r.get("calmar", 0),
            "trades": len(r["tr_trace"]),
            "config": r["config"],
            "currencies": r.get("currencies", []),
            "tr_trace": r["tr_trace"],  # Trade-Sequenz für Analyse speichern
            "trades_detailed": r.get("trades_detailed", []),  # Volle Trade-Details
            "monte_carlo": r.get("monte_carlo", {}),
            "fold_stability": r.get("fold_stability", 0),
            "nested_cv": r.get("nested_cv", {}),  # Nested CV Info
            "smoothness": r.get("smoothness", {}),  # Equity-Smoothness
            "long_short_stats": r.get("long_short_stats", {}),  # Separate L/S Statistiken
        } for r in elite_results]),
    }

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Assets für den Bot
    assets_path = os.path.join(run_path, "assets.json")
    with open(assets_path, "w") as f:
        json.dump(convert_numpy(final_assets), f, indent=2)

    # Strategie-Metadaten (falls noch nicht gespeichert)
    strategy_path = os.path.join(run_path, "strategy.json")
    if strategy_metadata and not os.path.exists(strategy_path):
        with open(strategy_path, "w") as f:
            json.dump(strategy_metadata, f, indent=2)

    # Grid-Details Fallback: on_result_ready schreibt bereits vollständige Dateien.
    # Hier nur schreiben wenn noch kein Verzeichnis existiert.
    if all_results:
        grid_dir = os.path.join(run_path, "grid_details")

        for r in all_results:
            if not r or not r.get("grid_results"):
                continue

            sym = r["symbol"]
            sym_dir = os.path.join(grid_dir, sym)
            if os.path.exists(sym_dir):
                continue

            os.makedirs(sym_dir, exist_ok=True)

            config_data = {
                "symbol": sym,
                "status": r.get("status", "unknown"),
                "total_combinations": len(r["grid_results"]),
            }
            if r.get("status") == "ok":
                config_data["selected_config"] = {
                    "tp_mult": r["config"]["tp_mult"],
                    "sl_mult": r["config"]["sl_mult"],
                    "conf_thresh": r["config"]["conf_thresh"],
                    "risk_per_trade": r["config"]["risk_per_trade"],
                    "features": r["config"]["features"],
                }
                config_data["selected_metrics"] = {
                    "pnl": r["pnl"],
                    "win_rate": r["win_rate"],
                    "rrr": r["rrr"],
                    "sharpe": r.get("sharpe", 0),
                    "calmar": r.get("calmar", 0),
                    "trades": len(r.get("tr_trace", [])),
                }
            with open(os.path.join(sym_dir, "config.json"), "w") as f:
                json.dump(config_data, f, indent=2)

            grid_results_data = {
                "total_combinations": len(r["grid_results"]),
                "grid_results": convert_numpy(r["grid_results"]),
            }
            with open(os.path.join(sym_dir, "grid_results.json"), "w") as f:
                json.dump(grid_results_data, f, indent=2)

    # Menschenlesbare Zusammenfassung
    summary_path = os.path.join(run_path, "summary.txt")
    _write_summary(summary_path, run_path, description, strategy_metadata,
                   raw_results, filtered_results, elite_results, final_assets, table_data,
                   all_results=all_results)

    return results_path, assets_path, summary_path


def _write_summary(summary_path, run_path, description, strategy_metadata,
                   raw_results, filtered_results, elite_results, final_assets, table_data,
                   all_results=None):
    """Schreibt die menschenlesbare Zusammenfassung."""
    with open(summary_path, "w") as f:
        f.write(f"{'='*70}\n")
        f.write("OPTIMIZER RUN SUMMARY\n")
        f.write(f"{'='*70}\n\n")

        f.write(f"Run ID:      {os.path.basename(run_path)}\n")
        f.write(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timeframe:   {TIMEFRAME}\n")

        if description:
            f.write(f"Description: {description}\n")
        f.write("\n")

        # Strategie-Metadaten
        if strategy_metadata:
            f.write(f"{'='*70}\n")
            f.write("STRATEGIE\n")
            f.write(f"{'='*70}\n\n")

            f.write(f"Name:        {strategy_metadata.get('name', '-')}\n")
            f.write(f"Kategorie:   {strategy_metadata.get('category', '-')}\n")
            if strategy_metadata.get('tags'):
                f.write(f"Tags:        {', '.join(strategy_metadata['tags'])}\n")
            f.write("\n")

            if strategy_metadata.get('hypothesis'):
                f.write("HYPOTHESE\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{strategy_metadata['hypothesis']}\n\n")

            if strategy_metadata.get('expected_outcome'):
                f.write("ERWARTETES ERGEBNIS\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{strategy_metadata['expected_outcome']}\n\n")

            if strategy_metadata.get('changes'):
                f.write("ÄNDERUNGEN (vs Baseline)\n")
                f.write(f"{'-'*40}\n")
                for change in strategy_metadata['changes']:
                    f.write(f"  [{change.get('component', '?')}] {change.get('before', '?')} → {change.get('after', '?')}\n")
                f.write("\n")

            if strategy_metadata.get('baseline_run'):
                f.write(f"Baseline:    {strategy_metadata['baseline_run']}\n\n")

            # Modell-Info
            model = strategy_metadata.get('model', {})
            if not isinstance(model, dict):
                model = {}
            f.write("MODELL\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  Typ:          {model.get('type', '-')}\n")
            f.write(f"  Architektur:  {model.get('architecture', '-')}\n")
            if model.get('hyperparameters'):
                f.write(f"  Hyperparams:  {model['hyperparameters']}\n")
            f.write("\n")

            # Feature-Info
            features = strategy_metadata.get('features', {})
            f.write("FEATURES\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  Technical:    {'Ja' if features.get('technical_indicators') else 'Nein'}\n")
            f.write(f"  Macro:        {'Ja' if features.get('macro_indicators') else 'Nein'}\n")
            f.write(f"  Time:         {'Ja' if features.get('time_features') else 'Nein'}\n")
            f.write(f"  Multi-TF:     {'Ja' if features.get('multi_timeframe') else 'Nein'}\n")
            f.write(f"  Selection:    {features.get('feature_selection', '-')}\n")
            if features.get('custom_features'):
                f.write(f"  Custom:       {', '.join(features['custom_features'])}\n")
            f.write("\n")

            # Simulation-Info
            sim = strategy_metadata.get('simulation', {})
            f.write("SIMULATION\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  TP/SL Basis:  {sim.get('tp_sl_basis', '-')}\n")
            f.write(f"  Trailing:     {'Ja' if sim.get('trailing_stop') else 'Nein'}\n")
            f.write(f"  Slippage:     {sim.get('slippage_model', '-')}\n")
            f.write(f"  Regime:       {'Ja' if sim.get('regime_filter') else 'Nein'}\n")
            f.write("\n")

            if strategy_metadata.get('notes'):
                f.write("NOTIZEN\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{strategy_metadata['notes']}\n\n")

        # Technische Konfiguration
        f.write(f"{'='*70}\n")
        f.write("TECHNISCHE KONFIGURATION\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Walk-Forward Folds: {WALK_FORWARD_FOLDS}\n")
        f.write(f"OOS Size:           {OOS_SIZE}\n")
        f.write(f"Min Trades:         {MIN_TRADES}\n")
        f.write(f"Corr Threshold:     {CORR_THRESHOLD}\n")
        f.write("\n")

        # Ergebnisse
        f.write(f"{'='*70}\n")
        f.write("ERGEBNISSE\n")
        f.write(f"{'='*70}\n\n")
        total_processed = len(all_results) if all_results else len(raw_results)
        f.write(f"Assets Processed:   {total_processed}\n")
        f.write(f"Significant (ok):   {len(raw_results)}\n")
        f.write(f"After Corr Filter:  {len(filtered_results)}\n")
        f.write(f"Elite (Top 10):     {len(elite_results)}\n")
        f.write(f"Profitable:         {len(final_assets)}\n")
        f.write("\n")

        if table_data:
            f.write("ELITE ASSETS\n")
            f.write(f"{'-'*120}\n")
            f.write(f"{'Asset':<10} {'Kelly':>8} {'WinRate':>8} {'RRR':>6} {'Sharpe':>7} {'Calmar':>7} {'Trades':>7} {'L/S':>12} {'Return':>10} {'MaxDD':>6} {'p-val':>6} {'Folds':>6} {'Status':>6}\n")
            f.write(f"{'-'*120}\n")

            for row in table_data:
                # row hat 13 Spalten: Asset, Kelly, WinRate, RRR, Sharpe, Calmar, Trades, L/S, Return, MaxDD, p-val, Folds, Status
                f.write(f"{row[0]:<10} {row[1]:>8} {row[2]:>8} {row[3]:>6} {row[4]:>7} {row[5]:>7} {row[6]:>7} {row[7]:>12} {row[8]:>10} {row[9]:>6} {row[10]:>6} {row[11]:>6} {row[12]:>6}\n")

        f.write("\n")
        f.write("PROFITABLE ASSETS\n")
        f.write(f"{'-'*40}\n")
        if final_assets:
            for sym in final_assets.keys():
                f.write(f"  - {sym}\n")
        else:
            f.write("  (keine)\n")


def list_runs(tags=None):
    """
    Listet alle vorhandenen Test-Runs auf.

    Args:
        tags: Filtert nach Tags (optional, Liste)
    """
    if not os.path.exists(RESULTS_BASE_PATH):
        return []

    runs = []
    for run_id in sorted(os.listdir(RESULTS_BASE_PATH), reverse=True):
        run_path = os.path.join(RESULTS_BASE_PATH, run_id)
        if os.path.isdir(run_path):
            run_info = {"run_id": run_id, "path": run_path}

            # Config laden
            config_path = os.path.join(run_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                    run_info["timestamp"] = config.get("timestamp")
                    run_info["description"] = config.get("description")
                    run_info["timeframe"] = config.get("timeframe")
                    run_info["account"] = config.get("account_name")

            # Strategy-Metadaten laden
            strategy_path = os.path.join(run_path, "strategy.json")
            if os.path.exists(strategy_path):
                with open(strategy_path) as f:
                    strategy = json.load(f)
                    run_info["strategy_name"] = strategy.get("name")
                    run_info["tags"] = strategy.get("tags", [])
                    run_info["hypothesis"] = strategy.get("hypothesis")
                    model = strategy.get("model", {})
                    run_info["model_architecture"] = model.get("architecture") if isinstance(model, dict) else None

            # Assets zählen
            assets_path = os.path.join(run_path, "assets.json")
            if os.path.exists(assets_path):
                with open(assets_path) as f:
                    assets = json.load(f)
                    run_info["profitable_count"] = len(assets)

            # Filter anwenden
            if tags:
                run_tags = run_info.get("tags", [])
                if not any(t in run_tags for t in tags):
                    continue

            runs.append(run_info)

    return runs


def load_run(run_id):
    """Lädt die Ergebnisse eines bestimmten Runs."""
    run_path = os.path.join(RESULTS_BASE_PATH, run_id)

    if not os.path.exists(run_path):
        return None

    result = {"run_id": run_id, "path": run_path}

    config_path = os.path.join(run_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            result["config"] = json.load(f)

    strategy_path = os.path.join(run_path, "strategy.json")
    if os.path.exists(strategy_path):
        with open(strategy_path) as f:
            result["strategy"] = json.load(f)

    results_path = os.path.join(run_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            result["results"] = json.load(f)

    assets_path = os.path.join(run_path, "assets.json")
    if os.path.exists(assets_path):
        with open(assets_path) as f:
            result["assets"] = json.load(f)

    return result


def compare_runs(run_ids):
    """Vergleicht mehrere Runs miteinander."""
    runs = [load_run(rid) for rid in run_ids]
    runs = [r for r in runs if r is not None]

    if not runs:
        return None

    comparison = {
        "runs": [],
        "all_symbols": set(),
    }

    for run in runs:
        strategy = run.get("strategy", {})
        run_info = {
            "run_id": run["run_id"],
            "timestamp": run.get("config", {}).get("timestamp"),
            "description": run.get("config", {}).get("description"),
            "timeframe": run.get("config", {}).get("timeframe"),
            "strategy_name": strategy.get("name"),
            "category": strategy.get("category"),
            "hypothesis": strategy.get("hypothesis"),
            "model_architecture": strategy["model"].get("architecture") if isinstance(strategy.get("model"), dict) else None,
            "profitable_count": len(run.get("assets", {})),
            "profitable_symbols": list(run.get("assets", {}).keys()),
        }

        if run.get("results", {}).get("elite_results"):
            elite = run["results"]["elite_results"]
            run_info["elite_stats"] = {
                "avg_sharpe": sum(e.get("sharpe", 0) for e in elite) / len(elite) if elite else 0,
                "avg_win_rate": sum(e.get("win_rate", 0) for e in elite) / len(elite) if elite else 0,
                "total_trades": sum(e.get("trades", 0) for e in elite),
            }

        comparison["runs"].append(run_info)
        comparison["all_symbols"].update(run_info["profitable_symbols"])

    comparison["all_symbols"] = sorted(comparison["all_symbols"])

    return comparison
