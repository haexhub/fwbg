"""The data files of a strategy run: datasource, timeframe and asset filters.

``select_run_files`` points the module-level data configuration (``DATA_PATH``,
``TIMEFRAME``, its tf config and the worker environment) at the strategy and
returns the CSV files to process. A run that processes several strategies one
after another (a portfolio of sleeves) calls it before each strategy.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from typing import Any

from fwbg.core.config import StrategyConfig
from fwbg_sdk.enums import Timeframe
from fwbg.data import config as data_config


def select_run_files(
    strategy: StrategyConfig,
    strategy_metadata: Mapping[str, Any] | None,
    asset_filter: list[str] | None = None,
) -> list[str] | None:
    """Configure the data globals for *strategy* and return its files (None: nothing to run)."""
    asset_filter = list(asset_filter) if asset_filter else None
    # Datasource aus Strategy (Data-Source-Registry); CLI --data-path hat höchste Priorität
    data_config.use_data_path(strategy.datasource, (strategy_metadata or {}).get("_cli_data_path"))

    # Timeframe aus Strategy übernehmen (überschreibt Modul-Globals)
    if strategy.timeframe:
        # Auf die kanonische Langform normalisieren (HOUR_1, DAY_1, …), damit
        # Config-Lookup, Datei-Globbing und die Worker denselben String sehen.
        # Eine unbekannte Bezeichnung soll laut scheitern statt still auf HOUR
        # zurückzufallen (der frühere Fallback ließ DAY_1 mit oos_size=4000 laufen
        # → 49500 Balken nötig → jeder Run brach als insufficient_data ab).
        tf = Timeframe.from_str(strategy.timeframe).canonical
        # CRITICAL: also set the env var, not just the module globals. Optimizer
        # workers run in a ProcessPoolExecutor; on Python 3.14 the default start
        # method is "forkserver" (not "fork"), so workers re-import data_config
        # fresh and would otherwise fall back to the HOUR defaults — inflating
        # test_period_years ~4x and halving the annualized Sharpe on M15. The
        # re-import reads TIMEFRAME from the environment, so propagate it there.
        os.environ["TIMEFRAME"] = tf
        data_config.TIMEFRAME = tf
        tf_cfg = data_config.resolve_tf_config(tf)
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
    source_tf = None
    if not files:
        from fwbg.data.resample import find_fallback_files

        files, source_tf = find_fallback_files(data_config.DATA_PATH, data_config.TIMEFRAME)
        if files:
            print(f"No {data_config.TIMEFRAME} files found — will resample from {source_tf}")
    # Like TIMEFRAME above: workers re-import data_config, so the env carries it.
    data_config.RESAMPLE_FROM = source_tf
    if source_tf:
        os.environ["RESAMPLE_FROM"] = source_tf
    else:
        os.environ.pop("RESAMPLE_FROM", None)

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
                        ASSET_CONFIG.get(os.path.basename(f).split("_")[0], {}).get("class")
                        in allowed_classes
                        for _ in [1]
                    )
                ]

    # Filter nach bestimmten Assets wenn angegeben
    if asset_filter:
        files = [f for f in files if any(a in f for a in asset_filter)]
    if not files:
        print(f"Keine Dateien für Timeframe {data_config.TIMEFRAME} gefunden!")
        print(f"Verfügbare Dateien: {glob.glob(f'{data_config.DATA_PATH}/*.csv')[:5]}...")
        return None
    return files
