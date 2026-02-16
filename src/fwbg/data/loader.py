"""
Daten laden und Makro-Indikatoren integrieren
"""
import os
import pandas as pd
import numpy as np

from .config import TARGET_TZ, DATA_PATH


def _has_header(path):
    """Prüft, ob die CSV-Datei einen Header hat."""
    with open(path, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:
            return True  # Leere Datei - pandas default
        first_col = first_line.split(',')[0]
        # Ein Header hat typischerweise Text wie "Time", "Date", "Open" etc.
        # Kein Header hat Datum wie "2024-01-01" oder Zahl
        # Datum erkennen: enthält "-" und besteht aus digits/hyphens
        is_date = '-' in first_col and all(c in '0123456789- :' for c in first_col)
        return not is_date  # Hat Header wenn NICHT Datum


def _validate_ohlc(df, path):
    """Validate OHLC data after loading."""
    sym = os.path.basename(path).split("_")[0]
    ohlc = ["O", "H", "L", "C"]
    present = [c for c in ohlc if c in df.columns]

    # Check OHLC columns are numeric
    for col in present:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{sym}: Column '{col}' is not numeric (dtype={df[col].dtype})")

    if present:
        # Check for inf values
        numeric_df = df[present]
        inf_counts = np.isinf(numeric_df).sum()
        has_inf = inf_counts[inf_counts > 0]
        if len(has_inf) > 0:
            raise ValueError(f"{sym}: OHLC contains inf values: {has_inf.to_dict()}")

        # Check for non-positive prices
        min_vals = numeric_df.min()
        non_positive = min_vals[min_vals <= 0]
        if len(non_positive) > 0:
            raise ValueError(f"{sym}: OHLC contains non-positive values: {non_positive.to_dict()}")


def load_data_aligned(path, is_sentiment=False):
    """Lädt OHLC-Daten aus CSV mit Zeitzone-Alignment."""
    try:
        # Prüfe ob Header vorhanden
        has_header = _has_header(path)
        df_raw = pd.read_csv(path, header=0 if has_header else None)

        # Bei Header: erste Spalte ist Index 0
        # Ohne Header: prüfe ob erste Spalte numerisch ist (Index-Spalte)
        start = 0
        if not has_header:
            # Ohne Header: erste Spalte ist immer Datum
            start = 0
        else:
            # Mit Header: prüfe ob erste Datenspalte numerisch ist
            start = 1 if str(df_raw.iloc[0, 0]).isdigit() else 0

        if len(df_raw.columns) >= start + 6:
            # 6 Spalten: T, O, H, L, C, V
            df = df_raw.iloc[
                :, [start, start + 1, start + 2, start + 3, start + 4, start + 5]
            ].copy()
            df.columns = ["T", "O", "H", "L", "C", "V"]
        elif len(df_raw.columns) >= start + 5:
            # 5 Spalten: T, O, H, L, C (kein Volume)
            df = df_raw.iloc[
                :, [start, start + 1, start + 2, start + 3, start + 4]
            ].copy()
            df.columns = ["T", "O", "H", "L", "C"]
            df["V"] = 0  # Dummy Volume
        else:
            df = df_raw.iloc[:, [start, start + 1]].copy()
            df.columns = ["T", "C"]
            df["O"] = df["H"] = df["L"] = df["C"]
        df["T"] = pd.to_datetime(df["T"])
        if is_sentiment:
            if df["T"].dt.tz is None:
                df["T"] = df["T"].dt.tz_localize("UTC")
            df["T"] = df["T"].dt.tz_convert(TARGET_TZ)
        else:
            # Keine TZ-Lokalisierung - behandle Daten als naive Timestamps
            # Das vermeidet DST-Probleme (ambiguous/nonexistent times)
            pass
        # Stelle sicher dass der Index keine TZ hat
        if df["T"].dt.tz is not None:
            df["T"] = df["T"].dt.tz_localize(None)
        df = df.set_index("T")

        # Validate OHLC data
        _validate_ohlc(df, path)

        return df
    except Exception as e:
        print(f"Fehler beim Laden von {path}: {e}")
        return None


def load_macro_csv(path):
    """
    Lädt eine Makro-CSV-Datei mit flexibler Spalten-Erkennung.
    Unterstützt: DATE, Datetime, Time als Index-Spalte.
    """
    if not os.path.exists(path):
        return None

    try:
        raw_df = pd.read_csv(path, nrows=1)
        cols = list(raw_df.columns)

        # Finde Datums-Spalte (case-insensitive)
        date_col = None
        for candidate in ["DATE", "Datetime", "datetime", "Time", "time", "Date"]:
            if candidate in cols:
                date_col = candidate
                break

        if not date_col:
            return None

        macro_df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
        return macro_df
    except Exception:
        return None


# ============================================================================
# Generic Data Loading Orchestrator
# ============================================================================

def run_data_loading(df, data_loading_configs):
    """
    Generic data-loading orchestrator.

    For each config:
    1. Resolve DataSource (source → CSV/REST/DB)
    2. Call source.load(items) → LoadResult
    3. Align raw data to DataFrame index (Daily→Intraday, Forward-Fill)
    4. Add base columns to DataFrame
    5. Call DATA_LOADING plugin → Computation
    """
    import logging
    from fwbg.core.data_sources import get_data_source
    from fwbg.core.registry import get_data_loader
    from fwbg.pipeline.context import PipelineContext

    log = logging.getLogger(__name__)

    if not data_loading_configs:
        return df

    for cfg in data_loading_configs:
        source_name = cfg.get("source")
        plugin_name = cfg.get("name")
        params = cfg.get("params", {})
        items = params.get("indicators", {})

        # No indicators specified → use plugin defaults
        if not items and plugin_name:
            try:
                cls = get_data_loader(plugin_name)
                defaults = cls().get_default_params()
                items = defaults.get("indicators", {})
                params["indicators"] = items
            except ValueError:
                pass

        # 1. Load raw data from DataSource
        if source_name and items:
            source = get_data_source(source_name)
            result = source.load(items)

            # 2. Align daily data to intraday index via forward-fill
            # IMPORTANT: Use PREVIOUS day's close to prevent lookahead bias.
            # On day D, the daily close is only available after market close,
            # so intraday bars on day D must use day D-1's value.
            date_series = df.index.date
            prev_date = pd.Series(date_series, index=df.index).map(
                lambda d: pd.Timestamp(d) - pd.Timedelta(days=1)
            )

            for prefix, raw_df in result.data.items():
                if "Close" in raw_df.columns:
                    lookup = raw_df["Close"].to_dict()
                    df[f"macro_{prefix}"] = prev_date.map(
                        lambda d, lk=lookup: lk.get(d, np.nan)
                    ).ffill()

            log.debug(
                f"Loaded {len(result.data)} items from source '{source_name}'"
            )

        # 3. Run computation plugin
        if plugin_name:
            try:
                cls = get_data_loader(plugin_name)
                ctx = PipelineContext(df=df, symbol="", asset_class="")
                ctx = cls().execute(ctx, **params)
                df = ctx.df
            except ValueError:
                log.debug(f"DataLoader plugin '{plugin_name}' not found, skipping")

    return df
