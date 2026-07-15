"""Chart data endpoints — OHLCV data and indicator computation for charting UI."""
import inspect
import math
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from fwbg.data.resample import (
    TIMEFRAME_ORDER as _TIMEFRAME_ORDER,
    resample_ohlcv as _resample_ohlcv,
    parse_symbol_timeframe as _parse_symbol_timeframe,
)

router = APIRouter(prefix="/chart", tags=["chart"])


def _call_plugin_method(method, params):
    """Call a plugin column method, forwarding params if the method accepts them."""
    sig = inspect.signature(method)
    if len(sig.parameters) > 0 and params is not None:
        return method(params)
    return method()


# ---------------------------------------------------------------------------
# Timeframe hierarchy & resampling
# ---------------------------------------------------------------------------


def _derivable_timeframes(native_tfs: list[str]) -> list[str]:
    """Given a list of native (on-disk) timeframes, return all timeframes that
    can be produced — native ones plus any higher timeframes derivable via
    resampling from the lowest available native timeframe."""
    if not native_tfs:
        return []

    # Find lowest native timeframe index
    indices = [_TIMEFRAME_ORDER.index(tf) for tf in native_tfs if tf in _TIMEFRAME_ORDER]
    if not indices:
        return native_tfs  # unknown timeframes, return as-is

    lowest_idx = min(indices)
    # All timeframes at or above the lowest native timeframe are producible
    result = []
    for i, tf in enumerate(_TIMEFRAME_ORDER):
        if i >= lowest_idx:
            result.append(tf)
    return result


def _forward_fill_to_chart_tf(indicator_df, chart_df, columns: list[str]):
    """Forward-fill indicator columns from a higher timeframe onto the chart timeframe.

    Each higher-TF value is held constant for all lower-TF bars until the next
    higher-TF bar produces a new value.
    """
    for col in columns:
        if col in indicator_df.columns:
            chart_df[col] = indicator_df[col].reindex(chart_df.index, method="ffill")
    return chart_df


def _best_native_file(ds, symbol: str, target_tf: str):
    """Find the best native file to load for a given target timeframe.
    Prefers exact match, then the lowest available timeframe that can be
    resampled up to the target."""
    # Try exact match first
    path = ds.get_file_path(symbol, target_tf)
    if path.exists():
        return path, target_tf

    target_idx = _TIMEFRAME_ORDER.index(target_tf) if target_tf in _TIMEFRAME_ORDER else -1
    if target_idx < 0:
        return None, None

    # Find lowest available native timeframe below the target
    best_path, best_tf = None, None
    for tf in _TIMEFRAME_ORDER:
        tf_idx = _TIMEFRAME_ORDER.index(tf)
        if tf_idx >= target_idx:
            break
        candidate = ds.get_file_path(symbol, tf)
        if candidate.exists():
            if best_path is None:
                best_path, best_tf = candidate, tf

    return best_path, best_tf


# ---------------------------------------------------------------------------
# GET /api/chart/sources — list available CSV data sources with symbols
# ---------------------------------------------------------------------------

@router.get("/sources")
def list_sources() -> list[dict]:
    """List available data sources with their symbols and timeframes."""
    from fwbg.core.data_sources import get_all_data_sources, CSVSourceConfig
    from fwbg.data.assets import AssetRegistry

    registry = AssetRegistry()
    sources = get_all_data_sources()
    result = []

    for name, source in sources.items():
        if not isinstance(source, CSVSourceConfig):
            continue
        if not source.exists():
            continue

        # Scan directory for {SYMBOL}_{TIMEFRAME}.csv files
        symbols_map: dict[str, list[str]] = defaultdict(list)
        for f in source.list_files("*.csv"):
            parsed = _parse_symbol_timeframe(f.stem)
            if not parsed:
                continue
            symbol, timeframe = parsed
            symbols_map[symbol].append(timeframe)

        symbols_list = []
        for symbol in sorted(symbols_map):
            asset = registry.get(symbol)
            native_tfs = symbols_map[symbol]
            all_tfs = _derivable_timeframes(native_tfs)
            symbols_list.append({
                "symbol": symbol,
                "timeframes": all_tfs,
                "asset_class": asset.asset_class,
                "point": asset.point,
                "spread": asset.spread,
            })

        result.append({
            "name": name,
            "type": "csv",
            "description": source.description,
            "symbols": symbols_list,
        })

    return result


# ---------------------------------------------------------------------------
# GET /api/chart/ohlcv — load OHLCV data from a CSV data source
# ---------------------------------------------------------------------------

def _resolve_source(source: Optional[str]) -> str:
    """Resolve the data-source name when the client sent none.

    Exactly one configured source → use it. None or several → 422 with the
    configured names; a frozen default here would break on every workspace
    whose source is named differently.
    """
    if source:
        return source
    from fwbg.core.data_sources import list_data_sources

    names = list_data_sources()
    if len(names) == 1:
        return names[0]
    if not names:
        raise HTTPException(
            422, "No 'source' specified and no data sources are configured."
        )
    raise HTTPException(
        422,
        f"No 'source' specified and {len(names)} data sources are configured "
        f"({', '.join(names)}); pass 'source' explicitly.",
    )


def _safe_float(v) -> Optional[float]:
    """Convert value to float, returning None for NaN/inf."""
    if v is None:
        return None
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


@router.get("/ohlcv")
def get_ohlcv(
    symbol: str = Query(...),
    timeframe: str = Query("HOUR"),
    source: Optional[str] = Query(None),
    limit: int = Query(5000, le=999999),
    offset: int = Query(0, ge=0),
    drop_flat_bars: bool = Query(False),
) -> dict:
    """Load OHLCV data from a CSV data source for charting."""
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    from fwbg.data.loader import load_data_aligned

    source = _resolve_source(source)
    try:
        ds = get_data_source(source)
    except ValueError as e:
        raise HTTPException(404, str(e))

    if not isinstance(ds, CSVSourceConfig):
        raise HTTPException(400, f"Source '{source}' is not a CSV source. Use POST for broker data.")

    # Try exact file first, then find a lower timeframe to resample from
    path = ds.get_file_path(symbol, timeframe)
    native_tf = timeframe
    if not path.exists():
        path, native_tf = _best_native_file(ds, symbol, timeframe)
        if not path:
            raise HTTPException(404, f"Data file not found: {symbol}_{timeframe} in {source}")

    df = load_data_aligned(str(path))
    if df is None or df.empty:
        raise HTTPException(500, f"Failed to load data: {symbol}_{timeframe}")

    # Drop flat bars (O==H==L==C, weekends/holidays for index data)
    if drop_flat_bars:
        df = df[~((df["O"] == df["H"]) & (df["H"] == df["L"]) & (df["L"] == df["C"]))]

    # Resample if we loaded a lower timeframe
    if native_tf != timeframe:
        df = _resample_ohlcv(df, timeframe)

    total = len(df)
    end_idx = total - offset
    start_idx = max(0, end_idx - limit)
    df_slice = df.iloc[start_idx:end_idx]

    data = []
    for ts, row in df_slice.iterrows():
        entry = {
            "timestamp": int(ts.timestamp() * 1000),
            "open": _safe_float(row["O"]),
            "high": _safe_float(row["H"]),
            "low": _safe_float(row["L"]),
            "close": _safe_float(row["C"]),
        }
        if "V" in row.index:
            entry["volume"] = _safe_float(row["V"])
        data.append(entry)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "total": total,
        "count": len(data),
        "data": data,
    }


# ---------------------------------------------------------------------------
# POST /api/chart/ohlcv — load OHLCV via broker adapter (credentials in body)
# ---------------------------------------------------------------------------

class BrokerOhlcvRequest(BaseModel):
    symbol: str
    timeframe: str = "HOUR"
    limit: int = 5000
    broker_type: str = "ig"
    credentials: dict


@router.post("/ohlcv")
def get_ohlcv_broker(body: BrokerOhlcvRequest) -> dict:
    """Load OHLCV data from a broker adapter using provided credentials."""
    from fwbg_sdk import Symbol, Timeframe

    # Map timeframe string to Timeframe enum
    tf_map = {
        "MINUTE_1": Timeframe.M1,
        "MINUTE_5": Timeframe.M5,
        "MINUTE_15": Timeframe.M15,
        "MINUTE_30": Timeframe.M30,
        "HOUR": Timeframe.H1,
        "DAY": Timeframe.D1,
    }
    tf = tf_map.get(body.timeframe)
    if tf is None:
        raise HTTPException(400, f"Unsupported timeframe: {body.timeframe}")

    # Map symbol string to Symbol enum
    try:
        sym = Symbol[body.symbol]
    except KeyError:
        raise HTTPException(400, f"Unknown symbol: {body.symbol}")

    # Create broker adapter
    adapter = _create_broker_adapter(body.broker_type, body.credentials)

    try:
        adapter.connect()
        df = adapter.get_historical_bars(sym, tf, limit=body.limit)
    except Exception as e:
        raise HTTPException(500, f"Broker data fetch failed: {e}")
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass

    if df is None or df.empty:
        raise HTTPException(404, f"No data returned for {body.symbol} {body.timeframe}")

    total = len(df)
    data = []
    for ts, row in df.iterrows():
        entry = {
            "timestamp": int(ts.timestamp() * 1000),
            "open": _safe_float(row["O"]),
            "high": _safe_float(row["H"]),
            "low": _safe_float(row["L"]),
            "close": _safe_float(row["C"]),
        }
        if "V" in row.index:
            entry["volume"] = _safe_float(row["V"])
        data.append(entry)

    return {
        "symbol": body.symbol,
        "timeframe": body.timeframe,
        "total": total,
        "count": len(data),
        "data": data,
    }


def _create_broker_adapter(broker_type: str, credentials: dict):
    """Create a broker adapter from type string and credentials dict."""
    if broker_type == "ig":
        from fwbg_broker_ig import IGBrokerAdapter
        return IGBrokerAdapter(
            username=credentials["username"],
            password=credentials["password"],
            api_key=credentials["api_key"],
            env=credentials.get("acc_type", "DEMO"),
        )
    raise HTTPException(400, f"Unsupported broker type: {broker_type}")


# ---------------------------------------------------------------------------
# POST /api/chart/indicator — compute indicator on OHLCV data
# ---------------------------------------------------------------------------

class IndicatorRequest(BaseModel):
    symbol: str
    timeframe: str = "HOUR"
    source: Optional[str] = None
    fqn: str
    params: Optional[dict] = None
    limit: int = 5000
    offset: int = 0
    credentials: Optional[dict] = None
    broker_type: Optional[str] = None
    indicator_timeframe: Optional[str] = None
    drop_flat_bars: bool = False


@router.post("/indicator")
def compute_indicator(body: IndicatorRequest) -> dict:
    """Compute a single indicator plugin on OHLCV data for chart overlay."""
    from fwbg.api.deps import get_plugin_registry
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    from fwbg.data.loader import load_data_aligned

    # --- Load OHLCV data ---
    if body.credentials and body.broker_type:
        # Broker source
        from fwbg_sdk import Symbol, Timeframe
        tf_map = {
            "MINUTE_1": Timeframe.M1, "MINUTE_5": Timeframe.M5,
            "MINUTE_15": Timeframe.M15, "MINUTE_30": Timeframe.M30,
            "HOUR": Timeframe.H1, "DAY": Timeframe.D1,
        }
        tf = tf_map.get(body.timeframe)
        if tf is None:
            raise HTTPException(400, f"Unsupported timeframe: {body.timeframe}")
        try:
            sym = Symbol[body.symbol]
        except KeyError:
            raise HTTPException(400, f"Unknown symbol: {body.symbol}")

        adapter = _create_broker_adapter(body.broker_type, body.credentials)
        try:
            adapter.connect()
            df = adapter.get_historical_bars(sym, tf, limit=body.limit + body.offset)
        except Exception as e:
            raise HTTPException(500, f"Broker data fetch failed: {e}")
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass
    else:
        # CSV source
        source = _resolve_source(body.source)
        try:
            ds = get_data_source(source)
        except ValueError as e:
            raise HTTPException(404, str(e))
        if not isinstance(ds, CSVSourceConfig):
            raise HTTPException(400, f"Source '{source}' is not CSV. Provide credentials for broker.")
        path = ds.get_file_path(body.symbol, body.timeframe)
        native_tf = body.timeframe
        if not path.exists():
            path, native_tf = _best_native_file(ds, body.symbol, body.timeframe)
            if not path:
                raise HTTPException(404, f"Data not found: {body.symbol}_{body.timeframe}")
        df = load_data_aligned(str(path))
        if native_tf != body.timeframe and df is not None and not df.empty:
            df = _resample_ohlcv(df, body.timeframe)

    if df is None or df.empty:
        raise HTTPException(500, "Failed to load data")

    # Drop flat bars (O==H==L==C, weekends/holidays for index data)
    if body.drop_flat_bars:
        df = df[~((df["O"] == df["H"]) & (df["H"] == df["L"]) & (df["L"] == df["C"]))]

    # --- MTF: load indicator-timeframe data if different from chart TF ---
    chart_df = None
    ind_tf = body.indicator_timeframe
    if ind_tf and ind_tf != body.timeframe:
        chart_idx = _TIMEFRAME_ORDER.index(body.timeframe) if body.timeframe in _TIMEFRAME_ORDER else -1
        ind_idx = _TIMEFRAME_ORDER.index(ind_tf) if ind_tf in _TIMEFRAME_ORDER else -1
        if ind_idx < 0 or chart_idx < 0:
            raise HTTPException(400, f"Unknown timeframe: {ind_tf}")
        if ind_idx <= chart_idx:
            raise HTTPException(400, f"indicator_timeframe ({ind_tf}) must be higher than chart timeframe ({body.timeframe})")
        chart_df = df
        if body.credentials and body.broker_type:
            raise HTTPException(501, "MTF via broker not yet supported")
        else:
            ind_path = ds.get_file_path(body.symbol, ind_tf)
            ind_native_tf = ind_tf
            if not ind_path.exists():
                ind_path, ind_native_tf = _best_native_file(ds, body.symbol, ind_tf)
                if not ind_path:
                    raise HTTPException(404, f"Data not found for indicator timeframe: {body.symbol}_{ind_tf}")
            ind_df = load_data_aligned(str(ind_path))
            if ind_native_tf != ind_tf and ind_df is not None and not ind_df.empty:
                ind_df = _resample_ohlcv(ind_df, ind_tf)
            if ind_df is None or ind_df.empty:
                raise HTTPException(500, "Failed to load indicator-timeframe data")
            df = ind_df

    # --- Get plugin and compute ---
    registry = get_plugin_registry()
    try:
        plugin_cls = registry.get(body.fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {body.fqn}")

    plugin = plugin_cls()
    params = body.params or plugin.get_default_params()

    # Convert point-based params from points (user input) to raw price.
    # Users enter e.g. "6" meaning 6 points; multiply by the asset's point size.
    point_params = ["breakout_threshold_abs", "min_range_height"]
    if any(params.get(p, 0) > 0 for p in point_params):
        from fwbg.data.assets import AssetRegistry
        asset = AssetRegistry().get(body.symbol)
        for p in point_params:
            if params.get(p, 0) > 0:
                params[p] = params[p] * asset.point

    result_df = plugin.compute(df, **params)

    # --- Extract feature columns ---
    feature_cols = _call_plugin_method(plugin.get_feature_columns, params)
    available_cols = [c for c in feature_cols if c in result_df.columns and not c.startswith("_")]

    # --- Overlay columns (price-scale, for main chart) ---
    overlay_cols = []
    if hasattr(plugin, "get_overlay_columns"):
        overlay_cols = [c for c in _call_plugin_method(plugin.get_overlay_columns, params)
                        if c in result_df.columns]

    # --- Classify columns via plugin methods (needed before undo-shift) ---
    plugin_signal_cols = set(_call_plugin_method(plugin.get_signal_columns, params)) if hasattr(plugin, "get_signal_columns") else set()

    # --- Adjust shift for chart display ---
    # Non-signal columns: undo shift_features (+1) so values appear at the
    # bar where they were computed.
    # Signal columns: shift +1 so signal markers align with trade entries
    # (signal fires at bar N close, entry at bar N+1 open → show at N+1).
    # Signal columns are NOT shifted by the indicator (no lookahead shift
    # needed since simulate_pro_trade already enters at idx+1).
    for col in available_cols:
        if col in plugin_signal_cols:
            result_df[col] = result_df[col].shift(1)
        else:
            result_df[col] = result_df[col].shift(-1)
    for col in overlay_cols:
        result_df[col] = result_df[col].shift(-1)

    # --- MTF: forward-fill indicator columns to chart timeframe ---
    # Continuous columns (plot + overlay) are forward-filled so each lower-TF bar
    # holds the last higher-TF value. Signal columns are NOT forward-filled —
    # they only appear at the higher-TF bar boundary (point values, not blocks).
    if chart_df is not None:
        # MTF: all indicator columns (including signals) are forward-filled
        # onto the chart timeframe.  A daily supertrend value of 1.0 is valid
        # for every M15 bar of that day, not just at midnight.
        all_ind_cols = list(set(available_cols + overlay_cols))
        _forward_fill_to_chart_tf(result_df, chart_df, all_ind_cols)
        result_df = chart_df

    # --- Slice ---
    # Cap total cells (columns × bars) to ~3M to keep JSON response
    # under ~50 MB (Node.js proxy limit).  With 160 columns this allows
    # ~18k bars; with 5 columns up to 600k bars.
    MAX_CELLS = 3_000_000
    n_cols = max(1, len(available_cols) + len(overlay_cols))
    max_bars = MAX_CELLS // n_cols
    effective_limit = min(body.limit, max_bars)
    total = len(result_df)
    end_idx = total - body.offset
    start_idx = max(0, end_idx - effective_limit)
    result_slice = result_df.iloc[start_idx:end_idx]
    plugin_plot_cols = set(_call_plugin_method(plugin.get_plot_columns, params)) if hasattr(plugin, "get_plot_columns") else set()

    # --- Build response ---
    columns_data = {}
    plot_columns = []
    signal_columns = []
    for col in available_cols:
        values = result_slice[col].tolist()
        clean = [
            None if (v != v or v == float("inf") or v == float("-inf")) else float(v)
            for v in values
        ]
        columns_data[col] = clean

        # Use plugin classification if available, fall back to heuristic
        if plugin_signal_cols or plugin_plot_cols:
            if col in plugin_signal_cols:
                signal_columns.append(col)
            else:
                plot_columns.append(col)
        else:
            # Fallback heuristic for plugins without get_signal_columns()
            unique_vals = {v for v in clean if v is not None}
            if not unique_vals <= {-1.0, 0.0, 1.0}:
                plot_columns.append(col)
            elif unique_vals:
                signal_columns.append(col)

    timestamps = [int(ts.timestamp() * 1000) for ts in result_slice.index]

    # --- Overlay data (absolute price-scale values for main chart) ---
    overlay_data = {}
    for col in overlay_cols:
        values = result_slice[col].tolist()
        overlay_data[col] = [
            None if (v != v or v == float("inf") or v == float("-inf")) else float(v)
            for v in values
        ]

    response = {
        "fqn": body.fqn,
        "columns": available_cols,
        "plot_columns": plot_columns,
        "signal_columns": signal_columns,
        "overlay_columns": overlay_cols,
        "overlay_data": overlay_data,
        "timestamps": timestamps,
        "data": columns_data,
    }

    # Include range zones if the indicator provides them (e.g. Opening Range)
    raw_zones = getattr(plugin, "_range_zones", None)
    if raw_zones:
        ts_start = timestamps[0] if timestamps else 0
        ts_end = timestamps[-1] if timestamps else 0
        response["range_zones"] = [
            z for z in raw_zones
            if z["end_ts"] >= ts_start and z["start_ts"] <= ts_end
        ]

    return response
