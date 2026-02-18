"""Chart data endpoints — OHLCV data and indicator computation for charting UI."""
import math
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/chart", tags=["chart"])


# ---------------------------------------------------------------------------
# Timeframe hierarchy & resampling
# ---------------------------------------------------------------------------

# Ordered from lowest to highest resolution
_TIMEFRAME_ORDER = ["MINUTE_1", "MINUTE_5", "MINUTE_15", "MINUTE_30", "HOUR", "HOUR_4", "DAY"]

# Pandas resample rule for each timeframe
_RESAMPLE_RULE: dict[str, str] = {
    "MINUTE_1": "1min",
    "MINUTE_5": "5min",
    "MINUTE_15": "15min",
    "MINUTE_30": "30min",
    "HOUR": "1h",
    "HOUR_4": "4h",
    "DAY": "1D",
}


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


def _resample_ohlcv(df, target_tf: str):
    """Resample an OHLCV DataFrame to a higher timeframe."""
    import pandas as pd

    rule = _RESAMPLE_RULE.get(target_tf)
    if not rule:
        return df

    agg = {"O": "first", "H": "max", "L": "min", "C": "last"}
    if "V" in df.columns:
        agg["V"] = "sum"

    resampled = df.resample(rule).agg(agg).dropna(subset=["O"])
    return resampled


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
            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            symbol, timeframe = parts
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
    source: str = Query("forexsb"),
    limit: int = Query(5000, le=50000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Load OHLCV data from a CSV data source for charting."""
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    from fwbg.data.loader import load_data_aligned

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
    source: str = "forexsb"
    fqn: str
    params: Optional[dict] = None
    limit: int = 5000
    offset: int = 0
    credentials: Optional[dict] = None
    broker_type: Optional[str] = None


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
        try:
            ds = get_data_source(body.source)
        except ValueError as e:
            raise HTTPException(404, str(e))
        if not isinstance(ds, CSVSourceConfig):
            raise HTTPException(400, f"Source '{body.source}' is not CSV. Provide credentials for broker.")
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

    # --- Get plugin and compute ---
    registry = get_plugin_registry()
    try:
        plugin_cls = registry.get(body.fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {body.fqn}")

    plugin = plugin_cls()
    params = body.params or plugin.get_default_params()
    result_df = plugin.compute(df, **params)

    # --- Extract feature columns ---
    feature_cols = plugin.get_feature_columns()
    available_cols = [c for c in feature_cols if c in result_df.columns and not c.startswith("_")]

    # --- Undo shift_features() for chart display ---
    # Indicators shift by +1 for ML lookahead prevention.
    # For charting, we want the value at the bar it belongs to.
    for col in available_cols:
        result_df[col] = result_df[col].shift(-1)

    # --- Slice ---
    total = len(result_df)
    end_idx = total - body.offset
    start_idx = max(0, end_idx - body.limit)
    result_slice = result_df.iloc[start_idx:end_idx]

    # --- Build response ---
    columns_data = {}
    plot_columns = []
    for col in available_cols:
        values = result_slice[col].tolist()
        clean = [
            None if (v != v or v == float("inf") or v == float("-inf")) else float(v)
            for v in values
        ]
        columns_data[col] = clean

        # Classify: continuous columns are plottable; binary/tri-state signals are not
        unique_vals = {v for v in clean if v is not None}
        if not unique_vals <= {-1.0, 0.0, 1.0}:
            plot_columns.append(col)

    timestamps = [int(ts.timestamp() * 1000) for ts in result_slice.index]

    return {
        "fqn": body.fqn,
        "columns": available_cols,
        "plot_columns": plot_columns,
        "timestamps": timestamps,
        "data": columns_data,
    }
