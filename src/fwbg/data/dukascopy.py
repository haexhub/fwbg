"""Dukascopy historical-data downloader → fwbg CSV (``T,O,H,L,C,V``).

Wraps `dukascopy-python` (which handles the instrument catalogue, per-instrument
price scaling and the compressed ``.bi5`` tick decoding) and writes ready-to-
backtest CSVs straight into a CSV source's datasource directory — so a download
needs no manual ETL/``prepare`` step afterwards.

Output matches ``CSVSourceConfig.prepare``: columns ``T,O,H,L,C,V`` with
``T`` formatted ``%Y-%m-%d %H:%M:%S`` (UTC) and ``V`` falling back to 0.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import dukascopy_python as dk
import dukascopy_python.instruments as dk_instruments

log = logging.getLogger(__name__)


class DukascopyError(RuntimeError):
    """Raised for unsupported timeframes/instruments or download failures."""


# Friendly timeframe key -> (dukascopy interval const, fwbg filename label).
TIMEFRAMES: dict[str, tuple[str, str]] = {
    "MINUTE_1": (dk.INTERVAL_MIN_1, "MINUTE_1"),
    "MINUTE_5": (dk.INTERVAL_MIN_5, "MINUTE_5"),
    "MINUTE_15": (dk.INTERVAL_MIN_15, "MINUTE_15"),
    "MINUTE_30": (dk.INTERVAL_MIN_30, "MINUTE_30"),
    "HOUR_1": (dk.INTERVAL_HOUR_1, "HOUR_1"),
    "HOUR_4": (dk.INTERVAL_HOUR_4, "HOUR_4"),
    "DAY_1": (dk.INTERVAL_DAY_1, "DAY_1"),
}

def _normalize(symbol: str) -> str:
    return symbol.replace("/", "").replace("_", "").replace("-", "").upper()


def _build_instrument_index() -> dict[str, str]:
    """Map normalized symbol (e.g. ``EURUSD``) -> dukascopy id (e.g. ``EUR/USD``)."""
    index: dict[str, str] = {}
    for name in dir(dk_instruments):
        if not name.startswith("INSTRUMENT_"):
            continue
        value = getattr(dk_instruments, name)
        if isinstance(value, str):
            index.setdefault(_normalize(value), value)
    return index


_INSTRUMENTS = _build_instrument_index()


# Asset-class label derived from the ``INSTRUMENT_<GROUP>_…`` constant-name prefix.
# Anything not listed here (country prefixes like US/UK/GERMANY/…) is a single stock.
_GROUP_LABELS: dict[str, str] = {
    "FX": "Forex",
    "VCCY": "Krypto",
    "CMD": "Rohstoffe",
    "IDX": "Indizes",
    "ETF": "ETF",
    "BND": "Anleihen",
}

# Per-instrument history metadata bundled from the dukascopy-node catalogue:
# normalized symbol -> {description, minute, hourly, daily} (history-start dates).
_META_PATH = Path(__file__).with_name("dukascopy_meta.json")


def _build_group_index() -> dict[str, str]:
    """Map normalized symbol -> friendly asset-class label (e.g. ``Forex``)."""
    groups: dict[str, str] = {}
    for name in dir(dk_instruments):
        if not name.startswith("INSTRUMENT_"):
            continue
        value = getattr(dk_instruments, name)
        if not isinstance(value, str):
            continue
        parts = name.split("_")
        prefix = parts[1] if len(parts) > 1 else ""
        groups.setdefault(_normalize(value), _GROUP_LABELS.get(prefix, "Aktien"))
    return groups


@lru_cache(maxsize=1)
def instrument_catalogue() -> list[dict]:
    """Downloadable instruments joined with their per-timeframe history starts.

    Returns only instruments that are *both* resolvable by the installed
    ``dukascopy-python`` library and present in the bundled history metadata, so
    the UI never offers something that can't be fetched or whose available range
    is unknown. Each entry::

        {
          "symbol": "EURUSD",            # ready to pass straight to download()
          "id": "EUR/USD",
          "description": "Euro vs US Dollar",
          "group": "Forex",
          "historyStart": {"minute": "2003-05-04", "hourly": "2003-05-04",
                           "daily": "1973-03-01"},
        }

    The three ``historyStart`` granularities map onto our timeframes: minute
    candles (MINUTE_*), hourly candles (HOUR_*) and daily candles (DAY_1).
    """
    meta: dict = json.loads(_META_PATH.read_text())
    groups = _build_group_index()
    out: list[dict] = []
    for norm_key, instrument_id in _INSTRUMENTS.items():
        m = meta.get(norm_key)
        if m is None:
            continue  # no history metadata -> not adaptively selectable
        out.append(
            {
                "symbol": norm_key,
                "id": instrument_id,
                "description": m.get("description") or instrument_id,
                "group": groups.get(norm_key, "Aktien"),
                "historyStart": {
                    "minute": m.get("minute"),
                    "hourly": m.get("hourly"),
                    "daily": m.get("daily"),
                },
            }
        )
    out.sort(key=lambda r: (r["group"], r["description"]))
    return out


def available_timeframes() -> list[str]:
    return list(TIMEFRAMES.keys())


def resolve_instrument(symbol: str) -> str:
    """Resolve a user symbol to a dukascopy instrument id. Accepts ``EURUSD``,
    ``EUR/USD``, ``EUR_USD`` or the raw dukascopy id."""
    key = _normalize(symbol)
    if key in _INSTRUMENTS:
        return _INSTRUMENTS[key]
    raise DukascopyError(f"unknown Dukascopy instrument: {symbol!r}")


def _fetch_with_progress(instrument, interval, side, start, end, on_frac):
    """Like ``dk.fetch`` but reports progress in [0, 1] via *on_frac* as bars stream
    in (estimated from how far the latest bar's timestamp has advanced through the
    requested range). Falls back to a plain ``dk.fetch`` (single 0→1 step) if the
    library's internal streaming helpers are unavailable or change.
    """
    import pandas as pd

    stream = getattr(dk, "_stream", None)
    interval_units = getattr(dk, "_interval_units", None)
    cols_for_unit = getattr(dk, "_get_dataframe_columns_for_timeunit", None)
    if stream is None or interval_units is None or cols_for_unit is None:
        df = dk.fetch(instrument, interval, side, start, end)
        on_frac(1.0)
        return df

    start_ms = start.timestamp() * 1000
    span_ms = max(end.timestamp() * 1000 - start_ms, 1.0)
    try:
        columns = cols_for_unit(interval_units[interval])
        rows = []
        last = -1.0
        for row in stream(
            instrument=instrument,
            interval=interval,
            offer_side=side,
            start=start,
            end=end,
            max_retries=7,
            limit=30_000,
        ):
            rows.append(row)
            frac = (row[0] - start_ms) / span_ms
            if frac - last >= 0.01:
                last = frac
                on_frac(min(max(frac, 0.0), 1.0))
        df = pd.DataFrame(data=rows, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        on_frac(1.0)
        return df
    except Exception:  # noqa: BLE001 — degrade gracefully if lib internals change
        log.warning("dukascopy: streaming progress failed, using plain fetch", exc_info=True)
        df = dk.fetch(instrument, interval, side, start, end)
        on_frac(1.0)
        return df


def download(
    out_dir: Path | str,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    offer_side: str = "bid",  # deprecated/ignored: OHLC is now mid = (bid+ask)/2
    manual_spread: float | None = None,
    progress_cb=None,
) -> list[dict]:
    """Download bid+ask bars per symbol, write mid-priced ``{SYMBOL}_{TF}.csv``
    and record a conservative bid-ask spread for backtesting.

    The written OHLC is the unbiased mid price ``(bid+ask)/2``. The spread is
    measured as the **90th percentile** of the per-bar ``ask_close - bid_close``
    gap (a single fixed value that already leans pessimistic, so brief spread
    spikes are roughly covered) and persisted per symbol via
    :func:`fwbg.data.assets.save_asset_spread`. If ``manual_spread`` is given it is
    stored as a user override that wins over the measured value in backtests.
    ``offer_side`` is accepted for backwards compatibility but ignored.

    Returns one result dict per symbol: ``{symbol, file, rows, spread[, warning]}``.
    """
    import numpy as np
    import pandas as pd

    from fwbg.data.assets import save_asset_spread

    if timeframe not in TIMEFRAMES:
        raise DukascopyError(
            f"unsupported timeframe {timeframe!r}; choose from {available_timeframes()}"
        )
    interval, tf_label = TIMEFRAMES[timeframe]

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end <= start:
        raise DukascopyError("end must be after start")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = len(symbols)
    results: list[dict] = []
    for i, raw_symbol in enumerate(symbols):
        instrument = resolve_instrument(raw_symbol)
        clean = _normalize(raw_symbol)
        filename = f"{clean}_{tf_label}.csv"
        dest = out_path / filename

        def report(local: float, phase: str):
            if progress_cb is None:
                return
            overall = (i + min(max(local, 0.0), 1.0)) / total if total else 1.0
            progress_cb(
                {
                    "percent": round(overall * 100, 1),
                    "symbol": clean,
                    "phase": phase,
                    "symbol_index": i + 1,
                    "symbol_total": total,
                }
            )

        log.info("dukascopy: fetching %s %s %s..%s (bid+ask)", instrument, timeframe, start, end)
        # bid ≈ first 45% of this symbol, ask the next 45%, write/spread the last 10%.
        report(0.0, "bid")
        bid = _fetch_with_progress(
            instrument, interval, dk.OFFER_SIDE_BID, start, end,
            lambda f: report(0.45 * f, "bid"),
        )
        report(0.45, "ask")
        ask = _fetch_with_progress(
            instrument, interval, dk.OFFER_SIDE_ASK, start, end,
            lambda f: report(0.45 + 0.45 * f, "ask"),
        )
        report(0.9, "write")

        if bid is None or bid.empty or ask is None or ask.empty:
            results.append(
                {"symbol": clean, "file": filename, "rows": 0, "warning": "no data in range"}
            )
            continue

        # Align both sides on their common timestamps before combining.
        bid, ask = bid.align(ask, join="inner", axis=0)
        if bid.empty:
            results.append(
                {"symbol": clean, "file": filename, "rows": 0,
                 "warning": "no overlapping bid/ask bars"}
            )
            continue

        ts = pd.DatetimeIndex(bid.index)
        ts = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")

        mid = {
            c: (bid[c].to_numpy() + ask[c].to_numpy()) / 2.0
            for c in ("open", "high", "low", "close")
        }
        if "volume" in bid.columns and "volume" in ask.columns:
            volume = (bid["volume"].to_numpy() + ask["volume"].to_numpy()) / 2.0
        else:
            volume = 0

        out = pd.DataFrame(
            {
                "T": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "O": mid["open"],
                "H": mid["high"],
                "L": mid["low"],
                "C": mid["close"],
                "V": volume,
            }
        )
        out.to_csv(dest, index=False)

        # Conservative spread: 90th percentile of the per-bar ask-bid close gap.
        gap = ask["close"].to_numpy() - bid["close"].to_numpy()
        gap = gap[np.isfinite(gap)]
        gap = gap[gap >= 0]
        spread = float(np.percentile(gap, 90)) if gap.size else 0.0
        if spread > 0:
            save_asset_spread(clean, spread, manual=False)
        if manual_spread and manual_spread > 0:
            save_asset_spread(clean, float(manual_spread), manual=True)

        log.info("dukascopy: wrote %s (%d bars, spread_p90=%.6g)", dest, len(out), spread)
        result = {"symbol": clean, "file": filename, "rows": int(len(out)), "spread": spread}
        if manual_spread and manual_spread > 0:
            result["manual_spread"] = float(manual_spread)
        results.append(result)

    return results
