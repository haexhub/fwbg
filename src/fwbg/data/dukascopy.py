"""Dukascopy historical-data downloader → fwbg CSV (``T,O,H,L,C,V``).

Wraps `dukascopy-python` (which handles the instrument catalogue, per-instrument
price scaling and the compressed ``.bi5`` tick decoding) and writes ready-to-
backtest CSVs straight into a CSV source's datasource directory — so a download
needs no manual ETL/``prepare`` step afterwards.

Output matches ``CSVSourceConfig.prepare``: columns ``T,O,H,L,C,V`` with
``T`` formatted ``%Y-%m-%d %H:%M:%S`` (UTC) and ``V`` falling back to 0.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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

_OFFER_SIDES: dict[str, str] = {
    "bid": dk.OFFER_SIDE_BID,
    "ask": dk.OFFER_SIDE_ASK,
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


def available_timeframes() -> list[str]:
    return list(TIMEFRAMES.keys())


def resolve_instrument(symbol: str) -> str:
    """Resolve a user symbol to a dukascopy instrument id. Accepts ``EURUSD``,
    ``EUR/USD``, ``EUR_USD`` or the raw dukascopy id."""
    key = _normalize(symbol)
    if key in _INSTRUMENTS:
        return _INSTRUMENTS[key]
    raise DukascopyError(f"unknown Dukascopy instrument: {symbol!r}")


def download(
    out_dir: Path | str,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    offer_side: str = "bid",
) -> list[dict]:
    """Download OHLC bars for each symbol and write ``{SYMBOL}_{TF}.csv``.

    Returns one result dict per symbol: ``{symbol, file, rows[, warning]}``.
    """
    import pandas as pd

    if timeframe not in TIMEFRAMES:
        raise DukascopyError(
            f"unsupported timeframe {timeframe!r}; choose from {available_timeframes()}"
        )
    interval, tf_label = TIMEFRAMES[timeframe]

    side = _OFFER_SIDES.get(offer_side.lower())
    if side is None:
        raise DukascopyError(f"unsupported offer_side {offer_side!r}; use 'bid' or 'ask'")

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end <= start:
        raise DukascopyError("end must be after start")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for raw_symbol in symbols:
        instrument = resolve_instrument(raw_symbol)
        clean = _normalize(raw_symbol)
        filename = f"{clean}_{tf_label}.csv"
        dest = out_path / filename

        log.info("dukascopy: fetching %s %s %s..%s", instrument, timeframe, start, end)
        df = dk.fetch(instrument, interval, side, start, end)

        if df is None or df.empty:
            results.append(
                {"symbol": clean, "file": filename, "rows": 0, "warning": "no data in range"}
            )
            continue

        idx = df.index
        ts = pd.DatetimeIndex(idx)
        ts = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")
        out = pd.DataFrame(
            {
                "T": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "O": df["open"].to_numpy(),
                "H": df["high"].to_numpy(),
                "L": df["low"].to_numpy(),
                "C": df["close"].to_numpy(),
                "V": df["volume"].to_numpy() if "volume" in df.columns else 0,
            }
        )
        out.to_csv(dest, index=False)
        log.info("dukascopy: wrote %s (%d bars)", dest, len(out))
        results.append({"symbol": clean, "file": filename, "rows": int(len(out))})

    return results
