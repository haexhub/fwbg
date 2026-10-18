"""Freeze Dukascopy bid/ask bars using fwbg's existing downloader helpers."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_fingerprint


def adjust_etf(bars: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply dividend adjustment and known splits, preserving Dukascopy OHLC.

    Yahoo Close is split-adjusted. Dukascopy may retain an unadjusted split.
    Only the published cumulative split factor or 1 is admissible; arbitrary
    price fitting is forbidden. Independent closing quotes must agree closely.
    """
    reference = reference.copy()
    reference.index = pd.DatetimeIndex(reference.index).tz_localize(None).normalize()
    reference = reference.loc[~reference.index.duplicated()]
    expected = reference.loc[bars.index.min() : bars.index.max()]
    missing = expected.index.difference(bars.index)
    if len(missing):
        raise ValueError(f"Dukascopy missing {len(missing)} reference trading sessions")
    extra = bars.index.difference(expected.index)
    frame = bars.loc[expected.index].copy()
    splits = reference["Stock Splits"].replace(0, 1)
    future_splits = splits.iloc[::-1].cumprod().iloc[::-1] / splits
    future_splits = future_splits.reindex(frame.index)
    ref = expected
    mid = (frame.bid_close + frame.ask_close) / 2
    raw_error = np.abs(np.log(mid / ref.Close))
    split_error = np.abs(np.log(mid / future_splits / ref.Close))
    split_factor = pd.Series(
        np.where(split_error < raw_error, 1 / future_splits, 1), index=frame.index
    )
    error = abs(mid * split_factor / ref.Close - 1)
    if error.quantile(0.99) > 0.03 or error.max() > 0.10:
        raise ValueError(
            f"ETF reference disagreement: p99={error.quantile(.99):.2%}, max={error.max():.2%}"
        )
    factor = split_factor * ref["Adj Close"] / ref.Close
    for column in frame.columns:
        if column.startswith(("bid_", "ask_")):
            frame[column] *= factor
    return frame, {
        "reference_error_p99": float(error.quantile(0.99)),
        "reference_error_max": float(error.max()),
        "non_reference_session_rows_removed": [str(date.date()) for date in extra],
        "split_adjusted_rows": int(split_factor.ne(1).sum()),
        "adjustment": "Yahoo dividend factor plus independently validated published split factor; Dukascopy OHLC and spreads retained",
    }


def prepare_swing_data(protocol_path: Path) -> dict:
    """Freeze protocol markets' Dukascopy bars and return their data manifest.

    Validate ETF adjustments against Yahoo references, remove flat weekend
    placeholders, and record each market's files or download error on disk.
    """
    from fwbg.data.dukascopy import _fetch_with_progress, dk, resolve_instrument

    protocol = json.loads(protocol_path.read_text())
    output = (protocol_path.parent / protocol["data_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.fromisoformat(protocol["download_start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(protocol["download_end_exclusive"]).replace(tzinfo=timezone.utc)
    manifest: dict = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": file_fingerprint(protocol_path),
        "markets": {},
        "errors": {},
    }
    for market in protocol["markets"]:
        symbol = market["symbol"]
        print(f"Downloading {symbol} bid/ask from Dukascopy", flush=True)
        folder = output / symbol
        folder.mkdir()
        try:
            sides = {}
            for side, code in (("bid", dk.OFFER_SIDE_BID), ("ask", dk.OFFER_SIDE_ASK)):
                raw = _fetch_with_progress(
                    resolve_instrument(market["dukascopy"]),
                    dk.INTERVAL_DAY_1,
                    code,
                    start,
                    end,
                    lambda fraction: None,
                )
                if raw.empty:
                    raise ValueError("empty Dukascopy data")
                raw.to_csv(folder / f"{side}_raw.csv")
                raw.index = (
                    pd.DatetimeIndex(raw.index).tz_convert("UTC").tz_localize(None).normalize()
                )
                raw = raw.loc[
                    (raw.index >= pd.Timestamp(start).tz_localize(None))
                    & (raw.index < pd.Timestamp(end).tz_localize(None))
                ]
                sides[side] = raw[["open", "high", "low", "close"]].add_prefix(f"{side}_")
            if not sides["bid"].index.equals(sides["ask"].index):
                raise ValueError("bid and ask timestamps differ")
            bars = pd.concat(sides.values(), axis=1).sort_index()
            details: dict = {}
            if market["yahoo"]:
                import yfinance as yf

                ref = yf.Ticker(market["yahoo"]).history(
                    start=protocol["download_start"],
                    end=protocol["download_end_exclusive"],
                    auto_adjust=False,
                    actions=True,
                    repair=False,
                )
                ref.to_csv(folder / "yahoo_reference.csv")
                bars, details = adjust_etf(bars, ref)
            # Flat weekend bars are feed placeholders, not executable sessions.
            mid = bars[["bid_open", "bid_high", "bid_low", "bid_close"]]
            flat_weekend = (bars.index.dayofweek >= 5) & mid.max(axis=1).eq(mid.min(axis=1))
            details["flat_weekend_rows_removed"] = int(flat_weekend.sum())
            bars = bars.loc[~flat_weekend]
            details["weekend_bars"] = int((bars.index.dayofweek >= 5).sum())
            bars.rename_axis("date").to_csv(folder / "bars.csv")
            manifest["markets"][symbol] = {
                **details,
                "start": str(bars.index[0].date()),
                "end": str(bars.index[-1].date()),
                "rows": len(bars),
                "files": {p.name: file_fingerprint(p) for p in folder.glob("*.csv")},
            }
            print(f"Prepared {symbol}: {len(bars)} bars", flush=True)
        except Exception as error:
            manifest["errors"][symbol] = f"{type(error).__name__}: {error}"
            print(f"DATA EXCLUDED {symbol}: {error}", flush=True)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
    manifest["status"] = "complete" if not manifest["errors"] else "complete_with_data_exclusions"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    prepare_swing_data(args.protocol)
