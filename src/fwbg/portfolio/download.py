"""Optional Yahoo/ECB research-data preparation; never called by a backtest.

The five US-listed funds are historical asset-class proxies, not a claim that
these exact products can be bought through a particular retail broker.
"""

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .data import file_fingerprint
from .engine import validate_daily_index

ASSETS = {
    "SPY": "US equities",
    "EFA": "Developed equities excluding US/Canada",
    "EEM": "Emerging-market equities",
    "IEF": "US Treasury bonds, 7-10 years",
    "GLD": "Gold",
}
ECB_SERIES = {
    "fx": "EXR/D.USD.EUR.SP00.A",
}


def _ecb_series(key: str, start: str, end: str, raw_path: Path):
    """Save an ECB response and return its finite, non-holiday series and URL."""
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        + key
        + "?"
        + urlencode(
            {
                "startPeriod": start,
                "endPeriod": end,
                "format": "csvdata",
            }
        )
    )
    with urlopen(url, timeout=60) as response:
        payload = response.read().decode("utf-8-sig")
    frame = pd.read_csv(StringIO(payload))
    if frame.empty or not {"TIME_PERIOD", "OBS_VALUE"}.issubset(frame.columns):
        raise ValueError(f"ECB returned no usable observations for {key}")
    raw_path.write_text(payload)
    # ECB explicitly encodes some TARGET holidays as H/NaN. They are not
    # observations; all other missing values still fail validation below.
    if "OBS_STATUS" in frame:
        frame = frame.loc[~(frame["OBS_VALUE"].isna() & frame["OBS_STATUS"].eq("H"))]
    values = pd.to_numeric(frame["OBS_VALUE"], errors="raise").to_numpy()
    series = pd.Series(values, index=pd.to_datetime(frame["TIME_PERIOD"])).sort_index()
    validate_daily_index(series.index, key)
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"ECB returned non-finite observations for {key}")
    return series, url


def convert_to_eur(usd: pd.DataFrame, usd_per_eur: pd.Series) -> pd.DataFrame:
    """Use the latest already-published ECB fixing, with a seven-day age limit.

    This is a valuation proxy, not a simultaneous executable FX quote.
    """
    validate_daily_index(usd.index, "USD prices")
    validate_daily_index(usd_per_eur.index, "FX rates")
    if not np.isfinite(usd_per_eur.to_numpy()).all() or (usd_per_eur <= 0).any():
        raise ValueError("FX rates must be positive and finite")
    locations = usd_per_eur.index.get_indexer(usd.index, method="ffill")
    if (locations < 0).any() or ((usd.index - usd_per_eur.index[locations]).days > 7).any():
        raise ValueError("missing or stale EUR/USD reference rates")
    fx = pd.Series(usd_per_eur.to_numpy()[locations], index=usd.index)
    return usd.div(fx, axis=0)


def prepare_example(output: Path, start: str, end: str) -> dict:
    """Download a frozen input bundle. End is exclusive; existing folders are refused."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".portfolio-download-", dir=output.parent) as directory:
        staged = Path(directory) / "bundle"
        metadata = _prepare_example(staged, start, end)
        staged.rename(output)
    return metadata


def _prepare_example(output: Path, start: str, end: str) -> dict:
    """Write adjusted ETF prices in EUR, raw sources and metadata to a new bundle."""
    import yfinance as yf

    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if not start_date < end_date <= today:
        raise ValueError("download requires start < end <= today (exclude incomplete sessions)")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    print("Downloading adjusted ETF prices: " + ", ".join(ASSETS), flush=True)
    downloaded = yf.download(
        list(ASSETS),
        start=start,
        end=end,
        auto_adjust=True,
        repair=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if downloaded is None or downloaded.empty:
        raise ValueError("Yahoo returned no ETF data")
    usd = downloaded["Close"].reindex(columns=list(ASSETS))
    if usd.isna().all().any():
        raise ValueError("Yahoo returned an empty asset series")
    usd.index = pd.DatetimeIndex(usd.index).tz_localize(None).normalize()
    usd.index.name = "date"
    # Keep IPO/inception truncation visible. Never backfill before inception.
    starts = {a: usd[a].first_valid_index().date().isoformat() for a in ASSETS}
    usd = usd.loc[max(starts.values()) :]
    if usd.isna().any().any() or not np.isfinite(usd.to_numpy()).all() or (usd <= 0).any().any():
        raise ValueError("ETF prices have internal gaps/nonpositive values; inspect the provider")
    usd.to_csv(raw / "adjusted_close_usd.csv")
    sources, series = {}, {}
    for name, key in ECB_SERIES.items():
        print(f"Downloading ECB {name}", flush=True)
        # Include history for as-of FX joins.
        series[name], sources[name] = _ecb_series(
            key,
            (start_date - pd.Timedelta(days=14)).date().isoformat(),
            end,
            raw / f"ecb_{name}.csv",
        )
    eur = convert_to_eur(usd, series["fx"])
    eur.to_csv(output / "prices_eur.csv")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_currency": "EUR",
        "assets": ASSETS,
        "source": "Yahoo Finance via yfinance; ECB Data API",
        "yfinance_version": yf.__version__,
        "requested_start": start,
        "requested_end_exclusive": end,
        "asset_first_dates": starts,
        "common_start": eur.index[0].date().isoformat(),
        "common_end": eur.index[-1].date().isoformat(),
        "ecb_urls": sources,
        "notes": [
            "Adjusted ETF closes proxy reinvested distributions, splits and fund expenses; no extra dividend credit.",
            "USD prices divided by same-day/latest ECB USD-per-EUR fixing; not simultaneous executable FX quotes.",
            "Uninvested cash earns zero; no interest-rate data are downloaded or used.",
            "Historical provider snapshot, not a point-in-time archive; no taxes, leverage or CFD funding.",
            "US-listed funds are research proxies; actual instrument availability/costs require separate validation.",
        ],
        "raw_files": [
            {**file_fingerprint(p), "path": str(p.relative_to(output))}
            for p in sorted(raw.iterdir())
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
