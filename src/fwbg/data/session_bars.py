"""Daily bars of one trading session (e.g. the US cash session) built from intraday bars.

A source bar joins a session only when it lies entirely inside one of the
session's open windows (``TradingHours.session_bounds``). A session counts only
when its first bar starts at the open, its last bar ends at the close and at least
half of its bars exist, so early closes and holes do not pass for full sessions.
Each output bar is labelled with the local date of its session's close (the
trading date), as a naive 00:00 timestamp like other daily bars.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .trading_hours import TradingHours
from .validation import DataCheckError, DataCheckReport, check_bars

MIN_SHARE_OF_BARS = 0.5


@dataclass(frozen=True)
class SessionBars:
    bars: pd.DataFrame  # index: trading date; columns O, H, L, C, V
    sessions: int  # sessions seen in the source
    incomplete: int  # of those, dropped as incomplete
    source_checks: DataCheckReport


def build_session_bars(
    df: pd.DataFrame, *, symbol: str, hours: TradingHours, bar_minutes: int
) -> SessionBars:
    """One bar per complete session of *hours* from intraday OHLC(V) bars *df*.

    The source passes the run's data checks first (filler bars dropped); a data
    error raises :class:`DataCheckError`.
    """
    if bar_minutes >= 1440:
        raise ValueError(f"session bars need intraday source bars, not {bar_minutes}-minute bars")
    clean, report = check_bars(df, symbol=symbol, bar_minutes=bar_minutes, hours=None)
    if report.errors:
        raise DataCheckError(report)
    opens, closes = hours.session_bounds(clean.index, bar_minutes)
    inside = ~opens.isna()
    bars = clean[inside].copy()
    local_starts = (
        (bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index)
        .tz_convert(hours.tz)
        .tz_localize(None)
    )
    bars["open_at"], bars["close_at"], bars["start"] = opens[inside], closes[inside], local_starts
    if "V" not in bars:
        bars["V"] = 0.0
    g = bars.groupby("open_at", sort=True)
    step = pd.Timedelta(minutes=bar_minutes)
    out = pd.DataFrame(
        {
            "O": g["O"].first(),
            "H": g["H"].max(),
            "L": g["L"].min(),
            "C": g["C"].last(),
            "V": g["V"].sum(),
            "first": g["start"].min(),
            "last": g["start"].max(),
            "close_at": g["close_at"].first(),
            "n": g.size(),
        }
    )
    expected = (out["close_at"] - out.index) / step
    complete = (
        (out["first"] == out.index)
        & (out["last"] + step == out["close_at"])
        & (out["n"] >= MIN_SHARE_OF_BARS * expected)
    )
    result = out[complete]
    result.index = pd.DatetimeIndex(result["close_at"].dt.normalize(), name="T")
    return SessionBars(
        bars=result[["O", "H", "L", "C", "V"]],
        sessions=len(out),
        incomplete=int((~complete).sum()),
        source_checks=report,
    )


def write_session_bars(
    built: SessionBars,
    *,
    out_dir: Path,
    symbol: str,
    session: str,
    source: Path,
    hours: TradingHours,
) -> Path:
    """Write ``{symbol}_DAY_1.csv`` (fwbg CSV) and a ``.session.json`` provenance file
    next to it; create a CSV datasource ``config.json`` when the folder has none."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{symbol}_DAY_1.csv"
    frame = built.bars.copy()
    frame.index = frame.index.strftime("%Y-%m-%d %H:%M:%S")
    frame.index.name = "T"
    frame.to_csv(csv)
    provenance = {
        "symbol": symbol,
        "session": session,
        "time_zone": hours.tz,
        "windows": [
            {"days": sorted(w.days), "start_minute": w.start, "end_minute": w.end}
            for w in hours.windows
        ],
        "source": str(source),
        "label": "local date of the session close",
        "sessions": built.sessions,
        "incomplete_dropped": built.incomplete,
        "bars": len(built.bars),
        "source_checks": built.source_checks.to_dict(),
    }
    csv.with_suffix(".session.json").write_text(json.dumps(provenance, indent=2))
    config = out_dir.parent / "config.json"
    if not config.exists():
        config.write_text(
            json.dumps(
                {
                    "type": "csv",
                    "name": out_dir.parent.name,
                    "description": f"{session} session daily bars built by fwbg data session-bars",
                    "path": str(out_dir),
                    "file_pattern": "{symbol}_{timeframe}.csv",
                    "timeframe_map": {},
                },
                indent=2,
            )
        )
    return csv
