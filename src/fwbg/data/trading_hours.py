"""Trading hours of assets: when an instrument can be traded.

Definitions live in ``trading_hours.json`` per asset class and per symbol group
(a group wins over the class); a symbol's ``trading_hours`` entry in the
workspace's ``_asset_meta.json`` wins over both. Holidays and early closes are not
modelled: data checks report them as missing bars, not as errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from .assets import AssetRegistry, load_asset_overrides

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEK_MINUTES = 7 * 1440
_HHMM = re.compile(r"^([01]\d|2[0-4]):([0-5]\d)$")
_DEFINITIONS = Path(__file__).with_name("trading_hours.json")
CURRENCIES = frozenset(
    "USD EUR GBP JPY CHF CAD AUD NZD SEK NOK DKK PLN HUF CZK TRY ZAR MXN SGD HKD CNH".split()
)


class TradingHoursError(ValueError):
    """A trading-hours definition is malformed."""


@dataclass(frozen=True)
class Window:
    """Open on each of *days* (0 = Monday) from *start* to *end* (minutes after local
    midnight); with ``end <= start`` the window closes on the next day."""

    days: frozenset[int]
    start: int
    end: int

    def intervals(self) -> list[tuple[int, int]]:
        """Open intervals as minutes of the week (may run past the week's end)."""
        length = (self.end - self.start) % 1440 or 1440
        return [(d * 1440 + self.start, d * 1440 + self.start + length) for d in sorted(self.days)]


@dataclass(frozen=True)
class TradingHours:
    """One session of an asset: a time zone and its weekly open windows."""

    tz: str
    windows: tuple[Window, ...]

    def _week_minutes(self, starts: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Local start times and their minute of the week. Naive timestamps are UTC."""
        idx = starts if starts.tz is not None else starts.tz_localize("UTC")
        local = idx.tz_convert(self.tz)
        m0 = np.asarray(local.dayofweek * 1440 + local.hour * 60 + local.minute, dtype=np.int64)
        return local, m0

    def open_minutes(self, starts: pd.DatetimeIndex, bar_minutes: int) -> np.ndarray:
        """Minutes of each bar ``[start, start + bar_minutes)`` the session is open."""
        _, m0 = self._week_minutes(starts)
        m1 = m0 + bar_minutes
        total = np.zeros(len(m0), dtype=np.int64)
        for window in self.windows:
            for start, end in window.intervals():
                for shift in (-WEEK_MINUTES, 0, WEEK_MINUTES):
                    total += np.clip(
                        np.minimum(m1, end + shift) - np.maximum(m0, start + shift), 0, None
                    )
        return np.minimum(total, bar_minutes)

    def session_bounds(
        self, starts: pd.DatetimeIndex, bar_minutes: int
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """Local open and close of the session each bar lies in entirely; NaT for other bars.

        Works on minutes of the week, never on clock times, so a bar just before
        midnight cannot pass for one inside a window that ends in the afternoon.
        """
        local, m0 = self._week_minutes(starts)
        m1 = m0 + bar_minutes
        offset = np.full(len(m0), -1, dtype=np.int64)  # minutes since the session opened
        length = np.zeros(len(m0), dtype=np.int64)
        for window in self.windows:
            for start, end in window.intervals():
                for shift in (-WEEK_MINUTES, 0, WEEK_MINUTES):
                    inside = (offset < 0) & (m0 >= start + shift) & (m1 <= end + shift)
                    offset[inside] = m0[inside] - start - shift
                    length[inside] = end - start
        naive = local.tz_localize(None).floor("min")
        found = offset >= 0
        opens = pd.DatetimeIndex(np.where(found, naive - pd.to_timedelta(offset, "min"), pd.NaT))
        closes = pd.DatetimeIndex(np.where(found, opens + pd.to_timedelta(length, "min"), pd.NaT))
        return opens, closes

    def overlaps(self, starts: pd.DatetimeIndex, bar_minutes: int) -> np.ndarray:
        """True for each bar that is open for at least one minute."""
        return self.open_minutes(starts, bar_minutes) > 0


def _minutes(value: Any, what: str) -> int:
    match = _HHMM.match(value) if isinstance(value, str) else None
    if not match or (match.group(1) == "24" and match.group(2) != "00"):
        raise TradingHoursError(f"{what} must be 'HH:MM' (00:00-24:00), not {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def _days(value: Any) -> frozenset[int]:
    if not isinstance(value, str) or not value.strip():
        raise TradingHoursError(f"days must be a string like 'Mon-Fri' or 'Mon,Wed', not {value!r}")
    days: set[int] = set()
    for part in value.split(","):
        names = [n.strip().capitalize() for n in part.split("-")]
        if len(names) > 2 or any(n not in DAY_NAMES for n in names):
            raise TradingHoursError(f"unknown day or range {part.strip()!r} in {value!r}")
        first, last = DAY_NAMES.index(names[0]), DAY_NAMES.index(names[-1])
        days.update((first + i) % 7 for i in range((last - first) % 7 + 1))
    return frozenset(days)


def parse_trading_hours(raw: Any) -> TradingHours:
    """Validate one session definition (``{"tz": ..., "windows": [...]}``)."""
    if not isinstance(raw, Mapping):
        raise TradingHoursError(f"a session must be an object with tz and windows, not {raw!r}")
    tz = raw.get("tz")
    if not isinstance(tz, str):
        raise TradingHoursError(f"tz must be an IANA time zone name, not {tz!r}")
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise TradingHoursError(f"unknown time zone {tz!r}") from exc
    windows = raw.get("windows")
    if not isinstance(windows, list) or not windows:
        raise TradingHoursError("windows must be a non-empty list")
    parsed = []
    for w in windows:
        if not isinstance(w, Mapping):
            raise TradingHoursError(f"a window must be an object, not {w!r}")
        parsed.append(
            Window(
                _days(w.get("days")),
                _minutes(w.get("start"), "start"),
                _minutes(w.get("end"), "end"),
            )
        )
    return TradingHours(tz, tuple(parsed))


def _sessions(raw: Any, where: str) -> dict[str, TradingHours]:
    if not isinstance(raw, Mapping):
        raise TradingHoursError(f"{where}: sessions must be an object of named sessions")
    try:
        return {name: parse_trading_hours(spec) for name, spec in raw.items()}
    except TradingHoursError as exc:
        raise TradingHoursError(f"{where}: {exc}") from exc


@lru_cache(maxsize=1)
def _builtin() -> tuple[dict[str, dict[str, TradingHours]], dict[str, dict[str, TradingHours]]]:
    with _DEFINITIONS.open() as f:
        data = json.load(f)
    classes = {name: _sessions(spec, f"class {name}") for name, spec in data["classes"].items()}
    symbols: dict[str, dict[str, TradingHours]] = {}
    for name, group in data["groups"].items():
        sessions = _sessions(group["sessions"], f"group {name}")
        symbols.update({symbol: sessions for symbol in group["symbols"]})
    return classes, symbols


def _asset_class(symbol: str) -> str | None:
    """The registry's class for symbols it knows; FOREX for any pair of ISO codes.

    The registry hands unknown symbols a FOREX default; that must not give an
    index or an ETF forex hours, so unknown symbols get a class only by their name.
    """
    registry = AssetRegistry()
    if symbol in registry.all_symbols():
        return registry.get(symbol).asset_class
    if len(symbol) == 6 and symbol[:3] in CURRENCIES and symbol[3:] in CURRENCIES:
        return "FOREX"
    return None


def asset_sessions(symbol: str) -> dict[str, TradingHours]:
    """Named sessions of *symbol* (``tradable`` and e.g. ``cash``); empty when unknown.

    A ``trading_hours`` entry for the symbol in ``_asset_meta.json`` wins over the
    built-in symbol groups, which win over the asset class.
    """
    override = load_asset_overrides().get(symbol, {}).get("trading_hours")
    if override is not None:
        return _sessions(override, f"_asset_meta.json trading_hours of {symbol}")
    classes, symbols = _builtin()
    if symbol in symbols:
        return dict(symbols[symbol])
    return dict(classes.get(_asset_class(symbol) or "", {}))
