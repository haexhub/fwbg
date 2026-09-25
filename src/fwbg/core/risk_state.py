"""Pure, replay-safe state for the live bot circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from typing import Any, Iterable
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ClosedTradeEvent:
    """A broker-confirmed closed trade with net PnL (fees already included)."""

    event_id: str
    closed_at: datetime
    net_pnl: float


@dataclass
class _DaySummary:
    pnl: float = 0.0
    consecutive_losses: int = 0


@dataclass
class RiskState:
    """Track realized daily PnL and loss streaks without side effects.

    ``observe`` is the synchronization boundary with a broker. Until a valid,
    recent observation is received, ``can_trade`` returns ``False``.
    """

    account_timezone: str = "UTC"
    max_daily_loss_percent: float = 0.05
    pause_after_consecutive_losses: int = 3
    pause_duration_minutes: int = 60
    max_observation_age_seconds: int = 300

    known: bool = field(default=False, init=False)
    unknown_reason: str | None = field(default="broker state has not been observed", init=False)
    observed_at: datetime | None = field(default=None, init=False)
    daily_start_balance: float = field(default=0.0, init=False)
    active_day: date | None = field(default=None, init=False)
    daily_pnl: float = field(default=0.0, init=False)
    consecutive_losses: int = field(default=0, init=False)
    rejected_orders: int = field(default=0, init=False)
    pause_until: datetime | None = field(default=None, init=False)
    _seen_event_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _events_by_day: dict[date, dict[str, ClosedTradeEvent]] = field(
        default_factory=dict, init=False, repr=False
    )
    _summaries: dict[date, _DaySummary] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._timezone = (
            ZoneInfo(self.account_timezone)
            if isinstance(self.account_timezone, str)
            else self.account_timezone
        )

    def local_day(self, value: datetime | None = None) -> date:
        """Return the account-local calendar day for ``value``."""
        value = value or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(self._timezone).date()

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def mark_unknown(self, reason: str) -> None:
        """Block trading until the next valid broker observation."""
        self.known = False
        self.unknown_reason = reason

    def _recompute_day(self, day: date) -> None:
        events = sorted(
            self._events_by_day.get(day, {}).values(),
            key=lambda event: (self._aware(event.closed_at), event.event_id),
        )
        pnl = sum(event.net_pnl for event in events)
        streak = 0
        for event in events:
            if event.net_pnl < 0:
                streak += 1
            elif event.net_pnl > 0:
                streak = 0
        self._summaries[day] = _DaySummary(pnl=float(pnl), consecutive_losses=streak)
        if day == self.active_day:
            self.daily_pnl = float(pnl)
            self.consecutive_losses = streak

    def process_close_event(
        self,
        event: ClosedTradeEvent | str,
        closed_at: datetime | None = None,
        net_pnl: float | None = None,
    ) -> bool:
        """Record one close event; return False when its ID was already seen."""
        if isinstance(event, ClosedTradeEvent):
            close_event = event
        else:
            if closed_at is None or net_pnl is None:
                raise ValueError("event_id, closed_at, and net_pnl are required")
            close_event = ClosedTradeEvent(event, closed_at, float(net_pnl))

        event_id = str(close_event.event_id).strip()
        close_time = self._aware(close_event.closed_at)
        pnl = float(close_event.net_pnl)
        if not event_id or not isfinite(pnl):
            raise ValueError("closed trade event requires a stable ID and finite PnL")
        if event_id in self._seen_event_ids:
            return False

        day = self.local_day(close_time)
        self._seen_event_ids.add(event_id)
        self._events_by_day.setdefault(day, {})[event_id] = ClosedTradeEvent(
            event_id, close_time, pnl
        )
        self._recompute_day(day)
        return True

    # Explicit alias for callers that prefer event terminology.
    record_close = process_close_event

    def observe(
        self,
        balance: float,
        events: Iterable[ClosedTradeEvent] | None,
        observed_at: datetime | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Apply one broker snapshot and return whether it is trustworthy."""
        now = self._aware(now or datetime.now(timezone.utc))
        observed_at = self._aware(observed_at or now)
        balance = float(balance)
        age = (now - observed_at).total_seconds()
        if not isfinite(balance) or balance <= 0:
            self.mark_unknown("broker balance is unavailable")
            return False
        if age < -5 or age > self.max_observation_age_seconds:
            self.mark_unknown("broker risk state is stale")
            return False
        if events is None:
            self.mark_unknown("closed trade history is unavailable")
            return False

        day = self.local_day(now)
        if self.active_day != day:
            self.active_day = day
            self.daily_start_balance = balance
            self.pause_until = None
        for event in events:
            try:
                self.process_close_event(event)
            except (TypeError, ValueError) as exc:
                self.mark_unknown(f"invalid closed trade event: {exc}")
                return False

        self.observed_at = observed_at
        self.known = True
        self.unknown_reason = None
        summary = self._summaries.get(day, _DaySummary())
        self.daily_pnl = summary.pnl
        self.consecutive_losses = summary.consecutive_losses
        return True

    def can_trade(self, now: datetime | None = None) -> bool:
        """Return whether the state allows a new entry."""
        if not self.known or self.observed_at is None:
            return False
        now = self._aware(now or datetime.now(timezone.utc))
        if (now - self.observed_at).total_seconds() > self.max_observation_age_seconds:
            self.mark_unknown("broker risk state is stale")
            return False
        if self.active_day != self.local_day(now):
            # A fresh observation must set the new day's opening balance.
            self.mark_unknown("new account-local day requires broker observation")
            return False
        if self.pause_until and now < self.pause_until:
            return False
        if (
            self.daily_start_balance > 0
            and -self.daily_pnl / self.daily_start_balance >= self.max_daily_loss_percent
        ):
            self.pause_until = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            return False
        if self.pause_after_consecutive_losses > 0 and self.consecutive_losses >= self.pause_after_consecutive_losses:
            self.pause_until = now + timedelta(minutes=self.pause_duration_minutes)
            return False
        return True

    def count_rejection(self) -> None:
        """Track rejected orders without changing realized-risk state."""
        self.rejected_orders += 1

    @property
    def state(self) -> str:
        return "known" if self.known else "unknown"

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable restart snapshot."""
        return {
            "account_timezone": self.account_timezone,
            "max_daily_loss_percent": self.max_daily_loss_percent,
            "pause_after_consecutive_losses": self.pause_after_consecutive_losses,
            "pause_duration_minutes": self.pause_duration_minutes,
            "max_observation_age_seconds": self.max_observation_age_seconds,
            "known": self.known,
            "unknown_reason": self.unknown_reason,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "daily_start_balance": self.daily_start_balance,
            "active_day": self.active_day.isoformat() if self.active_day else None,
            "rejected_orders": self.rejected_orders,
            "events": [
                {
                    "event_id": event.event_id,
                    "closed_at": event.closed_at.isoformat(),
                    "net_pnl": event.net_pnl,
                }
                for events in self._events_by_day.values()
                for event in events.values()
            ],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "RiskState":
        state = cls(
            account_timezone=snapshot.get("account_timezone", "UTC"),
            max_daily_loss_percent=float(snapshot.get("max_daily_loss_percent", 0.05)),
            pause_after_consecutive_losses=int(
                snapshot.get("pause_after_consecutive_losses", 3)
            ),
            pause_duration_minutes=int(snapshot.get("pause_duration_minutes", 60)),
            max_observation_age_seconds=int(
                snapshot.get("max_observation_age_seconds", 300)
            ),
        )
        for item in snapshot.get("events", []):
            state.process_close_event(
                str(item["event_id"]),
                datetime.fromisoformat(item["closed_at"]),
                float(item["net_pnl"]),
            )
        state.known = bool(snapshot.get("known", False))
        state.unknown_reason = snapshot.get("unknown_reason")
        observed_at = snapshot.get("observed_at")
        state.observed_at = datetime.fromisoformat(observed_at) if observed_at else None
        state.daily_start_balance = float(snapshot.get("daily_start_balance", 0.0))
        active_day = snapshot.get("active_day")
        state.active_day = date.fromisoformat(active_day) if active_day else None
        state.rejected_orders = int(snapshot.get("rejected_orders", 0))
        if state.active_day:
            state._recompute_day(state.active_day)
        return state


__all__ = ["ClosedTradeEvent", "RiskState"]
