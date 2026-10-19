"""Daily long-only entry/exit accounting on bid/ask total-return OHLC.

This is a cash/spot margin-account model, not a CFD or per-trade label model.
Signals and trail updates use closed bars; gaps fill at the opening bid.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from .engine import validate_daily_index

RULES = (
    "rsi_sma",
    "rsi_level",
    "breakout_channel",
    "breakout_trail",
    "pullback_ema",
    "pullback_trail",
    "buy_hold",
)


@dataclass(frozen=True)
class SwingSettings:
    rule: str
    start: str = "2019-01-01"
    end: str = "2025-12-31"
    capital: float = 10000
    leverage: float = 1
    cost_bps: float = 5
    cost_multiplier: float = 1
    borrow_rate: float = 0.08

    def validate(self) -> None:
        """Reject unsupported rules, leverage, capital, costs, funding and dates."""
        if self.rule not in RULES:
            raise ValueError("unknown swing rule")
        if self.leverage not in (1, 1.5):
            raise ValueError("only predeclared 1x and 1.5x leverage supported")
        if not all(
            np.isfinite(x)
            for x in (self.capital, self.cost_bps, self.cost_multiplier, self.borrow_rate)
        ):
            raise ValueError("settings must be finite")
        if (
            self.capital <= 0
            or self.cost_bps < 0
            or self.cost_multiplier < 1
            or not 0 <= self.borrow_rate < 1
        ):
            raise ValueError("invalid capital/cost/funding assumptions")
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("start must precede end")


@dataclass
class SwingResult:
    equity: pd.Series
    ledger: pd.DataFrame
    signals: pd.DataFrame
    positions: pd.DataFrame
    bankrupt: bool = False


def validate_bars(bars: pd.DataFrame) -> None:
    """Require valid daily bid/ask OHLC, nonnegative open/close spreads and gaps <=7 days."""
    validate_daily_index(bars.index, "swing bars")
    columns = [
        f"{side}_{field}" for side in ("bid", "ask") for field in ("open", "high", "low", "close")
    ]
    values = bars[columns]
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise ValueError("invalid OHLC values")
    for side in ("bid", "ask"):
        if (
            (
                bars[f"{side}_high"]
                < bars[[f"{side}_open", f"{side}_close", f"{side}_low"]].max(axis=1)
            )
            | (bars[f"{side}_low"] > bars[[f"{side}_open", f"{side}_close"]].min(axis=1))
        ).any():
            raise ValueError("OHLC high/low inconsistency")
    if (bars.ask_open < bars.bid_open).any() or (bars.ask_close < bars.bid_close).any():
        raise ValueError("negative opening/closing spread")
    if (bars.index.to_series().diff().dt.days.dropna() > 7).any():
        raise ValueError("unexplained gap longer than seven calendar days")


def indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute causal indicators and entry/exit signals from completed bid bars."""
    close = bars.bid_close
    frame = pd.DataFrame(index=bars.index)
    for period in (5, 200):
        frame[f"sma{period}"] = close.rolling(period).mean()
    for period in (20, 50, 200):
        frame[f"ema{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()
    frame["rsi"] = RSIIndicator(close, window=2).rsi()
    frame["atr"] = AverageTrueRange(
        bars.bid_high, bars.bid_low, close, window=14
    ).average_true_range()
    frame["upper"] = bars.bid_high.shift(1).rolling(55).max()
    frame["lower"] = bars.bid_low.shift(1).rolling(20).min()
    frame["rsi_entry"] = (close > frame.sma200) & (frame.rsi < 10)
    frame["breakout_entry"] = close > frame.upper
    frame["pullback_entry"] = (
        (frame.ema50 > frame.ema200)
        & (close.shift(1) < frame.ema20.shift(1))
        & (close > frame.ema20)
    )
    frame["rsi_sma_exit"] = close > frame.sma5
    frame["rsi_level_exit"] = frame.rsi > 70
    frame["breakout_channel_exit"] = close < frame.lower
    frame["pullback_ema_exit"] = close < frame.ema50
    return frame


def rebalance(
    cash: float, quantity: float, bid: float, ask: float, leverage: float, fee: float
) -> tuple[float, float, float, float, float]:
    """Solve post-cost target exposure exactly, marking existing assets at bid."""
    nav = cash + quantity * bid
    if leverage == 0:
        target = 0.0
    elif leverage * nav / bid >= quantity:
        unit_cost = ask - bid + ask * fee
        target = leverage * (nav + quantity * unit_cost) / (bid + leverage * unit_cost)
    else:
        unit_cost = bid * fee
        target = leverage * (nav - quantity * unit_cost) / (bid - leverage * unit_cost)
    delta = target - quantity
    fill = ask if delta > 0 else bid
    commission = abs(delta) * fill * fee
    return cash - delta * fill - commission, target, delta, fill, commission


def run_swing(bars: pd.DataFrame, settings: SwingSettings) -> SwingResult:
    """Simulate a daily long-only rule with bid/ask fills and margin borrowing.

    Execute prior-bar signals at the next open, apply stops before updating
    trails, and return equity, orders, signals, positions and bankruptcy status.
    """
    settings.validate()
    end = settings.end
    if end is not None:
        bars = bars.loc[bars.index <= pd.Timestamp(end)]
    validate_bars(bars)
    features = indicators(bars)
    eligible = np.flatnonzero(
        (bars.index >= pd.Timestamp(settings.start)) & (np.arange(len(bars)) >= 201)
    )
    if not len(eligible) or bars.index[eligible[0]] > pd.Timestamp(settings.start) + pd.Timedelta(
        days=7
    ):
        raise ValueError("insufficient warm-up or missing evaluation start")
    first = int(eligible[0])
    dates = bars.index
    bid_open = bars.bid_open.to_numpy()
    ask_open = bars.ask_open.to_numpy()
    bid_low = bars.bid_low.to_numpy()
    bid_high = bars.bid_high.to_numpy()
    bid_close = bars.bid_close.to_numpy()
    atr = features.atr.to_numpy()
    trailing = settings.rule.endswith("trail")
    if settings.rule == "buy_hold":
        enter = np.ones(len(bars), dtype=bool)
    else:
        family = settings.rule.split("_")[0]
        enter = features[f"{family}_entry"].to_numpy()
    exit_signal = features.get(f"{settings.rule}_exit", pd.Series(False, index=dates)).to_numpy()
    fee = settings.cost_bps * settings.cost_multiplier / 10000
    cash, quantity = float(settings.capital), 0.0
    stop, high_water, sessions = float("nan"), 0.0, 0
    equity = [settings.capital]
    ledger, signal_rows, positions = [], [], []
    pending_exit = None
    bankrupt = False
    for i in range(first, len(bars)):
        day = dates[i]
        elapsed = (day - dates[i - 1]).days
        funding = max(-cash, 0.0) * settings.borrow_rate * elapsed / 365
        cash -= funding
        if bankrupt or cash + quantity * bid_open[i] <= 0:
            bankrupt = True
            cash, quantity = 0.0, 0.0
            equity.append(0.0)
            positions.append(
                {"date": day, "quantity": 0.0, "cash": 0.0, "funding": funding, "exposure": 0.0}
            )
            continue
        # Spread stress multiplies observed opening spread, never future close spread.
        ask = bid_open[i] + (ask_open[i] - bid_open[i]) * settings.cost_multiplier
        reason = None
        target = None
        stop_at_open = quantity > 0 and trailing and bid_open[i] <= stop
        if stop_at_open or (quantity > 0 and pending_exit):
            reason = "gap_stop" if stop_at_open else pending_exit
            target = 0.0
        elif quantity == 0 and enter[i - 1]:
            reason, target = "entry", settings.leverage
        elif quantity > 0 and settings.leverage > 1:
            reason, target = "leverage_rebalance", settings.leverage
        pending_exit = None
        if target is not None:
            cash, quantity, delta, fill, commission = rebalance(
                cash, quantity, bid_open[i], ask, target, fee
            )
            if abs(delta) > 1e-10:
                ledger.append(
                    {
                        "date": day,
                        "reason": reason,
                        "quantity_change": delta,
                        "fill": fill,
                        "commission": commission,
                        "spread_cost": max(delta, 0) * (ask - bid_open[i]),
                    }
                )
            if reason == "entry":
                sessions = 0
                high_water = bid_open[i]
                stop = fill - 3 * atr[i - 1] if trailing else float("nan")
            elif target == 0:
                sessions = 0
                stop = float("nan")
        # Stop from before this bar (or entry ATR) is eligible during this bar.
        if quantity > 0 and trailing and bid_low[i] <= stop:
            fill = min(stop, bid_open[i])
            commission = quantity * fill * fee
            ledger.append(
                {
                    "date": day,
                    "reason": "intraday_stop",
                    "quantity_change": -quantity,
                    "fill": fill,
                    "commission": commission,
                    "spread_cost": 0.0,
                }
            )
            cash += quantity * fill - commission
            quantity, sessions, stop = 0.0, 0, float("nan")
        if quantity > 0:
            sessions += 1
            if settings.rule.startswith("rsi") and sessions >= 10:
                pending_exit = "time_exit"
            elif exit_signal[i]:
                pending_exit = "signal_exit"
            if trailing:
                high_water = max(high_water, bid_high[i])
                stop = max(stop, high_water - 3 * atr[i])
        nav = cash + quantity * bid_close[i]
        if nav <= 0:
            bankrupt = True
            nav, cash, quantity = 0.0, 0.0, 0.0
        equity.append(nav)
        positions.append(
            {
                "date": day,
                "quantity": quantity,
                "cash": cash,
                "funding": funding,
                "exposure": quantity * bid_close[i] / nav if nav else 0.0,
            }
        )
        signal_rows.append(
            {
                "date": day,
                "entry_signal": bool(enter[i]),
                "exit_signal": bool(exit_signal[i]),
                "next_open_exit": pending_exit,
                "next_bar_stop": stop,
                "held_sessions": sessions,
            }
        )
    return SwingResult(
        pd.Series(equity, index=dates[first - 1 :], name="strategy").rename_axis("date"),
        pd.DataFrame(
            ledger,
            columns=["date", "reason", "quantity_change", "fill", "commission", "spread_cost"],
        ).astype(
            {
                "date": "datetime64[ns]",
                "reason": object,
                "quantity_change": float,
                "fill": float,
                "commission": float,
                "spread_cost": float,
            }
        ),
        pd.DataFrame(signal_rows),
        pd.DataFrame(positions),
        bankrupt,
    )
