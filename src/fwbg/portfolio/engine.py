"""Long-only monthly portfolios on daily total-return prices in one currency.

Signals use completed month-end closes. Orders execute at the next session's
close, AFTER existing holdings earn that session's return. Uninvested cash earns zero; positions
are marked daily; rebalance fees are funded from the portfolio itself.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rules import STRATEGIES, allocation


@dataclass(frozen=True)
class PortfolioSettings:
    weights: dict[str, float]
    equity_benchmark: str
    lookback_months: int = 10
    initial_capital: float = 10_000.0
    one_way_cost_bps: float = 5.0
    start: str | None = None
    end: str | None = None
    strategy: str = "sma"
    selection_count: int = 2
    equity_assets: tuple[str, ...] = ()
    defensive_asset: str | None = None

    def validate(self) -> None:
        """Reject invalid allocation weights, strategy parameters, costs and dates."""
        if not self.weights or not all(
            isinstance(k, str) and k and math.isfinite(v) and v > 0 for k, v in self.weights.items()
        ):
            raise ValueError("weights must contain positive finite weights and asset names")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-10):
            raise ValueError("weights must sum to 1 (unlevered portfolio)")
        if self.equity_benchmark not in self.weights:
            raise ValueError("equity_benchmark must be one of the weighted assets")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {self.strategy}")
        minimum = 2 if self.strategy in {"sma", "inverse_volatility"} else 1
        if type(self.lookback_months) is not int or self.lookback_months < minimum:
            raise ValueError(f"lookback_months must be an integer >= {minimum}")
        if type(self.selection_count) is not int or self.selection_count < 1:
            raise ValueError("selection_count must be a positive integer")
        if self.strategy in {"relative_momentum", "dual_momentum", "reversal"}:
            if self.selection_count > len(self.weights):
                raise ValueError("selection_count exceeds the asset universe")
        if self.strategy == "equity_dual_momentum":
            if (
                not self.equity_assets
                or len(set(self.equity_assets)) != len(self.equity_assets)
                or not set(self.equity_assets).issubset(self.weights)
                or self.defensive_asset not in self.weights
                or self.defensive_asset in self.equity_assets
            ):
                raise ValueError("equity/defensive assets must be distinct members of weights")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive and finite")
        if not math.isfinite(self.one_way_cost_bps) or not 0 <= self.one_way_cost_bps < 1000:
            raise ValueError("one_way_cost_bps must be in [0, 1000)")
        for value in (self.start, self.end):
            if value is not None:
                stamp = pd.Timestamp(value)
                if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
                    raise ValueError("start/end must be timezone-free dates")
        if self.start and self.end and pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("start must precede end")


@dataclass
class PortfolioResult:
    equity: pd.DataFrame
    weights: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame


def validate_daily_index(index: pd.Index, label: str) -> None:
    """Require nonempty, sorted, unique, timezone-free daily date labels."""
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"{label}: a nonempty daily DatetimeIndex is required")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{label}: dates must be sorted, unique and non-null")
    if index.tz is not None or not index.equals(index.normalize()):
        raise ValueError(f"{label}: dates must be timezone-free daily labels")


def _rebalance(values, cash, target, cost_rate):
    """Solve NAV_after + fees(NAV_after) = NAV_before, without borrowing."""
    nav = float(values.sum() + cash)
    low, high = 0.0, nav
    for _ in range(55):
        after = (low + high) / 2
        fees = cost_rate * np.abs(target * after - values).sum()
        if after + fees > nav:
            high = after
        else:
            low = after
    new_values = target * ((low + high) / 2)
    notionals = new_values - values
    fees = cost_rate * np.abs(notionals)
    new_cash = nav - new_values.sum() - fees.sum()
    if new_cash < -max(1e-8, nav * 1e-12):
        raise ArithmeticError("rebalance would borrow cash")
    return new_values, max(0.0, new_cash), notionals, fees


def run_portfolio(prices: pd.DataFrame, settings: PortfolioSettings) -> PortfolioResult:
    """Evaluate trend, monthly static allocation, equity buy-and-hold and cash.

    Input prices must already include distributions/splits and FX conversion.
    No forward-filling of risky assets, leverage, shorts or optimizer is used.
    """
    settings.validate()
    validate_daily_index(prices.index, "prices")
    assets = list(settings.weights)
    if prices.columns.has_duplicates or set(prices.columns) != set(assets):
        raise ValueError("prices columns must exactly match weights")
    prices = prices.loc[:, assets].astype(float)
    if settings.end:
        prices = prices.loc[: settings.end]
    if len(prices) < 2 or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("prices must contain positive finite daily prices without missing values")
    if (prices.index.to_series().diff().dt.days.dropna() > 7).any():
        raise ValueError("prices contain a gap longer than 7 calendar days")

    periods = prices.index.to_period("M")
    boundaries = np.flatnonzero(periods[1:] != periods[:-1]) + 1
    # Only boundaries observed in the input complete a month. Never promote the
    # last available (possibly partial-month) bar to a new trading signal.
    history_months = settings.lookback_months + (settings.strategy != "sma")
    eligible = boundaries[history_months - 1 :]
    if settings.start:
        eligible = eligible[prices.index[eligible] >= pd.Timestamp(settings.start)]
    if not len(eligible):
        raise ValueError("not enough completed months for warm-up and evaluation")
    first = int(eligible[0])
    dates = prices.index[first - 1 :]
    base = np.array([settings.weights[a] for a in assets])
    benchmark = np.zeros(len(assets))
    benchmark[assets.index(settings.equity_benchmark)] = 1.0
    names = ("trend", "static", "equity_buy_hold", "cash")
    values = {name: np.zeros(len(assets)) for name in names}
    cash = dict.fromkeys(names, settings.initial_capital)
    rows = [{name: settings.initial_capital for name in names}]
    weight_rows, signal_rows, trade_rows = [], [], []
    boundary_lookup = {int(pos): n for n, pos in enumerate(boundaries)}
    price_values = prices.to_numpy()

    for i in range(first, len(prices)):
        date = prices.index[i]
        ratios = price_values[i] / price_values[i - 1]
        targets = {}
        if i in boundary_lookup:
            n = boundary_lookup[i]
            month_end_positions = boundaries[n - history_months + 1 : n + 1] - 1
            target, statistic = allocation(
                price_values[month_end_positions], base, assets, settings
            )
            targets = {"trend": target, "static": base}
            for j, asset in enumerate(assets):
                signal_rows.append(
                    {
                        "decision_date": prices.index[i - 1],
                        "execution_date": date,
                        "asset": asset,
                        "close": price_values[i - 1, j],
                        "sma" if settings.strategy == "sma" else "score": statistic[j],
                        "target_weight": float(target[j]),
                    }
                )
        if i == first:
            targets["equity_buy_hold"] = benchmark

        row = {}
        for name in names:
            values[name] *= ratios
            if name in targets:
                values[name], cash[name], notionals, fees = _rebalance(
                    values[name], cash[name], targets[name], settings.one_way_cost_bps / 10_000
                )
                for j, asset in enumerate(assets):
                    if abs(notionals[j]) > 1e-8:
                        trade_rows.append(
                            {
                                "portfolio": name,
                                "decision_date": prices.index[i - 1],
                                "execution_date": date,
                                "asset": asset,
                                "notional": notionals[j],
                                "cost": fees[j],
                            }
                        )
            row[name] = float(values[name].sum() + cash[name])
            weight_rows.append(
                {
                    "date": date,
                    "portfolio": name,
                    **{a: values[name][j] / row[name] for j, a in enumerate(assets)},
                    "cash": cash[name] / row[name],
                }
            )
        rows.append(row)

    trades = pd.DataFrame(
        trade_rows,
        columns=["portfolio", "decision_date", "execution_date", "asset", "notional", "cost"],
    )
    return PortfolioResult(
        pd.DataFrame(rows, index=dates).rename_axis("date"),
        pd.DataFrame(weight_rows),
        pd.DataFrame(signal_rows),
        trades,
    )


def summarize(equity: pd.DataFrame) -> dict[str, dict]:
    """Calendar-time CAGR, daily drawdown and cash-excess Sharpe; no significance claim."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or len(equity) < 3:
        raise ValueError("evaluation requires at least three valuation dates")
    returns = equity.pct_change().iloc[1:]
    observations_per_year = len(returns) / years
    output = {}
    for name in equity:
        excess = returns[name] - returns["cash"]
        stdev = float(excess.std(ddof=1))
        drawdown = equity[name] / equity[name].cummax() - 1
        output[name] = {
            "start": equity.index[0].date().isoformat(),
            "end": equity.index[-1].date().isoformat(),
            "years": years,
            "final_equity": float(equity[name].iloc[-1]),
            "total_return": float(equity[name].iloc[-1] / equity[name].iloc[0] - 1),
            "cagr": float((equity[name].iloc[-1] / equity[name].iloc[0]) ** (1 / years) - 1),
            "max_drawdown": float(drawdown.min()),
            "annual_volatility": float(returns[name].std(ddof=1) * np.sqrt(observations_per_year)),
            "sharpe_excess_cash": (
                float(excess.mean() / stdev * np.sqrt(observations_per_year))
                if stdev > 1e-12
                else None
            ),
        }
    return output
