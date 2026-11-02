"""Portfolio of frozen strategy sleeves: one gated run, one holdout look.

A portfolio spec names strategy files (sleeves). Each sleeve runs through the
per-symbol pipeline (``process_symbol``: walk-forward, frozen unified setting).
Its trades enter the portfolio unsized, as net returns on one unit of notional
(``pnl_raw / entry_price``, spread and financing included), booked on their exit
date; a sleeve with several symbols weights them equally. The sleeves' own Kelly
sizing is not used: sizing happens once, at the portfolio level, by causal
inverse-volatility weights (trailing ``lookback_days``, shifted one day) scaled to
``target_volatility`` (portfolio scale at most ``max_scale``; a sleeve's notional
at most its ``max_notional``).

An iteration reports the combination of the sleeves' walk-forward trades. A
holdout look develops every sleeve on ``[start, cutoff)`` and evaluates its frozen
setting once on ``[cutoff, end)`` whatever the sleeve's own development status:
the portfolio, not the sleeve, is the hypothesis. Reported holdout metrics come
from the portfolio's returns after the cutoff alone; their weights use volatility
estimated on the development trades before it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fwbg.core.config import StrategyConfig
from fwbg.data import config as data_config
from fwbg.results import storage as results_storage

from .holdout import frozen_trades_after_cutoff
from .process import process_symbol
from .run_files import select_run_files

PORTFOLIO_SYMBOL = "PORTFOLIO"
SPEC_KEYS = frozenset(
    {
        "name",
        "description",
        "sleeves",
        "lookback_days",
        "min_days",
        "target_volatility",
        "max_scale",
        "start_date",
        "end_date",
    }
)
SLEEVE_KEYS = frozenset({"name", "strategy_file", "max_notional"})
WINDOW_KEYS = ("start_date", "end_date", "holdout_cutoff")
DEFAULTS = {"lookback_days": 252, "min_days": 126, "target_volatility": 0.10, "max_scale": 5.0}
SLICES = 8
SIGNIFICANCE = 0.05

BeforeRun = Callable[[Mapping[str, Any], list[str]], dict[str, Any]]
LoadStrategy = Callable[[str], dict[str, Any] | None]
Prepare = Callable[[Mapping[str, Any]], dict[str, Any]]


class PortfolioSpecError(ValueError):
    """The portfolio spec or one of its sleeves cannot be used."""


def load_spec(path: str | Path, load_strategy: LoadStrategy, prepare: Prepare) -> dict[str, Any]:
    """Read a portfolio spec and resolve its sleeves into the metadata the gate fingerprints.

    The result holds the portfolio parameters, the window and ``sleeves``: name ->
    {"strategy": prepared strategy metadata, "max_notional": float | None}.
    """
    spec_path = Path(path)
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PortfolioSpecError(f"cannot read portfolio spec {spec_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PortfolioSpecError("a portfolio spec is a JSON object")
    unknown = set(raw) - SPEC_KEYS
    if unknown:
        raise PortfolioSpecError(f"unknown portfolio keys: {sorted(unknown)}")
    sleeves = raw.get("sleeves")
    if not isinstance(sleeves, list) or len(sleeves) < 2:
        raise PortfolioSpecError("a portfolio needs at least two sleeves")
    metadata: dict[str, Any] = {
        "name": raw.get("name") or spec_path.stem,
        "portfolio": {key: raw.get(key, default) for key, default in DEFAULTS.items()},
        "timeframe": "DAY_1",  # the combination's clock
        "sleeves": {},
    }
    _check_parameters(metadata["portfolio"])
    for key in ("start_date", "end_date"):
        if raw.get(key) is not None:
            metadata[key] = raw[key]
    for entry in sleeves:
        name, strategy, max_notional = _sleeve(entry, spec_path.parent, load_strategy, prepare)
        if name in metadata["sleeves"]:
            raise PortfolioSpecError(f"duplicate sleeve name {name!r}")
        metadata["sleeves"][name] = {"strategy": strategy, "max_notional": max_notional}
    return metadata


def _check_parameters(params: Mapping[str, Any]) -> None:
    for key in ("lookback_days", "min_days"):
        if not isinstance(params[key], int) or params[key] < 2:
            raise PortfolioSpecError(f"{key} must be an integer >= 2")
    if params["min_days"] > params["lookback_days"]:
        raise PortfolioSpecError("min_days must not exceed lookback_days")
    for key in ("target_volatility", "max_scale"):
        if not isinstance(params[key], (int, float)) or params[key] <= 0:
            raise PortfolioSpecError(f"{key} must be a positive number")


def _sleeve(
    entry: Any, base: Path, load_strategy: LoadStrategy, prepare: Prepare
) -> tuple[str, dict[str, Any], float | None]:
    if not isinstance(entry, dict) or set(entry) - SLEEVE_KEYS or "strategy_file" not in entry:
        raise PortfolioSpecError(
            f"a sleeve is {{name, strategy_file, max_notional?}}, got {entry!r}"
        )
    file = Path(entry["strategy_file"])
    file = file if file.is_absolute() else base / file
    loaded = load_strategy(str(file))
    if not loaded:
        raise PortfolioSpecError(f"cannot load sleeve strategy {file}")
    for key in WINDOW_KEYS:
        if loaded.get(key) is not None:
            raise PortfolioSpecError(
                f"sleeve {file.name} sets {key}; the portfolio spec sets the window"
            )
    name = str(entry.get("name") or file.stem)
    max_notional = entry.get("max_notional")
    if max_notional is not None and (
        not isinstance(max_notional, (int, float)) or max_notional <= 0
    ):
        raise PortfolioSpecError(f"sleeve {name}: max_notional must be a positive number")
    loaded.pop("_strategy_dir", None)
    return name, prepare(loaded), None if max_notional is None else float(max_notional)


def sleeve_files(metadata: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each sleeve's data files (configures the data globals per sleeve on the way)."""
    files: dict[str, list[str]] = {}
    for name, sleeve in metadata["sleeves"].items():
        strategy_metadata = sleeve["strategy"]
        found = select_run_files(StrategyConfig.from_dict(strategy_metadata), strategy_metadata)
        if not found:
            raise PortfolioSpecError(f"sleeve {name}: no data files")
        files[name] = found
    return files


def symbol_of(path: str) -> str:
    return os.path.basename(path).split("_")[0].upper()


def _sleeve_strategy(metadata: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The sleeve's metadata with the portfolio's (gated) window and cutoff."""
    strategy = copy.deepcopy(dict(metadata["sleeves"][name]["strategy"]))
    for key in WINDOW_KEYS:
        if metadata.get(key) is not None:
            strategy[key] = metadata[key]
    return strategy


def trade_returns(trades: list[Mapping[str, Any]]) -> pd.Series:
    """Net return per unit of notional of each trade, indexed by its exit date."""
    rows = [
        (pd.Timestamp(t["exit_time"]).normalize(), float(t["pnl_raw"]) / float(t["entry_price"]))
        for t in trades
        if t.get("exit_time") and t.get("entry_price")
    ]
    if len(rows) != len(trades):
        raise ValueError("trades without exit time or entry price cannot be dated")
    if not rows:
        return pd.Series(dtype=float)
    days, values = zip(*rows)
    index = pd.DatetimeIndex(days)
    if index.tz is not None:
        index = index.tz_localize(None)
    return pd.Series(values, index=index).groupby(level=0).sum()


def _run_symbol(csv_path: str, strategy: StrategyConfig) -> dict[str, Any]:
    """One sleeve symbol: development trades and, for a holdout look, the trades after it."""
    cutoff = strategy.holdout_cutoff
    develop = (
        dataclasses.replace(strategy, end_date=cutoff, holdout_cutoff=None) if cutoff else strategy
    )
    development = process_symbol(csv_path, develop)
    status = development.get("status")
    out: dict[str, Any] = {
        "symbol": development.get("symbol") or symbol_of(csv_path),
        "development_status": status,
        "development_trades": list(development.get("trades_detailed") or []),
    }
    if status not in ("ok", "not_significant"):
        out["reason"] = development.get("reason") or development.get("error") or str(status)
    if not cutoff:
        return out
    look = frozen_trades_after_cutoff(csv_path, strategy, development)
    if isinstance(look, dict):  # not evaluated
        out["holdout"] = look["holdout"]
        return out
    out["holdout"], out["holdout_trades"] = look
    return out


def combine(
    streams: Mapping[str, pd.Series],
    params: Mapping[str, Any],
    max_notional: Mapping[str, float | None],
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily portfolio returns and the sleeves' effective notional (causal weights)."""
    first = min(s.index.min() for s in streams.values())
    last = max(s.index.max() for s in streams.values())
    days = pd.bdate_range(first, last)
    returns = pd.DataFrame({k: s.reindex(days, fill_value=0.0) for k, s in streams.items()})
    returns = returns.fillna(0.0)
    window, min_days = params["lookback_days"], params["min_days"]
    annual = np.sqrt(252)
    vol = returns.rolling(window, min_periods=min_days).std().shift(1) * annual
    weights = params["target_volatility"] / len(streams) / vol.where(vol > 0)
    raw = (weights * returns).sum(axis=1, min_count=len(streams))
    port_vol = raw.rolling(window, min_periods=min_days).std().shift(1) * annual
    scale = (params["target_volatility"] / port_vol.where(port_vol > 0)).clip(
        upper=params["max_scale"]
    )
    notional = weights.mul(scale, axis=0)
    caps = pd.Series({k: (np.inf if v is None else v) for k, v in max_notional.items()})
    notional = notional.clip(upper=caps, axis=1)
    valid = notional.notna().all(axis=1)
    portfolio = (notional[valid] * returns[valid]).sum(axis=1)
    return portfolio, notional[valid]


def sign_permutation_p(values: np.ndarray, n: int = 5000, seed: int = 42) -> float:
    """One-sided p of the sum of *values* under random signs (monthly returns)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 10:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(values)))
    random_sums = (signs * np.abs(values)).sum(axis=1)
    return float((np.sum(random_sums >= values.sum()) + 1) / (n + 1))


def portfolio_metrics(returns: pd.Series, trades: int) -> dict[str, Any]:
    if len(returns) < 2 or returns.std() == 0:
        return {"trades": trades, "sharpe": None}
    equity = (1 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    slices = [chunk.sum() > 0 for chunk in np.array_split(returns.to_numpy(), SLICES) if len(chunk)]
    return {
        "trades": trades,
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)),
        "annual_return": round(float(equity.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "max_drawdown": round(float((1 - equity / equity.cummax()).max()), 4),
        "win_rate": (
            round(float((returns[returns != 0] > 0).mean()), 4) if (returns != 0).any() else None
        ),
        "p_value": sign_permutation_p(returns.resample("ME").sum().to_numpy()),
        "fold_stability": float(np.mean(slices)),
        "test_period_years": round(years, 2),
        "n_returns": int(len(returns)),
    }


def run_portfolio(
    spec_path: str,
    *,
    load_strategy: LoadStrategy,
    prepare: Prepare,
    run_id: str,
    before_run: BeforeRun | None,
    description: str | None = None,
) -> dict[str, Any] | None:
    """Gate, run and write one portfolio run; return its result (None: nothing ran)."""
    metadata = load_spec(spec_path, load_strategy, prepare)
    files = sleeve_files(metadata)
    if before_run is not None:
        symbols = sorted({symbol_of(f) for paths in files.values() for f in paths})
        metadata = before_run(metadata, symbols)
    cutoff = metadata.get("holdout_cutoff")
    run_path = Path(results_storage.create_run_directory(run_id, description, metadata))
    print(f"\nRun ID: {run_id}\nResults: {run_path}/\nPortfolio: {metadata['name']}")

    sleeves: dict[str, dict[str, Any]] = {}
    for name in metadata["sleeves"]:
        sleeve_metadata = _sleeve_strategy(metadata, name)
        strategy = StrategyConfig.from_dict(sleeve_metadata)
        paths = select_run_files(strategy, sleeve_metadata) or []
        print(f"Sleeve {name}: {', '.join(symbol_of(p) for p in paths)} ({data_config.TIMEFRAME})")
        sleeves[name] = {"symbols": [_run_symbol(p, strategy) for p in paths]}
    result = _assemble(metadata, sleeves, cutoff)
    _write(run_path, result)
    _print(result)
    return result


def _sleeve_stream(sleeve: Mapping[str, Any], key: str) -> pd.Series | None:
    """Equal-weight mean of the sleeve's symbols' daily returns (None if one has no trades)."""
    series = []
    for sym in sleeve["symbols"]:
        trades = sym.get(key)
        if trades is None:
            return None
        series.append(trade_returns(trades))
    if not series or all(s.empty for s in series):
        return None
    return pd.concat(series, axis=1).fillna(0.0).mean(axis=1)


def _assemble(
    metadata: Mapping[str, Any], sleeves: dict[str, dict[str, Any]], cutoff: str | None
) -> dict[str, Any]:
    summary = {
        name: [
            {k: v for k, v in sym.items() if not k.endswith("_trades")}
            | {
                "development_trade_count": len(sym["development_trades"]),
                "holdout_trade_count": len(sym.get("holdout_trades") or []),
            }
            for sym in sleeve["symbols"]
        ]
        for name, sleeve in sleeves.items()
    }
    result: dict[str, Any] = {"symbol": PORTFOLIO_SYMBOL, "sleeves": summary}
    looked = any(
        sym.get("holdout", {}).get("evaluated") for s in sleeves.values() for sym in s["symbols"]
    )
    if cutoff:
        result["holdout"] = {"cutoff": cutoff, "evaluated": bool(looked)}
    development = {name: _sleeve_stream(s, "development_trades") for name, s in sleeves.items()}
    missing = [name for name, stream in development.items() if stream is None]
    if cutoff:
        after = {name: _sleeve_stream(s, "holdout_trades") for name, s in sleeves.items()}
        missing += [
            name for name, stream in after.items() if stream is None and name not in missing
        ]
    if missing:
        result.update(
            status="error", reason=f"sleeves without trades: {', '.join(sorted(missing))}"
        )
        if cutoff:
            result["holdout"]["reason"] = result["reason"]
        return result
    caps = {name: s["max_notional"] for name, s in metadata["sleeves"].items()}
    params = metadata["portfolio"]
    streams: dict[str, pd.Series] = {k: v for k, v in development.items() if v is not None}
    if cutoff:
        cut = pd.Timestamp(cutoff)
        later: dict[str, pd.Series] = {k: v for k, v in after.items() if v is not None}
        streams = {
            name: pd.concat([dev[dev.index < cut], later[name][later[name].index >= cut]])
            for name, dev in streams.items()
        }
    returns, notional = combine(streams, params, caps)
    if cutoff:
        returns, notional = returns[returns.index >= cut], notional[notional.index >= cut]
        counts = sum(
            sym.get("holdout_trades") and len(sym["holdout_trades"]) or 0
            for s in sleeves.values()
            for sym in s["symbols"]
        )
    else:
        start = returns.index[0] if len(returns) else None
        counts = sum(
            (
                int((trade_returns(sym["development_trades"]).index >= start).sum())
                if start is not None
                else 0
            )
            for s in sleeves.values()
            for sym in s["symbols"]
        )
    metrics = portfolio_metrics(returns, counts)
    p_value = metrics.get("p_value")
    status = "ok" if p_value is not None and p_value < SIGNIFICANCE else "not_significant"
    if metrics.get("sharpe") is None:
        status = "no_unified_trades"
    result.update(
        status=status,
        metrics=metrics,
        monte_carlo={"p_value": p_value, "is_significant": status == "ok"},
        walk_forward={"total_trades": counts, "fold_stability": metrics.get("fold_stability")},
        mean_notional={k: round(float(v), 4) for k, v in notional.mean().items()},
        sleeve_sharpe={
            name: (float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else None)
            for name, s in pd.DataFrame(streams).reindex(returns.index).fillna(0.0).items()
        },
        sleeve_correlation=pd.DataFrame(streams)
        .reindex(returns.index)
        .fillna(0.0)
        .corr()
        .round(3)
        .to_dict(),
        daily=pd.DataFrame({"portfolio": returns}).join(notional.add_prefix("notional_")),
    )
    return result


def _write(run_path: Path, result: Mapping[str, Any]) -> None:
    details = run_path / "grid_details" / PORTFOLIO_SYMBOL
    details.mkdir(parents=True, exist_ok=True)
    daily = result.get("daily")
    if isinstance(daily, pd.DataFrame):
        daily.to_csv(details / "daily.csv", index_label="date")
    payload = {k: v for k, v in result.items() if k != "daily"}
    (details / "config.json").write_text(json.dumps(payload, indent=2, default=str))
    if "metrics" in result:
        (details / "unified_metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    summary = {"portfolio": payload.get("status"), "metrics": payload.get("metrics")}
    (run_path / "results.json").write_text(json.dumps(summary, indent=2, default=str))


def _print(result: Mapping[str, Any]) -> None:
    print(f"\nPORTFOLIO - {str(result['status']).upper()}")
    if result.get("reason"):
        print(f"  Reason: {result['reason']}")
    metrics = result.get("metrics") or {}
    if metrics.get("sharpe") is not None:
        print(
            f"  Sharpe={metrics['sharpe']:.2f} annual={metrics['annual_return']}% "
            f"maxDD={metrics['max_drawdown']:.3f} p={metrics['p_value']:.3f} "
            f"trades={metrics['trades']} slices+={metrics['fold_stability']:.2f}"
        )
        print(
            f"  Sleeve Sharpe: { {k: round(v, 2) for k, v in result['sleeve_sharpe'].items() if v is not None} }"
        )
        print(f"  Mean notional: {result['mean_notional']}")
