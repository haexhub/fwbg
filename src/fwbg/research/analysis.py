"""Put an MFE/MAE exit analysis on record before it reads data.

``fwbg analyze`` and ``POST /api/exploration/exit-optimization`` both go
through :func:`record_analysis`, so an analysis is gated alike from the CLI
and the dashboard, and an identical one is a duplicate whichever path ran it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fwbg.api.workspace import get_research_dir
from fwbg.research.card import CARD_ID_RE, CardError, load_card
from fwbg.research.gate import Denial, DenialCode
from fwbg.research.ledger import Trial
from fwbg.research.runs import GateDenied
from fwbg.research.study import register_trial


def analysis_config(
    exit_strategy: str,
    exit_params: Mapping[str, Any],
    max_bars: int,
    timeframe: str,
    datasource: str | None = None,
    data_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """The configuration an analysis is fingerprinted by; unset fields are left out."""
    config = {
        "analysis": "mfe_mae",
        "exit_strategies": [{"name": exit_strategy, "params": dict(exit_params)}],
        "max_bars": max_bars,
        "timeframe": timeframe,
        "datasource": datasource,
        "_cli_data_path": data_path,
        "start_date": start_date,
        "end_date": end_date,
    }
    return {key: value for key, value in config.items() if value is not None}


def record_analysis(
    research_id: str | None,
    config: Mapping[str, Any],
    symbols: Sequence[str],
    note: str | None,
    reason: str | None = None,
) -> tuple[Trial, dict[str, str | None]]:
    """Record the analysis as a trial and return it with the window it may read.

    Like a gated run, the analysis is recorded before it reads any data, so a
    refusal leaves the data unseen, and an open-ended window ends at the card's
    holdout cutoff (end exclusive). It is an iteration without metrics: it
    evaluates no strategy, so it adds no evaluation to the card's trial count,
    but its configuration and window are on record and a repeat needs *reason*.

    Raises:
        GateDenied: the register refused the analysis; nothing was recorded.
    """
    if not research_id:
        message = "a research id is required: the register must know which data shaped a strategy"
        raise GateDenied([Denial(DenialCode.MISSING_RESEARCH_ID, message)])
    config = dict(config)
    if "end_date" not in config:
        config["end_date"] = _holdout_cutoff(research_id)
    trial = register_trial(research_id, config, symbols, {}, note or "", reason=reason)
    return trial, {"start_date": config.get("start_date"), "end_date": config["end_date"]}


def _holdout_cutoff(research_id: str) -> str:
    """The card's holdout cutoff as YYYY-MM-DD; raises GateDenied without a readable card."""
    try:
        card = None
        if CARD_ID_RE.fullmatch(research_id):
            card = load_card(get_research_dir(create=False), research_id)
    except (OSError, CardError) as exc:
        denial = Denial(DenialCode.REGISTER_UNREADABLE, f"research register: {exc}")
        raise GateDenied([denial]) from exc
    if card is None:
        raise GateDenied([Denial(DenialCode.CARD_NOT_FOUND, f"no research card {research_id}")])
    return card.holdout_cutoff.isoformat()
