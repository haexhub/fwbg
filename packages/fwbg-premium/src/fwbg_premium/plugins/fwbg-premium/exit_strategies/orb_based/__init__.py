"""
ORB-Based Exit Strategy Plugin.

SL = indicator-provided *_sl_dist column, fallback to ATR * sl_mult.
TP = ATR * tp_mult (default) or range * tp_mult (tp_mode="range").

Design rationale:
  - SL is anchored to the structural level that invalidates the trade thesis.
  - When sl_dist_column is explicitly set (via exit_params), the column value
    is the ABSOLUTE SL distance — used as-is, no sl_mult applied.
    E.g. pdl_sl_dist = pd_range/2 → SL exactly at PDL (long) or PDH (short).
  - When auto-detected (no explicit sl_dist_column), sl_mult is applied as a
    buffer multiplier on the raw distance (1.0 = exact boundary, >1.0 = buffer).

tp_mode (exit_params):
  - "atr" (default): TP = ATR * tp_mult. Trailing uses ATR * trail_atr_mult.
  - "range": TP = range_column * tp_mult. Trailing uses range as trail distance.
    Typical config: tp_mult=1.0 (TP = full range), sl_mult=1.4 (SL = 70% of range
    from midpoint entry = 20% beyond range boundary), breakeven_trigger=0.5.
"""
from typing import Dict, Any, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

from fwbg_sdk import BaseExitStrategy, register_exit_strategy
from fwbg.simulation import _simulate_trade_numba, _simulate_trade_trailing_numba, compute_session_mask
from fwbg.simulation.numba_core import _simulate_trade_session_numba
from fwbg.core import GridParams

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext


@register_exit_strategy("orb_based")
class OrbExitStrategy(BaseExitStrategy):
    """
    Exit strategy for ORB (Opening Range Breakout) setups.

    SL: When sl_dist_column is explicitly set in exit_params, uses that column
        as the ABSOLUTE SL distance (no multiplier). When auto-detected, applies
        sl_mult as buffer multiplier. Falls back to ATR * sl_mult when no column.

    TP modes (set via tp_mode in exit_params):
      - "atr" (default): TP = ATR * tp_mult. Trail distance = ATR * trail_atr_mult.
      - "range": TP = range_column * tp_mult. Trail distance = range_column value.
        Example: entry at midpoint, orb_sl_dist = range/2.
        tp_mult=1.0 → TP = full range. sl_mult=1.4 → SL = 70% of range from entry.
    """

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: Union[GridParams, None] = None,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        atr_period: int = 14,
        min_tp_pips: int = 8,
        min_sl_pips: int = 5,
        timeout_bars: int = None,
        return_durations: bool = False,
        **kwargs,
    ) -> tuple:
        """Compute win/loss targets using ORB-range SL and configurable TP source."""
        if params is not None:
            tp_mult = params.tp_value
            sl_mult = params.sl_value
            if params.timeout_bars is not None:
                timeout_bars = params.timeout_bars
            if params.extra:
                atr_period = params.extra.get("atr_period", atr_period)
                min_tp_pips = params.extra.get("min_tp_pips", min_tp_pips)
                min_sl_pips = params.extra.get("min_sl_pips", min_sl_pips)

        exit_params = ctx.exit_params if ctx.exit_params else {}
        sl_dist_column = exit_params.get("sl_dist_column")
        tp_mode = exit_params.get("tp_mode", "atr")
        # Per-combo override from model_hyperparameters (grid search selects rl variant)
        hp_sl_dist = ctx.model_hyperparameters.get("sl_dist_column")
        if hp_sl_dist:
            sl_dist_column = hp_sl_dist

        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)

        atr_v = self._get_atr(df, atr_period)
        sl_dist_v = self._get_sl_dist(df, atr_v, sl_mult, sl_dist_column)

        # Range-based TP: use *_range column instead of ATR
        use_range_tp = tp_mode == "range"
        range_v = self._get_range(df, exit_params) if use_range_tp else None

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips
        slippage = ctx.spread * 0.5
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)
        timeout_val = timeout_bars if timeout_bars else 0

        # Session-aware exits: only exit during session hours.
        # Trades may run through off-session periods (overnight holds).
        # Prefer exit_session hours (wider CFD window), fall back to session hours.
        in_session = None
        s_start = getattr(ctx, "exit_session_start_hour", None)
        if s_start is None:
            s_start = getattr(ctx, "session_start_hour", None)
        s_end = getattr(ctx, "exit_session_end_hour", None)
        if s_end is None:
            s_end = getattr(ctx, "session_end_hour", None)
        if isinstance(s_start, int) and isinstance(s_end, int):
            in_session = compute_session_mask(
                df.index, s_start, s_end,
                ohlc=(opn_v, hgh_v, low_v, cls_v),
            )
        use_session = in_session is not None

        # === EXIT MODIFIER DISPATCH ===
        exit_modifier_name = getattr(ctx, "exit_modifier", None)
        modifier_params = getattr(ctx, "exit_modifier_params", {}) or {}
        use_trailing = bool(exit_modifier_name)
        breakeven_trigger = modifier_params.get("breakeven_trigger", 0.5) if use_trailing else 0.0
        trail_atr_mult = modifier_params.get("trail_atr_mult", 0.5) if use_trailing else 0.0

        n = len(cls_v)
        targets_long = np.zeros(n, dtype=np.float64)
        targets_short = np.zeros(n, dtype=np.float64)
        if return_durations:
            durations_long = np.zeros(n, dtype=np.int64)
            durations_short = np.zeros(n, dtype=np.int64)

        for i in range(n - 1):
            if use_range_tp:
                tp_distance = max(range_v[i] * tp_mult, min_tp_distance)
            else:
                tp_distance = max(atr_v[i] * tp_mult, min_tp_distance)
            sl_distance = max(sl_dist_v[i], min_sl_distance)

            if use_trailing:
                if use_range_tp:
                    trail_distance = range_v[i] if range_v[i] > 0.0 else 0.0
                else:
                    trail_distance = atr_v[i] * trail_atr_mult if trail_atr_mult > 0.0 else 0.0
                result_long, exit_long, _, _ = _simulate_trade_trailing_numba(
                    opn_v, cls_v, hgh_v, low_v, i, 1,
                    tp_distance, sl_distance, ctx.spread, slippage,
                    max_bars, timeout_val, breakeven_trigger, trail_distance, 0.0,
                )
            elif use_session:
                result_long, exit_long, _, _ = _simulate_trade_session_numba(
                    opn_v, cls_v, hgh_v, low_v, i, 1,
                    tp_distance, sl_distance, ctx.spread, slippage,
                    max_bars, timeout_val, in_session,
                )
            else:
                result_long, exit_long, _, _ = _simulate_trade_numba(
                    opn_v, cls_v, hgh_v, low_v, i, 1,
                    tp_distance, sl_distance, ctx.spread, slippage,
                    max_bars, timeout_val,
                )
            if result_long == 1.0:
                targets_long[i] = 1.0
            if return_durations:
                durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

            if use_trailing:
                if use_range_tp:
                    trail_distance = range_v[i] if range_v[i] > 0.0 else 0.0
                else:
                    trail_distance = atr_v[i] * trail_atr_mult if trail_atr_mult > 0.0 else 0.0
                result_short, exit_short, _, _ = _simulate_trade_trailing_numba(
                    opn_v, cls_v, hgh_v, low_v, i, -1,
                    tp_distance, sl_distance, ctx.spread, slippage,
                    max_bars, timeout_val, breakeven_trigger, trail_distance, 0.0,
                )
            elif use_session:
                result_short, exit_short, _, _ = _simulate_trade_session_numba(
                    opn_v, cls_v, hgh_v, low_v, i, -1,
                    tp_distance, sl_distance, ctx.spread, slippage,
                    max_bars, timeout_val, in_session,
                )
            else:
                result_short, exit_short, _, _ = _simulate_trade_numba(
                    opn_v, cls_v, hgh_v, low_v, i, -1,
                    tp_distance, sl_distance, ctx.spread, slippage,
                    max_bars, timeout_val,
                )
            if result_short == 1.0:
                targets_short[i] = 1.0
            if return_durations:
                durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

        if return_durations:
            return targets_long, targets_short, durations_long, durations_short
        return targets_long, targets_short

    def resolve_distances(
        self,
        df: pd.DataFrame,
        tp: float,
        sl: float,
        ctx: "SimulationContext",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """TP/SL distances. TP source depends on tp_mode (atr or range)."""
        exit_params = ctx.exit_params if ctx.exit_params else {}
        atr_period = exit_params.get("atr_period", 14)
        min_tp_pips = exit_params.get("min_tp_pips", 8)
        min_sl_pips = exit_params.get("min_sl_pips", 5)
        tp_mode = exit_params.get("tp_mode", "atr")
        sl_dist_column = exit_params.get("sl_dist_column")
        # Per-combo override from model_hyperparameters (grid search selects rl variant)
        hp_sl_dist = ctx.model_hyperparameters.get("sl_dist_column")
        if hp_sl_dist:
            sl_dist_column = hp_sl_dist

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips

        atr_v = self._get_atr(df, atr_period)
        sl_raw = self._get_sl_dist(df, atr_v, sl, sl_dist_column)

        if tp_mode == "range":
            range_v = self._get_range(df, exit_params)
            tp_dists = np.maximum(range_v * tp, min_tp_distance)
        else:
            tp_dists = np.maximum(atr_v * tp, min_tp_distance)
        sl_dists = np.maximum(sl_raw, min_sl_distance)
        return tp_dists, sl_dists

    def get_cache_key(self, params: dict) -> str:
        tp_mult = params.get("tp_mult", 0)
        sl_mult = params.get("sl_mult", 0)
        timeout = params.get("timeout_bars")
        timeout_str = str(timeout) if timeout else "none"
        return f"orb_tp{tp_mult:.2f}_sl{sl_mult:.2f}_to{timeout_str}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "tp_mode": "atr",
            "tp_mult": 2.0,
            "sl_mult": 1.0,
            "atr_period": 14,
            "min_tp_pips": 8,
            "min_sl_pips": 5,
            "timeout_bars": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "tp_mode": {
                "type": "choice",
                "default": "atr",
                "description": "TP source: 'atr' = ATR-based, 'range' = range column (for ORB setups with midpoint entry)",
                "choices": ["atr", "range"],
            },
            "tp_mult": {
                "type": "float",
                "default": 2.0,
                "description": (
                    "TP multiplier. In tp_mode='atr': ATR * tp_mult. "
                    "In tp_mode='range': range * tp_mult (1.0 = full range as TP)."
                ),
                "min": 0.5,
                "max": 10.0,
                "step": 0.5,
            },
            "sl_mult": {
                "type": "float",
                "default": 1.0,
                "description": (
                    "Multiplier on orb_sl_dist. "
                    "1.0 = SL at opposite ORB boundary, >1.0 adds buffer beyond. "
                    "E.g. 1.4 with entry at midpoint = SL at 70% of range = 20% beyond boundary. "
                    "Fallback to ATR * sl_mult if no orb_sl_dist column present."
                ),
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for TP calculation (tp_mode=atr) and SL fallback",
                "min": 1,
                "max": 200,
                "step": 1,
            },
            "min_tp_pips": {
                "type": "int",
                "default": 8,
                "description": "Minimum TP in spread multiples (floor for low-vol environments)",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "min_sl_pips": {
                "type": "int",
                "default": 5,
                "description": "Minimum SL in spread multiples (floor for low-vol environments)",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "timeout_bars": {
                "type": "int",
                "default": None,
                "description": "Close trade after N bars if neither TP nor SL is hit",
                "min": 1,
                "max": 500,
                "step": 1,
                "required": False,
            },
        }

    # --- Private helpers ---

    @staticmethod
    def _get_atr(df: pd.DataFrame, atr_period: int) -> np.ndarray:
        """Return ATR array: use precomputed column if available, else compute."""
        if "_atr" in df.columns:
            atr_v = df["_atr"].values.astype(np.float64)
        elif "vol_atr" in df.columns:
            atr_v = df["vol_atr"].values.astype(np.float64)
        else:
            import ta
            atr_series = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=atr_period
            )
            atr_v = atr_series.values.astype(np.float64)
        return np.nan_to_num(atr_v, nan=0.0)

    @staticmethod
    def _get_range(df: pd.DataFrame, exit_params: dict = None) -> np.ndarray:
        """Return range array for tp_mode='range'.

        Looks for range_column in exit_params, then auto-detects *_range columns
        (excluding *_vs_atr variants). Falls back to zeros.
        """
        ep = exit_params or {}
        range_column = ep.get("range_column")
        if range_column and range_column in df.columns:
            return np.nan_to_num(df[range_column].values.astype(np.float64), nan=0.0)
        # Auto-detect: first column ending in _range (but not _vs_atr or _vs_range)
        col = next(
            (c for c in df.columns
             if (c == "orb_range" or c.endswith("_orb_range") or c.endswith("_range"))
             and "_vs_" not in c),
            None,
        )
        if col is not None:
            return np.nan_to_num(df[col].values.astype(np.float64), nan=0.0)
        return np.zeros(len(df), dtype=np.float64)

    @staticmethod
    def _get_sl_dist(
        df: pd.DataFrame,
        atr_v: np.ndarray,
        sl_mult: float,
        sl_dist_column: str = None,
    ) -> np.ndarray:
        """Return SL distances per bar.

        When sl_dist_column is explicitly set: column value is the ABSOLUTE SL
        distance (no sl_mult applied). NaN rows fall back to ATR * sl_mult.

        When auto-detected (no explicit sl_dist_column): applies sl_mult as
        buffer multiplier on the raw distance. NaN rows fall back to ATR * sl_mult.

        No column found at all: pure ATR * sl_mult.
        """
        explicit = sl_dist_column and sl_dist_column in df.columns
        if explicit:
            sl_col = sl_dist_column
        else:
            sl_col = next(
                (c for c in df.columns if c == "orb_sl_dist" or c.endswith("_sl_dist")),
                None,
            )
        if sl_col is not None:
            raw = df[sl_col].values.astype(np.float64)
            fallback = atr_v * sl_mult
            if explicit:
                return np.where(np.isnan(raw), fallback, raw).astype(np.float64)
            return np.where(np.isnan(raw), fallback, raw * sl_mult).astype(np.float64)
        return (atr_v * sl_mult).astype(np.float64)


__all__ = ["OrbExitStrategy"]
