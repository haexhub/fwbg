import pandas as pd


def build_state_text(row: pd.Series) -> str:
    # ASSUMPTION: only single-bar state is sent today; no multi-bar OHLC window yet.
    """Format one OHLC bar and its nonmissing indicators as Jev state text."""
    parts = [f"O={row['O']:.5g} H={row['H']:.5g} L={row['L']:.5g} C={row['C']:.5g}"]
    for name, value in row.items():
        if name in ("O", "H", "L", "C"):
            continue
        if pd.isna(value):
            continue
        parts.append(f"{name}={value:.5g}")
    return " ".join(parts)


def build_questions(tp_pips: float, sl_pips: float, horizon_bars: int) -> dict:
    """Build matching long and short barrier questions using pip distances."""
    def _question(direction: str) -> dict:
        """Describe one trade direction and its win criterion."""
        return {
            "type": "noul",
            "instructions": (
                f"Opening a {direction} position now with take-profit "
                f"{tp_pips} pips and stop-loss {sl_pips} pips: is take-profit "
                f"reached before stop-loss, within the next {horizon_bars} bars?"
            ),
            "criteria": {
                "true": f"Take-profit reached before stop-loss, within {horizon_bars} bars",
                "false": f"Stop-loss reached first, or neither reached within {horizon_bars} bars",
            },
        }

    return {
        "is_long_win": _question("long"),
        "is_short_win": _question("short"),
    }
