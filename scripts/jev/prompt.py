import pandas as pd


def build_state_text(row: pd.Series, ohlc_window: pd.DataFrame | None) -> str:
    parts = [f"O={row['O']:.5g} H={row['H']:.5g} L={row['L']:.5g} C={row['C']:.5g}"]
    for name, value in row.items():
        if name in ("O", "H", "L", "C"):
            continue
        if pd.isna(value):
            continue
        parts.append(f"{name}={value:.5g}")
    return " ".join(parts)


def build_questions(tp_pips: float, sl_pips: float, horizon_bars: int) -> dict:
    def _question(direction: str) -> dict:
        return {
            "type": "noul",
            "instructions": (
                f"Opening a {direction} position now with take-profit "
                f"{tp_pips} pips and stop-loss {sl_pips} pips: is take-profit "
                f"reached before stop-loss or before {horizon_bars} bars pass?"
            ),
            "criteria": {
                "true": "Take-profit reached first",
                "false": "Stop-loss reached first, or neither within the horizon",
            },
        }

    return {
        "is_long_win": _question("long"),
        "is_short_win": _question("short"),
    }
