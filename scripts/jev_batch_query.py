"""
Batch-Query beider Jev-Provider (jev_semif, jev_official) gegen historische
EUR/USD H1-Bars, gecacht nach Zeitstempel/Provider/Fenstergroesse.

Siehe docs/plans/2026-09-24-jev-signal-provider-design.md fuer Kontext.

Ablauf:
1. EUR/USD H1 (60 Tage, yfinance) laden
2. fwbg-core Indikatoren (momentum, volatility, price_action, supertrend)
   berechnen -- "trend"-Plugin fehlt im Repo (leeres Verzeichnis, kein
   __init__.py), supertrend dient hier als Ersatz
3. Ground-Truth-Label (triple barrier: up_first/down_first/neither) pro Bar
   berechnen, Barrieren = 1.5x aktueller ATR(14) (dynamisch pro Bar) --
   die fixed-Exit-Strategy-Defaults (TP=30/SL=20 Spread-Multiplikatoren)
   ergaben empirisch 88% "neither" (Barrieren zu weit fuer H1/8-10h), siehe
   docs/plans/2026-09-24-jev-signal-provider-design.md fuer den Vergleich
4. Pro Bar einen Prompt bauen (Indikator-Snapshot + rohes OHLC-Fenster) und
   beide Provider abfragen, Ergebnis nach data/jev_cache/*.csv cachen
   (resumable: bereits gecachte (timestamp, provider, window) werden geskippt)
"""

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]
import yfinance as yf

from fwbg.core.registry import get_indicator

CACHE_DIR = Path("data/jev_cache")
CACHE_FILE = CACHE_DIR / "eurusd_h1_results.csv"

YF_TICKER = "EURUSD=X"
YF_INTERVAL = "1h"
YF_PERIOD = "60d"  # yfinance max lookback for hourly bars

EURUSD_SPREAD = 0.00018  # src/fwbg/data/assets.py DEFAULT_ASSETS["EURUSD"]
EURUSD_POINT = 0.0001
ATR_MULT = 1.5  # barrier distance = ATR_MULT * ATR(14) at entry bar
TIMEOUT_BARS = (
    8  # horizon in H1 bars; empirically ~29/34/37% class split at ATR_MULT=1.5
)

SEMIF_URL = "https://llms.itemis.cloud/typesafe/v1/systemone"
OFFICIAL_URL = "https://openrouter.ai/api/v1/systemone"
OFFICIAL_MODEL = "typesafe/jev-1.13"

INDICATOR_PLUGINS = ["momentum", "volatility", "price_action", "supertrend"]


def _load_keys(path: Path = Path("keys.json")) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise ValueError(
            f"{path} must contain a JSON object with string keys and values"
        )
    return raw


def load_semif_key(path: Path = Path("keys.json")) -> str:
    return _load_keys(path)["semif-itemis"]


def load_official_key(path: Path = Path("keys.json")) -> str:
    return _load_keys(path)["openrouter"]


def fetch_ohlc() -> pd.DataFrame:
    """EUR/USD H1 OHLC, letzte 60 Tage (yfinance-Limit fuer Stundenbars)."""
    print(f"Loading {YF_TICKER} {YF_INTERVAL} ({YF_PERIOD})...")
    df = yf.download(
        YF_TICKER,
        period=YF_PERIOD,
        interval=YF_INTERVAL,
        progress=False,
        auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "O", "High": "H", "Low": "L", "Close": "C"})
    df = df[["O", "H", "L", "C"]].astype(float)
    df.index.name = "T"
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Haengt fwbg-core-Indikatorspalten an (jedes Plugin shiftet bereits
    intern gegen Lookahead, siehe shift_features() in den jeweiligen
    __init__.py)."""
    out = df.copy()
    for name in INDICATOR_PLUGINS:
        cls = get_indicator(name)
        feats = cls().compute(df)
        for col in feats.columns:
            out[col] = feats[col]
    return out


def triple_barrier_labels(
    df: pd.DataFrame, atr_mult: float, timeout_bars: int
) -> list[str | None]:
    """Ground truth pro Bar: welche Barriere (Close[i] +/- atr_mult*ATR[i])
    wird zuerst in den naechsten timeout_bars Bars beruehrt. Barrieren sind
    pro Bar dynamisch (aktuelle Vola), nicht fix -- sonst dominiert "neither"
    in ruhigen Marktphasen und die Frage wird trivial. Bewusst ohne
    Spread/Slippage-Modellierung (reine Preisschwellen-Frage, identisch zu
    dem was Jev im Prompt gefragt wird)."""
    n = len(df)
    highs = df["H"].values
    lows = df["L"].values
    closes = df["C"].values
    atr = df["_atr"].values
    labels: list[str | None] = [None] * n

    for i in range(n - timeout_bars):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        entry = closes[i]
        upper = entry + atr_mult * atr[i]
        lower = entry - atr_mult * atr[i]
        label = "neither"
        for j in range(i + 1, i + 1 + timeout_bars):
            up_hit = highs[j] >= upper
            down_hit = lows[j] <= lower
            if up_hit and down_hit:
                # Match the native simulator's conservative stop-loss rule
                # when no sub-bar data is available to establish an order.
                label = "down_first"
                break
            if up_hit:
                label = "up_first"
                break
            if down_hit:
                label = "down_first"
                break
        labels[i] = label
    return labels


def build_state_text(df: pd.DataFrame, i: int, window: int) -> str:
    """Rohes OHLC-Fenster (letzte `window` Bars) + aktueller Indikator-Snapshot."""
    start = max(0, i - window + 1)
    recent = df.iloc[start : i + 1]
    bars_txt = ", ".join(
        f"{ts.strftime('%m-%d %H:%M')}: O={r.O:.5f} H={r.H:.5f} L={r.L:.5f} C={r.C:.5f}"
        for ts, r in recent.iterrows()
    )
    indicator_cols = [c for c in df.columns if c not in ("O", "H", "L", "C")]
    row = df.iloc[i]
    indicators_txt = ", ".join(
        f"{c}={row[c]:.5f}" for c in indicator_cols if pd.notna(row[c])
    )
    return (
        f"EUR/USD H1 bars (last {len(recent)}): {bars_txt}. "
        f"Indicators at last close: {indicators_txt}."
    )


def build_question(
    entry_price: float,
    tp_dist: float,
    sl_dist: float,
    horizon_bars: int,
    reverse_order: bool = False,
) -> dict[str, object]:
    """Explizite Ziel-/Stop-Preise + Pip-Angabe statt roher Deltas -- ein
    Choice-Logit-Readout-Modell soll den Referenzpreis nicht selbst aus dem
    OHLC-Fenster inferieren muessen.

    reverse_order: fuer den Positions-Bias-Diagnosetest -- vertauscht die
    Reihenfolge der criteria-Keys (down_first vor up_first), um zu pruefen
    ob jev_official's ~95%-down_first-Bias an der Frage/Formulierung haengt
    oder an der reinen Positionierung im JSON."""
    target = entry_price + tp_dist
    stop = entry_price - sl_dist
    tp_pips = tp_dist / EURUSD_POINT
    sl_pips = sl_dist / EURUSD_POINT
    criteria = {
        "up_first": (
            f"Price rises to at least {target:.5f} "
            f"(+{tp_pips:.1f} pips) before falling to {stop:.5f}"
        ),
        "down_first": (
            f"Price falls to at least {stop:.5f} "
            f"(-{sl_pips:.1f} pips) before rising to {target:.5f}"
        ),
        "neither": (
            f"Neither {target:.5f} nor {stop:.5f} is reached within the horizon"
        ),
    }
    if reverse_order:
        criteria = {
            "down_first": criteria["down_first"],
            "up_first": criteria["up_first"],
            "neither": criteria["neither"],
        }
    return {
        "barrier_outcome": {
            "type": "choice",
            "instructions": (
                f"Current EUR/USD price is {entry_price:.5f}. "
                f"Which barrier is reached first in the next {horizon_bars} bars?"
            ),
            "criteria": criteria,
        }
    }


def _session_for(ts: pd.Timestamp) -> str:
    hour = ts.tz_convert("UTC").hour
    if 0 <= hour < 7:
        return "Asian session (typically quiet)"
    if 7 <= hour < 12:
        return "London session"
    if 12 <= hour < 16:
        return "London/New York overlap (high liquidity)"
    if 16 <= hour < 21:
        return "New York session"
    return "late New York / pre-Asian (thin liquidity)"


def _rsi_desc(rsi: float) -> str:
    if rsi < 30:
        return "oversold"
    if rsi < 45:
        return "bearish-leaning"
    if rsi < 55:
        return "neutral"
    if rsi < 70:
        return "bullish-leaning"
    return "overbought"


def _vol_desc(pct_rank: float) -> str:
    if pct_rank < 0.33:
        return "low"
    if pct_rank < 0.66:
        return "normal"
    return "elevated"


def build_state_text_narrative(df: pd.DataFrame, i: int, window: int) -> str:
    """Kuratierte, semantisch beschriebene Zusammenfassung statt rohem
    59-Spalten-Dump -- Hypothese: Jev ist auf kompakte, textartige
    Zustandsbeschreibungen trainiert (Routing/Klassifikation-artige Tasks),
    nicht auf dichte quantitative Finanz-Feature-Vektoren mit 5 Nachkommastellen."""
    row = df.iloc[i]
    start = max(0, i - window + 1)
    window_df = df.iloc[start : i + 1]
    net_pips = (row["C"] - window_df["C"].iloc[0]) / EURUSD_POINT
    window_high = window_df["H"].max()
    window_low = window_df["L"].min()
    last_closes = ", ".join(f"{c:.5f}" for c in window_df["C"].iloc[-5:])

    trend_desc = "uptrend" if row["st_direction"] == 1 else "downtrend"
    structure = row["pa_trend_structure"]
    rsi = row["mom_rsi_14"]
    vol_pct = row["vol_atr_pct_14_rank"]

    if row["pa_bullish_streak"] > 0:
        streak_desc = f"{int(row['pa_bullish_streak'])} consecutive bullish candles"
    elif row["pa_bearish_streak"] > 0:
        streak_desc = f"{int(row['pa_bearish_streak'])} consecutive bearish candles"
    else:
        streak_desc = "no clear candle streak"

    return (
        f"EUR/USD is trading at {row['C']:.5f}, during the {_session_for(df.index[i])}. "
        f"Over the last {len(window_df)} hours price moved from {window_df['C'].iloc[0]:.5f} "
        f"to {row['C']:.5f} ({net_pips:+.1f} pips), ranging between {window_low:.5f} and "
        f"{window_high:.5f}. Last 5 closes: {last_closes}. "
        f"Short-term trend (Supertrend): {trend_desc}. "
        f"Medium-term structure score: {structure:+.0f}/10 "
        f"(net higher-highs/higher-lows vs. lower-highs/lower-lows over the "
        f"recent window; positive favors bullish structure). "
        f"Momentum: RSI(14)={rsi:.0f}, {_rsi_desc(rsi)}. "
        f"Volatility: {_vol_desc(vol_pct)} (current ATR at the {vol_pct * 100:.0f}th percentile "
        f"of the last 14 periods). "
        f"Price action: {streak_desc}, sitting at {row['pa_range_pos'] * 100:.0f}% of the "
        f"recent trading range."
    )


def build_question_qualitative(
    horizon_bars: int, reverse_order: bool = False
) -> dict[str, object]:
    """Keine Preis-Arithmetik mehr im Prompt (kein Ziel-/Stop-Preisvergleich) --
    reine qualitative Richtungseinschaetzung. Barrieren-Ground-Truth wird
    weiterhin extern berechnet (triple_barrier_labels), nur nicht mehr dem
    Modell als Vergleichswert vorgegeben.

    reverse_order: siehe build_question() -- der orderbias-Diagnosetest zeigt
    einen massiven Positions-Bias bei jev_official (dominante Antwort = Key
    an Position 2, unabhaengig vom Inhalt). Aufrufer sollten dies alternierend
    pro Bar setzen (z.B. i % 2 == 0), damit der Bias sich in der Aggregat-
    statistik aufhebt statt ein Ergebnis systematisch zu verzerren."""
    criteria = {
        "up_first": (
            "A clear sustained upward move is more likely than a "
            "downward move or a range-bound/sideways market"
        ),
        "down_first": (
            "A clear sustained downward move is more likely than an "
            "upward move or a range-bound/sideways market"
        ),
        "neither": (
            "Price is more likely to stay range-bound with no clear "
            "sustained directional move"
        ),
    }
    if reverse_order:
        criteria = {
            "down_first": criteria["down_first"],
            "up_first": criteria["up_first"],
            "neither": criteria["neither"],
        }
    return {
        "directional_bias": {
            "type": "choice",
            "instructions": (
                f"Over the next {horizon_bars} hours, which is the more "
                f"likely price behavior for EUR/USD?"
            ),
            "criteria": criteria,
        }
    }


def _post_with_retry(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    retries: int = 3,
) -> dict[str, object]:
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                if not isinstance(result, dict):
                    return {
                        "error": True,
                        "status": 200,
                        "body": "response is not a JSON object",
                    }
                return result
            if resp.status_code == 429 and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return {"error": True, "status": resp.status_code, "body": resp.text}
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return {"error": True, "status": None, "body": str(e)}
    return {"error": True, "status": None, "body": "exhausted retries"}


def call_semif(
    semif_key: str, state: str, questions: Mapping[str, object]
) -> dict[str, object]:
    return _post_with_retry(
        SEMIF_URL,
        {"Authorization": f"Bearer {semif_key}", "Content-Type": "application/json"},
        {"state": state, "questions": questions},
    )


def call_official(
    official_key: str, state: str, questions: Mapping[str, object]
) -> dict[str, object]:
    return _post_with_retry(
        OFFICIAL_URL,
        {"Authorization": f"Bearer {official_key}", "Content-Type": "application/json"},
        {"model": OFFICIAL_MODEL, "state": state, "questions": questions},
    )


def load_cache(cache_file: Path) -> pd.DataFrame:
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["timestamp"])
    return pd.DataFrame(
        columns=[
            "timestamp",
            "provider",
            "window",
            "label",
            "choice",
            "prob_up_first",
            "prob_down_first",
            "prob_neither",
            "confidence",
            "input_tokens",
            "output_tokens",
            "cost",
            "error",
        ]
    )


def append_cache_row(cache_file: Path, row: Mapping[str, object]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not cache_file.exists()
    df_row.to_csv(cache_file, mode="a", header=write_header, index=False)


def run_batch(
    df: pd.DataFrame,
    labels: list[str | None],
    bar_indices: list[int],
    windows: list[int],
    semif_key: str,
    official_key: str,
    *,
    cache_file: Path = CACHE_FILE,
    question_key: str = "barrier_outcome",
    build_state_fn: Callable[[pd.DataFrame, int, int], str] = build_state_text,
    build_question_fn: Callable[[pd.DataFrame, int], dict[str, object]] | None = None,
) -> None:
    """build_question_fn(df, i) -> questions dict; defaults to the ATR-based
    explicit-price barrier question. question_key must match the top-level
    key of that dict (the provider echoes it back in `answers`)."""
    if build_question_fn is None:

        def build_question_fn(df: pd.DataFrame, i: int) -> dict[str, object]:
            entry_price = df["C"].iloc[i]
            barrier_dist = ATR_MULT * df["_atr"].iloc[i]
            return build_question(entry_price, barrier_dist, barrier_dist, TIMEOUT_BARS)

    cache = load_cache(cache_file)
    already = set(
        zip(cache["timestamp"].astype(str), cache["provider"], cache["window"])
    )
    total = len(bar_indices) * len(windows) * 2
    done = 0

    for i in bar_indices:
        ts = df.index[i]
        label = labels[i]
        if label is None:
            continue
        questions = build_question_fn(df, i)

        for window in windows:
            state = build_state_fn(df, i, window)

            for provider_name, key, call_fn in (
                ("jev_semif", semif_key, call_semif),
                ("jev_official", official_key, call_official),
            ):
                done += 1
                key_tuple = (str(ts), provider_name, window)
                if key_tuple in already:
                    continue

                result = call_fn(key, state, questions)
                row = {
                    "timestamp": ts,
                    "provider": provider_name,
                    "window": window,
                    "label": label,
                    "choice": None,
                    "prob_up_first": None,
                    "prob_down_first": None,
                    "prob_neither": None,
                    "confidence": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "cost": None,
                    "error": None,
                }
                if result.get("error"):
                    row["error"] = f"{result.get('status')}: {result.get('body')}"
                else:
                    answers = result.get("answers")
                    if not isinstance(answers, dict):
                        raise ValueError("provider response has no answers object")
                    answer = answers.get(question_key)
                    if not isinstance(answer, dict):
                        raise ValueError(
                            f"provider response has no {question_key!r} answer"
                        )
                    choice = answer.get("choice")
                    probs = answer.get("probabilities")
                    if not isinstance(choice, str) or not isinstance(probs, dict):
                        raise ValueError(
                            "provider answer has an invalid choice/probabilities shape"
                        )
                    probability_values = {
                        name: probs.get(name)
                        for name in ("up_first", "down_first", "neither")
                    }
                    if any(
                        not isinstance(value, (int, float))
                        or not np.isfinite(value)
                        or not 0 <= value <= 1
                        for value in probability_values.values()
                    ):
                        raise ValueError(
                            "provider answer contains invalid probabilities"
                        )
                    usage = result.get("usage", {})
                    if not isinstance(usage, dict):
                        usage = {}
                    row.update(
                        {
                            "choice": choice,
                            "prob_up_first": probability_values["up_first"],
                            "prob_down_first": probability_values["down_first"],
                            "prob_neither": probability_values["neither"],
                            "confidence": answer.get("confidence"),
                            "input_tokens": usage.get("input_tokens"),
                            "output_tokens": usage.get("output_tokens"),
                            "cost": usage.get("cost"),
                        }
                    )
                append_cache_row(cache_file, row)
                print(
                    f"  [{done}/{total}] {ts} {provider_name} w={window} -> "
                    f"{row['choice'] or row['error']}"
                )


if __name__ == "__main__":
    import sys

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "sensitivity"

    semif_key = load_semif_key()
    official_key = load_official_key()

    ohlc = fetch_ohlc()
    features = compute_features(ohlc)
    labels = triple_barrier_labels(features, ATR_MULT, TIMEOUT_BARS)

    valid_indices = [i for i, label in enumerate(labels) if label is not None]

    sample = valid_indices[:: max(1, len(valid_indices) // 40)][:40]

    if mode == "sensitivity":
        # ~40 gleichmaessig verteilte Bars, 3 Fenstergroessen, beide Provider
        run_batch(features, labels, sample, [10, 20, 50], semif_key, official_key)
    elif mode == "full":
        window = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        run_batch(features, labels, valid_indices, [window], semif_key, official_key)
    elif mode == "orderbias":
        # Diagnose: vertauscht nur die criteria-Key-Reihenfolge, alles
        # andere (Bars, Fenster, Frage-Inhalt) identisch zu config 3.
        def reversed_question(df, i):
            entry_price = df["C"].iloc[i]
            barrier_dist = ATR_MULT * df["_atr"].iloc[i]
            return build_question(
                entry_price,
                barrier_dist,
                barrier_dist,
                TIMEOUT_BARS,
                reverse_order=True,
            )

        run_batch(
            features,
            labels,
            sample,
            [20],
            semif_key,
            official_key,
            cache_file=CACHE_DIR / "eurusd_h1_results_orderbias.csv",
            build_question_fn=reversed_question,
        )
    elif mode == "narrative":
        # Kuratierte narrative State-Beschreibung + rein qualitative Frage
        # (keine Ziel-/Stop-Preis-Arithmetik mehr im Prompt).
        sample_order = {
            index: position % 2 == 0 for position, index in enumerate(sample)
        }

        def qualitative_question(df: pd.DataFrame, i: int) -> dict[str, object]:
            return build_question_qualitative(
                TIMEOUT_BARS, reverse_order=sample_order[i]
            )

        run_batch(
            features,
            labels,
            sample,
            [10, 20, 50],
            semif_key,
            official_key,
            cache_file=CACHE_DIR / "eurusd_h1_results_narrative.csv",
            question_key="directional_bias",
            build_state_fn=build_state_text_narrative,
            build_question_fn=qualitative_question,
        )
    else:
        print(
            f"Unknown mode: {mode} (use 'sensitivity', 'full <window>', "
            f"'orderbias' or 'narrative')"
        )
