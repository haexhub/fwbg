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
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from fwbg.core.registry import get_indicator

CACHE_DIR = "./data/jev_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = f"{CACHE_DIR}/eurusd_h1_results.csv"

YF_TICKER = "EURUSD=X"
YF_INTERVAL = "1h"
YF_PERIOD = "60d"  # yfinance max lookback for hourly bars

EURUSD_SPREAD = 0.00018  # src/fwbg/data/assets.py DEFAULT_ASSETS["EURUSD"]
EURUSD_POINT = 0.0001
ATR_MULT = 1.5  # barrier distance = ATR_MULT * ATR(14) at entry bar
TIMEOUT_BARS = 8  # horizon in H1 bars; empirically ~29/34/37% class split at ATR_MULT=1.5

SEMIF_URL = "https://llms.itemis.cloud/typesafe/v1/systemone"
OFFICIAL_URL = "https://openrouter.ai/api/v1/systemone"
OFFICIAL_MODEL = "typesafe/jev-1.13"

INDICATOR_PLUGINS = ["momentum", "volatility", "price_action", "supertrend"]


def _load_keys() -> dict:
    with open("keys.json") as f:
        return json.load(f)


def load_semif_key() -> str:
    return _load_keys()["semif-itemis"]


def load_official_key() -> str:
    return _load_keys()["openrouter"]


def fetch_ohlc() -> pd.DataFrame:
    """EUR/USD H1 OHLC, letzte 60 Tage (yfinance-Limit fuer Stundenbars)."""
    print(f"Loading {YF_TICKER} {YF_INTERVAL} ({YF_PERIOD})...")
    df = yf.download(YF_TICKER, period=YF_PERIOD, interval=YF_INTERVAL,
                      progress=False, auto_adjust=True)
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


def triple_barrier_labels(df: pd.DataFrame, atr_mult: float,
                          timeout_bars: int) -> list:
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
    labels = [None] * n

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
                # both touched in the same bar: resolve via close-proximity
                label = "up_first" if abs(highs[j] - upper) <= abs(lower - lows[j]) else "down_first"
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
    recent = df.iloc[start:i + 1]
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


def build_question(entry_price: float, tp_dist: float, sl_dist: float,
                    horizon_bars: int) -> dict:
    """Explizite Ziel-/Stop-Preise + Pip-Angabe statt roher Deltas -- ein
    Choice-Logit-Readout-Modell soll den Referenzpreis nicht selbst aus dem
    OHLC-Fenster inferieren muessen."""
    target = entry_price + tp_dist
    stop = entry_price - sl_dist
    tp_pips = tp_dist / EURUSD_POINT
    sl_pips = sl_dist / EURUSD_POINT
    return {
        "barrier_outcome": {
            "type": "choice",
            "instructions": (
                f"Current EUR/USD price is {entry_price:.5f}. "
                f"Which barrier is reached first in the next {horizon_bars} bars?"
            ),
            "criteria": {
                "up_first": (
                    f"Price rises to at least {target:.5f} "
                    f"(+{tp_pips:.1f} pips) before falling to {stop:.5f}"
                ),
                "down_first": (
                    f"Price falls to at least {stop:.5f} "
                    f"(-{sl_pips:.1f} pips) before rising to {target:.5f}"
                ),
                "neither": (
                    f"Neither {target:.5f} nor {stop:.5f} is reached "
                    f"within the horizon"
                ),
            },
        }
    }


def _post_with_retry(url: str, headers: dict, payload: dict, retries: int = 3) -> dict:
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()
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


def call_semif(semif_key: str, state: str, questions: dict) -> dict:
    return _post_with_retry(
        SEMIF_URL,
        {"Authorization": f"Bearer {semif_key}", "Content-Type": "application/json"},
        {"state": state, "questions": questions},
    )


def call_official(official_key: str, state: str, questions: dict) -> dict:
    return _post_with_retry(
        OFFICIAL_URL,
        {"Authorization": f"Bearer {official_key}", "Content-Type": "application/json"},
        {"model": OFFICIAL_MODEL, "state": state, "questions": questions},
    )


def load_cache() -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
    return pd.DataFrame(columns=[
        "timestamp", "provider", "window", "label", "choice",
        "prob_up_first", "prob_down_first", "prob_neither",
        "confidence", "input_tokens", "output_tokens", "cost", "error",
    ])


def append_cache_row(row: dict) -> None:
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(CACHE_FILE)
    df_row.to_csv(CACHE_FILE, mode="a", header=write_header, index=False)


def run_batch(df: pd.DataFrame, labels: list, bar_indices: list, windows: list,
              semif_key: str, official_key: str) -> None:
    cache = load_cache()
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
        entry_price = df["C"].iloc[i]
        barrier_dist = ATR_MULT * df["_atr"].iloc[i]
        questions = build_question(entry_price, barrier_dist, barrier_dist, TIMEOUT_BARS)

        for window in windows:
            state = build_state_text(df, i, window)

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
                    "timestamp": ts, "provider": provider_name, "window": window,
                    "label": label, "choice": None,
                    "prob_up_first": None, "prob_down_first": None, "prob_neither": None,
                    "confidence": None, "input_tokens": None, "output_tokens": None,
                    "cost": None, "error": None,
                }
                if result.get("error"):
                    row["error"] = f"{result.get('status')}: {result.get('body')}"
                else:
                    answer = result["answers"]["barrier_outcome"]
                    probs = answer["probabilities"]
                    usage = result.get("usage", {})
                    row.update({
                        "choice": answer["choice"],
                        "prob_up_first": probs.get("up_first"),
                        "prob_down_first": probs.get("down_first"),
                        "prob_neither": probs.get("neither"),
                        "confidence": answer.get("confidence"),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "cost": usage.get("cost"),
                    })
                append_cache_row(row)
                print(f"  [{done}/{total}] {ts} {provider_name} w={window} -> "
                      f"{row['choice'] or row['error']}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "sensitivity"

    semif_key = load_semif_key()
    official_key = load_official_key()

    ohlc = fetch_ohlc()
    features = compute_features(ohlc)
    labels = triple_barrier_labels(features, ATR_MULT, TIMEOUT_BARS)

    valid_indices = [i for i, l in enumerate(labels) if l is not None]

    if mode == "sensitivity":
        # ~40 gleichmaessig verteilte Bars, 3 Fenstergroessen, beide Provider
        sample = valid_indices[::max(1, len(valid_indices) // 40)][:40]
        run_batch(features, labels, sample, [10, 20, 50], semif_key, official_key)
    elif mode == "full":
        window = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        run_batch(features, labels, valid_indices, [window], semif_key, official_key)
    else:
        print(f"Unknown mode: {mode} (use 'sensitivity' or 'full <window>')")
