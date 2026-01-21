import pandas as pd
import ta
import numpy as np
import os
import yaml
import time
import warnings
from xgboost import XGBClassifier
from concurrent.futures import ProcessPoolExecutor

# --- WARNUNGEN UNTERDRÜCKEN ---
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- PARAMETER GRID ---
PARAM_GRID = {
    "conf_thresh": [0.55, 0.58, 0.60, 0.62, 0.65],
    "adx_thresh": [15, 20, 25],
    "sl_atr_mult": [2.0, 3.0, 4.0],
    "tp_atr_mult": [4.0, 6.0, 9.0],
}


def run_backtest_segment(df, p):
    if len(df) < 60:
        return 0
    features = ["RSI", "ADX", "SMA_50"]
    model = XGBClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        n_jobs=1,
    )
    split = int(len(df) * 0.6)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    model.fit(train_df[features], train_df["Target"])
    probs = model.predict_proba(test_df[features])[:, 1]
    pnl = 0
    for i in range(len(test_df)):
        if probs[i] >= p["conf_thresh"] and test_df["ADX"].iloc[i] > p["adx_thresh"]:
            pnl += 1 if test_df["Target"].iloc[i] == 1 else -1
    return pnl


def worker_task(args):
    src, name, epic, p_grid = args
    safe_epic = epic.replace(".", "_")
    path = f"data/{src}/{safe_epic}.csv"

    if not os.path.exists(path):
        return src, name, None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) < 800:
            return src, name, None

        df["RSI"] = ta.momentum.rsi(df["Close"], 14)
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
        df["ATR"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], 14
        )
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], 50)
        df["Target"] = (
            np.log(df["Close"] / df["Close"].shift(1)).shift(-1) > 0
        ).astype(int)
        df = df.dropna()

        segments = np.array_split(df, 4)
        valid_params = []

        for ct in p_grid["conf_thresh"]:
            for at in p_grid["adx_thresh"]:
                for sl in p_grid["sl_atr_mult"]:
                    for tp in p_grid["tp_atr_mult"]:
                        p = {
                            "conf_thresh": ct,
                            "adx_thresh": at,
                            "sl_atr_mult": sl,
                            "tp_atr_mult": tp,
                        }
                        seg_profits = [run_backtest_segment(seg, p) for seg in segments]
                        if all(res >= 0 for res in seg_profits):
                            valid_params.append({"p": p, "total": sum(seg_profits)})

        if not valid_params:
            return src, name, None

        finalists = []
        for cand in valid_params:
            p_curr = cand["p"]
            neighbors = [
                c["total"]
                for c in valid_params
                if abs(c["p"]["conf_thresh"] - p_curr["conf_thresh"]) <= 0.03
            ]
            stability = (
                (sum(neighbors) / len(neighbors)) / cand["total"]
                if cand["total"] > 0
                else 0
            )
            if stability >= 0.70:
                finalists.append({**cand, "stability": stability})

        if not finalists:
            return src, name, None
        best = max(finalists, key=lambda x: x["total"])
        return src, name, {**best, "rows": len(df)}

    except Exception as e:
        # Präzises Exception-Handling statt bare except
        print(f"   🔥 Fehler bei {name} ({src}): {e}")
        return src, name, None


if __name__ == "__main__":
    start_time = time.time()

    if not os.path.exists("config.yaml"):
        print("❌ config.yaml nicht gefunden!")
        exit()

    with open("config.yaml", "r") as f:
        pairs_cfg = yaml.safe_load(f).get("pairs", {})

    sources = ["yahoo", "stooq"]
    tasks = [
        (src, name, (p["epic"] if isinstance(p, dict) else p), PARAM_GRID)
        for src in sources
        for name, p in pairs_cfg.items()
    ]

    print(
        f"🚀 ULTIMATIVE OPTIMIERUNG: {len(tasks)} Aufgaben auf {os.cpu_count()} Kernen..."
    )

    results_map = {"yahoo": {}, "stooq": {}}
    with ProcessPoolExecutor() as executor:
        for src, name, res in executor.map(worker_task, tasks):
            if res:
                results_map[src][name] = res
                print(
                    f"💎 {src.upper():6} | {name:15} | Stab: {res['stability'] * 100:>5.1f}%"
                )

    for src in sources:
        if results_map[src]:
            print(f"\n🏆 ULTRAROBUSTE PARAMETER ({src.upper()})")
            print("-" * 30)
            final_output = {"pairs": {}}
            for name, res in results_map[src].items():
                epic = (
                    pairs_cfg[name]["epic"]
                    if isinstance(pairs_cfg[name], dict)
                    else pairs_cfg[name]
                )
                final_output["pairs"][name] = {
                    "epic": epic,
                    **res["p"],
                    "stability": round(res["stability"] * 100, 1),
                }
            print(yaml.dump(final_output, default_flow_style=False, sort_keys=False))

    print(f"\n⏱️ Gesamtdauer: {round((time.time() - start_time) / 60, 2)} Min")
