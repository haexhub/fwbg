import pandas as pd
import ta
import numpy as np
import os
import yaml
import time
import logging
import warnings
from datetime import datetime
from xgboost import XGBClassifier
from trading_ig import IGService

# --- SETUP ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.simplefilter(action="ignore", category=FutureWarning)


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


BASE_CFG = load_config()


class IGBot:
    def __init__(self):
        # Profil-Daten laden
        active_p = BASE_CFG.get("active_profile", "demo")
        profile_data = BASE_CFG.get("profiles", {}).get(active_p)

        if not profile_data:
            raise ValueError(f"❌ Profil '{active_p}' nicht in Config gefunden!")

        self.ig = IGService(
            profile_data["username"],
            profile_data["password"],
            profile_data["api_key"],
            profile_data["acc_type"],
        )
        self.ig.create_session()

        # Den Pre-Flight-Check aufrufen
        self.show_startup_summary(active_p)

    def show_startup_summary(self, profile_name):
        """Zeigt eine Übersicht der Risiko-Parameter beim Start."""
        mm = BASE_CFG.get("money_management", {})
        max_risk = mm.get("max_risk_pct", 0.02)
        kelly_m = mm.get("kelly_multiplier", 0.2)

        print("\n" + "=" * 50)
        print(f"🛡️  PRE-FLIGHT CHECK: PROFIL [{profile_name.upper()}]")
        print("=" * 50)
        print(f"Max. Risiko pro Trade: {max_risk * 100:.1f}% vom Kapital")
        print(f"Kelly-Multiplikator:   {kelly_m:.2f} (Vorsicht-Faktor)")
        print("-" * 50)
        print(f"{'Paar':<15} | {'Stabilität':<10} | {'Risk-Gewichtung':<15}")
        print("-" * 50)

        for name, p in BASE_CFG.get("pairs", {}).items():
            stab = p.get("stability", 80.0)
            # Berechnung des Stabilitäts-Faktors (wie in execute_trade)
            stab_factor = min(max(0.5, stab / 100.0), 1.25)
            # Effektives Risiko bei perfektem Signal
            eff_risk = max_risk * stab_factor

            print(
                f"{name:<15} | {stab:>9.1f}% | x{stab_factor:.2f} ({eff_risk * 100:.2f}%)"
            )

        print("=" * 50)
        print("🤖 Bot wartet auf Signale...\n")

    def get_market_data(self, epic):
        safe_epic = epic.replace(".", "_")
        source = BASE_CFG.get("bot", {}).get("data_source", "stooq")
        file_path = f"data/{source}/{safe_epic}.csv"

        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df["RSI"] = ta.momentum.rsi(df["Close"], 14)
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
        df["ATR"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], 14
        )
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], 50)
        df["Target"] = (
            np.log(df["Close"] / df["Close"].shift(1)).shift(-1) > 0
        ).astype(int)
        return df.dropna()

    def is_market_regime_ok(self, df, name):
        current_atr = df["ATR"].iloc[-1]
        hist_atr_mean = df["ATR"].tail(100).mean()
        if current_atr > (hist_atr_mean * 2.5) or df["ADX"].iloc[-1] < 12:
            logging.warning(f"⚠️ Regime-Block {name}: Volatilität oder Trend unpassend.")
            return False
        return True

    def analyze_and_predict(self, df):
        features = ["RSI", "ADX", "SMA_50"]
        model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0, n_jobs=-1
        )
        model.fit(df.tail(1000)[features], df.tail(1000)["Target"])
        prob = model.predict_proba(df.iloc[[-1]][features])[0, 1]
        return prob, {"atr": df["ATR"].iloc[-1], "close": df["Close"].iloc[-1]}

    def sync_open_positions(self):
        """Prüft, ob IG-Trades geschlossen wurden und loggt das Ergebnis."""
        try:
            ig_pos = self.ig.fetch_open_positions()
            active_ids = ig_pos["dealId"].tolist() if not ig_pos.empty else []

            if not os.path.exists("trade_history.csv"):
                return

            df = pd.read_csv("trade_history.csv")
            mask_just_closed = (df["pnl"] == 0) & (~df["deal_id"].isin(active_ids))

            if mask_just_closed.any():
                history = self.ig.fetch_transaction_history(max_results=30)
                for idx, row in df[mask_just_closed].iterrows():
                    match = history[history["dealId"] == row["deal_id"]]
                    df.at[idx, "pnl"] = (
                        float(match.iloc[0]["profitAndLoss"])
                        if not match.empty
                        else 0.01
                    )
                df.to_csv("trade_history.csv", index=False)
                logging.info("🏁 Trade-Historie synchronisiert.")
        except Exception as e:
            logging.error(f"Sync-Fehler: {e}")

    def execute_trade(self, epic, signal, stats, prob, p):
        try:
            # Kelly + Stabilität
            b = p["tp_atr_mult"] / p["sl_atr_mult"]
            kelly = prob - ((1 - prob) / b)
            if kelly <= 0:
                return

            stab_f = min(max(0.5, p.get("stability", 80.0) / 100.0), 1.25)
            risk = min(
                kelly * BASE_CFG["money_management"]["kelly_multiplier"] * stab_f,
                BASE_CFG["money_management"]["max_risk_pct"],
            )

            acc = self.ig.fetch_accounts()
            bal = float(acc[acc["accountType"] == "CFD"].iloc[0]["balance"])

            is_jpy = "JPY" in epic
            dist = max(p["sl_atr_mult"] * stats["atr"], 12.0 if is_jpy else 0.0012)
            size = round(
                max(0.2, (bal * risk) / (dist if not is_jpy else dist / 100)), 1
            )

            res = self.ig.create_open_position(
                currency_code="EUR",
                direction=signal,
                epic=epic,
                expiry="DFB",
                order_type="MARKET",
                size=size,
                force_open=False,
                guaranteed_stop=False,
            )
            deal_id = (
                res.get("dealId")
                if isinstance(res, dict)
                else getattr(res, "dealId", None)
            )

            if deal_id:
                time.sleep(2)
                exec_p = float(res.get("level", stats["close"]))
                sl = round(
                    exec_p - dist if signal == "BUY" else exec_p + dist,
                    2 if is_jpy else 4,
                )
                tp = round(
                    exec_p + (p["tp_atr_mult"] * stats["atr"])
                    if signal == "BUY"
                    else exec_p - (p["tp_atr_mult"] * stats["atr"]),
                    2 if is_jpy else 4,
                )
                self.ig.update_open_position(
                    deal_id=deal_id, limit_level=tp, stop_level=sl
                )

                # Sofort-Log
                log_df = pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "epic": epic,
                            "deal_id": deal_id,
                            "signal": signal,
                            "size": size,
                            "pnl": 0,
                        }
                    ]
                )
                log_df.to_csv(
                    "trade_history.csv",
                    mode="a",
                    index=False,
                    header=not os.path.exists("trade_history.csv"),
                )
                logging.info(f"✅ {signal} @ {exec_p} | SL: {sl} | TP: {tp}")
        except Exception as e:
            logging.error(f"Execution Error {epic}: {e}")

    def run(self):
        while True:
            logging.info("--- 🔄 Scan ---")
            self.sync_open_positions()
            active_epics = (
                self.ig.fetch_open_positions()["epic"].tolist()
                if not self.ig.fetch_open_positions().empty
                else []
            )

            for name, p in BASE_CFG["pairs"].items():
                if p["epic"] in active_epics:
                    continue
                df = self.get_market_data(p["epic"])
                if df is None or not self.is_market_regime_ok(df, name):
                    continue

                prob, stats = self.analyze_and_predict(df)
                if prob >= p["conf_thresh"]:
                    self.execute_trade(p["epic"], "BUY", stats, prob, p)
                elif prob <= (1 - p["conf_thresh"]):
                    self.execute_trade(p["epic"], "SELL", stats, prob, p)

            time.sleep(300)


if __name__ == "__main__":
    IGBot().run()
