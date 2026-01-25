import pandas as pd
import ta
import os
import json
import time
import threading
import logging
import sys
from datetime import datetime
from xgboost import XGBClassifier
from trading_ig import IGService
import yfinance as yf

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot_fortress.log"),
    ],
)
logger = logging.getLogger("FortressBot")


class EliteBot:
    SYMBOL_TO_EPIC = {
        "FTSE100": "IX.D.FTSE.DAILY.IP",
        "DOW30": "IX.D.DOW.DAILY.IP",
        "NAS100": "IX.D.NASDAQ.DAILY.IP",
        "DAX": "IX.D.DAX.DAILY.IP",
        "EURUSD": "CS.D.EURUSD.TODAY.IP",
        "GBPUSD": "CS.D.GBPUSD.TODAY.IP",
        "USDJPY": "CS.D.USDJPY.TODAY.IP",
        "USDCHF": "CS.D.USDCHF.TODAY.IP",
        "USDCAD": "CS.D.USDCAD.TODAY.IP",
        "AUDUSD": "CS.D.AUDUSD.TODAY.IP",
        "EURCAD": "CS.D.EURCAD.TODAY.IP",
        "XAUUSD": "CC.D.GOLD.USS.IP",
        "GOLD": "CC.D.GOLD.USS.IP",
        "XAGUSD": "CC.D.SILVER.USS.IP",
        "SILVER": "CC.D.SILVER.USS.IP",
        "BRENT": "CC.D.LCO.UNC.IP",
    }

    def __init__(self, account_dir):
        self._stop_event = threading.Event()
        self.account_dir = account_dir
        self.TARGET_TZ = "Europe/Berlin"
        self.load_configurations()
        self.ig = self.initialize_ig_session()

        logger.info("🧠 Training KI-Modelle...")
        self.models = {
            s: self.train_elite_model(s)
            for s in self.assets.keys()
            if self.train_elite_model(s) is not None
        }
        logger.info(f"🏰 Bot 6.6 scharf. {len(self.models)} Assets geladen.")

    def load_configurations(self):
        with open(f"{self.account_dir}/account_info.json", "r") as f:
            self.account_info = json.load(f)
        with open(f"{self.account_dir}/assets.json", "r") as f:
            self.assets = json.load(f)

    def initialize_ig_session(self):
        creds = self.account_info["credentials"]
        ig = IGService(
            creds["username"], creds["password"], creds["api_key"], creds["env"].upper()
        )
        try:
            ig.create_session()
            return ig
        except Exception as e:
            logger.error(f"❌ Login gescheitert: {e}")
            sys.exit(1)

    def load_data_aligned(self, path, is_sentiment=False):
        try:
            df_raw = pd.read_csv(path)
            start = 1 if str(df_raw.iloc[0, 0]).isdigit() else 0
            df = (
                df_raw.iloc[
                    :, [start, start + 1, start + 2, start + 3, start + 4]
                ].copy()
                if len(df_raw.columns) >= 5
                else df_raw.iloc[:, [start, start + 1]].copy()
            )
            df.columns = (
                ["T", "O", "H", "L", "C"] if len(df.columns) == 5 else ["T", "C"]
            )
            if "O" not in df.columns:
                df["O"] = df["H"] = df["L"] = df["C"]
            df["T"] = pd.to_datetime(df["T"])
            if is_sentiment:
                if df["T"].dt.tz is None:
                    df["T"] = df["T"].dt.tz_localize("UTC")
                df["T"] = df["T"].dt.tz_convert(self.TARGET_TZ)
            else:
                if df["T"].dt.tz is None:
                    df["T"] = df["T"].dt.tz_localize(
                        self.TARGET_TZ, ambiguous="infer", nonexistent="shift_forward"
                    )
            df["T"] = df["T"].dt.tz_localize(None)
            return df.set_index("T")
        except Exception:
            return None

    def calculate_indicators(self, df, feats):
        if "trend_adx" in feats:
            df["trend_adx"] = ta.trend.adx(df["H"], df["L"], df["C"])
        if "trend_cci" in feats:
            df["trend_cci"] = ta.trend.cci(df["H"], df["L"], df["C"])
        if "trend_ema" in feats:
            df["trend_ema"] = (df["C"] - ta.trend.ema_indicator(df["C"], 50)) / df["C"]
        if "mom_rsi" in feats:
            df["mom_rsi"] = ta.momentum.rsi(df["C"])
        if "mom_uo" in feats:
            df["mom_uo"] = ta.momentum.ultimate_oscillator(df["H"], df["L"], df["C"])
        if "vol_atr" in feats:
            df["vol_atr"] = ta.volatility.average_true_range(df["H"], df["L"], df["C"])
        if "vol_bbh" in feats:
            df["vol_bbh"] = (ta.volatility.bollinger_hband(df["C"]) - df["C"]) / df["C"]
        if "time_hr" in feats:
            df["time_hr"] = df.index.hour
        return df

    def train_elite_model(self, symbol):
        try:
            cfg = self.assets[symbol]
            df = self.load_data_aligned(f"./data/forexsb/{symbol}_HOUR.csv")
            for s in ["VIX", "DXY"]:
                if f"sent_{s.lower()}" in cfg["features"]:
                    s_path = f"./data/forexsb/{s.upper()}_HOUR.csv"
                    s_df = self.load_data_aligned(s_path, True)
                    df = df.join(
                        s_df["C"].rename(f"sent_{s.lower()}"), how="left"
                    ).fillna(0)
            df = self.calculate_indicators(df, cfg["features"])
            df["Target"] = (df["C"].shift(-1) > df["C"]).astype(int)
            df = df.dropna()
            m = XGBClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, random_state=42, verbosity=0
            )
            m.fit(df[cfg["features"]], df["Target"])
            return m
        except Exception:
            return None

    def execute_order(self, symbol, direction, prob):
        cfg = self.assets[symbol]
        epic = self.SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            return

        try:
            acc = self.ig.fetch_accounts()
            balance = float(acc.loc[0, "balance"])
            # Maximale Positionsgröße begrenzen (Anti-Wahnsinn-Sicherung)
            risk_cash = min(balance * cfg["kelly_risk"], balance * 0.05)

            df = self.load_data_aligned(f"./data/forexsb/{symbol}_HOUR.csv").tail(50)
            atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"]).iloc[-1]

            # Korrekte Berechnung der Pips/Punkte Distanz
            # sl_dist_pts ist die absolute Differenz in Broker-Einheiten
            sl_dist_pts = max(10, int((atr * cfg["sl_mult"]) / cfg["point_value"]))
            limit_dist_pts = int((atr * cfg["tp_mult"]) / cfg["point_value"])

            # Normierte Size Berechnung: Risk_Cash / (Distanz_in_Punkten)
            # IG verlangt bei vielen Indizes 1.0 pro Punkt, bei FX 10.0 pro Pip
            size = round(risk_cash / sl_dist_pts, 2)
            size = max(self.account_info["money_management"]["min_lot_size"], size)

            logger.info(
                f"🚀 {direction} {symbol} | Epic: {epic} | Size: {size} | SL: {sl_dist_pts}"
            )

            # KORREKTE IG METHODE: create_open_position
            response = self.ig.create_open_position(
                currency_code=self.account_info["metadata"]["currency"],
                direction=direction,
                epic=epic,
                expiry="DFB",
                order_type="MARKET",
                size=size,
                guaranteed_stop=False,
                stop_distance=sl_dist_pts,
                limit_distance=limit_dist_pts,
            )

            if response and "dealReference" in response:
                logger.info(f"✅ Order platziert! Ref: {response['dealReference']}")
            else:
                logger.error(f"❌ Abgelehnt: {response}")

        except Exception as e:
            logger.error(f"❌ Fehler bei {symbol}: {e}")

    def run(self):
        while not self._stop_event.is_set():
            if datetime.now().weekday() < 5:
                # Sentiment Refresh (Fix für float Warning)
                tickers = {"vix": "^VIX", "dxy": "DX-Y.NYB"}
                current_sent = {}
                for k, v in tickers.items():
                    data = yf.download(v, period="1d", interval="1h", progress=False)
                    if not data.empty:
                        current_sent[k] = float(data["Close"].iloc[-1])

                for sym, cfg in self.assets.items():
                    if sym not in self.models:
                        continue
                    try:
                        df = (
                            self.load_data_aligned(f"./data/forexsb/{sym}_HOUR.csv")
                            .tail(100)
                            .copy()
                        )
                        for k, v in current_sent.items():
                            if f"sent_{k}" in cfg["features"]:
                                df[f"sent_{k}"] = v

                        df = self.calculate_indicators(df, cfg["features"])
                        prob = self.models[sym].predict_proba(
                            df[cfg["features"]].iloc[[-1]]
                        )[0, 1]

                        if prob >= cfg["conf_thresh"]:
                            self.execute_order(sym, "BUY", prob)
                        elif prob <= (1 - cfg["conf_thresh"]):
                            self.execute_order(sym, "SELL", prob)
                    except Exception:
                        continue
            time.sleep(300)


if __name__ == "__main__":
    EliteBot("accounts/main_demo").run()
