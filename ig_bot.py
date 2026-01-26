import pandas as pd
import numpy as np
import ta
import os
import json
import time
import threading
import logging
import sys
import yfinance as yf
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from trading_ig import IGService

# Optimizer-Module für konsistente Feature-Berechnung
from optimizer.indicators import compute_indicator_pool
from optimizer.data_loader import load_macro_indicators, load_interest_rates

# --- LOGGING SETUP ---
LOG_DIR = os.environ.get("LOG_DIR", "logs")
STATS_DIR = os.environ.get("STATS_DIR", "stats_export")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log")),
    ],
)
logger = logging.getLogger("FortressBot")


RESTART_FILE = os.path.join(STATS_DIR, "restart_signal")


def check_restart_signal():
    """Check if a restart signal file exists and remove it."""
    if os.path.exists(RESTART_FILE):
        try:
            os.remove(RESTART_FILE)
            return True
        except Exception:
            pass
    return False


class EliteBot:
    # Default EPICs - werden zur Laufzeit aktualisiert falls nötig
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
        "XAUUSD": "CS.D.USCGC.TODAY.IP",
        "GOLD": "CS.D.USCGC.TODAY.IP",
        "XAGUSD": "CS.D.USCSI.TODAY.IP",
        "SILVER": "CS.D.USCSI.TODAY.IP",
        "BRENT": "EN.D.LCO.MONTH2.IP",
    }

    # Suchbegriffe für dynamische EPIC-Suche
    SYMBOL_SEARCH_TERMS = {
        "GOLD": "Gold",
        "XAUUSD": "Gold",
        "SILVER": "Silver",
        "XAGUSD": "Silver",
        "BRENT": "Brent",
    }

    def __init__(self, account_dir):
        self._stop_event = threading.Event()
        self.account_dir = account_dir
        self.account_id = os.path.basename(account_dir)
        self.TARGET_TZ = "Europe/Berlin"
        self.load_configurations()
        self.ig = self.initialize_ig_session()

        # Suche korrekte EPICs für Commodities
        self.resolve_epics()

        logger.info("🧠 Training KI-Modelle...")
        self.models = {}
        for s in self.assets.keys():
            model = self.train_elite_model(s)
            if model is not None:
                self.models[s] = model
            time.sleep(1)
        logger.info(f"🏰 Bot 6.6 scharf. {len(self.models)} Assets geladen.")
        self.write_status("RUNNING")

    def load_configurations(self):
        with open(f"{self.account_dir}/account_info.json", "r") as f:
            self.account_info = json.load(f)
        with open(f"{self.account_dir}/assets.json", "r") as f:
            self.assets = json.load(f)

    def write_status(self, status):
        """Write bot status for dashboard heartbeat monitoring."""
        status_dir = os.path.join(STATS_DIR, self.account_id)
        os.makedirs(status_dir, exist_ok=True)
        status_file = os.path.join(status_dir, "bot_status.json")

        status_data = {
            "last_heartbeat": datetime.now().isoformat(),
            "status": status,
            "active_pairs_count": len(self.models) if hasattr(self, "models") else 0,
            "active_epics": list(self.models.keys()) if hasattr(self, "models") else [],
            "account_id": self.account_id,
            "account_mode": self.account_info.get("credentials", {}).get("env", "DEMO"),
        }

        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)

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

    def resolve_epics(self):
        """Sucht korrekte EPICs für Commodities via IG API."""
        for symbol in self.assets.keys():
            if symbol in self.SYMBOL_SEARCH_TERMS:
                search_term = self.SYMBOL_SEARCH_TERMS[symbol]
                try:
                    time.sleep(0.5)
                    results = self.ig.search_markets(search_term)
                    if results is not None and not results.empty:
                        # Suche nach passendem Markt (Spot/Cash, kein Future)
                        for _, row in results.iterrows():
                            epic = row.get("epic", "")
                            name = row.get("instrumentName", "").lower()
                            # Bevorzuge Spot/Cash Märkte
                            if "spot" in name or "cash" in name or "usd" in name.lower():
                                self.SYMBOL_TO_EPIC[symbol] = epic
                                logger.info(f"📍 {symbol} -> {epic}")
                                break
                        else:
                            # Fallback: nimm ersten Treffer
                            epic = results.iloc[0]["epic"]
                            self.SYMBOL_TO_EPIC[symbol] = epic
                            logger.info(f"📍 {symbol} -> {epic} (fallback)")
                except Exception as e:
                    logger.warning(f"⚠️ EPIC-Suche für {symbol} fehlgeschlagen: {e}")

    def fetch_ig_historical(self, symbol, num_points=2000):
        """Holt historische OHLC-Daten von der IG API."""
        epic = self.SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            logger.warning(f"⚠️ Kein EPIC für {symbol}")
            return None

        # Rate limiting
        time.sleep(0.5)

        try:
            # IG API: fetch_historical_prices_by_epic
            # resolution: HOUR, numPoints: max 10000
            response = self.ig.fetch_historical_prices_by_epic(
                epic=epic,
                resolution="1H",
                numpoints=num_points,
            )

            if response is None or "prices" not in response:
                logger.warning(f"⚠️ Keine Daten von IG für {symbol}")
                return None

            prices = response["prices"]
            if not prices:
                return None

            # Konvertiere zu DataFrame
            data = []
            for p in prices:
                # IG gibt bid/ask - wir nehmen den Midpoint
                snap = p.get("snapshotTimeUTC") or p.get("snapshotTime")
                o = (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2
                h = (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2
                low = (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2
                c = (p["closePrice"]["bid"] + p["closePrice"]["ask"]) / 2
                data.append({"T": snap, "O": o, "H": h, "L": low, "C": c})

            df = pd.DataFrame(data)
            df["T"] = pd.to_datetime(df["T"])
            df = df.set_index("T").sort_index()

            return df

        except Exception as e:
            logger.warning(f"⚠️ IG API Fehler für {symbol}: {e}")
            return None

    def fetch_macro_data(self):
        """Holt aktuelle Makro-Daten von yfinance."""
        macro_tickers = {
            "macro_vix": "^VIX",
            "macro_hyg": "HYG",
            "macro_gold_fut": "GC=F",
            "macro_tlt": "TLT",
            "macro_dxy": "DX-Y.NYB",
        }

        macro_data = {}
        for name, ticker in macro_tickers.items():
            try:
                data = yf.download(ticker, period="60d", interval="1d", progress=False)
                if not data.empty:
                    # Letzter Wert und Changes
                    macro_data[name] = float(data["Close"].iloc[-1])
                    if len(data) > 1:
                        macro_data[f"{name}_chg_12h"] = float(
                            (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100
                        )
            except Exception:
                pass

        return macro_data

    def load_and_prepare_data(self, symbol):
        """
        Lädt Live-Daten von IG API und berechnet alle Features wie der Optimizer.
        """
        try:
            # Lade OHLC-Daten von IG API
            df = self.fetch_ig_historical(symbol)
            if df is None or df.empty:
                return None

            # Berechne alle technischen Indikatoren (Ichimoku, EMA, etc.)
            df = compute_indicator_pool(df)

            # Lade Makro-Indikatoren aus CSV (für historische Daten)
            df = load_macro_indicators(df)

            # Lade Zinsdaten
            df = load_interest_rates(df)

            # Aktualisiere mit Live-Makro-Daten für die letzte Zeile
            macro_live = self.fetch_macro_data()
            for key, value in macro_live.items():
                if key in df.columns:
                    df.loc[df.index[-1], key] = value

            return df

        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Laden von {symbol}: {e}")
            return None

    def train_elite_model(self, symbol):
        """Trainiert Modell mit Optimizer-Features."""
        try:
            cfg = self.assets[symbol]
            df = self.load_and_prepare_data(symbol)
            if df is None:
                logger.warning(f"⚠️ Keine Daten für {symbol}")
                return None

            # Prüfe ob alle Features vorhanden sind
            missing = [f for f in cfg["features"] if f not in df.columns]
            if missing:
                logger.warning(f"⚠️ {symbol}: Fehlende Features: {missing}")
                return None

            df["Target"] = (df["C"].shift(-1) > df["C"]).astype(int)
            df = df.dropna(subset=cfg["features"] + ["Target"])

            if len(df) < 1000:
                logger.warning(f"⚠️ {symbol}: Zu wenig Daten ({len(df)} Zeilen)")
                return None

            m = XGBClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, random_state=42, verbosity=0
            )
            m.fit(df[cfg["features"]], df["Target"])
            logger.info(f"✅ {symbol} trainiert mit {len(df)} Samples")
            return m
        except Exception as e:
            logger.error(f"❌ {symbol} Training fehlgeschlagen: {e}")
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

            df = self.fetch_ig_historical(symbol, num_points=100)
            if df is None:
                logger.error(f"❌ Keine Daten für {symbol}")
                return
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
            # Check for restart signal
            if check_restart_signal():
                logger.info("🔄 Restart signal detected, restarting bot...")
                self.write_status("RESTARTING")
                # Re-execute the script
                os.execv(sys.executable, [sys.executable] + sys.argv)

            self.write_status("RUNNING")
            if datetime.now().weekday() < 5:
                for sym, cfg in self.assets.items():
                    if sym not in self.models:
                        continue
                    try:
                        # Lade Daten mit Optimizer-Logik
                        df = self.load_and_prepare_data(sym)
                        if df is None or len(df) < 100:
                            continue

                        df = df.tail(500).copy()  # Genug Daten für Indikatoren

                        # Prüfe Features
                        missing = [f for f in cfg["features"] if f not in df.columns]
                        if missing:
                            continue

                        prob = self.models[sym].predict_proba(
                            df[cfg["features"]].iloc[[-1]]
                        )[0, 1]

                        if prob >= cfg["conf_thresh"]:
                            self.execute_order(sym, "BUY", prob)
                        elif prob <= (1 - cfg["conf_thresh"]):
                            self.execute_order(sym, "SELL", prob)
                    except Exception as e:
                        logger.warning(f"⚠️ {sym} Prediction fehler: {e}")
                        continue
            time.sleep(300)


def discover_accounts(accounts_dir="accounts"):
    """Discover all account directories that contain required config files."""
    accounts = []
    if not os.path.exists(accounts_dir):
        logger.warning(f"⚠️ Accounts directory '{accounts_dir}' does not exist")
        return accounts

    for name in os.listdir(accounts_dir):
        account_path = os.path.join(accounts_dir, name)
        if os.path.isdir(account_path):
            # Check if required config files exist
            account_info = os.path.join(account_path, "account_info.json")
            assets_file = os.path.join(account_path, "assets.json")
            if os.path.exists(account_info) and os.path.exists(assets_file):
                accounts.append(account_path)
            else:
                logger.warning(f"⚠️ Skipping '{name}': missing account_info.json or assets.json")

    return accounts


def run_bot_for_account(account_path):
    """Run a bot instance for a specific account."""
    try:
        bot = EliteBot(account_path)
        bot.run()
    except Exception as e:
        logger.error(f"❌ Bot for {account_path} crashed: {e}")


if __name__ == "__main__":
    accounts = discover_accounts()

    if not accounts:
        logger.error("❌ No valid accounts found in 'accounts/' directory")
        logger.info("💡 Each account needs: account_info.json and assets.json")
        sys.exit(1)

    logger.info(f"🔍 Found {len(accounts)} account(s): {accounts}")

    if len(accounts) == 1:
        # Single account - run directly
        run_bot_for_account(accounts[0])
    else:
        # Multiple accounts - run in parallel threads
        threads = []
        for account_path in accounts:
            t = threading.Thread(target=run_bot_for_account, args=(account_path,), daemon=True)
            t.start()
            threads.append(t)
            logger.info(f"🚀 Started bot for {account_path}")

        # Wait for all threads
        for t in threads:
            t.join()
