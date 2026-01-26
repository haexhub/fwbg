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
    # CFD EPICs - keine Knock-Outs, keine Optionen
    SYMBOL_TO_EPIC = {
        # Indizes CFDs
        "FTSE100": "IX.D.FTSE.DAILY.IP",
        "DOW30": "IX.D.DOW.DAILY.IP",
        "NAS100": "IX.D.NASDAQ.DAILY.IP",
        "DAX": "IX.D.DAX.DAILY.IP",
        # Forex CFDs
        "EURUSD": "CS.D.EURUSD.TODAY.IP",
        "GBPUSD": "CS.D.GBPUSD.TODAY.IP",
        "USDJPY": "CS.D.USDJPY.TODAY.IP",
        "USDCHF": "CS.D.USDCHF.TODAY.IP",
        "USDCAD": "CS.D.USDCAD.TODAY.IP",
        "AUDUSD": "CS.D.AUDUSD.TODAY.IP",
        "EURCAD": "CS.D.EURCAD.TODAY.IP",
        # Commodities CFDs (Spot)
        "XAUUSD": "CS.D.CFDGOLD.CFD.IP",
        "GOLD": "CS.D.CFDGOLD.CFD.IP",
        "XAGUSD": "CS.D.CFDSILVER.CFD.IP",
        "SILVER": "CS.D.CFDSILVER.CFD.IP",
        "BRENT": "CC.D.LCO.UNC.IP",
    }

    def __init__(self, account_dir):
        self._stop_event = threading.Event()
        self.account_dir = account_dir
        self.account_id = os.path.basename(account_dir)
        self.TARGET_TZ = "Europe/Berlin"
        self.load_configurations()
        self.ig = self.initialize_ig_session()

        # Cache für OHLC-Daten und berechnete Features pro Symbol
        self.ohlc_cache = {}  # {symbol: DataFrame mit OHLC}
        self.features_cache = {}  # {symbol: DataFrame mit Features}
        self.last_bar_time = {}  # {symbol: letzter Timestamp}

        # Slippage-Warnungen für Dashboard (max. 20 Einträge)
        self.slippage_warnings = []  # [{symbol, timestamp, expected, actual, slippage_pct}]

        logger.info("🧠 Training KI-Modelle...")
        self.models = {}
        for s in self.assets.keys():
            model = self.train_elite_model(s)
            if model is not None:
                self.models[s] = model
            time.sleep(1)
        logger.info(f"🏰 Bot 7.0 (Cache-First) scharf. {len(self.models)} Assets geladen.")
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
            "slippage_warnings": self.slippage_warnings if hasattr(self, "slippage_warnings") else [],
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

    def fetch_ig_historical(self, symbol, num_points=1000):
        """Holt historische OHLC-Daten von der IG API."""
        epic = self.SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            logger.warning(f"⚠️ Kein EPIC für {symbol}")
            return None

        # Rate limiting - IG API erlaubt ~60 requests/min
        time.sleep(2)

        # Retry-Logik bei Rate Limiting (403)
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = self.ig.fetch_historical_prices_by_epic(
                    epic=epic,
                    resolution="1H",
                    numpoints=num_points,
                )
                break  # Erfolg
            except Exception as e:
                if "403" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    logger.warning(f"⚠️ Rate limit für {symbol}, warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise

        if response is None or "prices" not in response:
            logger.warning(f"⚠️ Keine Daten von IG für {symbol}")
            return None

        prices = response["prices"]
        if not prices:
            return None

        try:
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

    def update_ohlc_cache(self, symbol):
        """
        Inkrementelles Update: Holt nur neue Kerzen seit dem letzten Update.
        Beim ersten Aufruf werden alle historischen Daten geladen.
        """
        if symbol not in self.ohlc_cache:
            # Erster Aufruf: Lade alle historischen Daten
            logger.info(f"📊 {symbol}: Lade initiale historische Daten...")
            df = self.fetch_ig_historical(symbol, num_points=1000)
            if df is not None and not df.empty:
                self.ohlc_cache[symbol] = df
                self.last_bar_time[symbol] = df.index[-1]
                return True
            return False

        # Inkrementelles Update: Nur neue Kerzen holen
        last_time = self.last_bar_time.get(symbol)
        if last_time is None:
            return False

        # Prüfe ob eine neue Stunde begonnen hat
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        # Wenn wir noch in der gleichen Stunde sind, kein Update nötig
        if last_time >= current_hour:
            return True  # Cache ist aktuell

        # Hole nur die letzten paar Bars (sicherheitshalber etwas mehr)
        hours_since_last = max(2, int((now - last_time).total_seconds() / 3600) + 2)
        num_new_bars = min(hours_since_last, 24)  # Max 24 Bars auf einmal

        logger.info(f"📊 {symbol}: Hole {num_new_bars} neue Kerzen...")
        new_df = self.fetch_ig_historical(symbol, num_points=num_new_bars)

        if new_df is None or new_df.empty:
            return False

        # Merge: Alte Daten behalten, neue anhängen/überschreiben
        cached_df = self.ohlc_cache[symbol]

        # Kombiniere und entferne Duplikate (neue Daten haben Vorrang)
        combined = pd.concat([cached_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        # Behalte nur die letzten 1000 Bars (Memory-Limit)
        if len(combined) > 1000:
            combined = combined.tail(1000)

        self.ohlc_cache[symbol] = combined
        self.last_bar_time[symbol] = combined.index[-1]

        return True

    def get_features_for_prediction(self, symbol):
        """
        Holt aktuelle Features für Prediction.
        Nutzt Cache wenn möglich, berechnet Indikatoren nur bei neuen Daten.
        """
        # Update OHLC Cache (inkrementell)
        if not self.update_ohlc_cache(symbol):
            return None

        ohlc_df = self.ohlc_cache[symbol]
        if ohlc_df is None or len(ohlc_df) < 100:
            return None

        # Prüfe ob wir neue Features berechnen müssen
        last_ohlc_time = ohlc_df.index[-1]
        cached_features = self.features_cache.get(symbol)

        if cached_features is not None and len(cached_features) > 0:
            last_feature_time = cached_features.index[-1]
            if last_feature_time >= last_ohlc_time:
                # Features sind aktuell
                return cached_features

        # Berechne Features neu (nur wenn nötig)
        logger.info(f"🔄 {symbol}: Berechne Indikatoren...")
        df = ohlc_df.copy()

        # Berechne alle technischen Indikatoren
        df = compute_indicator_pool(df)

        # Lade Makro-Indikatoren aus CSV
        df = load_macro_indicators(df)

        # Lade Zinsdaten
        df = load_interest_rates(df)

        # Aktualisiere mit Live-Makro-Daten für die letzte Zeile
        macro_live = self.fetch_macro_data()
        for key, value in macro_live.items():
            if key in df.columns:
                df.loc[df.index[-1], key] = value

        # Cache Features
        self.features_cache[symbol] = df

        return df

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
        """
        Trainiert Modell mit Optimizer-Features.
        Füllt gleichzeitig den OHLC- und Feature-Cache.
        """
        try:
            cfg = self.assets[symbol]

            # Lade OHLC und fülle Cache
            logger.info(f"📊 {symbol}: Lade historische Daten...")
            ohlc_df = self.fetch_ig_historical(symbol, num_points=1000)
            if ohlc_df is None or ohlc_df.empty:
                logger.warning(f"⚠️ Keine Daten für {symbol}")
                return None

            # Cache füllen
            self.ohlc_cache[symbol] = ohlc_df
            self.last_bar_time[symbol] = ohlc_df.index[-1]

            # Features berechnen
            df = ohlc_df.copy()
            df = compute_indicator_pool(df)
            df = load_macro_indicators(df)
            df = load_interest_rates(df)

            # Aktualisiere mit Live-Makro-Daten für die letzte Zeile
            macro_live = self.fetch_macro_data()
            for key, value in macro_live.items():
                if key in df.columns:
                    df.loc[df.index[-1], key] = value

            # Feature-Cache füllen
            self.features_cache[symbol] = df

            # Prüfe ob alle Features vorhanden sind
            missing = [f for f in cfg["features"] if f not in df.columns]
            if missing:
                logger.warning(f"⚠️ {symbol}: Fehlende Features: {missing}")
                return None

            df["Target"] = (df["C"].shift(-1) > df["C"]).astype(int)
            df_train = df.dropna(subset=cfg["features"] + ["Target"])

            if len(df_train) < 500:  # Reduziert von 1000 - wir haben nur 1000 Bars
                logger.warning(f"⚠️ {symbol}: Zu wenig Daten ({len(df_train)} Zeilen)")
                return None

            m = XGBClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, random_state=42, verbosity=0
            )
            m.fit(df_train[cfg["features"]], df_train["Target"])
            logger.info(f"✅ {symbol} trainiert mit {len(df_train)} Samples, Cache gefüllt")
            return m
        except Exception as e:
            logger.error(f"❌ {symbol} Training fehlgeschlagen: {e}")
            return None

    def verify_execution_price(self, symbol, deal_reference, expected_price, max_slippage_pct=0.5):
        """
        Prüft ob der tatsächliche Ausführungspreis im akzeptablen Rahmen liegt.

        Args:
            symbol: Trading-Symbol
            deal_reference: IG Deal Reference
            expected_price: Erwarteter Preis (aus Cache)
            max_slippage_pct: Maximale akzeptable Slippage in Prozent (default 0.5%)

        Returns:
            tuple: (is_ok, actual_price, slippage_pct)
        """
        try:
            # Hole Deal-Bestätigung von IG
            confirmation = self.ig.fetch_deal_by_deal_reference(deal_reference)

            if confirmation is None:
                logger.warning(f"⚠️ Keine Deal-Bestätigung für {deal_reference}")
                return (False, None, None)

            # Extrahiere tatsächlichen Ausführungspreis
            actual_price = confirmation.get("level") or confirmation.get("openLevel")
            deal_status = confirmation.get("dealStatus")
            reason = confirmation.get("reason", "")

            if deal_status != "ACCEPTED":
                logger.error(f"❌ Deal abgelehnt: {deal_status} - {reason}")
                return (False, actual_price, None)

            if actual_price is None:
                logger.warning("⚠️ Kein Ausführungspreis in Bestätigung")
                return (False, None, None)

            actual_price = float(actual_price)

            # Berechne Slippage
            if expected_price and expected_price > 0:
                slippage_pct = abs(actual_price - expected_price) / expected_price * 100
            else:
                slippage_pct = 0.0

            # Prüfe ob Slippage akzeptabel
            if slippage_pct > max_slippage_pct:
                logger.warning(
                    f"⚠️ HOHE SLIPPAGE für {symbol}: {slippage_pct:.2f}% "
                    f"(erwartet: {expected_price:.5f}, tatsächlich: {actual_price:.5f})"
                )
                return (False, actual_price, slippage_pct)

            logger.info(
                f"✅ {symbol} Slippage OK: {slippage_pct:.3f}% "
                f"(erwartet: {expected_price:.5f}, tatsächlich: {actual_price:.5f})"
            )
            return (True, actual_price, slippage_pct)

        except Exception as e:
            logger.error(f"❌ Fehler bei Slippage-Check für {symbol}: {e}")
            return (False, None, None)

    def execute_order_fast(self, symbol, direction, prob, cached_atr):
        """
        Schnelle Order-Ausführung ohne API-Calls für Preisdaten.
        ATR kommt aus dem Cache. Prüft Slippage nach Ausführung.
        """
        cfg = self.assets[symbol]
        epic = self.SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            return

        try:
            acc = self.ig.fetch_accounts()
            balance = float(acc.loc[0, "balance"])
            # Maximale Positionsgröße begrenzen (Anti-Wahnsinn-Sicherung)
            risk_cash = min(balance * cfg["kelly_risk"], balance * 0.05)

            # ATR aus Cache - kein API-Call nötig!
            atr = cached_atr

            # Korrekte Berechnung der Pips/Punkte Distanz
            sl_dist_pts = max(10, int((atr * cfg["sl_mult"]) / cfg["point_value"]))
            limit_dist_pts = int((atr * cfg["tp_mult"]) / cfg["point_value"])

            # Normierte Size Berechnung
            size = round(risk_cash / sl_dist_pts, 2)
            size = max(self.account_info["money_management"]["min_lot_size"], size)

            # Erwarteter Preis aus Cache (letzter Close)
            ohlc_df = self.ohlc_cache.get(symbol)
            expected_price = float(ohlc_df["C"].iloc[-1]) if ohlc_df is not None else None

            logger.info(
                f"🚀 {direction} {symbol} | Epic: {epic} | Size: {size} | "
                f"SL: {sl_dist_pts} | Prob: {prob:.2f} | Expected: {expected_price}"
            )

            # Market Order - sofort ausführen
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
                deal_ref = response["dealReference"]
                logger.info(f"📝 Order gesendet! Ref: {deal_ref}")

                # SLIPPAGE CHECK: Prüfe ob Ausführungspreis akzeptabel
                time.sleep(0.5)  # Kurz warten auf Deal-Bestätigung
                is_ok, actual_price, slippage = self.verify_execution_price(
                    symbol, deal_ref, expected_price, max_slippage_pct=1.0
                )

                if is_ok:
                    logger.info(f"✅ Order bestätigt! {symbol} @ {actual_price}")
                else:
                    logger.warning(f"⚠️ Order-Warnung für {symbol}: Slippage={slippage}%")
                    # Slippage-Warnung für Dashboard speichern
                    self.slippage_warnings.append({
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "expected_price": expected_price,
                        "actual_price": actual_price,
                        "slippage_pct": slippage,
                        "direction": direction,
                    })
                    # Max 20 Warnungen behalten
                    if len(self.slippage_warnings) > 20:
                        self.slippage_warnings = self.slippage_warnings[-20:]
                    # Status sofort aktualisieren
                    self.write_status("RUNNING")
            else:
                logger.error(f"❌ Abgelehnt: {response}")

        except Exception as e:
            logger.error(f"❌ Fehler bei {symbol}: {e}")

    def execute_order(self, symbol, direction, prob):
        """Legacy-Methode - nutzt API-Call für ATR (langsamer)."""
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

    def update_cache_background(self):
        """
        Aktualisiert den Cache für alle Symbole im Hintergrund.
        Wird nach den Signalprüfungen ausgeführt.
        """
        for sym in self.models.keys():
            try:
                self.update_ohlc_cache(sym)
                # Berechne auch Features neu wenn nötig
                self.get_features_for_prediction(sym)
            except Exception as e:
                logger.warning(f"⚠️ Cache-Update für {sym} fehlgeschlagen: {e}")
            time.sleep(1)  # Kleine Pause zwischen Symbolen

    def run(self):
        """
        Hauptloop mit Cache-First Architektur:
        1. Signale aus Cache prüfen (schnell, keine API-Calls)
        2. Bei Signal: Order SOFORT absetzen
        3. Cache im Hintergrund updaten
        """
        # Initiales Cache-Füllen beim Start (bereits durch train_elite_model passiert)
        logger.info("📊 Initialisiere Feature-Cache für alle Assets...")
        for sym in self.models.keys():
            try:
                # Cache wurde beim Training gefüllt, aber Features noch nicht
                if sym in self.ohlc_cache:
                    self.get_features_for_prediction(sym)
            except Exception as e:
                logger.warning(f"⚠️ Feature-Cache für {sym} fehlgeschlagen: {e}")

        last_signal_hour = {}  # Verhindert mehrfache Signale pro Stunde

        while not self._stop_event.is_set():
            # Check for restart signal
            if check_restart_signal():
                logger.info("🔄 Restart signal detected, restarting bot...")
                self.write_status("RESTARTING")
                os.execv(sys.executable, [sys.executable] + sys.argv)

            self.write_status("RUNNING")
            now = datetime.now()
            current_hour = now.replace(minute=0, second=0, microsecond=0)

            # Nur an Werktagen handeln
            if now.weekday() < 5:
                signals_to_execute = []

                # PHASE 1: Signale aus Cache prüfen (SCHNELL - keine API-Calls)
                for sym, cfg in self.assets.items():
                    if sym not in self.models:
                        continue

                    # Verhindere mehrfache Signale in derselben Stunde
                    if last_signal_hour.get(sym) == current_hour:
                        continue

                    try:
                        # Nutze gecachte Features (kein API-Call!)
                        df = self.features_cache.get(sym)
                        if df is None or len(df) < 100:
                            continue

                        # Prüfe Features
                        missing = [f for f in cfg["features"] if f not in df.columns]
                        if missing:
                            continue

                        # Prediction aus Cache
                        prob = self.models[sym].predict_proba(
                            df[cfg["features"]].iloc[[-1]]
                        )[0, 1]

                        # ATR aus Cache für schnelle Order-Ausführung
                        ohlc_df = self.ohlc_cache.get(sym)
                        if ohlc_df is not None and len(ohlc_df) >= 14:
                            cached_atr = ta.volatility.average_true_range(
                                ohlc_df["H"], ohlc_df["L"], ohlc_df["C"]
                            ).iloc[-1]
                        else:
                            cached_atr = None

                        # Signal erkannt?
                        if prob >= cfg["conf_thresh"]:
                            signals_to_execute.append((sym, "BUY", prob, cached_atr))
                            last_signal_hour[sym] = current_hour
                        elif prob <= (1 - cfg["conf_thresh"]):
                            signals_to_execute.append((sym, "SELL", prob, cached_atr))
                            last_signal_hour[sym] = current_hour

                    except Exception as e:
                        logger.warning(f"⚠️ {sym} Signal-Check fehlgeschlagen: {e}")
                        continue

                # PHASE 2: Orders SOFORT ausführen (minimale Latenz)
                for sym, direction, prob, cached_atr in signals_to_execute:
                    if cached_atr is not None:
                        self.execute_order_fast(sym, direction, prob, cached_atr)
                    else:
                        # Fallback: Legacy-Methode mit API-Call
                        self.execute_order(sym, direction, prob)

                # PHASE 3: Cache im Hintergrund aktualisieren
                # Nur einmal pro Stunde, kurz nach der vollen Stunde
                if now.minute < 10:  # In den ersten 10 Minuten jeder Stunde
                    self.update_cache_background()

            # Kurzer Sleep - wir prüfen öfter, aber Signale nur 1x pro Stunde
            time.sleep(60)  # Jede Minute prüfen (statt 5 Minuten)


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
