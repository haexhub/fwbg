from datetime import datetime
from trading_ig import IGService
from xgboost import XGBClassifier
import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import ta
import threading
import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_account_configs() -> list[dict]:
    """Load all account configurations from accounts/*.json files."""
    accounts = []
    accounts_dir = "accounts"

    if not os.path.exists(accounts_dir):
        os.makedirs(accounts_dir)
        return []

    for config_file in glob.glob(os.path.join(accounts_dir, "*.json")):
        # Skip example files
        if ".example." in config_file:
            continue

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                config["_config_file"] = config_file
                accounts.append(config)
        except Exception as e:
            print(f"Error loading {config_file}: {e}")

    return accounts


def create_account_logger(account_id: str, log_file: str) -> logging.Logger:
    """Create a dedicated logger for each account with its own log file."""
    logger = logging.getLogger(f"igbot.{account_id}")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler (account-specific log file)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


class IGAccountBot:
    """
    Bot instance for a single IG account.
    Each account runs independently with its own:
    - Configuration (credentials, strategy, pairs, etc.)
    - IG API session
    - Log file
    - Trade history
    - Status file
    """

    def __init__(self, config: dict):
        self.config = config
        self.account_id = config["id"]
        self.account_name = config["name"]
        self.credentials = config["credentials"]
        self.bot_settings = config.get("bot", {})
        self.xgb_settings = config.get("xgb_settings", {})
        self.money_management = config.get("money_management", {})
        self.strategy = config.get("strategy", {})
        self.pairs = config.get("pairs", {})

        # Create account-specific directories
        self.stats_dir = f"stats_export/{self.account_id}"
        os.makedirs(self.stats_dir, exist_ok=True)

        # Create account-specific logger
        log_file = self.bot_settings.get("log_file", f"bot_{self.account_id}.log")
        self.logger = create_account_logger(self.account_id, log_file)

        # Initialize IG service for this account
        self.ig = IGService(
            self.credentials["username"],
            self.credentials["password"],
            self.credentials["api_key"],
            self.credentials["acc_type"],
        )
        self.ig.create_session()

        self.logger.info(f"Account initialized: {self.credentials['acc_type']}")

    def show_startup_summary(self):
        """Zeigt eine Übersicht der Risiko-Parameter beim Start."""
        max_risk = self.money_management.get("max_risk_pct", 0.02)
        kelly_m = self.money_management.get("kelly_multiplier", 0.2)

        print("\n" + "=" * 50)
        print(f"🛡️  PRE-FLIGHT CHECK: [{self.account_name.upper()}]")
        print("=" * 50)
        print(f"Account Type:          {self.credentials.get('acc_type', 'UNKNOWN')}")
        print(f"Max. Risiko pro Trade: {max_risk * 100:.1f}% vom Kapital")
        print(f"Kelly-Multiplikator:   {kelly_m:.2f} (Vorsicht-Faktor)")
        print("-" * 50)
        print(f"{'Paar':<15} | {'Stabilität':<10} | {'Risk-Gewichtung':<15}")
        print("-" * 50)

        for name, p in self.pairs.items():
            stab = p.get("stability", 80.0)
            stab_factor = min(max(0.5, stab / 100.0), 1.25)
            eff_risk = max_risk * stab_factor
            print(f"{name:<15} | {stab:>9.1f}% | x{stab_factor:.2f} ({eff_risk * 100:.2f}%)")

        print("=" * 50)
        print("🤖 Bot wartet auf Signale...\n")

    def get_market_data(self, epic):
        """Lädt Marktdaten aus der lokalen CSV-Datei und berechnet Indikatoren."""
        safe_epic = epic.replace(".", "_")
        source = self.bot_settings.get("data_source", "stooq")
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
        """Prüft, ob das Marktregime für Trading geeignet ist."""
        current_atr = df["ATR"].iloc[-1]
        hist_atr_mean = df["ATR"].tail(100).mean()
        if current_atr > (hist_atr_mean * 2.5) or df["ADX"].iloc[-1] < 12:
            self.logger.warning(f"⚠️ Regime-Block {name}: Volatilität oder Trend unpassend.")
            return False
        return True

    def check_margin_availability(self, size, price, epic):
        """
        Prüft, ob genug Kapital für die Margin vorhanden ist.
        Gibt True zurück, wenn der Trade finanziell möglich ist.
        """
        try:
            acc = self.ig.fetch_accounts()
            cfd_acc = acc[acc["accountType"] == "CFD"].iloc[0]
            available_cash = float(cfd_acc["available"])

            is_index = epic.startswith("IX")
            margin_factor = 0.05 if is_index else 0.035
            total_value = size * price
            required_margin = total_value * margin_factor
            needed_with_buffer = required_margin * 1.10

            if available_cash > needed_with_buffer:
                return True
            else:
                self.logger.warning(
                    f"⚠️ Margin-Check fehlgeschlagen: Benötigt ca. {needed_with_buffer:.2f}€, Vorhanden: {available_cash:.2f}€"
                )
                return False

        except Exception as e:
            self.logger.error(f"Fehler beim Margin-Check: {e}")
            return False

    def update_bot_status(self, active_epics):
        """Erstellt eine Status-Datei für das Web-Dashboard."""
        status_data = {
            "last_heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "RUNNING",
            "active_pairs_count": len(active_epics),
            "active_epics": active_epics,
            "account_id": self.account_id,
            "account_name": self.account_name,
            "account_mode": self.credentials.get("acc_type", "unknown"),
        }
        with open(f"{self.stats_dir}/bot_status.json", "w") as f:
            json.dump(status_data, f, indent=4)

    def analyze_and_predict(self, df):
        """Trainiert das XGBoost-Modell und gibt eine Wahrscheinlichkeit zurück."""
        features = ["RSI", "ADX", "SMA_50"]

        xgb_params = {
            "n_estimators": self.xgb_settings.get("n_estimators", 50),
            "max_depth": self.xgb_settings.get("max_depth", 3),
            "learning_rate": self.xgb_settings.get("learning_rate", 0.05),
            "n_jobs": self.xgb_settings.get("n_jobs", -1),
            "random_state": self.xgb_settings.get("random_state", 42),
            "verbosity": 0,
        }

        model = XGBClassifier(**xgb_params)
        model.fit(df.tail(1000)[features], df.tail(1000)["Target"])
        prob = model.predict_proba(df.iloc[[-1]][features])[0, 1]
        return prob, {"atr": df["ATR"].iloc[-1], "close": df["Close"].iloc[-1]}

    def sync_open_positions(self):
        """Prüft, ob IG-Trades geschlossen wurden und loggt das Ergebnis."""
        try:
            ig_pos = self.ig.fetch_open_positions()
            active_ids = ig_pos["dealId"].tolist() if not ig_pos.empty else []

            trade_history_file = f"{self.stats_dir}/trade_history.csv"

            if not os.path.exists(trade_history_file):
                return

            df = pd.read_csv(trade_history_file)
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
                df.to_csv(trade_history_file, index=False)
                self.logger.info("🏁 Trade-Historie synchronisiert.")
        except Exception as e:
            self.logger.error(f"Sync-Fehler: {e}")

    def execute_trade(self, epic, signal, stats, prob, pair_config):
        """Führt einen Trade aus mit Kelly-Sizing und Risikomanagement."""
        try:
            p = pair_config
            b = p["tp_atr_mult"] / p["sl_atr_mult"]
            kelly = prob - ((1 - prob) / b)
            if kelly <= 0:
                return

            stab_f = min(max(0.5, p.get("stability", 80.0) / 100.0), 1.25)
            risk = min(
                kelly * self.money_management.get("kelly_multiplier", 0.2) * stab_f,
                self.money_management.get("max_risk_pct", 0.02),
            )

            acc = self.ig.fetch_accounts()
            bal = float(acc[acc["accountType"] == "CFD"].iloc[0]["balance"])

            is_jpy = "JPY" in epic
            dist = max(p["sl_atr_mult"] * stats["atr"], 12.0 if is_jpy else 0.0012)

            min_size = self.bot_settings.get("risk_size", 0.5)
            size = round(
                max(min_size, (bal * risk) / (dist if not is_jpy else dist / 100)), 1
            )

            if not self.check_margin_availability(size, stats["close"], epic):
                self.logger.info(f"🚫 Trade abgebrochen: Nicht genügend Margin für {epic}")
                return

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

                trade_history_file = f"{self.stats_dir}/trade_history.csv"
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
                    trade_history_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(trade_history_file),
                )
                self.logger.info(f"✅ {signal} @ {exec_p} | SL: {sl} | TP: {tp}")
        except Exception as e:
            self.logger.error(f"Execution Error {epic}: {e}")

    def run(self):
        """Main loop for this account - runs continuously in its own thread."""
        self.show_startup_summary()

        while True:
            # Re-load config to check if still active
            try:
                with open(self.config["_config_file"], "r", encoding="utf-8") as f:
                    current_config = json.load(f)
                    if not current_config.get("isActive", True):
                        self.logger.info("Account deactivated, pausing...")
                        self.update_bot_status([])
                        time.sleep(60)
                        continue
            except Exception:
                pass

            self.logger.info("--- 🔄 Scan ---")
            self.sync_open_positions()

            try:
                open_pos = self.ig.fetch_open_positions()
                active_epics = open_pos["epic"].tolist() if not open_pos.empty else []
            except Exception as e:
                self.logger.error(f"Failed to fetch positions: {e}")
                active_epics = []

            self.update_bot_status(active_epics)

            for name, pair_config in self.pairs.items():
                epic = pair_config["epic"]
                if epic in active_epics:
                    continue

                df = self.get_market_data(epic)
                if df is None or not self.is_market_regime_ok(df, name):
                    continue

                # Use pair-specific thresholds, fallback to strategy defaults
                conf_thresh = pair_config.get(
                    "conf_thresh", self.strategy.get("conf_thresh", 0.55)
                )

                prob, stats = self.analyze_and_predict(df)
                if prob >= conf_thresh:
                    self.execute_trade(epic, "BUY", stats, prob, pair_config)
                elif prob <= (1 - conf_thresh):
                    self.execute_trade(epic, "SELL", stats, prob, pair_config)

            time.sleep(300)


class MultiAccountBot:
    """
    Main bot that manages multiple IG accounts.
    Each active account runs in its own thread for parallel execution.
    """

    def __init__(self):
        self.accounts: list[IGAccountBot] = []
        self.threads: list[threading.Thread] = []
        os.makedirs("stats_export", exist_ok=True)

    def _initialize_accounts(self):
        """Initialize bot instances for all active accounts."""
        configs = load_account_configs()

        print("\n" + "=" * 60)
        print("🛡️  MULTI-ACCOUNT BOT STARTING")
        print("=" * 60)

        if not configs:
            print("  ⚠️  No account configs found in accounts/ directory")
            print("  Create accounts/*.json files (see accounts/demo.example.json)")
            print("=" * 60)
            raise ValueError("No account configurations found")

        for config in configs:
            account_id = config.get("id", "unknown")
            account_name = config.get("name", account_id)

            if not config.get("isActive", True):
                print(f"  ⏸️  Skipping inactive account: {account_name}")
                continue

            try:
                account_bot = IGAccountBot(config)
                self.accounts.append(account_bot)
                print(f"  ✅ Initialized: {account_name} ({config.get('credentials', {}).get('acc_type', 'UNKNOWN')})")
            except Exception as e:
                print(f"  ❌ Failed to initialize {account_name}: {e}")

        print("=" * 60)

        if not self.accounts:
            raise ValueError("No active accounts found! Set isActive: true in account configs")

        print(f"\n🚀 Starting {len(self.accounts)} account(s) in parallel...\n")

    def run(self):
        """Start all account bots in parallel threads."""
        self._initialize_accounts()

        for account in self.accounts:
            thread = threading.Thread(
                target=account.run,
                name=f"bot-{account.account_id}",
                daemon=True
            )
            self.threads.append(thread)
            thread.start()

        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")


if __name__ == "__main__":
    MultiAccountBot().run()
