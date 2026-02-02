"""
FWBG Trading Bot - Broker-agnostischer ML-basierter Trading Bot.

Der Bot ist fokussiert auf:
- ML Model Training (XGBoost)
- Feature-Berechnung via Plugin-System
- Signal-Generierung (BUY/SELL)
- Delegation der Ausführung an BrokerAdapter

Der Bot kennt KEINE broker-spezifischen Details wie EPICs oder API-Calls.
Alles läuft über das BrokerAdapter Interface.

Beispiel:
    from fwbg.bot import TradingBot
    from fwbg.adapters import IGBrokerAdapter

    adapter = IGBrokerAdapter(username="...", password="...", api_key="...")

    bot = TradingBot(
        adapter=adapter,
        assets_config={"EURUSD": {...}, "XAUUSD": {...}},
        account_config={...}
    )

    bot.run()  # Startet den Bot (Streaming oder Polling)
"""
import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field

import pandas as pd
import ta
from xgboost import XGBClassifier

from fwbg.adapters.broker import BrokerAdapter, OrderSide, BarData
from fwbg.builtins.indicators import compute_indicator_pool
from fwbg.builtins.utils import load_macro_indicators, load_interest_rates

logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """Konfiguration für ein einzelnes Asset (vom Optimizer)."""
    symbol: str
    features: List[str]
    conf_thresh: float = 0.55
    kelly_risk: float = 0.005
    point_value: float = 0.0001
    sl_mult: float = 25.0
    tp_mult: float = 40.0
    good_hours: List[int] = field(default_factory=lambda: list(range(8, 17)))
    dd_scaling: Dict[str, float] = field(default_factory=dict)
    ensemble: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, symbol: str, data: Dict[str, Any]) -> "AssetConfig":
        """Erstellt AssetConfig aus Dictionary."""
        return cls(
            symbol=symbol,
            features=data.get("features", []),
            conf_thresh=data.get("conf_thresh", 0.55),
            kelly_risk=data.get("kelly_risk", 0.005),
            point_value=data.get("point_value", 0.0001),
            sl_mult=data.get("sl_mult", 25.0),
            tp_mult=data.get("tp_mult", 40.0),
            good_hours=data.get("good_hours", list(range(8, 17))),
            dd_scaling=data.get("dd_scaling", {}),
            ensemble=data.get("ensemble", {}),
        )


@dataclass
class Signal:
    """Trading Signal vom Bot."""
    symbol: str
    direction: OrderSide
    probability: float
    timestamp: datetime
    size: float = 0.0
    stop_distance: float = 0.0
    limit_distance: float = 0.0


class TradingBot:
    """
    Broker-agnostischer ML Trading Bot.

    Verantwortlich für:
    - ML Model Training und Prediction
    - Feature-Berechnung
    - Signal-Generierung

    NICHT verantwortlich für:
    - Daten-Beschaffung (→ BrokerAdapter)
    - Order-Ausführung (→ BrokerAdapter)
    - Broker-spezifische Logik (→ BrokerAdapter)
    """

    def __init__(
        self,
        adapter: BrokerAdapter,
        assets_config: Dict[str, Dict[str, Any]],
        account_config: Dict[str, Any],
        stats_dir: str = "stats_export",
        use_streaming: bool = True,
        min_training_samples: int = 500,
    ):
        """
        Args:
            adapter: BrokerAdapter für Daten und Order-Ausführung
            assets_config: Asset-Konfigurationen vom Optimizer
            account_config: Account-Konfiguration (currency, min_lot_size, etc.)
            stats_dir: Verzeichnis für Status-Export
            use_streaming: Live-Streaming nutzen wenn verfügbar
            min_training_samples: Minimum Samples für Model-Training
        """
        self.adapter = adapter
        self.account_config = account_config
        self.stats_dir = stats_dir
        self.use_streaming = use_streaming
        self.min_training_samples = min_training_samples

        # Parse asset configs
        self.assets: Dict[str, AssetConfig] = {}
        for symbol, cfg in assets_config.items():
            self.assets[symbol] = AssetConfig.from_dict(symbol, cfg)

        # ML Models
        self.models: Dict[str, XGBClassifier] = {}

        # OHLC und Feature Cache
        self.ohlc_cache: Dict[str, pd.DataFrame] = {}
        self.features_cache: Dict[str, pd.DataFrame] = {}
        self.last_bar_time: Dict[str, datetime] = {}

        # Signal Tracking
        self.last_signal_hour: Dict[str, datetime] = {}
        self.slippage_warnings: List[Dict[str, Any]] = []

        # Control
        self._stop_event = threading.Event()
        self._running = False

        # Account info
        self.account_id = account_config.get("account_id", "default")
        self.currency = account_config.get("currency", "EUR")
        self.min_lot_size = account_config.get("min_lot_size", 0.1)
        self.max_risk_percent = account_config.get("max_risk_percent", 0.05)

    def initialize(self) -> bool:
        """
        Initialisiert den Bot: Verbindung, Daten laden, Models trainieren.

        Returns:
            True wenn erfolgreich
        """
        logger.info("🚀 Initializing Trading Bot...")

        # Adapter verbinden
        if not self.adapter.is_connected:
            if not self.adapter.connect():
                logger.error("❌ Failed to connect adapter")
                return False

        # Models trainieren
        logger.info("🧠 Training ML models...")
        trained = 0
        for symbol, cfg in self.assets.items():
            if self._train_model(symbol, cfg):
                trained += 1
            time.sleep(1)  # Rate limiting

        if trained == 0:
            logger.error("❌ No models trained")
            return False

        logger.info(f"✅ Bot initialized with {trained} models")
        self._write_status("RUNNING")
        return True

    def _train_model(self, symbol: str, cfg: AssetConfig) -> bool:
        """
        Trainiert ein XGBoost Model für ein Symbol.

        Füllt gleichzeitig den OHLC und Feature Cache.
        """
        try:
            logger.info(f"📊 {symbol}: Loading historical data...")

            # Daten vom Adapter holen
            df = self.adapter.get_historical_bars(symbol, timeframe="1H", limit=1000)

            if df is None or df.empty:
                logger.warning(f"⚠️ {symbol}: No data available")
                return False

            if len(df) < self.min_training_samples:
                logger.warning(f"⚠️ {symbol}: Not enough data ({len(df)} < {self.min_training_samples})")
                return False

            # Cache füllen
            self.ohlc_cache[symbol] = df
            self.last_bar_time[symbol] = df.index[-1]

            # Features berechnen
            df = self._compute_features(df)
            self.features_cache[symbol] = df

            # Prüfe ob alle Features vorhanden
            missing = [f for f in cfg.features if f not in df.columns]
            if missing:
                logger.warning(f"⚠️ {symbol}: Missing features: {missing[:5]}...")
                return False

            # Target erstellen
            df["Target"] = (df["C"].shift(-1) > df["C"]).astype(int)
            df_train = df.dropna(subset=cfg.features + ["Target"])

            if len(df_train) < self.min_training_samples:
                logger.warning(f"⚠️ {symbol}: Not enough training samples ({len(df_train)})")
                return False

            # Model trainieren
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                n_jobs=-1,
                random_state=42,
                verbosity=0
            )
            model.fit(df_train[cfg.features], df_train["Target"])

            self.models[symbol] = model
            logger.info(f"✅ {symbol}: Model trained with {len(df_train)} samples")
            return True

        except Exception as e:
            logger.error(f"❌ {symbol}: Training failed: {e}")
            return False

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet alle Features für einen DataFrame."""
        df = df.copy()

        # Technische Indikatoren via Plugin-System
        df = compute_indicator_pool(df)

        # Makro-Indikatoren
        df = load_macro_indicators(df)

        # Zinsdaten
        df = load_interest_rates(df)

        return df

    def run(self):
        """
        Startet den Bot.

        Nutzt Streaming wenn verfügbar und aktiviert, sonst Polling.
        """
        if not self.initialize():
            return

        self._running = True

        if self.use_streaming:
            self._run_streaming()
        else:
            self._run_polling()

    def _run_streaming(self):
        """Streaming-Modus: Reagiert auf Live-Bar-Updates."""
        logger.info("📡 Starting streaming mode...")

        # Subscriptions für alle Symbole
        subscribed = 0
        for symbol in self.models.keys():
            if self.adapter.subscribe_bars(symbol, "1H", callback=self._on_bar):
                subscribed += 1

        if subscribed == 0:
            logger.warning("⚠️ No streaming subscriptions, falling back to polling")
            self._run_polling()
            return

        logger.info(f"📡 Streaming active for {subscribed} symbols")

        # Main loop - nur Health-Monitoring
        while not self._stop_event.is_set():
            self._write_status("RUNNING")

            if self._check_restart_signal():
                logger.info("🔄 Restart signal detected")
                break

            time.sleep(60)

        self._cleanup()

    def _run_polling(self):
        """Polling-Modus: Prüft regelmäßig auf neue Signale."""
        logger.info("📊 Starting polling mode...")

        while not self._stop_event.is_set():
            self._write_status("RUNNING")

            if self._check_restart_signal():
                logger.info("🔄 Restart signal detected")
                break

            now = datetime.now()

            # Nur an Werktagen handeln
            if now.weekday() < 5:
                self._check_all_signals()

            time.sleep(60)

        self._cleanup()

    def _on_bar(self, bar: BarData):
        """
        Callback für neue Bars vom Streaming.

        Args:
            bar: Neue Bar-Daten
        """
        symbol = bar.symbol

        if symbol not in self.models:
            return

        # Cache aktualisieren
        self._update_cache_with_bar(symbol, bar)

        # Signal prüfen
        self._check_signal(symbol)

    def _update_cache_with_bar(self, symbol: str, bar: BarData):
        """Aktualisiert den Cache mit einer neuen Bar."""
        if symbol not in self.ohlc_cache:
            return

        cache_df = self.ohlc_cache[symbol]

        # Neue Bar hinzufügen
        if bar.timestamp not in cache_df.index:
            cache_df.loc[bar.timestamp] = [bar.open, bar.high, bar.low, bar.close]
            cache_df = cache_df.sort_index()

            # Auf 1000 Bars begrenzen
            if len(cache_df) > 1000:
                cache_df = cache_df.tail(1000)

            self.ohlc_cache[symbol] = cache_df
            self.last_bar_time[symbol] = bar.timestamp

            # Features neu berechnen
            self.features_cache[symbol] = self._compute_features(cache_df)

    def _check_all_signals(self):
        """Prüft Signale für alle Symbole (Polling-Modus)."""
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        for symbol in self.models.keys():
            # Verhindere mehrfache Signale pro Stunde
            if self.last_signal_hour.get(symbol) == current_hour:
                continue

            # Cache aktualisieren wenn nötig
            self._update_cache_if_needed(symbol)

            # Signal prüfen
            self._check_signal(symbol)

    def _update_cache_if_needed(self, symbol: str):
        """Aktualisiert den Cache wenn eine neue Stunde begonnen hat."""
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        last_time = self.last_bar_time.get(symbol)
        if last_time is not None and last_time >= current_hour:
            return  # Cache ist aktuell

        # Neue Daten holen
        try:
            df = self.adapter.get_historical_bars(symbol, timeframe="1H", limit=24)
            if df is not None and not df.empty:
                # Mit bestehendem Cache mergen
                if symbol in self.ohlc_cache:
                    old_df = self.ohlc_cache[symbol]
                    combined = pd.concat([old_df, df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index().tail(1000)
                    self.ohlc_cache[symbol] = combined
                else:
                    self.ohlc_cache[symbol] = df

                self.last_bar_time[symbol] = self.ohlc_cache[symbol].index[-1]
                self.features_cache[symbol] = self._compute_features(self.ohlc_cache[symbol])

        except Exception as e:
            logger.warning(f"⚠️ {symbol}: Cache update failed: {e}")

    def _check_signal(self, symbol: str):
        """
        Prüft ob ein Trading-Signal generiert werden soll.

        Bei Signal wird eine Order über den Adapter platziert.
        """
        cfg = self.assets.get(symbol)
        if not cfg:
            return

        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        # Verhindere mehrfache Signale pro Stunde
        if self.last_signal_hour.get(symbol) == current_hour:
            return

        # Prüfe good_hours
        if now.hour not in cfg.good_hours:
            return

        try:
            df = self.features_cache.get(symbol)
            if df is None or len(df) < 100:
                return

            # Prüfe Features
            missing = [f for f in cfg.features if f not in df.columns]
            if missing:
                return

            # Prediction
            model = self.models.get(symbol)
            if model is None:
                return

            prob = model.predict_proba(df[cfg.features].iloc[[-1]])[0, 1]

            # Signal-Logik
            direction = None
            if prob >= cfg.conf_thresh:
                direction = OrderSide.BUY
            elif prob <= (1 - cfg.conf_thresh):
                direction = OrderSide.SELL

            if direction:
                self.last_signal_hour[symbol] = current_hour
                self._execute_signal(symbol, direction, prob, cfg)

        except Exception as e:
            logger.error(f"❌ {symbol}: Signal check failed: {e}")

    def _execute_signal(
        self,
        symbol: str,
        direction: OrderSide,
        probability: float,
        cfg: AssetConfig
    ):
        """
        Führt ein Signal aus - delegiert an den Adapter.
        """
        try:
            # Account Info für Position Sizing
            account = self.adapter.get_account_info()
            balance = account.balance

            if balance <= 0:
                logger.warning(f"⚠️ {symbol}: No balance available")
                return

            # ATR aus Cache für SL/TP Berechnung
            ohlc_df = self.ohlc_cache.get(symbol)
            if ohlc_df is None or len(ohlc_df) < 14:
                logger.warning(f"⚠️ {symbol}: Not enough data for ATR")
                return

            atr = ta.volatility.average_true_range(
                ohlc_df["H"], ohlc_df["L"], ohlc_df["C"]
            ).iloc[-1]

            # Position Sizing
            risk_cash = min(balance * cfg.kelly_risk, balance * self.max_risk_percent)
            sl_dist = max(10, int((atr * cfg.sl_mult) / cfg.point_value))
            tp_dist = int((atr * cfg.tp_mult) / cfg.point_value)
            size = max(self.min_lot_size, round(risk_cash / sl_dist, 2))

            logger.info(
                f"🎯 {symbol} SIGNAL: {direction.value} "
                f"(prob={probability:.3f}, size={size}, sl={sl_dist}, tp={tp_dist})"
            )

            # Order über Adapter ausführen
            result = self.adapter.submit_order(
                symbol=symbol,
                direction=direction,
                size=size,
                stop_distance=sl_dist,
                limit_distance=tp_dist,
            )

            if result.success:
                logger.info(f"✅ {symbol}: Order filled @ {result.fill_price}")
            else:
                logger.warning(f"⚠️ {symbol}: Order rejected - {result.message}")

        except Exception as e:
            logger.error(f"❌ {symbol}: Signal execution failed: {e}")

    def stop(self):
        """Stoppt den Bot."""
        logger.info("⏹️ Stopping bot...")
        self._stop_event.set()
        self._running = False

    def _cleanup(self):
        """Cleanup beim Beenden."""
        self._write_status("STOPPED")

        # Streaming Subscriptions beenden
        for symbol in self.models.keys():
            self.adapter.unsubscribe_bars(symbol)

    def _write_status(self, status: str):
        """Schreibt Bot-Status für Dashboard."""
        try:
            status_dir = os.path.join(self.stats_dir, self.account_id)
            os.makedirs(status_dir, exist_ok=True)
            status_file = os.path.join(status_dir, "bot_status.json")

            status_data = {
                "last_heartbeat": datetime.now().isoformat(),
                "status": status,
                "active_pairs_count": len(self.models),
                "active_symbols": list(self.models.keys()),
                "account_id": self.account_id,
                "slippage_warnings": self.slippage_warnings[-20:],
            }

            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to write status: {e}")

    def _check_restart_signal(self) -> bool:
        """Prüft ob ein Restart-Signal existiert."""
        restart_file = os.path.join(self.stats_dir, "restart_signal")
        if os.path.exists(restart_file):
            try:
                os.remove(restart_file)
                return True
            except Exception:
                pass
        return False


def run_bot_from_config(
    adapter: BrokerAdapter,
    account_dir: str,
    use_streaming: bool = True
):
    """
    Startet einen Bot aus Konfigurations-Dateien.

    Args:
        adapter: Verbundener BrokerAdapter
        account_dir: Verzeichnis mit account_info.json und assets.json
        use_streaming: Streaming nutzen wenn verfügbar
    """
    # Configs laden
    with open(os.path.join(account_dir, "account_info.json")) as f:
        account_config = json.load(f)

    with open(os.path.join(account_dir, "assets.json")) as f:
        assets_config = json.load(f)

    # Account ID aus Verzeichnis
    account_config["account_id"] = os.path.basename(account_dir)

    # Bot erstellen und starten
    bot = TradingBot(
        adapter=adapter,
        assets_config=assets_config,
        account_config=account_config,
        use_streaming=use_streaming,
    )

    bot.run()


__all__ = ["TradingBot", "AssetConfig", "Signal", "run_bot_from_config"]
