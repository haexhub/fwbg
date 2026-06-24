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

from fwbg_sdk.models import BaseModel, TrainingContext
from fwbg.adapters.broker import BrokerAdapter, OrderSide, BarData
from fwbg.core import get_model
from fwbg.pipeline import compute_indicator_pool
from fwbg.data.loader import run_data_loading

logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """Konfiguration für ein einzelnes Asset (vom Optimizer)."""
    symbol: str
    features: List[str]
    conf_thresh: float = 0.55
    risk_per_trade: float = 0.005
    point_value: float = 0.0001
    sl_mult: float = 25.0
    tp_mult: float = 40.0
    ensemble: Dict[str, Any] = field(default_factory=dict)
    strategy_slug: str | None = None

    @classmethod
    def from_dict(cls, symbol: str, data: Dict[str, Any]) -> "AssetConfig":
        """Erstellt AssetConfig aus Dictionary."""
        return cls(
            symbol=symbol,
            features=data.get("features", []),
            conf_thresh=data.get("conf_thresh", 0.55),
            risk_per_trade=data.get("risk_per_trade", 0.005),
            point_value=data.get("point_value", 0.0001),
            sl_mult=data.get("sl_mult", 25.0),
            tp_mult=data.get("tp_mult", 40.0),
            ensemble=data.get("ensemble", {}),
            strategy_slug=data.get("strategy_slug"),
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

        # Strategy slug (M6a): when set, telemetry (Task 3) writes under
        # data/account-trades/<strategy_slug>/. None → legacy mode.
        # Derive from the first asset's slug (1-strategy-per-bot).
        self.strategy_slug: str | None = next(
            (a.strategy_slug for a in self.assets.values() if a.strategy_slug),
            None,
        )

        # ML Models
        self.models: Dict[str, BaseModel] = {}

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
        self._killed = False  # Kill-Switch Status

        # Account info
        self.account_id = account_config.get("account_id", "default")
        self.currency = account_config.get("currency", "EUR")
        self.min_lot_size = account_config.get("min_lot_size", 0.1)
        self.max_risk_percent = account_config.get("max_risk_percent", 0.05)

        # Risk Management - Position Limits
        self.max_concurrent_positions = account_config.get("max_concurrent_positions", 5)
        self.max_position_per_symbol = account_config.get("max_position_per_symbol", 1)
        self.max_total_exposure_percent = account_config.get("max_total_exposure_percent", 0.20)

        # Circuit Breaker - Drawdown Protection
        self.circuit_breaker_enabled = account_config.get("circuit_breaker_enabled", True)
        self.max_daily_loss_percent = account_config.get("max_daily_loss_percent", 0.05)
        self.pause_after_consecutive_losses = account_config.get("pause_after_consecutive_losses", 3)
        self.pause_duration_minutes = account_config.get("pause_duration_minutes", 60)

        # Circuit Breaker State
        self._daily_pnl = 0.0
        self._daily_start_balance = 0.0
        self._consecutive_losses = 0
        self._pause_until: datetime = None
        self._last_day: int = -1

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
            model_class = get_model("xgboost")
            model = model_class()
            training_context = TrainingContext()
            model.train(
                df_train[cfg.features], df_train["Target"].values,
                training_context,
                n_estimators=100, max_depth=5, random_state=42,
            )

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

        # Data Loading (Macro etc.) via Plugin-System
        if hasattr(self, '_data_loading_configs') and self._data_loading_configs:
            df = run_data_loading(df, self._data_loading_configs)

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
            # Kill-Switch prüfen
            if self._check_kill_switch():
                logger.critical("🚨 Kill switch activated in streaming mode")
                break

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
            # Kill-Switch prüfen
            if self._check_kill_switch():
                logger.critical("🚨 Kill switch activated in polling mode")
                break

            self._write_status("RUNNING")

            if self._check_restart_signal():
                logger.info("🔄 Restart signal detected")
                break

            now = datetime.now()

            # Nur an Werktagen handeln (wenn nicht pausiert)
            if now.weekday() < 5 and not self._check_circuit_breaker():
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

            prob = model.predict_probability(df[cfg.features].iloc[[-1]])[0, 1]

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

        Prüft vorher:
        - Kill-Switch
        - Circuit Breaker
        - Position Limits
        """
        # Kill-Switch Check
        if self._check_kill_switch():
            logger.warning(f"⚠️ {symbol}: Trading disabled (kill switch)")
            return

        # Circuit Breaker Check
        if self._check_circuit_breaker():
            logger.warning(f"⚠️ {symbol}: Trading paused (circuit breaker)")
            return

        # Position Limits Check
        if not self._check_position_limits(symbol):
            return

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
            risk_cash = min(balance * cfg.risk_per_trade, balance * self.max_risk_percent)
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

                # Slippage tracken
                if result.fill_price:
                    expected_price = self.ohlc_cache.get(symbol, pd.DataFrame()).get("C", pd.Series()).iloc[-1] if symbol in self.ohlc_cache else None
                    if expected_price:
                        slippage = abs(result.fill_price - expected_price)
                        if slippage > 0:
                            self.slippage_warnings.append({
                                "symbol": symbol,
                                "time": datetime.now().isoformat(),
                                "expected": expected_price,
                                "actual": result.fill_price,
                                "slippage": slippage
                            })
            else:
                logger.warning(f"⚠️ {symbol}: Order rejected - {result.message}")
                # Rejected order als Loss für Circuit Breaker zählen
                self._consecutive_losses += 1

        except Exception as e:
            logger.error(f"❌ {symbol}: Signal execution failed: {e}")

    def stop(self):
        """Stoppt den Bot."""
        logger.info("⏹️ Stopping bot...")
        self._stop_event.set()
        self._running = False

    def kill(self, close_positions: bool = True):
        """
        Emergency Kill Switch - Stoppt sofort allen Handel.

        Args:
            close_positions: Wenn True, werden alle offenen Positionen geschlossen
        """
        logger.critical("🚨 KILL SWITCH ACTIVATED")
        self._killed = True
        self._stop_event.set()
        self._running = False

        if close_positions:
            logger.warning("🚨 Closing ALL positions...")
            try:
                closed = self.adapter.close_all_positions()
                logger.warning(f"🚨 Closed {closed} positions")
            except Exception as e:
                logger.error(f"❌ Failed to close positions: {e}")

        self._write_status("KILLED")

    def _check_kill_switch(self) -> bool:
        """
        Prüft ob Kill-Switch ausgelöst werden soll.

        Returns:
            True wenn Handel gestoppt werden soll
        """
        if self._killed:
            return True

        # Prüfe auf Kill-Signal-Datei
        kill_file = os.path.join(self.stats_dir, "kill_signal")
        if os.path.exists(kill_file):
            try:
                os.remove(kill_file)
            except Exception:
                pass
            self.kill(close_positions=True)
            return True

        return False

    def _check_circuit_breaker(self) -> bool:
        """
        Prüft ob Circuit Breaker ausgelöst wurde.

        Returns:
            True wenn Handel pausiert werden soll
        """
        if not self.circuit_breaker_enabled:
            return False

        now = datetime.now()

        # Reset tägliche Statistik um Mitternacht
        if now.day != self._last_day:
            self._last_day = now.day
            self._daily_pnl = 0.0
            try:
                account = self.adapter.get_account_info()
                self._daily_start_balance = account.balance
            except Exception:
                pass

        # Prüfe ob Pause noch aktiv
        if self._pause_until and now < self._pause_until:
            return True

        # Prüfe täglichen Verlust
        if self._daily_start_balance > 0:
            daily_loss_percent = -self._daily_pnl / self._daily_start_balance
            if daily_loss_percent >= self.max_daily_loss_percent:
                logger.warning(
                    f"🚨 Circuit breaker: Daily loss limit reached "
                    f"({daily_loss_percent:.1%} >= {self.max_daily_loss_percent:.1%})"
                )
                self._pause_until = now.replace(hour=23, minute=59, second=59)
                return True

        # Prüfe consecutive losses
        if self._consecutive_losses >= self.pause_after_consecutive_losses:
            logger.warning(
                f"🚨 Circuit breaker: {self._consecutive_losses} consecutive losses, "
                f"pausing for {self.pause_duration_minutes} minutes"
            )
            self._pause_until = now + pd.Timedelta(minutes=self.pause_duration_minutes)
            self._consecutive_losses = 0
            return True

        return False

    def _check_position_limits(self, symbol: str) -> bool:
        """
        Prüft ob Position-Limits eingehalten werden.

        Args:
            symbol: Symbol für das eine neue Position eröffnet werden soll

        Returns:
            True wenn eine neue Position erlaubt ist
        """
        try:
            positions = self.adapter.get_positions()
            account = self.adapter.get_account_info()

            # Maximale Anzahl gleichzeitiger Positionen
            if len(positions) >= self.max_concurrent_positions:
                logger.warning(
                    f"⚠️ Position limit reached: {len(positions)}/{self.max_concurrent_positions}"
                )
                return False

            # Maximale Positionen pro Symbol
            symbol_positions = sum(1 for p in positions if p.symbol == symbol)
            if symbol_positions >= self.max_position_per_symbol:
                logger.warning(f"⚠️ {symbol}: Already have {symbol_positions} position(s)")
                return False

            # Maximale Gesamt-Exposure
            if account.balance > 0:
                total_exposure = sum(abs(p.size * p.entry_price) for p in positions)
                exposure_percent = total_exposure / account.balance
                if exposure_percent >= self.max_total_exposure_percent:
                    logger.warning(
                        f"⚠️ Total exposure limit reached: "
                        f"{exposure_percent:.1%} >= {self.max_total_exposure_percent:.1%}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Position limit check failed: {e}")
            return False  # Im Zweifel keine neue Position

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
    use_streaming: bool = True,
    strategy_slug: str | None = None,
):
    """
    Startet einen Bot aus Konfigurations-Dateien.

    Args:
        adapter: Verbundener BrokerAdapter
        account_dir: Verzeichnis mit account_info.json und assets.json
        use_streaming: Streaming nutzen wenn verfügbar
        strategy_slug: Optional. When set, propagates into every AssetConfig
            so the bot can isolate paper-trade telemetry under
            ``data/account-trades/<strategy_slug>/``. Kwarg wins over the
            ``strategy_slug`` key inside ``account_info.json``.
            ``None`` → legacy mode (no telemetry written).
    """
    # Configs laden
    with open(os.path.join(account_dir, "account_info.json")) as f:
        account_config = json.load(f)

    with open(os.path.join(account_dir, "assets.json")) as f:
        assets_config = json.load(f)

    # Account ID aus Verzeichnis
    account_config["account_id"] = os.path.basename(account_dir)

    # Resolve strategy_slug: kwarg wins over the optional file-key.
    effective_slug = strategy_slug if strategy_slug is not None else account_config.get("strategy_slug")
    if effective_slug is not None:
        for sym_cfg in assets_config.values():
            sym_cfg["strategy_slug"] = effective_slug

    # Bot erstellen und starten
    bot = TradingBot(
        adapter=adapter,
        assets_config=assets_config,
        account_config=account_config,
        use_streaming=use_streaming,
    )

    bot.run()


__all__ = ["TradingBot", "AssetConfig", "Signal", "run_bot_from_config"]
