"""
IG Markets Broker Adapter.

Implementiert das BrokerAdapter-Interface für IG Markets.
Unterstützt REST API für historische Daten und Lightstreamer für Streaming.
"""
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from threading import Lock
import logging
import os
import time
import pandas as pd

try:
    from trading_ig import IGService
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False

try:
    from trading_ig.stream import IGStreamService
    from lightstreamer.client import Subscription
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from fwbg.adapters.broker import (
    BrokerAdapter, OrderSide, OrderType, OrderStatus,
    OrderResult, Position, AccountInfo, BarData,
    Symbol, Timeframe,
)
from .mappings import (
    SYMBOL_TO_EPIC,
    SYMBOL_TO_YFINANCE,
    SYMBOL_POINT_VALUE,
    TIMEFRAME_TO_RESOLUTION,
    TIMEFRAME_TO_YF_INTERVAL,
)
from .streaming import IGCandleListener

log = logging.getLogger(__name__)


class _IGCredentials:
    """Container for IG credentials with a safe repr that never leaks secrets."""

    __slots__ = ("username", "_password", "_api_key")

    def __init__(self, username: str, password: str, api_key: str):
        self.username = username
        self._password = password
        self._api_key = api_key

    @property
    def password(self) -> str:
        return self._password

    @property
    def api_key(self) -> str:
        return self._api_key

    def __repr__(self) -> str:
        return f"_IGCredentials(username={self.username!r}, password=<redacted>, api_key=<redacted>)"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self.username and self._password and self._api_key)


class IGBrokerAdapter(BrokerAdapter):
    """
    IG Markets Broker Adapter.

    Implementiert das vollständige BrokerAdapter-Interface:
    - Historische Daten via REST API (mit yfinance Fallback)
    - Live-Streaming via Lightstreamer
    - Order-Ausführung via REST API
    - Position- und Account-Management
    """

    adapter_type: str = "ig"

    MIN_STOP_POINTS = 1
    MAX_STOP_POINTS = int(os.environ.get("FWBG_IG_MAX_STOP_POINTS", "10000"))

    def __init__(
        self,
        username: str,
        password: str,
        api_key: str,
        env: str = "DEMO",
        currency: str = "EUR",
        use_yfinance_fallback: bool = True,
        rate_limit_delay: float = 2.0,
        **kwargs
    ):
        """
        Args:
            username: IG Benutzername
            password: IG Passwort
            api_key: IG API Key
            env: "DEMO" oder "LIVE"
            currency: Kontowährung
            use_yfinance_fallback: Bei IG-Fehlern auf yfinance zurückfallen
            rate_limit_delay: Pause zwischen API-Calls in Sekunden
        """
        if not IG_AVAILABLE:
            raise ImportError("trading_ig nicht installiert: pip install trading-ig")

        super().__init__(**kwargs)

        # Credentials are kept wrapped to prevent accidental leakage via repr/str
        # (mirrors src/fwbg/adapters/broker/ig/adapter.py).
        self._credentials = _IGCredentials(username, password, api_key)
        self.env = env.upper()
        self.currency = currency
        self.use_yfinance_fallback = use_yfinance_fallback
        self.rate_limit_delay = rate_limit_delay

        self._ig: Optional[IGService] = None
        self._stream_service: Optional[Any] = None
        self._subscriptions: Dict[Symbol, Any] = {}
        self._lock = Lock()
        self._last_request_time = 0.0

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> bool:
        """Verbindet mit der IG API."""
        try:
            self._ig = IGService(
                self._credentials.username,
                self._credentials.password,
                self._credentials.api_key,
                self.env
            )
            self._ig.create_session()
            self._connected = True
            self.log_info(f"Connected to IG {self.env}")
            return True
        except Exception as e:
            self.log_error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Trennt die Verbindung."""
        self._stop_streaming()
        if self._ig:
            try:
                self._ig.logout()
            except Exception:
                pass
            self._ig = None
        self._connected = False
        self.log_info("Disconnected from IG")

    def _rate_limit(self):
        """Enforced Rate Limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    # =========================================================================
    # Symbol Mapping
    # =========================================================================

    def get_broker_symbol(self, symbol: Symbol) -> Optional[str]:
        """Konvertiert Symbol zu IG Epic."""
        return SYMBOL_TO_EPIC.get(symbol)

    def get_point_value(self, symbol: Symbol) -> float:
        """Gibt Point Value für ein Symbol zurück."""
        return SYMBOL_POINT_VALUE.get(symbol, 0.0001)

    @property
    def is_paper(self) -> bool:
        """True iff connected to the IG DEMO environment."""
        return self.env == "DEMO"

    # =========================================================================
    # Historical Data
    # =========================================================================

    def get_historical_bars(
        self,
        symbol: Symbol,
        timeframe: Timeframe = Timeframe.H1,
        limit: int = 1000,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        Lädt historische OHLC-Daten.

        Versucht zuerst IG API, fällt bei Fehler auf yfinance zurück.
        """
        df = self._fetch_ig_historical(symbol, timeframe, limit)

        if (df is None or df.empty) and self.use_yfinance_fallback:
            df = self._fetch_yfinance_historical(symbol, timeframe, limit)

        if df is None:
            return pd.DataFrame(columns=["O", "H", "L", "C"])

        if start and len(df) > 0:
            df = df[df.index >= start]
        if end and len(df) > 0:
            df = df[df.index <= end]

        return df

    def _fetch_ig_historical(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """Holt historische Daten von IG API."""
        if not self._ig:
            return None

        epic = SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            self.log_warning(f"No EPIC mapping for {symbol}")
            return None

        self._rate_limit()
        resolution = TIMEFRAME_TO_RESOLUTION.get(timeframe, "HOUR")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._ig.fetch_historical_prices_by_epic(
                    epic=epic,
                    resolution=resolution,
                    numpoints=limit,
                )
                break
            except Exception as e:
                error_str = str(e)
                if "exceeded-account-historical-data-allowance" in error_str:
                    self.log_warning(f"IG historical data limit for {symbol}")
                    return None
                if "403" in error_str and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    self.log_warning(f"Rate limit for {symbol}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                self.log_warning(f"IG API error for {symbol}: {e}")
                return None

        if response is None or "prices" not in response:
            return None

        prices = response["prices"]
        if not prices:
            return None

        try:
            data = []
            for p in prices:
                snap = p.get("snapshotTimeUTC") or p.get("snapshotTime")
                o = (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2
                h = (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2
                low = (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2
                c = (p["closePrice"]["bid"] + p["closePrice"]["ask"]) / 2
                data.append({"T": snap, "O": o, "H": h, "L": low, "C": c})

            df = pd.DataFrame(data)
            df["T"] = pd.to_datetime(df["T"])
            df = df.set_index("T").sort_index()

            self.log_info(f"{symbol}: {len(df)} bars from IG")
            return df

        except Exception as e:
            self.log_warning(f"Error parsing IG data for {symbol}: {e}")
            return None

    def _fetch_yfinance_historical(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """Fallback: Holt historische Daten von yfinance."""
        if not YFINANCE_AVAILABLE:
            return None

        ticker = SYMBOL_TO_YFINANCE.get(symbol)
        if not ticker:
            self.log_warning(f"No yfinance ticker for {symbol}")
            return None

        try:
            yf_interval = TIMEFRAME_TO_YF_INTERVAL.get(timeframe, "1h")

            if "m" in yf_interval:
                minutes = int(yf_interval.replace("m", ""))
                days_needed = min((limit * minutes // 60 // 24) + 5, 7)
            elif "h" in yf_interval:
                days_needed = min((limit // 24) + 5, 60)
            else:
                days_needed = limit + 5

            self.log_info(f"{symbol}: Loading from yfinance ({ticker})")
            data = yf.download(
                ticker,
                period=f"{days_needed}d",
                interval=yf_interval,
                progress=False,
                auto_adjust=True
            )

            if data is None or data.empty:
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = data.rename(columns={
                "Open": "O", "High": "H", "Low": "L", "Close": "C"
            })[["O", "H", "L", "C"]]

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            if len(df) > limit:
                df = df.tail(limit)

            self.log_info(f"{symbol}: {len(df)} bars from yfinance")
            return df

        except Exception as e:
            self.log_warning(f"yfinance error for {symbol}: {e}")
            return None

    # =========================================================================
    # Current Price
    # =========================================================================

    def get_current_price(self, symbol: Symbol) -> Optional[Dict[str, float]]:
        """Ruft aktuellen Preis ab."""
        if not self._ig:
            return None

        epic = SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            return None

        self._rate_limit()

        try:
            market_info = self._ig.fetch_market_by_epic(epic)
            if market_info:
                snapshot = market_info.get("snapshot", {})
                instrument = market_info.get("instrument", {})

                # IG gibt Forex-Preise in Points zurück, nicht als Dezimalwert
                # Der echte Kurs steht in currencies[].baseExchangeRate
                currencies = instrument.get("currencies", [])
                if currencies and instrument.get("type") == "CURRENCIES":
                    # Für Forex: baseExchangeRate ist der echte Kurs
                    base_rate = currencies[0].get("baseExchangeRate")
                    if base_rate:
                        # Spread aus snapshot berechnen (in Points)
                        bid_points = snapshot.get("bid", 0)
                        ask_points = snapshot.get("offer", 0)
                        spread_points = ask_points - bid_points
                        # Point Value für dieses Symbol
                        point_value = self.get_point_value(symbol)
                        spread = spread_points * point_value / 10  # Points zu Preis
                        mid = float(base_rate)
                        return {
                            "bid": mid - spread / 2,
                            "ask": mid + spread / 2,
                            "mid": mid,
                        }

                # Für andere Instrumente (Indizes, Commodities): snapshot direkt
                bid = snapshot.get("bid")
                ask = snapshot.get("offer")
                if bid and ask:
                    return {
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": (float(bid) + float(ask)) / 2,
                    }
        except Exception as e:
            self.log_error(f"Price fetch failed for {symbol}: {e}")

        return None

    # =========================================================================
    # Order Execution
    # =========================================================================

    def _submit_order_impl(
        self,
        symbol: Symbol,
        direction: OrderSide,
        size: float,
        stop_distance: float = None,
        limit_distance: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """Sendet eine Order an IG.

        Wird nur über den Stop-Loss-Gate der Basisklasse (submit_order) oder von
        close_position (Exit, stop_distance=None) aufgerufen.
        """
        if not self._ig:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message="Not connected to IG"
            )

        epic = SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=f"No EPIC mapping for: {symbol}"
            )

        if stop_distance is not None and not (self.MIN_STOP_POINTS <= stop_distance <= self.MAX_STOP_POINTS):
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=f"stop_distance {stop_distance} outside allowed range [{self.MIN_STOP_POINTS}, {self.MAX_STOP_POINTS}]",
            )
        if limit_distance is not None and not (self.MIN_STOP_POINTS <= limit_distance <= self.MAX_STOP_POINTS):
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=f"limit_distance {limit_distance} outside allowed range [{self.MIN_STOP_POINTS}, {self.MAX_STOP_POINTS}]",
            )

        self._rate_limit()

        try:
            # Entries kommen durch den Basisklassen-Gate: stop_distance ist > 0.
            # None ist nur bei Exits (close_position) möglich → kein Stop senden
            # (kein stiller 50er-Default mehr).
            sl_dist = int(round(stop_distance)) if stop_distance is not None else None
            tp_dist = (
                int(round(limit_distance)) if limit_distance
                else (sl_dist * 2 if sl_dist is not None else None)
            )

            self.log_info(
                f"Sending order: {direction.value} {symbol} "
                f"Size={size} SL={sl_dist} TP={tp_dist}"
            )

            # Stop-Loss wird atomar im selben Request mit dem Entry gesendet.
            response = self._ig.create_open_position(
                currency_code=self.currency,
                direction=direction.value,
                epic=epic,
                expiry="DFB",
                order_type=order_type.value,
                size=size,
                guaranteed_stop=False,
                stop_distance=sl_dist,
                limit_distance=tp_dist,
            )

            if response and "dealReference" in response:
                deal_ref = response["dealReference"]
                time.sleep(0.5)
                self._rate_limit()
                confirmation = self._ig.fetch_deal_by_deal_reference(deal_ref)

                if confirmation:
                    deal_status = confirmation.get("dealStatus")
                    if deal_status == "ACCEPTED":
                        fill_price = confirmation.get("level") or confirmation.get("openLevel") or 0.0
                        return OrderResult(
                            success=True,
                            order_id=deal_ref,
                            status=OrderStatus.FILLED,
                            fill_price=float(fill_price),
                            filled_quantity=size,
                            message="Order filled",
                            raw_response=confirmation,
                        )
                    else:
                        reason = confirmation.get("reason", "Unknown rejection")
                        return OrderResult(
                            success=False,
                            order_id=deal_ref,
                            status=OrderStatus.REJECTED,
                            message=reason,
                            raw_response=confirmation,
                        )
                else:
                    return OrderResult(
                        success=True,
                        order_id=deal_ref,
                        status=OrderStatus.PENDING,
                        message="Order sent, awaiting confirmation",
                        raw_response=response,
                    )
            else:
                return OrderResult(
                    success=False,
                    status=OrderStatus.REJECTED,
                    message=str(response),
                )

        except Exception as e:
            self.log_error(f"Order error: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=str(e),
            )

    # =========================================================================
    # Position Management
    # =========================================================================

    def get_positions(self) -> List[Position]:
        """Ruft offene Positionen ab."""
        if not self._ig:
            return []

        self._rate_limit()

        try:
            positions_df = self._ig.fetch_open_positions()
            if positions_df is None or len(positions_df) == 0:
                return []

            positions = []
            for _, row in positions_df.iterrows():
                direction_str = row.get("direction", "BUY")
                direction = OrderSide.BUY if direction_str == "BUY" else OrderSide.SELL

                epic = row.get("epic", "")
                symbol = self._epic_to_symbol(epic)

                stop_level = row.get("stopLevel")
                limit_level = row.get("limitLevel")
                positions.append(Position(
                    symbol=symbol,
                    direction=direction,
                    size=float(row.get("size", 0)),
                    entry_price=float(row.get("openLevel", 0)),
                    current_price=float(row.get("level", 0)),
                    unrealized_pnl=float(row.get("profit", 0)),
                    position_id=str(row.get("dealId", "")),
                    stop_level=stop_level,
                    limit_level=limit_level,
                    stop_loss=float(stop_level) if stop_level is not None else None,
                    take_profit=float(limit_level) if limit_level is not None else None,
                    currency=row.get("currency", self.currency),
                ))
            return positions

        except Exception as e:
            self.log_error(f"Failed to get positions: {e}")
            return []

    def _epic_to_symbol(self, epic: str) -> str:
        """Konvertiert IG Epic zurück zu Symbol-String."""
        for sym, ep in SYMBOL_TO_EPIC.items():
            if ep == epic:
                return str(sym)
        return epic

    # =========================================================================
    # Account Info
    # =========================================================================

    def get_account_info(self) -> AccountInfo:
        """Ruft Kontoinformationen ab."""
        if not self._ig:
            return AccountInfo(balance=0, equity=0, currency=self.currency)

        self._rate_limit()

        try:
            accounts = self._ig.fetch_accounts()
            if accounts is not None and len(accounts) > 0:
                return AccountInfo(
                    balance=float(accounts.loc[0, "balance"]),
                    equity=float(accounts.loc[0, "balance"]) + float(accounts.loc[0, "profitLoss"]),
                    margin_used=float(accounts.loc[0, "deposit"]),
                    margin_available=float(accounts.loc[0, "available"]),
                    currency=self.currency,
                )
        except Exception as e:
            self.log_error(f"Failed to get account info: {e}")

        return AccountInfo(balance=0, equity=0, currency=self.currency)

    # =========================================================================
    # Streaming
    # =========================================================================

    def subscribe_bars(
        self,
        symbol: Symbol,
        timeframe: Timeframe = Timeframe.H1,
        callback: Callable[[BarData], None] = None,
    ) -> bool:
        """Abonniert Live-Bars via Lightstreamer."""
        if not STREAMING_AVAILABLE:
            self.log_warning("Streaming not available - install trading-ig[streaming]")
            return False

        epic = SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            self.log_warning(f"No EPIC for {symbol}")
            return False

        try:
            if not self._stream_service:
                self._init_streaming()

            item = f"CHART:{epic}:HOUR"

            subscription = Subscription(
                mode="MERGE",
                items=[item],
                fields=["UTM", "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
                        "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE", "CONS_END"]
            )

            listener = IGCandleListener(self, symbol, callback)
            subscription.addListener(listener)

            self._stream_service.subscribe(subscription)
            self._subscriptions[symbol] = subscription

            if callback:
                self.add_bar_callback(symbol, callback)

            self.log_info(f"Subscribed to {symbol} streaming")
            return True

        except Exception as e:
            self.log_error(f"Subscription failed for {symbol}: {e}")
            return False

    def _init_streaming(self):
        """Initialisiert Streaming Service."""
        if not self._ig:
            raise RuntimeError("Not connected to IG")

        session_result = self._ig.create_session()
        acc_id = None
        if isinstance(session_result, dict):
            acc_id = session_result.get('currentAccountId')

        if not acc_id:
            raise ValueError("Could not get account ID from IG session")

        self._stream_service = IGStreamService(self._ig)
        self._stream_service.acc_number = acc_id
        self._stream_service.create_session()

        self.log_info("Streaming service initialized")

    def _stop_streaming(self):
        """Stoppt Streaming Service."""
        if self._stream_service:
            try:
                self._stream_service.unsubscribe_all()
                self._stream_service.disconnect()
            except Exception:
                pass
            self._stream_service = None
        self._subscriptions.clear()
        self._bar_callbacks.clear()
