"""
IG Markets Execution Adapter.

Führt Orders über die IG REST API aus.
Nutzt das Event-System für Signal-Empfang und Order-Updates.

Beispiel:
    adapter = IGExecutionAdapter(
        username="...",
        password="...",
        api_key="...",
        env="DEMO"
    )

    with adapter:
        # Automatisch subscribed zu SignalEvents
        # Publiziert OrderFilledEvent/OrderRejectedEvent
        pass
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import time

try:
    from trading_ig import IGService
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False

from . import (
    ExecutionAdapter, Order, Position, AccountInfo,
    OrderType, OrderSide
)
from fwbg.core.events import (
    SignalEvent, OrderFilledEvent, OrderRejectedEvent
)

log = logging.getLogger(__name__)


class IGExecutionAdapter(ExecutionAdapter):
    """
    IG Markets Execution Adapter.

    Empfängt SignalEvents vom MessageBus und führt Orders über IG aus.
    Publiziert OrderFilledEvent/OrderRejectedEvent nach Ausführung.
    """

    adapter_type: str = "ig"

    # Symbol -> Epic Mapping
    SYMBOL_TO_EPIC = {
        # Indizes CFDs
        "FTSE100": "IX.D.FTSE.DAILY.IP",
        "DOW30": "IX.D.DOW.DAILY.IP",
        "NAS100": "IX.D.NASDAQ.DAILY.IP",
        "DAX": "IX.D.DAX.DAILY.IP",
        "SPX500": "IX.D.SPTRD.DAILY.IP",
        # Forex CFDs
        "EURUSD": "CS.D.EURUSD.TODAY.IP",
        "GBPUSD": "CS.D.GBPUSD.TODAY.IP",
        "USDJPY": "CS.D.USDJPY.TODAY.IP",
        "USDCHF": "CS.D.USDCHF.TODAY.IP",
        "USDCAD": "CS.D.USDCAD.TODAY.IP",
        "AUDUSD": "CS.D.AUDUSD.TODAY.IP",
        "NZDUSD": "CS.D.NZDUSD.TODAY.IP",
        "EURCAD": "CS.D.EURCAD.TODAY.IP",
        "EURGBP": "CS.D.EURGBP.TODAY.IP",
        "EURJPY": "CS.D.EURJPY.TODAY.IP",
        # Commodities CFDs
        "XAUUSD": "CS.D.CFDGOLD.CFD.IP",
        "GOLD": "CS.D.CFDGOLD.CFD.IP",
        "XAGUSD": "CS.D.CFDSILVER.CFD.IP",
        "SILVER": "CS.D.CFDSILVER.CFD.IP",
        "BRENT": "CC.D.LCO.UNC.IP",
        "WTI": "CC.D.CL.UNC.IP",
    }

    # Point Value für korrekte Pips-Berechnung
    SYMBOL_POINT_VALUE = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
        "USDCHF": 0.0001,
        "USDCAD": 0.0001,
        "AUDUSD": 0.0001,
        "NZDUSD": 0.0001,
        "XAUUSD": 0.01,
        "GOLD": 0.01,
        "DAX": 1.0,
        "DOW30": 1.0,
        "NAS100": 1.0,
        "SPX500": 0.1,
        "FTSE100": 1.0,
    }

    def __init__(
        self,
        username: str,
        password: str,
        api_key: str,
        env: str = "DEMO",
        currency: str = "EUR",
        min_lot_size: float = 0.1,
        max_risk_percent: float = 0.05,
        **kwargs
    ):
        """
        Args:
            username: IG Benutzername
            password: IG Passwort
            api_key: IG API Key
            env: "DEMO" oder "LIVE"
            currency: Kontowährung
            min_lot_size: Minimale Lotgröße
            max_risk_percent: Maximales Risiko pro Trade
        """
        if not IG_AVAILABLE:
            raise ImportError(
                "trading_ig ist nicht installiert. pip install trading-ig"
            )

        super().__init__(**kwargs)

        self.username = username
        self.password = password
        self.api_key = api_key
        self.env = env.upper()
        self.currency = currency
        self.min_lot_size = min_lot_size
        self.max_risk_percent = max_risk_percent

        self._ig: Optional[IGService] = None
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # Rate limiting

    def connect(self) -> bool:
        """Verbindet mit der IG API."""
        try:
            self._ig = IGService(
                self.username,
                self.password,
                self.api_key,
                self.env
            )
            self._ig.create_session()
            self._connected = True
            self.log_info(f"Connected to IG {self.env} Account")
            return True
        except Exception as e:
            self.log_error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Trennt die Verbindung."""
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
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_epic(self, symbol: str) -> Optional[str]:
        """Konvertiert Symbol zu IG Epic."""
        return self.SYMBOL_TO_EPIC.get(symbol.upper())

    def _get_point_value(self, symbol: str) -> float:
        """Gibt Point Value für ein Symbol zurück."""
        return self.SYMBOL_POINT_VALUE.get(symbol.upper(), 0.0001)

    def signal_to_order(self, signal: SignalEvent) -> Optional[Order]:
        """
        Konvertiert SignalEvent zu Order mit IG-spezifischer Logik.

        Berechnet Positionsgröße basierend auf Risiko.
        """
        side = OrderSide.BUY if signal.direction == "BUY" else OrderSide.SELL

        # Positionsgröße berechnen
        account = self.get_account_info()
        size = self._calculate_position_size(signal, account)

        return Order(
            symbol=signal.symbol,
            side=side,
            quantity=max(self.min_lot_size, size),
            order_type=OrderType.MARKET,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def _calculate_position_size(
        self,
        signal: SignalEvent,
        account: AccountInfo
    ) -> float:
        """Berechnet Positionsgröße basierend auf Risiko."""
        if account.balance <= 0:
            return self.min_lot_size

        risk_amount = account.balance * self.max_risk_percent

        # Bei Stop-Loss: Risiko basierend auf SL-Distanz
        if signal.stop_loss and signal.stop_loss > 0:
            point_value = self._get_point_value(signal.symbol)
            # Annahme: SL ist in Pips
            size = risk_amount / (signal.stop_loss / point_value)
            return round(size, 2)

        return self.min_lot_size

    def submit_order(self, order: Order) -> bool:
        """Sendet Order an IG."""
        if not self._ig:
            self._publish_order_rejected(
                order, "Not connected to IG"
            )
            return False

        epic = self._get_epic(order.symbol)
        if not epic:
            self._publish_order_rejected(
                order, f"Unknown symbol: {order.symbol}"
            )
            return False

        self._rate_limit()

        try:
            # SL/TP Distanzen berechnen
            point_value = self._get_point_value(order.symbol)
            sl_distance = self._calculate_distance(order.stop_loss, point_value)
            tp_distance = self._calculate_distance(order.take_profit, point_value)

            # Mindest-Distanzen
            sl_distance = max(10, sl_distance) if sl_distance else 50
            tp_distance = max(10, tp_distance) if tp_distance else sl_distance * 2

            self.log_info(
                f"Sending order: {order.side.value} {order.symbol} "
                f"Size={order.quantity} SL={sl_distance} TP={tp_distance}"
            )

            # Order platzieren
            response = self._ig.create_open_position(
                currency_code=self.currency,
                direction=order.side.value,
                epic=epic,
                expiry="DFB",  # Daily Funded Bet
                order_type="MARKET",
                size=order.quantity,
                guaranteed_stop=False,
                stop_distance=sl_distance,
                limit_distance=tp_distance,
            )

            if response and "dealReference" in response:
                deal_ref = response["dealReference"]
                order.order_id = deal_ref

                # Deal-Bestätigung abrufen
                time.sleep(0.5)
                self._rate_limit()

                confirmation = self._ig.fetch_deal_by_deal_reference(deal_ref)

                if confirmation:
                    deal_status = confirmation.get("dealStatus")
                    if deal_status == "ACCEPTED":
                        execution_price = (
                            confirmation.get("level") or
                            confirmation.get("openLevel")
                        )
                        self._publish_order_filled(
                            order,
                            execution_price or 0.0,
                            order.quantity,
                            deal_ref
                        )
                        return True
                    else:
                        reason = confirmation.get("reason", "Unknown rejection")
                        self._publish_order_rejected(order, reason)
                        return False
                else:
                    # Keine Bestätigung, aber Deal Reference erhalten
                    self._publish_order_filled(
                        order, 0.0, order.quantity, deal_ref
                    )
                    return True
            else:
                self._publish_order_rejected(order, str(response))
                return False

        except Exception as e:
            self.log_error(f"Order error: {e}")
            self._publish_order_rejected(order, str(e))
            return False

    def _calculate_distance(
        self,
        value: Optional[float],
        point_value: float
    ) -> int:
        """Berechnet Distanz in Punkten."""
        if value is None or value <= 0:
            return 0
        return int(value / point_value)

    def _publish_order_filled(
        self,
        order: Order,
        price: float,
        quantity: float,
        deal_id: str
    ):
        """Publiziert OrderFilledEvent."""
        event = OrderFilledEvent(
            symbol=order.symbol,
            order_id=order.order_id or deal_id,
            side=order.side.value,
            quantity=quantity,
            price=price,
            commission=0.0,
        )
        self.publish(event)
        self.log_info(
            f"Order filled: {order.side.value} {order.symbol} "
            f"@ {price} (Deal: {deal_id})"
        )

    def _publish_order_rejected(self, order: Order, reason: str):
        """Publiziert OrderRejectedEvent."""
        event = OrderRejectedEvent(
            symbol=order.symbol,
            order_id=order.order_id or "",
            reason=reason,
        )
        self.publish(event)
        self.log_warning(
            f"Order rejected: {order.symbol} - {reason}"
        )

    def cancel_order(self, order_id: str) -> bool:
        """Storniert eine Order (nicht unterstützt für Market Orders)."""
        self.log_warning("IG Market Orders cannot be cancelled")
        return False

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
                direction = row.get("direction", "BUY")
                side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL

                # Symbol aus Epic ermitteln
                epic = row.get("epic", "")
                symbol = self._epic_to_symbol(epic)

                positions.append(Position(
                    symbol=symbol,
                    side=side,
                    quantity=float(row.get("size", 0)),
                    entry_price=float(row.get("openLevel", 0)),
                    current_price=float(row.get("level", 0)),
                    unrealized_pnl=float(row.get("profit", 0)),
                    position_id=str(row.get("dealId", "")),
                    stop_loss=row.get("stopLevel"),
                    take_profit=row.get("limitLevel"),
                ))
            return positions

        except Exception as e:
            self.log_error(f"Failed to get positions: {e}")
            return []

    def _epic_to_symbol(self, epic: str) -> str:
        """Konvertiert IG Epic zurück zu Symbol."""
        for symbol, ep in self.SYMBOL_TO_EPIC.items():
            if ep == epic:
                return symbol
        return epic  # Fallback: Epic als Symbol

    def get_account_info(self) -> AccountInfo:
        """Ruft Kontoinformationen ab."""
        if not self._ig:
            return AccountInfo(
                balance=0, equity=0, currency=self.currency
            )

        self._rate_limit()

        try:
            accounts = self._ig.fetch_accounts()
            if accounts is not None and len(accounts) > 0:
                return AccountInfo(
                    balance=float(accounts.loc[0, "balance"]),
                    equity=float(accounts.loc[0, "balance"]) +
                           float(accounts.loc[0, "profitLoss"]),
                    margin_used=float(accounts.loc[0, "deposit"]),
                    margin_available=float(accounts.loc[0, "available"]),
                    currency=self.currency,
                )
        except Exception as e:
            self.log_error(f"Failed to get account info: {e}")

        return AccountInfo(
            balance=0, equity=0, currency=self.currency
        )

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Ruft aktuellen Preis ab."""
        if not self._ig:
            return None

        epic = self._get_epic(symbol)
        if not epic:
            return None

        self._rate_limit()

        try:
            market_info = self._ig.fetch_market_by_epic(epic)
            if market_info:
                snapshot = market_info.get("snapshot", {})
                bid = snapshot.get("bid")
                ask = snapshot.get("offer")
                if bid and ask:
                    return {
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": (float(bid) + float(ask)) / 2,
                    }
        except Exception as e:
            self.log_error(f"Failed to get price for {symbol}: {e}")

        return None

    def add_symbol_mapping(
        self,
        symbol: str,
        epic: str,
        point_value: float = 0.0001
    ):
        """
        Fügt ein neues Symbol-Mapping hinzu.

        Args:
            symbol: Asset-Symbol
            epic: IG Epic
            point_value: Punkt-Wert für Pips-Berechnung
        """
        self.SYMBOL_TO_EPIC[symbol.upper()] = epic
        self.SYMBOL_POINT_VALUE[symbol.upper()] = point_value


__all__ = ["IGExecutionAdapter"]
