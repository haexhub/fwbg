"""
Binance Execution Adapter für FWBG.

Installation:
    pip install fwbg-adapter-binance

    # Oder lokal:
    cd examples/adapter_package
    pip install -e .

Verwendung:
    from fwbg.core import discover_plugins, get_execution_adapter

    discover_plugins()

    BinanceAdapter = get_execution_adapter("binance")
    adapter = BinanceAdapter(api_key="...", api_secret="...")

    with adapter:
        # Trading...
        pass
"""
from typing import List, Optional
from fwbg.adapters import (
    ExecutionAdapter,
    Order, Position, AccountInfo,
    OrderType, OrderSide,
)
from fwbg.core.events import SignalEvent, OrderFilledEvent, OrderRejectedEvent

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


class BinanceAdapter(ExecutionAdapter):
    """
    Binance Spot/Futures Execution Adapter.

    Unterstützt:
    - Spot Trading
    - USDT-M Futures
    - Coin-M Futures
    """

    adapter_type = "binance"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        futures: bool = False,
        **kwargs
    ):
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance nicht installiert: pip install python-binance")

        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.futures = futures
        self._client: Optional[Client] = None

    def connect(self) -> bool:
        try:
            self._client = Client(
                self.api_key,
                self.api_secret,
                testnet=self.testnet,
            )
            # Verbindung testen
            self._client.get_account()
            self._connected = True
            mode = "Futures" if self.futures else "Spot"
            env = "Testnet" if self.testnet else "Live"
            self.log_info(f"Connected to Binance {mode} ({env})")
            return True
        except Exception as e:
            self.log_error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        self._client = None
        self._connected = False

    def signal_to_order(self, signal: SignalEvent) -> Optional[Order]:
        side = OrderSide.BUY if signal.direction == "BUY" else OrderSide.SELL

        # Position Sizing basierend auf Account Balance
        account = self.get_account_info()
        risk_amount = account.balance * 0.02  # 2% Risiko

        # Vereinfachte Berechnung
        quantity = 0.001  # Muss basierend auf Symbol angepasst werden

        return Order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def submit_order(self, order: Order) -> bool:
        if not self._client:
            return False

        try:
            if self.futures:
                result = self._client.futures_create_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    type="MARKET",
                    quantity=order.quantity,
                )
            else:
                result = self._client.create_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    type="MARKET",
                    quantity=order.quantity,
                )

            if result.get("status") == "FILLED":
                self.publish(OrderFilledEvent(
                    symbol=order.symbol,
                    order_id=str(result["orderId"]),
                    side=order.side.value,
                    quantity=float(result["executedQty"]),
                    price=float(result["fills"][0]["price"]) if result.get("fills") else 0,
                    commission=sum(float(f["commission"]) for f in result.get("fills", [])),
                ))
                return True

            return False

        except Exception as e:
            self.log_error(f"Order failed: {e}")
            self.publish(OrderRejectedEvent(
                symbol=order.symbol,
                order_id=order.order_id,
                reason=str(e),
            ))
            return False

    def cancel_order(self, order_id: str) -> bool:
        # Implementation...
        return False

    def get_positions(self) -> List[Position]:
        if not self._client:
            return []

        try:
            if self.futures:
                positions = self._client.futures_position_information()
                return [
                    Position(
                        symbol=p["symbol"],
                        side=OrderSide.BUY if float(p["positionAmt"]) > 0 else OrderSide.SELL,
                        quantity=abs(float(p["positionAmt"])),
                        entry_price=float(p["entryPrice"]),
                        current_price=float(p["markPrice"]),
                        unrealized_pnl=float(p["unRealizedProfit"]),
                    )
                    for p in positions
                    if float(p["positionAmt"]) != 0
                ]
            return []
        except Exception as e:
            self.log_error(f"Failed to get positions: {e}")
            return []

    def get_account_info(self) -> AccountInfo:
        if not self._client:
            return AccountInfo(balance=0, equity=0, currency="USDT")

        try:
            if self.futures:
                account = self._client.futures_account()
                return AccountInfo(
                    balance=float(account["totalWalletBalance"]),
                    equity=float(account["totalMarginBalance"]),
                    margin_used=float(account["totalInitialMargin"]),
                    margin_available=float(account["availableBalance"]),
                    currency="USDT",
                )
            else:
                account = self._client.get_account()
                usdt = next((a for a in account["balances"] if a["asset"] == "USDT"), None)
                balance = float(usdt["free"]) + float(usdt["locked"]) if usdt else 0
                return AccountInfo(
                    balance=balance,
                    equity=balance,
                    currency="USDT",
                )
        except Exception as e:
            self.log_error(f"Failed to get account: {e}")
            return AccountInfo(balance=0, equity=0, currency="USDT")


__all__ = ["BinanceAdapter"]
