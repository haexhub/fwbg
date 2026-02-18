# Phase 6: Risk Management

## Purpose

The risk management phase computes position sizes and risk controls based on trade history and performance metrics. Goal: Optimal capital allocation per trade, considering win probability and risk-reward ratio.

---

## Important: Not Executed by PipelineRunner

Risk managers are **not** orchestrated by the PipelineRunner. They are called **directly by the optimization code** after the trade simulation is complete.

---

## BaseRiskManager

Module: `fwbg_sdk.risk_managers`

```python
class BaseRiskManager(BasePlugin, ABC):
    phase = PluginPhase.RISK_MANAGEMENT

    @abstractmethod
    def compute_risk_params(self, trades: List[float], win_rate: float,
                           rrr: float, **params) -> Dict[str, Any]:
        """
        Computes risk parameters.

        Args:
            trades: List of trade returns
            win_rate: Win rate (0.0-1.0)
            rrr: Risk-reward ratio (TP/SL)

        Returns:
            Dict containing at least:
            - risk_per_trade: float (position size as fraction of capital)
            - trade_returns: List[float] (per-trade returns for metrics)
            - circuit_breaker: dict
            - risk_adjustment: dict
        """
```

- Import: `from fwbg_sdk import BaseRiskManager, register_risk_manager`
- Registration: `@register_risk_manager("name")`

---

## Return Value Structure

```python
{
    "risk_per_trade": 0.02,        # 2% of capital per trade
    "trade_returns": [...],         # All trade returns with adjusted sizing

    "circuit_breaker": {
        "pause_after_losses": 3,    # Pause after 3 consecutive losses
        "pause_bars": 10,           # Pause for 10 bars
        "enabled": True
    },

    "risk_adjustment": {
        "original_risk": 0.03,      # Raw Kelly result
        "scale_factor": 0.5,        # Scaled down (half-Kelly)
        "target_dd": 0.15           # Target drawdown of 15%
    }
}
```

---

## Available Plugins

### kelly (fwbg-core)

Kelly Criterion — computes the mathematically optimal position size based on win probability and risk-reward ratio:

```
Kelly% = WinRate - (1 - WinRate) / RRR
```

In practice, "half-Kelly" (50% of the theoretical optimum) is typically used, as full Kelly produces very aggressive position sizes.

### vol_targeted_kelly (fwbg-core)

Kelly Criterion with **volatility targeting** — dynamically scales position size by the ratio of target volatility to realized volatility:

```
Adjusted_Size = Kelly_Size × (target_vol / realized_vol)
```

In high-volatility environments, position size is reduced; in low-volatility environments, it is increased.

---

## Strategy JSON Configuration

Risk managers are not directly configured in the strategy JSON. They are selected via the `risk_manager` parameter in the strategy and automatically called with the trade results.

---

## Creating a Custom Risk Management Plugin

See [Plugin Development Guide](../plugin-development.md) for the complete guide.
