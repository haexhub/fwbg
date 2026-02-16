# Phase 6: Risk Management

## Zweck

Die Risk-Management-Phase berechnet Positionsgrößen und Risk-Controls basierend auf Trade-Historie und Performance-Metriken. Ziel: Optimale Kapitalallokation pro Trade unter Berücksichtigung der Gewinnwahrscheinlichkeit und des Risk-Reward-Verhältnisses.

---

## Wichtig: Nicht vom PipelineRunner ausgeführt

Risk Manager werden **nicht** vom PipelineRunner orchestriert. Sie werden **direkt vom Optimization-Code** aufgerufen, nachdem die Trade-Simulation abgeschlossen ist.

---

## BaseRiskManager

Basisklasse: `src/fwbg/plugins/risk_manager.py`

```python
class BaseRiskManager(BasePlugin, ABC):
    phase = PluginPhase.RISK_MANAGEMENT

    @abstractmethod
    def compute_risk_params(self, trades: List[float], win_rate: float,
                           rrr: float, **params) -> Dict[str, Any]:
        """
        Berechnet Risk-Parameter.

        Args:
            trades: Liste der Trade-Returns
            win_rate: Gewinnrate (0.0-1.0)
            rrr: Risk-Reward-Ratio (TP/SL)

        Returns:
            Dict mit mindestens:
            - risk_per_trade: float (Positionsgröße als Kapitalanteil)
            - trade_returns: List[float] (Per-Trade Returns für Metriken)
            - circuit_breaker: dict
            - risk_adjustment: dict
        """
```

- Registrierung: `@register_risk_manager("name")`

---

## Return-Value Struktur

```python
{
    "risk_per_trade": 0.02,        # 2% des Kapitals pro Trade
    "trade_returns": [...],         # Alle Trade-Returns mit angepasster Größe

    "circuit_breaker": {
        "pause_after_losses": 3,    # Nach 3 Verlusten in Folge pausieren
        "pause_bars": 10,           # Für 10 Bars pausieren
        "enabled": True
    },

    "risk_adjustment": {
        "original_risk": 0.03,      # Unbereinigtes Kelly-Ergebnis
        "scale_factor": 0.5,        # Herunterskaliert (Half-Kelly)
        "target_dd": 0.15           # Ziel-Drawdown von 15%
    }
}
```

---

## Verfügbare Plugins

### kelly (fwbg-core)

Kelly Criterion — berechnet die mathematisch optimale Positionsgröße basierend auf Gewinnwahrscheinlichkeit und Risk-Reward-Ratio:

```
Kelly% = WinRate - (1 - WinRate) / RRR
```

In der Praxis wird typischerweise "Half-Kelly" (50% des theoretischen Optimums) verwendet, da Full-Kelly sehr aggressive Positionsgrößen ergibt.

### vol_targeted_kelly (fwbg-core)

Kelly Criterion mit **Volatility Targeting** — skaliert die Positionsgröße dynamisch mit dem Verhältnis von Zielvolatilität zu realisierter Volatilität:

```
Adjusted_Size = Kelly_Size × (target_vol / realized_vol)
```

Bei hoher Volatilität wird die Position verkleinert, bei niedriger Volatilität vergrößert.

---

## Strategy-JSON Konfiguration

Risk Manager werden nicht direkt in der Strategy-JSON konfiguriert. Sie werden über den `risk_manager`-Parameter in der Strategy ausgewählt und automatisch mit den Trade-Ergebnissen aufgerufen.

---

## Eigenes Risk-Management-Plugin erstellen

Siehe [Plugin Development Guide](../plugin-development.md) für die vollständige Anleitung.
