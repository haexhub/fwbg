import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# --- EINSTELLUNGEN ---
TRADE_LOG = "trade_history.csv"
EXPECTED_WIN_RATE = 0.58  # Wert aus deinem Ultimate Optimizer (Beispiel)
EXPECTED_PROFIT_FACTOR = 1.5


def calculate_z_score(actual_wr, expected_wr, num_trades):
    """Berechnet, ob die Abweichung statistisch signifikant ist"""
    if num_trades == 0:
        return 0
    import math

    # Standardfehler der Proportion
    se = math.sqrt((expected_wr * (1 - expected_wr)) / num_trades)
    return (actual_wr - expected_wr) / se if se != 0 else 0


def run_dashboard():
    if not os.path.exists(TRADE_LOG):
        print("❌ Noch keine Trade-Historie vorhanden.")
        return

    # 1. Daten laden
    df = pd.read_csv(TRADE_LOG)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # 2. Metriken berechnen
    total_trades = len(df)
    winning_trades = len(df[df["pnl"] > 0])
    actual_wr = winning_trades / total_trades if total_trades > 0 else 0

    cumulative_pnl = df["pnl"].cumsum()
    max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()

    # 3. Model Drift Check (Z-Score)
    # Ein Z-Score < -1.96 bedeutet eine 95%ige Wahrscheinlichkeit, dass das Modell driftet
    z_score = calculate_z_score(actual_wr, EXPECTED_WIN_RATE, total_trades)

    # 4. Visualisierung
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Equity Kurve
    ax1.plot(
        df["timestamp"],
        cumulative_pnl,
        label="Echte Performance",
        color="blue",
        linewidth=2,
    )
    ax1.set_title(f"Live Equity Kurve (Trades: {total_trades})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Win-Rate Vergleich
    colors = ["green" if actual_wr >= EXPECTED_WIN_RATE else "red"]
    ax2.bar(
        ["Ist-WinRate", "Soll-WinRate"],
        [actual_wr, EXPECTED_WIN_RATE],
        color=["blue", "gray"],
    )
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Win-Rate Check (Drift Z-Score: {z_score:.2f})")

    # Status-Anzeige
    status_text = "✅ SYSTEM STABIL"
    status_color = "green"
    if z_score < -1.65:
        status_text = "⚠️ WARNUNG: MODEL DRIFT"
        status_color = "orange"
    if z_score < -2.33:
        status_text = "🚨 ALARM: PARAMETER UNGÜLTIG"
        status_color = "red"

    plt.figtext(
        0.5,
        0.02,
        status_text,
        ha="center",
        fontsize=14,
        bbox={"facecolor": status_color, "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_dashboard()
