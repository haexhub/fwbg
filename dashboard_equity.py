import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_equity_curve():
    file_path = "trade_history.csv"
    if not os.path.exists(file_path):
        print("❌ Noch keine Trades in der Historie gefunden.")
        return

    # Daten laden
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Kumulierten Gewinn berechnen
    # Wir nehmen an, wir starten bei 0 oder einem Startkapital
    df["cumulative_pnl"] = df["pnl"].cumsum()

    # Plot erstellen
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["timestamp"],
        df["cumulative_pnl"],
        marker="o",
        linestyle="-",
        color="#2ecc71",
        label="Kumulierter PnL",
    )

    # Null-Linie zur Orientierung
    plt.axhline(0, color="red", linewidth=0.8, linestyle="--")

    plt.title("Statistische Performance: Kumulierter Gewinn/Verlust", fontsize=14)
    plt.xlabel("Zeitpunkt", fontsize=12)
    plt.ylabel("PnL in EUR", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    print(f"📊 Trades analysiert: {len(df)}")
    print(f"💰 Aktueller Gesamt-PnL: {df['cumulative_pnl'].iloc[-1]:.2f} EUR")

    plt.show()


if __name__ == "__main__":
    plot_equity_curve()
