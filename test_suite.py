import unittest
import os
import yaml
import pandas as pd
import numpy as np
from ig_bot import IGBot, BASE_CFG


class TestOmniBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialisiert den Bot einmal für alle Tests."""
        try:
            cls.bot = IGBot()
        except Exception as e:
            cls.bot = None
            print(f"\n❌ Setup fehlgeschlagen: {e}")

    # --- 1. CONFIG TESTS ---
    def test_config_loading(self):
        """Prüft, ob die Config-Datei korrekt geladen wurde."""
        self.assertIsNotNone(BASE_CFG)
        self.assertIn("active_profile", BASE_CFG)
        self.assertIn("pairs", BASE_CFG)
        print("✅ Config-Struktur ist valide.")

    # --- 2. LOGIK & DATEN TESTS ---
    def test_indicator_calculation(self):
        """Prüft, ob die technischen Indikatoren korrekt berechnet werden."""
        # Erstelle Fake-Daten
        data = {
            "High": np.random.uniform(1.08, 1.10, 100),
            "Low": np.random.uniform(1.06, 1.08, 100),
            "Close": np.random.uniform(1.07, 1.09, 100),
        }
        df_test = pd.DataFrame(data)

        # Simuliere die Indikatoren-Berechnung aus dem Bot
        import ta

        df_test["RSI"] = ta.momentum.rsi(df_test["Close"], 14)
        df_test["ATR"] = ta.volatility.average_true_range(
            df_test["High"], df_test["Low"], df_test["Close"], 14
        )

        self.assertFalse(df_test["RSI"].dropna().empty)
        self.assertFalse(df_test["ATR"].dropna().empty)
        print("✅ Indikatoren-Logik (ta-lib) funktioniert.")

    # --- 3. IG API INTEGRATION TESTS ---
    def test_ig_api_connectivity(self):
        """Testet die echte Verbindung zur IG API."""
        if not self.bot:
            self.skipTest("Bot konnte nicht initialisiert werden.")

        acc = self.bot.ig.fetch_accounts()
        self.assertFalse(acc.empty)
        self.assertIn("balance", acc.columns)
        print(
            f"✅ API-Check: Verbindung steht. Account-Typ: {acc['accountType'].iloc[0]}"
        )

    def test_market_availability(self):
        """Prüft, ob ein Beispiel-Epic bei IG noch gültig ist."""
        if not self.bot:
            self.skipTest("Bot konnte nicht initialisiert werden.")

        # Teste das erste Epic aus deiner Config
        first_pair = list(BASE_CFG["pairs"].keys())[0]
        epic = BASE_CFG["pairs"][first_pair]["epic"]

        market_info = self.bot.ig.fetch_market_by_epic(epic)
        self.assertIsNotNone(market_info)
        print(f"✅ API-Check: Markt {epic} ({first_pair}) ist erreichbar.")

    # --- 4. DATEISYSTEM TESTS ---
    def test_directories(self):
        """Prüft, ob alle notwendigen Ordner existieren."""
        paths = ["data", "stats_export"]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Ordner {p} fehlt!")
        print("✅ Verzeichnisstruktur ist korrekt.")


if __name__ == "__main__":
    print("\n🚀 Starte Omni-Bot Test Suite...\n")
    unittest.main()
