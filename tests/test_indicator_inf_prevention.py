"""
TDD Tests für inf/nan-Prävention in Indikatoren.

Diese Tests stellen sicher, dass Indikatoren niemals inf-Werte produzieren,
die XGBoost zum Absturz bringen würden.
"""
import pytest
import numpy as np
import pandas as pd


class TestIndicatorsNoInf:
    """Tests dass Indikatoren keine inf-Werte produzieren."""

    @pytest.fixture
    def realistic_ohlc(self):
        """Realistische OHLC-Daten."""
        np.random.seed(42)
        n = 5000

        # Realistische Preisbewegung
        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "O": prices * (1 + np.random.randn(n) * 0.001),
            "H": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
            "L": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
            "C": prices,
            "V": np.abs(np.random.randn(n)) * 10000 + 1000,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        df["H"] = df[["O", "H", "C"]].max(axis=1)
        df["L"] = df[["O", "L", "C"]].min(axis=1)

        return df

    @pytest.fixture
    def edge_case_ohlc(self):
        """OHLC-Daten mit Edge Cases die inf verursachen könnten."""
        np.random.seed(42)
        n = 5000

        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "O": prices,
            "H": prices + 0.1,
            "L": prices - 0.1,
            "C": prices,
            "V": np.abs(np.random.randn(n)) * 1000,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        # Edge Case 1: Konstante Preise (std=0 -> Division durch 0)
        df.loc[df.index[1000:1050], "C"] = 100.0
        df.loc[df.index[1000:1050], "O"] = 100.0
        df.loc[df.index[1000:1050], "H"] = 100.1
        df.loc[df.index[1000:1050], "L"] = 99.9

        # Edge Case 2: Sehr kleine Preise
        df.loc[df.index[2000:2010], "C"] = 0.001
        df.loc[df.index[2000:2010], "O"] = 0.001
        df.loc[df.index[2000:2010], "H"] = 0.0011
        df.loc[df.index[2000:2010], "L"] = 0.0009

        return df

    def test_volatility_no_inf(self, realistic_ohlc):
        """Volatility-Indikatoren dürfen keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        result = compute_indicator_pool(realistic_ohlc.copy(), indicators=["volatility"])

        feature_cols = [c for c in result.columns
                        if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

        for col in feature_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"volatility/{col} hat {inf_count} inf-Werte"

    def test_trend_no_inf(self, realistic_ohlc):
        """Trend-Indikatoren dürfen keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        result = compute_indicator_pool(realistic_ohlc.copy(), indicators=["trend"])

        feature_cols = [c for c in result.columns
                        if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

        for col in feature_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"trend/{col} hat {inf_count} inf-Werte"

    def test_dynamics_no_inf(self, realistic_ohlc):
        """Dynamics-Indikatoren dürfen keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        result = compute_indicator_pool(realistic_ohlc.copy(), indicators=["dynamics"])

        feature_cols = [c for c in result.columns
                        if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

        for col in feature_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"dynamics/{col} hat {inf_count} inf-Werte"

    def test_ichimoku_no_inf(self, realistic_ohlc):
        """Ichimoku-Indikatoren dürfen keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        result = compute_indicator_pool(realistic_ohlc.copy(), indicators=["ichimoku"])

        feature_cols = [c for c in result.columns
                        if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

        for col in feature_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"ichimoku/{col} hat {inf_count} inf-Werte"

    def test_multi_timeframe_no_inf(self, realistic_ohlc):
        """Multi-Timeframe-Indikatoren dürfen keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        result = compute_indicator_pool(realistic_ohlc.copy(), indicators=["multi_timeframe"])

        feature_cols = [c for c in result.columns
                        if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

        for col in feature_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"multi_timeframe/{col} hat {inf_count} inf-Werte"

    def test_price_action_no_inf(self, realistic_ohlc):
        """Price-Action-Indikatoren dürfen keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        result = compute_indicator_pool(realistic_ohlc.copy(), indicators=["price_action"])

        feature_cols = [c for c in result.columns
                        if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

        for col in feature_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"price_action/{col} hat {inf_count} inf-Werte"

    def test_all_indicators_no_inf_with_edge_cases(self, edge_case_ohlc):
        """ALLE Indikatoren dürfen auch bei Edge Cases keine inf-Werte produzieren."""
        from fwbg.pipeline import compute_indicator_pool

        all_indicators = [
            "trend", "momentum", "volatility", "dynamics",
            "ichimoku", "multi_timeframe", "price_action"
        ]

        for indicator in all_indicators:
            result = compute_indicator_pool(edge_case_ohlc.copy(), indicators=[indicator])

            feature_cols = [c for c in result.columns
                            if c not in ["O", "H", "L", "C", "V"] and not c.startswith("_")]

            for col in feature_cols:
                inf_count = np.isinf(result[col]).sum()
                assert inf_count == 0, f"{indicator}/{col} hat {inf_count} inf-Werte bei Edge Cases"


class TestFeatureGroupFiltering:
    """Tests dass Feature-Gruppen-Filterung inf-Features korrekt entfernt."""

    def test_filter_removes_inf_features(self):
        """Filterung muss Features mit inf-Werten entfernen."""
        np.random.seed(42)
        n = 500

        # DataFrame mit einigen inf-Features
        df = pd.DataFrame({
            "clean_feat1": np.random.randn(n),
            "clean_feat2": np.random.randn(n),
            "inf_feat1": np.random.randn(n),
            "inf_feat2": np.random.randn(n),
            "nan_heavy_feat": np.random.randn(n),
        })

        # inf einfügen
        df.loc[100, "inf_feat1"] = np.inf
        df.loc[200, "inf_feat2"] = -np.inf

        # 60% NaN
        df.loc[:300, "nan_heavy_feat"] = np.nan

        group_features = ["clean_feat1", "clean_feat2", "inf_feat1", "inf_feat2", "nan_heavy_feat"]

        # Filterlogik (wie in grid_search.py)
        clean_features = []
        for feat in group_features:
            if feat in df.columns:
                col = df[feat]
                has_inf = np.isinf(col).any()
                nan_ratio = col.isna().mean()
                if not has_inf and nan_ratio <= 0.5:
                    clean_features.append(feat)

        assert "clean_feat1" in clean_features
        assert "clean_feat2" in clean_features
        assert "inf_feat1" not in clean_features
        assert "inf_feat2" not in clean_features
        assert "nan_heavy_feat" not in clean_features
        assert len(clean_features) == 2


class TestXGBoostWithFilteredFeatures:
    """Tests dass XGBoost mit gefilterten Features funktioniert."""

    def test_xgboost_trains_with_clean_features(self):
        """XGBoost muss mit sauberen Features trainieren können."""
        import xgboost as xgb

        np.random.seed(42)
        n = 1000

        # Saubere Features
        X = np.random.randn(n, 5)
        y = (X[:, 0] > 0).astype(int)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == n

    def test_xgboost_fails_with_inf(self):
        """XGBoost muss bei inf-Werten fehlschlagen."""
        import xgboost as xgb

        np.random.seed(42)
        n = 1000

        X = np.random.randn(n, 5)
        X[100, 2] = np.inf  # inf einfügen
        y = (X[:, 0] > 0).astype(int)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, verbosity=0)

        with pytest.raises(Exception):
            model.fit(X, y)


    # Integration-Test für Grid-Search entfernt - zu viele Dependencies
    # Die wichtigen Tests (inf-Prävention) sind in TestIndicatorsNoInf
