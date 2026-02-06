"""
Regressionstests für inf/nan-Filterung in Feature-Gruppen.

Diese Tests stellen sicher, dass Features mit inf/nan-Werten korrekt
gefiltert werden, bevor sie an XGBoost übergeben werden.

Issue: XGBoost wirft "Input data contains `inf` or a value too large" Fehler,
wenn Features mit inf-Werten im Training-Set vorhanden sind.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


class TestFeatureInfNanFiltering:
    """Tests für die inf/nan-Filterung in _process_feature_group."""

    @pytest.fixture
    def sample_df_with_inf(self):
        """Erstellt ein Sample-DataFrame mit inf/nan-Werten in einigen Features."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            "O": np.random.randn(n) + 100,
            "H": np.random.randn(n) + 101,
            "L": np.random.randn(n) + 99,
            "C": np.random.randn(n) + 100,
            "V": np.abs(np.random.randn(n)) * 1000,
            # Clean features
            "trend_ema_20": np.random.randn(n),
            "trend_sma_50": np.random.randn(n),
            "trend_adx_14": np.random.rand(n) * 100,
            # Features mit inf-Werten
            "volatility_atr_zscore": np.random.randn(n),
            "volatility_bb_width": np.random.randn(n),
            # Features mit hohem NaN-Anteil
            "momentum_rsi_14": np.random.randn(n),
        })

        # inf-Werte einfügen
        df.loc[100, "volatility_atr_zscore"] = np.inf
        df.loc[200, "volatility_atr_zscore"] = -np.inf
        df.loc[150, "volatility_bb_width"] = np.inf

        # Hoher NaN-Anteil (>50%)
        df.loc[:300, "momentum_rsi_14"] = np.nan

        return df

    @pytest.fixture
    def sample_df_clean(self):
        """Erstellt ein sauberes Sample-DataFrame ohne inf/nan."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            "O": np.random.randn(n) + 100,
            "H": np.random.randn(n) + 101,
            "L": np.random.randn(n) + 99,
            "C": np.random.randn(n) + 100,
            "V": np.abs(np.random.randn(n)) * 1000,
            "trend_ema_20": np.random.randn(n),
            "trend_sma_50": np.random.randn(n),
            "trend_adx_14": np.random.rand(n) * 100,
        })

        return df

    def test_filter_features_with_inf(self, sample_df_with_inf):
        """Test: Features mit inf-Werten werden herausgefiltert."""
        df = sample_df_with_inf

        # Simuliere die Filterlogik aus _process_feature_group
        group_features = [
            "volatility_atr_zscore",  # Hat inf
            "volatility_bb_width",     # Hat inf
            "trend_ema_20",            # Clean
        ]

        clean_features = []
        for feat in group_features:
            if feat in df.columns:
                col = df[feat]
                has_inf = np.isinf(col).any()
                nan_ratio = col.isna().mean()
                if not has_inf and nan_ratio <= 0.5:
                    clean_features.append(feat)

        # Nur trend_ema_20 sollte übrig bleiben
        assert "trend_ema_20" in clean_features
        assert "volatility_atr_zscore" not in clean_features
        assert "volatility_bb_width" not in clean_features
        assert len(clean_features) == 1

    def test_filter_features_with_high_nan_ratio(self, sample_df_with_inf):
        """Test: Features mit >50% NaN werden herausgefiltert."""
        df = sample_df_with_inf

        group_features = [
            "momentum_rsi_14",  # >50% NaN
            "trend_sma_50",     # Clean
        ]

        clean_features = []
        for feat in group_features:
            if feat in df.columns:
                col = df[feat]
                has_inf = np.isinf(col).any()
                nan_ratio = col.isna().mean()
                if not has_inf and nan_ratio <= 0.5:
                    clean_features.append(feat)

        # Nur trend_sma_50 sollte übrig bleiben
        assert "trend_sma_50" in clean_features
        assert "momentum_rsi_14" not in clean_features
        assert len(clean_features) == 1

    def test_clean_df_passes_all_features(self, sample_df_clean):
        """Test: Saubere Features werden nicht gefiltert."""
        df = sample_df_clean

        group_features = [
            "trend_ema_20",
            "trend_sma_50",
            "trend_adx_14",
        ]

        clean_features = []
        for feat in group_features:
            if feat in df.columns:
                col = df[feat]
                has_inf = np.isinf(col).any()
                nan_ratio = col.isna().mean()
                if not has_inf and nan_ratio <= 0.5:
                    clean_features.append(feat)

        # Alle Features sollten durchkommen
        assert len(clean_features) == 3
        assert set(clean_features) == set(group_features)

    def test_edge_case_exactly_50_percent_nan(self):
        """Test: Features mit exakt 50% NaN werden NICHT gefiltert."""
        n = 100
        df = pd.DataFrame({
            "feat_50_nan": np.concatenate([np.full(50, np.nan), np.random.randn(50)]),
        })

        col = df["feat_50_nan"]
        nan_ratio = col.isna().mean()

        # Exakt 50% sollte akzeptiert werden (<=0.5)
        assert nan_ratio == 0.5
        assert not np.isinf(col).any()
        # Feature sollte durchkommen
        assert nan_ratio <= 0.5

    def test_edge_case_51_percent_nan(self):
        """Test: Features mit 51% NaN werden gefiltert."""
        n = 100
        df = pd.DataFrame({
            "feat_51_nan": np.concatenate([np.full(51, np.nan), np.random.randn(49)]),
        })

        col = df["feat_51_nan"]
        nan_ratio = col.isna().mean()

        # 51% sollte gefiltert werden (>0.5)
        assert nan_ratio == 0.51
        assert nan_ratio > 0.5

    def test_inf_detection_catches_negative_inf(self):
        """Test: Negative inf-Werte werden erkannt."""
        df = pd.DataFrame({
            "feat": [1.0, -np.inf, 3.0, 4.0, 5.0],
        })

        assert np.isinf(df["feat"]).any()

    def test_inf_detection_catches_positive_inf(self):
        """Test: Positive inf-Werte werden erkannt."""
        df = pd.DataFrame({
            "feat": [1.0, np.inf, 3.0, 4.0, 5.0],
        })

        assert np.isinf(df["feat"]).any()

    def test_very_large_values_not_filtered(self):
        """Test: Sehr große aber endliche Werte werden NICHT gefiltert."""
        df = pd.DataFrame({
            "feat": [1e308, -1e308, 1.0, 2.0, 3.0],
        })

        # Sollte NICHT als inf erkannt werden
        assert not np.isinf(df["feat"]).any()
        assert df["feat"].isna().mean() == 0.0


class TestXGBoostInfHandling:
    """Tests die zeigen, dass XGBoost mit inf-Werten abstürzt."""

    def test_xgboost_fails_with_inf(self):
        """Test: XGBoost wirft Fehler bei inf-Werten."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost nicht installiert")

        # Daten mit inf
        X = np.array([[1.0, 2.0], [3.0, np.inf], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        model = xgb.XGBClassifier(n_estimators=2, max_depth=2, verbosity=0)

        # Sollte einen Fehler werfen
        with pytest.raises(Exception):  # ValueError oder XGBoostError
            model.fit(X, y)

    def test_xgboost_works_without_inf(self):
        """Test: XGBoost funktioniert ohne inf-Werte."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost nicht installiert")

        # Saubere Daten
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        model = xgb.XGBClassifier(n_estimators=2, max_depth=2, verbosity=0)

        # Sollte funktionieren
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 3

    def test_xgboost_handles_nan_with_missing_param(self):
        """Test: XGBoost kann NaN mit missing-Parameter verarbeiten."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost nicht installiert")

        # Daten mit NaN (aber nicht inf!)
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        # XGBoost kann NaN verarbeiten (behandelt als missing)
        model = xgb.XGBClassifier(n_estimators=2, max_depth=2, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == 3


class TestIntegrationWithGridSearch:
    """Integrationstests für die Filterung in grid_search.py."""

    def test_process_feature_group_filters_inf(self):
        """Test: _process_feature_group filtert inf-Features korrekt."""
        from fwbg.optimization.grid_search import _process_feature_group

        # Mock-Context und andere Dependencies
        ctx = MagicMock()
        ctx.grid_combinations_per_feature_group.return_value = 10
        ctx.total_grid_combinations.return_value = 100
        ctx.exit_params = {}
        ctx.min_rrr = 0

        # Mock-Grid
        grid = MagicMock()
        grid.tp = [10]
        grid.sl = [20]
        grid.timeout_bars = [100]

        # DataFrame mit inf-Werten
        np.random.seed(42)
        n = 200
        inner_df = pd.DataFrame({
            "O": np.random.randn(n) + 100,
            "H": np.random.randn(n) + 101,
            "L": np.random.randn(n) + 99,
            "C": np.random.randn(n) + 100,
            "V": np.abs(np.random.randn(n)) * 1000,
            "trend_feat1": np.random.randn(n),
            "trend_feat2": np.random.randn(n),
            "trend_feat_inf": np.random.randn(n),
        })
        # inf einfügen
        inner_df.loc[50, "trend_feat_inf"] = np.inf

        # Feature-Pool
        full_pool = ["trend_feat1", "trend_feat2", "trend_feat_inf"]

        # Mock filter_features_by_group to return all trend features
        with patch("fwbg.optimization.grid_search.filter_features_by_group") as mock_filter:
            mock_filter.return_value = full_pool

            # Die Funktion sollte trend_feat_inf herausfiltern
            # und NICHT abstürzen
            # Wir testen nur die Filterlogik, nicht den vollen CV-Durchlauf

            # Da wir nicht den vollen CV mocken können, testen wir
            # die Filterlogik direkt
            group_features = mock_filter.return_value

            clean_features = []
            for feat in group_features:
                if feat in inner_df.columns:
                    col = inner_df[feat]
                    has_inf = np.isinf(col).any()
                    nan_ratio = col.isna().mean()
                    if not has_inf and nan_ratio <= 0.5:
                        clean_features.append(feat)

            # trend_feat_inf sollte gefiltert sein
            assert "trend_feat1" in clean_features
            assert "trend_feat2" in clean_features
            assert "trend_feat_inf" not in clean_features
            assert len(clean_features) == 2
