"""Tests for Plateau Feature Selection plugin."""
import pytest

from fwbg.plugins import import_plugin_module

_plateau = import_plugin_module("fwbg-premium", "feature_selection", "plateau")
if _plateau is None:
    pytest.skip("fwbg-premium plateau plugin not available", allow_module_level=True)

find_feature_neighbors = _plateau.find_feature_neighbors
calculate_feature_plateau_score = _plateau.calculate_feature_plateau_score
calculate_param_plateau_score = _plateau.calculate_param_plateau_score
select_best_plateau_candidate = _plateau.select_best_plateau_candidate
select_plateau_features = _plateau.select_plateau_features


# --- Feature Neighbor Tests ---


class TestFindFeatureNeighbors:
    """Tests für die Nachbar-Erkennung."""

    def test_finds_numeric_suffix_neighbors(self):
        """Findet Nachbarn mit numerischem Suffix."""
        all_features = ["rsi_10", "rsi_12", "rsi_14", "rsi_16", "rsi_20"]

        neighbors = find_feature_neighbors("rsi_14", all_features)

        # Sollte nahe Werte finden
        assert "rsi_12" in neighbors or "rsi_16" in neighbors

    def test_finds_hour_suffix_neighbors(self):
        """Findet Nachbarn mit Stunden-Suffix."""
        all_features = ["macro_vix_chg_12h", "macro_vix_chg_24h", "macro_vix_chg_48h"]

        neighbors = find_feature_neighbors("macro_vix_chg_24h", all_features)

        assert "macro_vix_chg_12h" in neighbors or "macro_vix_chg_48h" in neighbors

    def test_finds_day_suffix_neighbors(self):
        """Findet Nachbarn mit Tages-Suffix."""
        all_features = ["returns_1d", "returns_3d", "returns_5d", "returns_7d"]

        neighbors = find_feature_neighbors("returns_5d", all_features)

        assert "returns_3d" in neighbors or "returns_7d" in neighbors

    def test_no_neighbors_for_unique_feature(self):
        """Keine Nachbarn für einzigartige Features."""
        all_features = ["special_feature", "rsi_14", "atr_10"]

        neighbors = find_feature_neighbors("special_feature", all_features)

        assert len(neighbors) == 0

    def test_does_not_include_self(self):
        """Feature ist nicht sein eigener Nachbar."""
        all_features = ["rsi_14", "rsi_16"]

        neighbors = find_feature_neighbors("rsi_14", all_features)

        assert "rsi_14" not in neighbors


# --- Plateau Score Calculation Tests ---


class TestPlateauScoreCalculation:
    """Tests für Plateau-Score Berechnung."""

    def test_calculates_scores_for_features(self):
        """Berechnet Scores für alle Features."""
        importances = {
            "rsi_14": 0.15,
            "rsi_12": 0.14,
            "rsi_16": 0.13,
            "noise": 0.01,
        }
        all_features = list(importances.keys())

        results = calculate_feature_plateau_score(importances, all_features)

        assert "rsi_14" in results
        assert "plateau_score" in results["rsi_14"]
        assert "stability" in results["rsi_14"]

    def test_stable_features_get_higher_plateau_score(self):
        """Features mit ähnlichen Nachbarn bekommen höheren Plateau-Score."""
        # Stabile Gruppe: alle ähnlich
        stable_importances = {
            "stable_10": 0.10,
            "stable_12": 0.10,
            "stable_14": 0.10,
            "stable_16": 0.10,
        }
        # Instabile Gruppe: große Variation
        unstable_importances = {
            "unstable_10": 0.05,
            "unstable_12": 0.15,
            "unstable_14": 0.10,
            "unstable_16": 0.02,
        }

        all_features = list(stable_importances.keys()) + list(unstable_importances.keys())
        all_importances = {**stable_importances, **unstable_importances}

        results = calculate_feature_plateau_score(all_importances, all_features)

        # Stabile Features sollten höhere Stability haben
        stable_stability = results["stable_14"]["stability"]
        unstable_stability = results["unstable_14"]["stability"]

        assert stable_stability >= unstable_stability

    def test_isolated_features_get_penalty(self):
        """Isolierte Features (ohne Nachbarn) bekommen Penalty."""
        importances = {
            "rsi_14": 0.10,
            "rsi_16": 0.10,
            "isolated": 0.10,  # Keine Nachbarn
        }
        all_features = list(importances.keys())

        results = calculate_feature_plateau_score(importances, all_features)

        # Isolierte Features bekommen niedrigeren Score trotz gleicher Importance
        # (wegen 0.8 Multiplikator)
        assert results["isolated"]["plateau_score"] < results["isolated"]["importance"]


class TestSelectPlateauFeatures:
    """Tests für select_plateau_features Funktion."""

    def test_selects_features(self):
        """Sollte Features nach Plateau-Score auswählen."""
        importances = {
            "rsi_10": 0.15,
            "rsi_12": 0.14,
            "rsi_14": 0.16,
            "rsi_16": 0.13,
            "noise_1": 0.02,
            "noise_2": 0.01,
        }
        all_features = list(importances.keys())

        selected = select_plateau_features(
            importances, all_features, top_n=3
        )

        assert len(selected) == 3
        # RSI Features sollten bevorzugt werden (haben Nachbarn)
        rsi_selected = [f for f in selected if f.startswith("rsi_")]
        assert len(rsi_selected) >= 2

    def test_respects_top_n(self):
        """Sollte top_n Limit einhalten."""
        importances = {"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4}
        all_features = list(importances.keys())

        selected = select_plateau_features(
            importances, all_features, top_n=2
        )

        assert len(selected) == 2

    def test_handles_empty_importances(self):
        """Sollte leeres Dict behandeln."""
        selected = select_plateau_features({}, [], top_n=5)
        assert selected == []


# --- Parameter Plateau Tests ---


class TestParamPlateauScore:
    """Tests für Grid-Search Parameter Plateau-Berechnung."""

    @pytest.fixture
    def sample_candidates(self):
        """Grid-Search Kandidaten mit verschiedenen Scores."""
        return [
            {"params": (10, 15, 0.55), "score": 0.58},
            {"params": (10, 15, 0.60), "score": 0.59},
            {"params": (10, 20, 0.55), "score": 0.57},
            {"params": (15, 15, 0.55), "score": 0.56},
            {"params": (15, 15, 0.60), "score": 0.61},  # Bester Score
            {"params": (15, 20, 0.55), "score": 0.55},
            {"params": (15, 20, 0.60), "score": 0.58},
            {"params": (20, 15, 0.55), "score": 0.54},
            {"params": (20, 15, 0.60), "score": 0.53},
        ]

    def test_adds_plateau_scores(self, sample_candidates):
        """Fügt Plateau-Scores zu Kandidaten hinzu."""
        grid_tp = [10, 15, 20]
        grid_sl = [15, 20]
        grid_ct = [0.55, 0.60]

        result = calculate_param_plateau_score(
            sample_candidates, grid_tp, grid_sl, grid_ct
        )

        for c in result:
            assert "plateau_score" in c
            assert "stability_score" in c
            assert "neighbor_count" in c

    def test_central_candidates_have_more_neighbors(self, sample_candidates):
        """Zentrale Grid-Positionen haben mehr Nachbarn."""
        grid_tp = [10, 15, 20]
        grid_sl = [15, 20]
        grid_ct = [0.55, 0.60]

        result = calculate_param_plateau_score(
            sample_candidates, grid_tp, grid_sl, grid_ct
        )

        # Finde Kandidat in der Mitte (15, 15, 0.55)
        central = next(c for c in result if c["params"] == (15, 15, 0.55))
        # Finde Kandidat am Rand (10, 15, 0.55)
        edge = next(c for c in result if c["params"] == (10, 15, 0.55))

        assert central["neighbor_count"] >= edge["neighbor_count"]

    def test_empty_candidates(self):
        """Behandelt leere Kandidatenliste."""
        result = calculate_param_plateau_score([], [10, 15], [15, 20], [0.55, 0.60])
        assert result == []


class TestSelectBestPlateauCandidate:
    """Tests für die Auswahl des besten Plateau-Kandidaten."""

    def test_selects_best_plateau_score(self):
        """Wählt Kandidat mit bestem Plateau-Score."""
        candidates = [
            {"params": (10, 15, 0.55), "score": 0.58},
            {"params": (10, 15, 0.60), "score": 0.59},
            {"params": (10, 20, 0.55), "score": 0.57},
            {"params": (15, 15, 0.55), "score": 0.56},
            {"params": (15, 15, 0.60), "score": 0.59},
            {"params": (15, 20, 0.55), "score": 0.58},
        ]
        grid_tp = [10, 15]
        grid_sl = [15, 20]
        grid_ct = [0.55, 0.60]

        best = select_best_plateau_candidate(
            candidates, grid_tp, grid_sl, grid_ct, min_neighbors=1
        )

        assert best is not None
        assert "plateau_score" in best

    def test_returns_none_for_empty_list(self):
        """Gibt None für leere Liste zurück."""
        best = select_best_plateau_candidate([], [10], [15], [0.55])
        assert best is None

    def test_fallback_to_best_score(self):
        """Fallback auf besten Score wenn keine Nachbarn."""
        # Nur ein Kandidat - keine Nachbarn möglich
        candidates = [
            {"params": (10, 15, 0.55), "score": 0.58},
        ]
        grid_tp = [10]
        grid_sl = [15]
        grid_ct = [0.55]

        best = select_best_plateau_candidate(
            candidates, grid_tp, grid_sl, grid_ct, min_neighbors=2
        )

        # Sollte den einzigen Kandidaten als Fallback wählen
        assert best is not None
        assert best["score"] == 0.58
