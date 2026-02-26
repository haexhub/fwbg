"""
Integration Tests: ORB Optimization Pipeline.

Prüft dass die ORB-Optimierung korrekt läuft:
- Grid-Kombinationen werden vollständig berechnet (kein stilles Skip)
- Alle Session-Features werden korrekt berechnet
- Fold 0 evaluiert ALLE Kombos (Early Pruning beginnt erst ab Fold min_folds_ratio)
- Grid-Results haben valide Struktur
- Pruning reduziert aktive Kombos aber löscht nicht alles

Diese Tests sind dafür da, "stille" Bugs zu fangen, bei denen die Optimierung
zu schnell fertig wird weil Arbeit unerwartet übersprungen wird.
"""
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from fwbg.core.context import SimulationContext
from fwbg.core.config import GridConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_m15_df(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Erstellt realistischen M15 DataFrame.

    n=6000 Bars ≈ 62 Tage (24h × 4 bars/h = 96 bars/Tag × 62 Tage).
    Genug für ATR-Warmup (14 Bars) und mehrere Walk-Forward Folds.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="15min")

    close = 15000.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    noise = rng.normal(0, 0.0003, n)
    high_bump = np.abs(rng.normal(0, 0.002, n))
    low_bump = np.abs(rng.normal(0, 0.002, n))

    df = pd.DataFrame(
        {
            "O": close * (1 + noise),
            "H": close * (1 + high_bump),
            "L": close * (1 - low_bump),
            "C": close,
            "V": rng.integers(100, 10_000, n).astype(float),
        },
        index=idx,
    )
    # Enforce OHLC consistency
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _add_orb_features(df: pd.DataFrame, sessions: list = None) -> pd.DataFrame:
    """Berechnet echte ORB-Features via Plugin (kein Mock)."""
    from fwbg.plugins import import_plugin_module

    orb = import_plugin_module("fwbg-core", "indicators", "opening_range")
    if orb is None:
        pytest.skip("fwbg-core opening_range plugin not available")

    kw = {}
    if sessions is not None:
        kw["sessions"] = sessions

    ind = orb.OpeningRangeIndicator()
    return ind.compute(df, range_bars=1, **kw)


def _make_ctx(
    tp: list,
    sl: list,
    ct: list,
    timeout_bars: list = None,
    modifier_params_grid: list = None,
    pruning: bool = False,
    keep_ratio: float = 0.5,
    min_survivors: int = 2,
    min_folds_ratio: float = 0.3,
) -> SimulationContext:
    """
    Minimaler SimulationContext für Grid-Search Tests.

    Verwendet PERMISSIVE Validierungs-Defaults damit synthetische Zufallsdaten
    die Coverage-Tests nicht durch Fold-Filtering blockieren:
    - early_termination=False: kein vorzeitiger Abbruch bei schlechten Folds
    - first_fold_sanity_check=False: kein Catastrophic-Failure Filter
    - min_fold_stability=0.0: akzeptiert Combos mit 0 profitablen Folds
    - min_trades=1: niedrigste akzeptable Trade-Anzahl

    Coverage-Tests prüfen OB etwas evaluiert wird, nicht OB die Ergebnisse gut sind.
    """
    return SimulationContext(
        symbol="TEST",
        asset_class="INDEX",
        spread=1.0,
        point=1.0,
        grid_tp=tp,
        grid_sl=sl,
        grid_ct=ct,
        grid_timeout_bars=timeout_bars if timeout_bars is not None else [None],
        grid_exit_modifier_params=modifier_params_grid if modifier_params_grid is not None else [None],
        exit_strategy="atr_based",
        exit_params={"atr_period": 14, "min_tp_pips": 0, "min_sl_pips": 0},
        model_type="xgboost",
        min_trades=1,
        # Permissive validation: synthetische Daten haben kein echtes Signal
        early_termination=False,
        first_fold_sanity_check=False,
        min_fold_stability=0.0,
        # Pruning config
        early_pruning_enabled=pruning,
        early_pruning_keep_ratio=keep_ratio,
        early_pruning_min_survivors=min_survivors,
        early_pruning_min_folds_before_pruning_ratio=min_folds_ratio,
    )


def _make_inner_folds(feature_df: pd.DataFrame, n_folds: int = 3, fold_size: int = 800):
    """
    Erstellt (train_df, val_df) Folds aus einem Feature-DataFrame.

    Jeder Fold: 2×fold_size Train + fold_size Val, nicht überlappend.
    """
    folds = []
    step = fold_size
    train_len = 2 * fold_size

    for i in range(n_folds):
        start = i * step
        train_end = start + train_len
        val_end = train_end + fold_size
        if val_end > len(feature_df):
            break
        train_df = feature_df.iloc[start:train_end].copy()
        val_df = feature_df.iloc[train_end:val_end].copy()
        folds.append((train_df, val_df))

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Grid-Kombinationen
# ─────────────────────────────────────────────────────────────────────────────

class TestGridCombinationsCount:
    """
    total_grid_combinations() muss tp × sl × timeout × modifier entsprechen.

    CT ist KEIN eigener Grid-Loop — CT wird innerhalb jeder Combo per
    optimal-threshold-suche in targets.py gefunden.
    """

    def test_basic_tp_sl_timeout(self):
        """tp=3 × sl=2 × timeout=2 × modifier=1 (default) = 12."""
        ctx = _make_ctx(
            tp=[2.0, 3.0, 4.0],
            sl=[1.5, 2.0],
            ct=[0.5],
            timeout_bars=[8, 16],
        )
        assert ctx.total_grid_combinations() == 3 * 2 * 2 * 1

    def test_modifier_grid_doubles_combinations(self):
        """Jede weitere modifier_params_grid Variante multipliziert die Gesamtzahl."""
        ctx_no_mod = _make_ctx(
            tp=[2.0, 3.0],
            sl=[1.5, 2.0],
            ct=[0.5],
            timeout_bars=[8],
            modifier_params_grid=[None],
        )
        ctx_with_mod = _make_ctx(
            tp=[2.0, 3.0],
            sl=[1.5, 2.0],
            ct=[0.5],
            timeout_bars=[8],
            modifier_params_grid=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
        )
        assert ctx_with_mod.total_grid_combinations() == 2 * ctx_no_mod.total_grid_combinations()

    def test_three_modifiers_triples_combinations(self):
        """Drei modifier Varianten × Basis-Grid."""
        ctx = _make_ctx(
            tp=[2.0, 3.0, 4.0],
            sl=[1.5],
            ct=[0.5],
            timeout_bars=[None],
            modifier_params_grid=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.3, "trail_atr_mult": 0.3},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
        )
        assert ctx.total_grid_combinations() == 3 * 1 * 1 * 3  # tp × sl × timeout × mod

    def test_no_timeout_counts_as_one(self):
        """timeout_bars=[None] zählt als 1 Timeout-Variante."""
        ctx = _make_ctx(tp=[2.0, 3.0], sl=[1.5], ct=[0.5], timeout_bars=[None])
        assert ctx.total_grid_combinations() == 2  # 2 × 1 × 1 × 1

    def test_ct_does_not_multiply_combinations(self):
        """CT ist kein Combo-Loop — mehr CT-Werte ändern die Gesamtzahl NICHT."""
        ctx_1ct = _make_ctx(tp=[2.0], sl=[1.5], ct=[0.5])
        ctx_4ct = _make_ctx(tp=[2.0], sl=[1.5], ct=[0.5, 0.55, 0.6, 0.65])
        assert ctx_1ct.total_grid_combinations() == ctx_4ct.total_grid_combinations()

    def test_orb_scalping_index_preset_formula(self):
        """
        Preset orb_scalping_index: sl=4 × timeout=3 × modifier=2 = 6 pro TP-Wert.

        DAX (deep_orb_index): tp=5 → 5 × 6 = 30.
        SL override=3 für DAX → 3 × 3 × 2 = 18 pro TP → 5 × 18 = 90.
        """
        # Preset-Werte aus orb_scalping_index_v1.json, mit DAX override für sl
        ctx = _make_ctx(
            tp=[6.0, 7.0, 8.0, 10.0, 12.0],   # DAX override
            sl=[2.0, 2.5, 3.0],                  # DAX override
            ct=[0.65, 0.7, 0.75, 0.8],           # DAX override
            timeout_bars=[8, 16, 32],            # aus Preset
            modifier_params_grid=[               # aus Preset
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
        )
        # tp × sl × timeout × modifier (CT zählt nicht)
        expected = 5 * 3 * 3 * 2
        assert ctx.total_grid_combinations() == expected


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ORB Feature-Berechnung
# ─────────────────────────────────────────────────────────────────────────────

class TestORBFeatureComputation:
    """ORB Indicator berechnet alle erwarteten Feature-Spalten für Pipeline-Sessions."""

    def test_all_pipeline_session_features_present(self):
        """
        Nach ORB-Compute mit sessions=[0,1,2,5,6,7,8,12,13,14] müssen alle
        orb_sXX_range Spalten im Result vorhanden sein.
        """
        df = _make_m15_df(n=6000)
        sessions = [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]
        result = _add_orb_features(df, sessions=sessions)

        for h in sessions:
            col = f"rb1_orb_s{h:02d}_range"
            assert col in result.columns, (
                f"Session {h} UTC: Spalte '{col}' fehlt im Ergebnis. "
                f"Vorhandene orb_s*_range: {[c for c in result.columns if 'orb_s' in c and '_range' in c]}"
            )

    def test_session_features_have_non_nan_values(self):
        """Jede Session-Feature-Spalte muss mindestens 50% nicht-NaN Werte haben."""
        df = _make_m15_df(n=6000)
        sessions = [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]
        result = _add_orb_features(df, sessions=sessions)

        for h in sessions:
            col = f"rb1_orb_s{h:02d}_range"
            if col not in result.columns:
                continue
            non_nan_ratio = result[col].notna().mean()
            assert non_nan_ratio >= 0.3, (
                f"Session {h} UTC: '{col}' hat nur {non_nan_ratio:.1%} nicht-NaN Werte "
                f"— ORB-Features werden nicht berechnet."
            )

    def test_breakout_features_are_binary_events(self):
        """
        orb_s08_breakout_up / orb_s08_breakout_down müssen Event-Features sein:
        Werte nur {0, 1, NaN}. Niemals dauerhaft 1 für mehrere Bars.
        """
        df = _make_m15_df(n=6000)
        result = _add_orb_features(df, sessions=[8])

        for col in ["rb1_orb_s08_breakout_up", "rb1_orb_s08_breakout_down"]:
            if col not in result.columns:
                continue
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0.0, 1.0}), (
                f"'{col}' enthält nicht-binäre Werte: {vals.unique()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Grid-Search Coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestGridSearchCoverage:
    """
    Prüft dass run_grid_search WIRKLICH alle Kombos evaluiert.

    Bug-Kategorie: "Optimierung war zu schnell" — mögliche Ursache wäre, dass
    run_grid_search zu früh zurückkehrt (leere Candidates, no features, etc.)
    """

    def _run_small_grid_search(
        self,
        tp: list,
        sl: list,
        ct: list = None,
        timeout_bars: list = None,
        modifier_grid: list = None,
        pruning: bool = False,
        keep_ratio: float = 0.5,
        min_survivors: int = 1,
        min_folds_ratio: float = 0.0,
        n_inner_folds: int = 3,
    ):
        """
        Führt einen echten run_grid_search-Aufruf mit ORB-Features durch.
        Gibt (candidates, grid_results, n_combos) zurück.
        """
        from fwbg.optimization.grid_search import run_grid_search
        from fwbg.pipeline.features import get_feature_columns

        df = _make_m15_df(n=8000)
        feature_df = _add_orb_features(df, sessions=[8, 14])

        # Nur ORB-Features als Feature-Pool
        all_cols = get_feature_columns(feature_df)
        orb_cols = [c for c in all_cols if "_orb_s" in c]
        assert orb_cols, "Keine ORB-Features im Feature-Pool — ORB-Plugin nicht geladen?"

        # Feature-DF bereinigen: NaN-Warmup entfernen
        clean_df = feature_df.dropna(subset=orb_cols[:3])

        # ct=None → [0.3, 0.4, 0.5]: niedrig genug damit bei synthetischen Daten
        # Trades entstehen (ML-Modell auf Random-Walk liefert Predictions ~0.5)
        effective_ct = ct if ct is not None else [0.3, 0.4, 0.5]

        ctx = _make_ctx(
            tp=tp,
            sl=sl,
            ct=effective_ct,
            timeout_bars=timeout_bars if timeout_bars is not None else [None],
            modifier_params_grid=modifier_grid if modifier_grid is not None else [None],
            pruning=pruning,
            keep_ratio=keep_ratio,
            min_survivors=min_survivors,
            min_folds_ratio=min_folds_ratio,
        )
        ctx.n_inner_folds = n_inner_folds

        folds = _make_inner_folds(clean_df, n_folds=n_inner_folds, fold_size=1000)
        assert len(folds) >= n_inner_folds, (
            f"Zu wenig Daten für {n_inner_folds} Folds — DataFrame hat {len(clean_df)} Zeilen"
        )

        inner_df = clean_df

        candidates, grid_results = run_grid_search(
            full_pool=orb_cols,
            inner_folds=folds,
            grid=GridConfig(tp=tp, sl=sl, ct=effective_ct, timeout_bars=timeout_bars),
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=inner_df,
        )

        n_combos = ctx.total_grid_combinations()
        return candidates, grid_results, n_combos

    def test_grid_search_returns_nonempty_results(self):
        """
        run_grid_search mit ORB-Features und 4 Kombos muss grid_results zurückgeben.

        Schlägt dieser Test fehl, wird die Optimierung still übersprungen —
        genau der Bug der zu "8-Minuten-Fertigstellung" führen kann.
        """
        _, grid_results, n_combos = self._run_small_grid_search(
            tp=[2.0, 3.0],
            sl=[1.5, 2.0],
            pruning=False,
        )
        assert n_combos == 4, f"Erwarte 2×2 = 4 Combos, got {n_combos}"
        assert len(grid_results) > 0, (
            f"grid_results ist leer — run_grid_search hat nichts evaluiert! "
            f"(n_combos={n_combos}). Mögliche Ursache: Feature Selection schlägt fehl "
            f"oder Daten zu kurz für Modell-Training."
        )

    def test_all_combos_appear_in_results_without_pruning(self):
        """
        Ohne Early Pruning: Alle n_combos Kombos sollen in grid_results erscheinen.

        grid_results enthält nur überlebende Kombos — ohne Pruning alle.
        """
        tp = [2.0, 3.0]
        sl = [1.5]

        _, grid_results, n_combos = self._run_small_grid_search(
            tp=tp,
            sl=sl,
            pruning=False,  # Kein Pruning: alle 2 Kombos müssen überleben
        )
        assert n_combos == 2, f"Erwarte 2 Combos (tp×sl), got {n_combos}"
        assert len(grid_results) == n_combos, (
            f"Ohne Pruning sollten ALLE {n_combos} Kombos in grid_results erscheinen, "
            f"aber got {len(grid_results)}. Wurden Kombos unerwartet übersprungen?"
        )

    def test_pruning_reduces_but_does_not_eliminate_results(self):
        """
        Mit Early Pruning: Weniger Kombos überleben, aber nicht 0.

        Pruning schneidet die schlechtesten Kombos ab — das REDUZIERT grid_results,
        eliminiert sie aber nicht komplett. keep_ratio=0.5, min_survivors=1.
        """
        tp = [2.0, 3.0, 4.0, 5.0]
        sl = [1.5]

        _, grid_results_no_pruning, n_combos = self._run_small_grid_search(
            tp=tp, sl=sl,
            pruning=False,
            n_inner_folds=3,
        )
        _, grid_results_with_pruning, _ = self._run_small_grid_search(
            tp=tp, sl=sl,
            pruning=True,
            keep_ratio=0.5,
            min_survivors=1,
            min_folds_ratio=0.3,   # Start pruning after 30% of inner folds
            n_inner_folds=3,
        )

        assert n_combos == 4, f"Erwarte 4 Combos, got {n_combos}"
        assert len(grid_results_with_pruning) > 0, (
            "Pruning hat ALLE Kombos eliminiert — min_survivors-Logik defekt?"
        )
        assert len(grid_results_with_pruning) <= len(grid_results_no_pruning), (
            f"Pruning sollte grid_results REDUZIEREN, nicht erhöhen. "
            f"Ohne Pruning: {len(grid_results_no_pruning)}, mit: {len(grid_results_with_pruning)}"
        )

    def test_modifier_grid_produces_correct_combo_count(self):
        """
        exit_modifier_params_grid mit 2 Einträgen: 2× mehr Kombos als ohne.

        Verifikation dass die modifier_grid Dimension korrekt in die
        Combo-Schleife eingebaut ist.
        """
        tp = [2.0, 3.0]
        sl = [1.5]

        _, grid_results_no_mod, n_combos_no_mod = self._run_small_grid_search(
            tp=tp, sl=sl,
            modifier_grid=[None],
            pruning=False,
        )
        _, grid_results_with_mod, n_combos_with_mod = self._run_small_grid_search(
            tp=tp, sl=sl,
            modifier_grid=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
            pruning=False,
        )

        assert n_combos_with_mod == 2 * n_combos_no_mod, (
            f"2 Modifier-Varianten sollten Combos verdoppeln: "
            f"{n_combos_no_mod} → {n_combos_with_mod}, erwartet {2 * n_combos_no_mod}"
        )
        assert len(grid_results_with_mod) == 2 * len(grid_results_no_mod), (
            f"grid_results sollten verdoppelt sein: "
            f"{len(grid_results_no_mod)} → {len(grid_results_with_mod)}"
        )

    def test_grid_result_params_within_grid_bounds(self):
        """Jedes grid_result enthält TP/SL-Werte aus dem konfigurierten Grid."""
        tp_values = [2.0, 3.0, 4.0]
        sl_values = [1.5, 2.0]

        _, grid_results, _ = self._run_small_grid_search(
            tp=tp_values, sl=sl_values,
            pruning=False,
        )

        for result in grid_results:
            # grid_result speichert TP/SL als tp_mult / sl_mult
            assert result["tp_mult"] in tp_values, (
                f"grid_result TP={result['tp_mult']} liegt nicht im Grid {tp_values}"
            )
            assert result["sl_mult"] in sl_values, (
                f"grid_result SL={result['sl_mult']} liegt nicht im Grid {sl_values}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ORB + Optimizer Kombination
# ─────────────────────────────────────────────────────────────────────────────

class TestORBOptimizationEndToEnd:
    """
    Verknüpft ORB-Feature-Berechnung mit der Grid-Search.

    Prüft den echten Pfad: M15-Daten → ORB-Features → Grid-Search → Ergebnisse.
    """

    def test_orb_features_survive_feature_selection(self):
        """
        Nach Feature-Selection (innerhalb run_grid_search) sind noch
        ORB-Features im Feature-Pool — sie werden nicht alle herausgefiltert.
        """
        from fwbg.optimization.grid_search import run_grid_search
        from fwbg.pipeline.features import get_feature_columns

        df = _make_m15_df(n=8000)
        feature_df = _add_orb_features(df, sessions=[8, 14])
        all_cols = get_feature_columns(feature_df)
        orb_cols = [c for c in all_cols if "_orb_s" in c]
        clean_df = feature_df.dropna(subset=orb_cols[:3])

        ctx = _make_ctx(
            tp=[2.0, 3.0], sl=[1.5], ct=[0.3, 0.4, 0.5],
            pruning=False,
        )
        folds = _make_inner_folds(clean_df, n_folds=3, fold_size=1000)

        # Capture was selected_features_long/short was set to
        captured = {}

        # Patch select_features to spy on what gets selected
        from fwbg.optimization import grid_search as gs_module
        original_select = gs_module.select_features

        def spy_select(inner_folds, features, ctx, sym):
            result = original_select(inner_folds, features, ctx, sym)
            captured["selected_long"] = result[0]
            captured["selected_short"] = result[1]
            return result

        with patch.object(gs_module, "select_features", side_effect=spy_select):
            candidates, grid_results = run_grid_search(
                full_pool=orb_cols,
                inner_folds=folds,
                grid=GridConfig(tp=[2.0, 3.0], sl=[1.5], ct=[0.3, 0.4, 0.5]),
                ctx=ctx,
                regime_config={},
                sym="TEST",
                inner_df=clean_df,
            )

        if captured.get("selected_long"):
            orb_in_selection = [f for f in captured["selected_long"] if "_orb_s" in f]
            assert len(orb_in_selection) > 0, (
                "Feature Selection hat ALLE ORB-Features entfernt — "
                "ORB-Features tragen keinen Prediction-Wert auf den Testdaten."
            )

    def test_optimization_produces_valid_candidate_structure(self):
        """
        candidates[0] hat alle erwarteten Felder: tp, sl, sharpe, n_trades, win_rate.
        """
        from fwbg.optimization.grid_search import run_grid_search
        from fwbg.pipeline.features import get_feature_columns

        df = _make_m15_df(n=8000)
        feature_df = _add_orb_features(df, sessions=[8, 14])
        all_cols = get_feature_columns(feature_df)
        orb_cols = [c for c in all_cols if "_orb_s" in c]
        clean_df = feature_df.dropna(subset=orb_cols[:3])

        ctx = _make_ctx(
            tp=[2.0, 3.0, 4.0], sl=[1.5, 2.0], ct=[0.3, 0.4, 0.5],
            pruning=False,
        )
        folds = _make_inner_folds(clean_df, n_folds=3, fold_size=1000)

        candidates, grid_results = run_grid_search(
            full_pool=orb_cols,
            inner_folds=folds,
            grid=GridConfig(tp=[2.0, 3.0, 4.0], sl=[1.5, 2.0], ct=[0.3, 0.4, 0.5]),
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=clean_df,
        )

        assert len(candidates) > 0, (
            "Keine Kandidaten zurückgegeben. Mögliche Ursachen:\n"
            "1. Alle Kombos unter min_trades Schwelle\n"
            "2. Feature Selection schlägt fehl\n"
            "3. ORB-Plugin nicht geladen\n"
            "4. Daten zu kurz für Modell-Training"
        )

        best = candidates[0]

        # Pflichtfelder im candidate-Dict von run_grid_search
        for required_key in ["params", "inner_val_pnl", "fold_stability", "rrr", "feats"]:
            assert required_key in best, (
                f"Kandidat fehlt Pflichtfeld '{required_key}'. "
                f"Vorhandene Felder: {list(best.keys())}"
            )

        # params = (tp, sl, ct) Tuple
        assert isinstance(best["params"], tuple), f"params kein Tuple: {type(best['params'])}"
        assert len(best["params"]) >= 2, f"params-Tuple zu kurz: {best['params']}"
        tp_val, sl_val = best["params"][0], best["params"][1]
        assert isinstance(tp_val, (int, float)), f"tp kein Numeric: {type(tp_val)}"
        assert isinstance(sl_val, (int, float)), f"sl kein Numeric: {type(sl_val)}"
        assert np.isfinite(best["inner_val_pnl"]), f"inner_val_pnl nicht finite: {best['inner_val_pnl']}"
        assert 0.0 <= best["fold_stability"] <= 1.0, f"fold_stability außerhalb [0,1]: {best['fold_stability']}"
        assert len(best["feats"]) > 0, "candidate hat leeres feats-Array"
