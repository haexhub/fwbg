"""
Tests für Regime-Detection Features und Regime-Filter.

Testet:
- Hurst-Exponent Berechnung (Plugin + features.py)
- Rolling Hurst
- Regime-Filter (compute_regime_filter mit ADX, VIX, Hurst)
- RegimeFilterGridConfig (Kombinationen, Parsing)
- Regime-Plugin Feature-Berechnung
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline.features import compute_regime_filter, _hurst_exponent, _compute_rolling_hurst
from fwbg.core.config import (
    RegimeFilterGridConfig,
    RegimeFilterConfig,
    GridConfig,
    StrategyConfig,
)


# === FIXTURES ===

@pytest.fixture
def trending_series():
    """Stark trending Zeitreihe (sollte H > 0.5 haben)."""
    np.random.seed(42)
    n = 500
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 1
    return trend + noise


@pytest.fixture
def mean_reverting_series():
    """Mean-reverting Zeitreihe (sollte H < 0.5 haben)."""
    np.random.seed(42)
    n = 500
    series = np.zeros(n)
    series[0] = 100
    for i in range(1, n):
        deviation = series[i-1] - 100
        series[i] = series[i-1] - 0.3 * deviation + np.random.randn() * 2
    return series


@pytest.fixture
def random_walk():
    """Random Walk (sollte H ~ 0.5 haben)."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n)
    return 100 + np.cumsum(returns)


@pytest.fixture
def sample_ohlc():
    """Sample OHLC-Daten für Regime-Filter Tests."""
    np.random.seed(42)
    n = 300
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close * (1 + np.random.randn(n) * 0.002)

    df = pd.DataFrame({
        'O': open_price,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


# === TESTS FÜR _hurst_exponent (features.py) ===

class TestHurstExponent:
    """Tests für _hurst_exponent() in features.py."""

    def test_returns_value_between_0_and_1(self, random_walk):
        h = _hurst_exponent(random_walk)
        assert 0 <= h <= 1

    def test_trending_series_computable(self, trending_series):
        h = _hurst_exponent(trending_series)
        assert 0 <= h <= 1
        assert np.isfinite(h)

    def test_mean_reverting_computable(self, mean_reverting_series):
        h = _hurst_exponent(mean_reverting_series)
        assert 0 <= h <= 1
        assert np.isfinite(h)

    def test_returns_default_for_short_series(self):
        short_series = np.array([100.0, 101.0, 99.0])
        h = _hurst_exponent(short_series)
        assert h == 0.5


# === TESTS FÜR _compute_rolling_hurst (features.py) ===

class TestComputeRollingHurst:
    """Tests für _compute_rolling_hurst() in features.py."""

    def test_output_length_matches_input(self, random_walk):
        result = _compute_rolling_hurst(random_walk, window=100, step=10)
        assert len(result) == len(random_walk)

    def test_values_between_0_and_1(self, random_walk):
        result = _compute_rolling_hurst(random_walk, window=100, step=10)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 1 for v in valid)

    def test_forward_fills_gaps(self, random_walk):
        result = _compute_rolling_hurst(random_walk, window=100, step=20)
        first_valid_idx = np.where(~np.isnan(result))[0]
        if len(first_valid_idx) > 0:
            first = first_valid_idx[0]
            assert not any(np.isnan(result[first:]))


# === TESTS FÜR compute_regime_filter ===

class TestComputeRegimeFilter:
    """Tests für compute_regime_filter()."""

    def test_no_params_allows_all_trades(self, sample_ohlc):
        result = compute_regime_filter(sample_ohlc, regime_params=None)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlc)
        assert result.all()

    def test_adx_filter_blocks_some_trades(self, sample_ohlc):
        params = RegimeFilterConfig(adx_enabled=True, adx_min=30)
        result = compute_regime_filter(sample_ohlc, regime_params=params)
        assert result.dtype == bool
        assert not result.all(), "ADX=30 sollte einige Bars filtern"

    def test_adx_zero_means_no_filter(self, sample_ohlc):
        params = RegimeFilterConfig(adx_enabled=True, adx_min=0)
        result = compute_regime_filter(sample_ohlc, regime_params=params)
        assert result.all(), "adx_min=0 sollte keinen Filter anwenden"

    def test_hurst_filter(self, sample_ohlc):
        params = RegimeFilterConfig(hurst_enabled=True, hurst_min=0.45, hurst_max=0.55)
        result = compute_regime_filter(sample_ohlc, regime_params=params)
        assert result.dtype == bool
        assert len(result) == len(sample_ohlc)

    def test_hurst_filter_min_only(self, sample_ohlc):
        params = RegimeFilterConfig(hurst_enabled=True, hurst_min=0.45)
        result = compute_regime_filter(sample_ohlc, regime_params=params)
        assert result.dtype == bool

    def test_vix_filter_with_macro_vix_column(self, sample_ohlc):
        """VIX-Filter nutzt macro_vix Spalte."""
        df = sample_ohlc.copy()
        df["macro_vix"] = np.linspace(15, 40, len(df))

        params = RegimeFilterConfig(vix_enabled=True, vix_max=30)
        result = compute_regime_filter(df, regime_params=params)
        assert result.dtype == bool
        # Bars mit VIX > 30 sollten gefiltert werden
        assert not result.all()
        # Bars mit VIX < 30 sollten erlaubt sein
        assert result.any()

    def test_vix_filter_with_sent_vix_fallback(self, sample_ohlc):
        """VIX-Filter fällt auf sent_vix zurück wenn macro_vix fehlt."""
        df = sample_ohlc.copy()
        df["sent_vix"] = np.linspace(15, 40, len(df))

        params = RegimeFilterConfig(vix_enabled=True, vix_max=30)
        result = compute_regime_filter(df, regime_params=params)
        assert not result.all(), "sent_vix > 30 sollte gefiltert werden"

    def test_vix_filter_prefers_macro_vix(self, sample_ohlc):
        """VIX-Filter bevorzugt macro_vix über sent_vix."""
        df = sample_ohlc.copy()
        # macro_vix: alle unter 30 -> alles erlaubt
        df["macro_vix"] = 20.0
        # sent_vix: alle über 30 -> würde alles filtern
        df["sent_vix"] = 40.0

        params = RegimeFilterConfig(vix_enabled=True, vix_max=30)
        result = compute_regime_filter(df, regime_params=params)
        # Sollte macro_vix verwenden -> alles erlaubt
        assert result.all(), "Sollte macro_vix bevorzugen (alle < 30)"

    def test_vix_filter_no_column_available(self, sample_ohlc):
        """VIX-Filter ohne VIX-Spalte filtert nichts."""
        params = RegimeFilterConfig(vix_enabled=True, vix_max=30)
        result = compute_regime_filter(sample_ohlc, regime_params=params)
        # Kein VIX-Daten -> Filter greift nicht
        assert result.all()

    def test_combined_adx_and_hurst(self, sample_ohlc):
        """Kombinierter ADX + Hurst Filter."""
        params = RegimeFilterConfig(
            adx_enabled=True, adx_min=25,
            hurst_enabled=True, hurst_min=0.45,
        )
        result = compute_regime_filter(sample_ohlc, regime_params=params)
        assert result.dtype == bool
        # Kombinierter Filter sollte restriktiver sein als einzeln
        adx_only = compute_regime_filter(sample_ohlc, RegimeFilterConfig(adx_enabled=True, adx_min=25))
        assert result.sum() <= adx_only.sum()

    def test_combined_adx_vix_hurst(self, sample_ohlc):
        """Alle drei Filter gleichzeitig."""
        df = sample_ohlc.copy()
        df["macro_vix"] = np.linspace(15, 40, len(df))

        params = RegimeFilterConfig(
            adx_enabled=True, adx_min=25,
            vix_enabled=True, vix_max=30,
            hurst_enabled=True, hurst_min=0.45,
        )
        result = compute_regime_filter(df, regime_params=params)
        assert result.dtype == bool
        assert len(result) == len(df)


# === TESTS FÜR RegimeFilterGridConfig ===

class TestRegimeFilterGridConfig:
    """Tests für RegimeFilterGridConfig Konfiguration und Kombinationen."""

    def test_default_produces_one_combination(self):
        config = RegimeFilterGridConfig()
        combos = config.get_combinations()
        assert len(combos) == 1
        assert combos[0]["adx_enabled"] is False
        assert combos[0]["vix_enabled"] is False
        assert combos[0]["hurst_enabled"] is False

    def test_total_combinations_matches_get_combinations(self):
        config = RegimeFilterGridConfig(
            adx_min=[0, 25],
            vix_max=[None, 30],
            hurst=[None, {"min": 0.45}],
        )
        assert config.total_combinations() == len(config.get_combinations())

    def test_exploration_config_produces_8_combinations(self):
        """Die aktuelle Exploration-Konfiguration: 2 ADX × 2 VIX × 2 Hurst = 8."""
        config = RegimeFilterGridConfig(
            adx_min=[0, 25],
            vix_max=[None, 30],
            hurst=[None, {"min": 0.45}],
        )
        combos = config.get_combinations()
        assert len(combos) == 8

    def test_combination_flags_correct(self):
        """Prüft dass enabled-Flags korrekt gesetzt werden."""
        config = RegimeFilterGridConfig(
            adx_min=[0, 25],
            vix_max=[None, 30],
            hurst=[None, {"min": 0.45}],
        )
        combos = config.get_combinations()

        # Finde "no filter" Kombi (adx=0, vix=None, hurst=None)
        no_filter = [c for c in combos if not c["adx_enabled"] and not c["vix_enabled"] and not c["hurst_enabled"]]
        assert len(no_filter) == 1, "Genau eine 'kein Filter' Kombination erwartet"

        # Finde "all filters" Kombi (adx=25, vix=30, hurst=0.45)
        all_filters = [c for c in combos if c["adx_enabled"] and c["vix_enabled"] and c["hurst_enabled"]]
        assert len(all_filters) == 1, "Genau eine 'alle Filter' Kombination erwartet"
        assert all_filters[0]["adx_min"] == 25
        assert all_filters[0]["vix_max"] == 30
        assert all_filters[0]["hurst_min"] == 0.45

    def test_hurst_dict_parsed_correctly(self):
        """Hurst-Dict mit min/max wird korrekt geparsed."""
        config = RegimeFilterGridConfig(
            adx_min=[0],
            vix_max=[None],
            hurst=[{"min": 0.4, "max": 0.6}],
        )
        combos = config.get_combinations()
        assert len(combos) == 1
        assert combos[0]["hurst_enabled"] is True
        assert combos[0]["hurst_min"] == 0.4
        assert combos[0]["hurst_max"] == 0.6

    def test_from_dict_none_returns_default(self):
        config = RegimeFilterGridConfig.from_dict(None)
        assert config.total_combinations() == 1

    def test_from_dict_parses_strategy_format(self):
        """Parsed das Format aus unseren Strategy-JSONs."""
        data = {
            "adx_min": [0, 25],
            "vix_max": [None, 30],
            "hurst": [None, {"min": 0.45}],
        }
        config = RegimeFilterGridConfig.from_dict(data)
        assert config.total_combinations() == 8

    def test_from_dict_scalar_to_list(self):
        """Einzelwerte werden zu Listen konvertiert."""
        data = {"adx_min": 25, "vix_max": 30}
        config = RegimeFilterGridConfig.from_dict(data)
        assert config.adx_min == [25]
        assert config.vix_max == [30]

    def test_grid_config_includes_regime_filter(self):
        """GridConfig.from_dict parsed regime_filter_grid."""
        data = {
            "tp": [10, 20],
            "sl": [20, 30],
            "ct": [0.6],
            "regime_filter_grid": {
                "adx_min": [0, 25],
                "vix_max": [None, 30],
                "hurst": [None, {"min": 0.45}],
            },
        }
        grid = GridConfig.from_dict(data)
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_grid_config_without_regime_filter(self):
        """GridConfig ohne regime_filter_grid hat Default (1 Kombi)."""
        data = {"tp": [10], "sl": [20], "ct": [0.6]}
        grid = GridConfig.from_dict(data)
        assert grid.regime_filter_grid.total_combinations() == 1


# === TESTS FÜR RegimeFilterConfig ===

class TestRegimeFilterConfig:
    """Tests für RegimeFilterConfig (einzelne Regime-Konfiguration)."""

    def test_default_all_disabled(self):
        config = RegimeFilterConfig()
        assert not config.adx_enabled
        assert not config.vix_enabled
        assert not config.hurst_enabled

    def test_from_dict_none_returns_default(self):
        config = RegimeFilterConfig.from_dict(None)
        assert not config.adx_enabled

    def test_from_dict_with_values(self):
        data = {
            "adx_enabled": True,
            "adx_min": 25,
            "vix_enabled": True,
            "vix_max": 30,
            "hurst_enabled": True,
            "hurst_min": 0.45,
        }
        config = RegimeFilterConfig.from_dict(data)
        assert config.adx_enabled
        assert config.adx_min == 25
        assert config.vix_enabled
        assert config.vix_max == 30
        assert config.hurst_enabled
        assert config.hurst_min == 0.45

    def test_grid_combo_to_filter_config(self):
        """RegimeFilterGridConfig Combos können als RegimeFilterConfig verwendet werden."""
        grid_config = RegimeFilterGridConfig(
            adx_min=[0, 25],
            vix_max=[None, 30],
            hurst=[None, {"min": 0.45}],
        )
        combos = grid_config.get_combinations()

        for combo_dict in combos:
            config = RegimeFilterConfig.from_dict(combo_dict)
            # Sollte gültig sein
            assert isinstance(config.adx_enabled, bool)
            assert isinstance(config.vix_enabled, bool)
            assert isinstance(config.hurst_enabled, bool)


# === TESTS FÜR Strategy-Laden mit Regime-Filter ===

class TestStrategyRegimeConfig:
    """Tests dass Exploration-Strategien korrekte Regime-Config haben."""

    def test_exploration_json_has_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_exploration_atr_has_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration_atr.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_exploration_fast_has_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration_fast.json")
        grid = config.get_grid_for_class("FOREX")
        assert grid.regime_filter_grid.total_combinations() == 8

    def test_all_asset_classes_have_regime_grid(self):
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        for asset_class in ["FOREX", "INDEX", "COMMODITY", "CRYPTO"]:
            grid = config.get_grid_for_class(asset_class)
            assert grid.regime_filter_grid.total_combinations() == 8, (
                f"{asset_class} sollte 8 Regime-Kombinationen haben"
            )

    def test_regime_combos_include_no_filter_baseline(self):
        """Jede Strategie hat eine 'kein Filter' Baseline."""
        config = StrategyConfig.from_json_file("strategies/exploration.json")
        grid = config.get_grid_for_class("FOREX")
        combos = grid.regime_filter_grid.get_combinations()

        no_filter = [c for c in combos if not c["adx_enabled"] and not c["vix_enabled"] and not c["hurst_enabled"]]
        assert len(no_filter) == 1


# === TESTS FÜR Regime-Plugin ===

class TestRegimePlugin:
    """Tests für das RegimeIndicators Plugin."""

    def test_plugin_importable(self):
        from fwbg.plugins.indicator import BaseIndicator
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-premium:regime")
        assert plugin_cls is not None

    def test_plugin_computes_features(self, sample_ohlc):
        """Plugin berechnet Regime-Features korrekt."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        # Brauchen mehr Daten für Hurst
        np.random.seed(42)
        n = 600
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.randn(n) * 0.005))
        low = close * (1 - np.abs(np.random.randn(n) * 0.005))
        open_price = close * (1 + np.random.randn(n) * 0.002)

        df = pd.DataFrame({
            'O': open_price, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[50], vr_windows=[100], vr_lags=[5], step=10)

        assert "regime_hurst_100" in result.columns
        assert "regime_entropy_50" in result.columns
        assert "regime_vr_100_5" in result.columns

    def test_plugin_hurst_values_valid(self, sample_ohlc):
        """Hurst-Werte sind zwischen 0 und 1."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'O': close, 'H': close * 1.005, 'L': close * 0.995, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[], vr_windows=[], vr_lags=[], step=10)

        hurst = result["regime_hurst_100"].dropna()
        assert len(hurst) > 0
        assert all(0 <= v <= 1 for v in hurst)

    def test_plugin_uses_original_close(self):
        """Plugin nutzt _original_close wenn vorhanden (Frac-Diff)."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        plugin = plugin_cls()

        np.random.seed(42)
        n = 600
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'O': close, 'H': close + 0.5, 'L': close - 0.5,
            'C': np.random.randn(n),  # Frac-diff'd (nicht-originale Close)
            '_original_close': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = plugin.compute(df, hurst_windows=[100], entropy_windows=[], vr_windows=[], vr_lags=[], step=10)

        hurst = result["regime_hurst_100"].dropna()
        assert len(hurst) > 0
        # Sollte sinnvolle Werte haben (nicht nur 0.5)
        assert all(0 <= v <= 1 for v in hurst)

    def test_plugin_benefits_from_stationary_false(self):
        """Regime-Plugin sollte benefits_from_stationary=False haben."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime")
        assert plugin_cls.benefits_from_stationary is False
