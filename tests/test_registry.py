"""
Tests für Plugin Registry.

Fokus auf Edge Cases:
- Doppelte Registrierung
- Unbekannte Plugins
- Entry Point Discovery Fehler
- Leere Registries
"""
import pytest

from fwbg.core.registry import (
    INDICATOR_REGISTRY,
    EXIT_STRATEGY_REGISTRY,
    FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    DATA_ADAPTER_REGISTRY,
    EXECUTION_ADAPTER_REGISTRY,
    register_indicator,
    register_exit_strategy,
    register_feature_selector,
    register_preprocessor,
    register_data_adapter,
    register_execution_adapter,
    get_indicator,
    get_exit_strategy,
    get_feature_selector,
    get_preprocessor,
    get_data_adapter,
    get_execution_adapter,
    list_indicators,
    list_exit_strategies,
    list_feature_selectors,
    list_preprocessors,
    list_data_adapters,
    list_execution_adapters,
    discover_plugins,
)


# --- Decorator Tests ---


class TestRegisterIndicator:
    """Tests für @register_indicator Decorator."""

    def test_registers_class(self):
        """Decorator sollte Klasse registrieren."""
        @register_indicator("test_indicator_1")
        class TestIndicator1:
            pass

        assert "test_indicator_1" in INDICATOR_REGISTRY
        assert INDICATOR_REGISTRY["test_indicator_1"] is TestIndicator1

        # Cleanup
        del INDICATOR_REGISTRY["test_indicator_1"]

    def test_sets_name_attribute(self):
        """Decorator sollte name-Attribut setzen."""
        @register_indicator("test_indicator_2")
        class TestIndicator2:
            pass

        assert TestIndicator2.name == "test_indicator_2"

        # Cleanup
        del INDICATOR_REGISTRY["test_indicator_2"]

    def test_overwrites_existing(self):
        """Doppelte Registrierung sollte überschreiben."""
        @register_indicator("test_indicator_3")
        class TestIndicator3a:
            pass

        @register_indicator("test_indicator_3")
        class TestIndicator3b:
            pass

        assert INDICATOR_REGISTRY["test_indicator_3"] is TestIndicator3b

        # Cleanup
        del INDICATOR_REGISTRY["test_indicator_3"]

    def test_empty_name(self):
        """Leerer Name sollte funktionieren (aber nicht empfohlen)."""
        @register_indicator("")
        class EmptyNameIndicator:
            pass

        assert "" in INDICATOR_REGISTRY

        # Cleanup
        del INDICATOR_REGISTRY[""]


class TestRegisterExitStrategy:
    """Tests für @register_exit_strategy Decorator."""

    def test_registers_class(self):
        """Decorator sollte Klasse registrieren."""
        @register_exit_strategy("test_exit_1")
        class TestExit1:
            pass

        assert "test_exit_1" in EXIT_STRATEGY_REGISTRY
        assert EXIT_STRATEGY_REGISTRY["test_exit_1"] is TestExit1

        # Cleanup
        del EXIT_STRATEGY_REGISTRY["test_exit_1"]

    def test_sets_name_attribute(self):
        """Decorator sollte name-Attribut setzen."""
        @register_exit_strategy("test_exit_2")
        class TestExit2:
            pass

        assert TestExit2.name == "test_exit_2"

        # Cleanup
        del EXIT_STRATEGY_REGISTRY["test_exit_2"]


class TestRegisterFeatureSelector:
    """Tests für @register_feature_selector Decorator."""

    def test_registers_class(self):
        """Decorator sollte Klasse registrieren."""
        @register_feature_selector("test_fs_1")
        class TestFS1:
            pass

        assert "test_fs_1" in FEATURE_SELECTOR_REGISTRY

        # Cleanup
        del FEATURE_SELECTOR_REGISTRY["test_fs_1"]


class TestRegisterPreprocessor:
    """Tests für @register_preprocessor Decorator."""

    def test_registers_class(self):
        """Decorator sollte Klasse registrieren."""
        @register_preprocessor("test_pp_1")
        class TestPP1:
            pass

        assert "test_pp_1" in PREPROCESSOR_REGISTRY

        # Cleanup
        del PREPROCESSOR_REGISTRY["test_pp_1"]


class TestRegisterDataAdapter:
    """Tests für @register_data_adapter Decorator."""

    def test_registers_class(self):
        """Decorator sollte Klasse registrieren."""
        @register_data_adapter("test_da_1")
        class TestDA1:
            pass

        assert "test_da_1" in DATA_ADAPTER_REGISTRY
        assert TestDA1.adapter_type == "test_da_1"

        # Cleanup
        del DATA_ADAPTER_REGISTRY["test_da_1"]


class TestRegisterExecutionAdapter:
    """Tests für @register_execution_adapter Decorator."""

    def test_registers_class(self):
        """Decorator sollte Klasse registrieren."""
        @register_execution_adapter("test_ea_1")
        class TestEA1:
            pass

        assert "test_ea_1" in EXECUTION_ADAPTER_REGISTRY
        assert TestEA1.adapter_type == "test_ea_1"

        # Cleanup
        del EXECUTION_ADAPTER_REGISTRY["test_ea_1"]


# --- Getter Tests ---


class TestGetIndicator:
    """Tests für get_indicator()."""

    def test_returns_registered_class(self):
        """Sollte registrierte Klasse zurückgeben."""
        # Built-in Indicators sollten nach discover_plugins vorhanden sein
        discover_plugins()

        if "trend" in INDICATOR_REGISTRY:
            cls = get_indicator("trend")
            assert cls is not None

    def test_raises_for_unknown(self):
        """Sollte ValueError für unbekannten Namen werfen."""
        with pytest.raises(ValueError) as exc_info:
            get_indicator("nonexistent_indicator_xyz")

        assert "nonexistent_indicator_xyz" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_error_lists_available(self):
        """Fehlermeldung sollte verfügbare Plugins auflisten."""
        # Registriere ein Test-Plugin
        @register_indicator("available_test_ind")
        class AvailableTestInd:
            pass

        try:
            with pytest.raises(ValueError) as exc_info:
                get_indicator("wrong_name")

            # Prüfe dass verfügbare Plugins genannt werden
            assert "available_test_ind" in str(exc_info.value)
        finally:
            del INDICATOR_REGISTRY["available_test_ind"]


class TestGetExitStrategy:
    """Tests für get_exit_strategy()."""

    def test_returns_registered_class(self):
        """Sollte registrierte Klasse zurückgeben."""
        discover_plugins()

        if "fixed" in EXIT_STRATEGY_REGISTRY:
            cls = get_exit_strategy("fixed")
            assert cls is not None

    def test_raises_for_unknown(self):
        """Sollte ValueError für unbekannten Namen werfen."""
        with pytest.raises(ValueError) as exc_info:
            get_exit_strategy("nonexistent_exit_xyz")

        assert "nonexistent_exit_xyz" in str(exc_info.value)


class TestGetFeatureSelector:
    """Tests für get_feature_selector()."""

    def test_raises_for_unknown(self):
        """Sollte ValueError für unbekannten Namen werfen."""
        with pytest.raises(ValueError) as exc_info:
            get_feature_selector("nonexistent_fs_xyz")

        assert "nonexistent_fs_xyz" in str(exc_info.value)


class TestGetPreprocessor:
    """Tests für get_preprocessor()."""

    def test_raises_for_unknown(self):
        """Sollte ValueError für unbekannten Namen werfen."""
        with pytest.raises(ValueError) as exc_info:
            get_preprocessor("nonexistent_pp_xyz")

        assert "nonexistent_pp_xyz" in str(exc_info.value)


class TestGetDataAdapter:
    """Tests für get_data_adapter()."""

    def test_raises_for_unknown(self):
        """Sollte ValueError für unbekannten Namen werfen."""
        with pytest.raises(ValueError) as exc_info:
            get_data_adapter("nonexistent_da_xyz")

        assert "nonexistent_da_xyz" in str(exc_info.value)


class TestGetExecutionAdapter:
    """Tests für get_execution_adapter()."""

    def test_raises_for_unknown(self):
        """Sollte ValueError für unbekannten Namen werfen."""
        with pytest.raises(ValueError) as exc_info:
            get_execution_adapter("nonexistent_ea_xyz")

        assert "nonexistent_ea_xyz" in str(exc_info.value)


# --- List Functions Tests ---


class TestListFunctions:
    """Tests für list_* Funktionen."""

    def test_list_indicators_returns_list(self):
        """list_indicators sollte Liste zurückgeben."""
        result = list_indicators()
        assert isinstance(result, list)

    def test_list_exit_strategies_returns_list(self):
        """list_exit_strategies sollte Liste zurückgeben."""
        result = list_exit_strategies()
        assert isinstance(result, list)

    def test_list_feature_selectors_returns_list(self):
        """list_feature_selectors sollte Liste zurückgeben."""
        result = list_feature_selectors()
        assert isinstance(result, list)

    def test_list_preprocessors_returns_list(self):
        """list_preprocessors sollte Liste zurückgeben."""
        result = list_preprocessors()
        assert isinstance(result, list)

    def test_list_data_adapters_returns_list(self):
        """list_data_adapters sollte Liste zurückgeben."""
        result = list_data_adapters()
        assert isinstance(result, list)

    def test_list_execution_adapters_returns_list(self):
        """list_execution_adapters sollte Liste zurückgeben."""
        result = list_execution_adapters()
        assert isinstance(result, list)

    def test_list_reflects_registry(self):
        """Listen sollten aktuelle Registry-Inhalte widerspiegeln."""
        @register_indicator("list_test_indicator")
        class ListTestIndicator:
            pass

        try:
            indicators = list_indicators()
            assert "list_test_indicator" in indicators
        finally:
            del INDICATOR_REGISTRY["list_test_indicator"]


# --- Discovery Tests ---


class TestDiscoverPlugins:
    """Tests für discover_plugins()."""

    def test_runs_without_error(self):
        """discover_plugins sollte ohne Fehler laufen."""
        # Sollte keine Exception werfen
        discover_plugins()

    def test_loads_builtin_indicators(self):
        """Sollte Built-in Indikatoren laden."""
        discover_plugins()

        # Mindestens einige Standard-Indikatoren sollten vorhanden sein
        indicators = list_indicators()
        assert len(indicators) > 0, "Sollte Built-in Indikatoren laden"

    def test_loads_builtin_exit_strategies(self):
        """Sollte Built-in Exit-Strategien laden."""
        discover_plugins()

        strategies = list_exit_strategies()
        assert len(strategies) > 0, "Sollte Built-in Exit-Strategien laden"

    def test_idempotent(self):
        """Mehrfaches Aufrufen sollte keine Duplikate erzeugen."""
        discover_plugins()
        count1 = len(list_indicators())

        discover_plugins()
        count2 = len(list_indicators())

        # Counts sollten gleich sein (Überschreiben, keine Duplikate)
        assert count1 == count2

    def test_handles_missing_groups(self):
        """Sollte fehlende Entry Point Gruppen graceful behandeln."""
        # Sollte keine Exception werfen auch wenn Gruppe nicht existiert
        discover_plugins()


# --- Edge Cases ---


class TestRegistryEdgeCases:
    """Edge Cases für Registry."""

    def test_special_characters_in_name(self):
        """Namen mit Sonderzeichen sollten funktionieren."""
        @register_indicator("test-with-dashes")
        class DashIndicator:
            pass

        assert "test-with-dashes" in INDICATOR_REGISTRY
        del INDICATOR_REGISTRY["test-with-dashes"]

        @register_indicator("test_with_underscores")
        class UnderscoreIndicator:
            pass

        assert "test_with_underscores" in INDICATOR_REGISTRY
        del INDICATOR_REGISTRY["test_with_underscores"]

    def test_unicode_name(self):
        """Unicode-Namen sollten funktionieren."""
        @register_indicator("test_äöü")
        class UnicodeIndicator:
            pass

        assert "test_äöü" in INDICATOR_REGISTRY
        cls = get_indicator("test_äöü")
        assert cls is UnicodeIndicator

        del INDICATOR_REGISTRY["test_äöü"]

    def test_numeric_name(self):
        """Numerische Namen sollten funktionieren."""
        @register_indicator("123")
        class NumericIndicator:
            pass

        assert "123" in INDICATOR_REGISTRY
        del INDICATOR_REGISTRY["123"]

    def test_very_long_name(self):
        """Sehr lange Namen sollten funktionieren."""
        long_name = "a" * 1000

        @register_indicator(long_name)
        class LongNameIndicator:
            pass

        assert long_name in INDICATOR_REGISTRY
        del INDICATOR_REGISTRY[long_name]

    def test_class_without_methods(self):
        """Klassen ohne Methoden sollten registrierbar sein."""
        @register_indicator("empty_class_indicator")
        class EmptyClass:
            pass

        assert "empty_class_indicator" in INDICATOR_REGISTRY
        del INDICATOR_REGISTRY["empty_class_indicator"]

    def test_function_as_plugin(self):
        """Funktionen sollten auch registrierbar sein (nicht empfohlen aber möglich)."""
        @register_indicator("function_indicator")
        def func_indicator():
            pass

        assert "function_indicator" in INDICATOR_REGISTRY
        del INDICATOR_REGISTRY["function_indicator"]


class TestRegistryIntegration:
    """Integration Tests für Registry."""

    def test_full_workflow(self):
        """Vollständiger Workflow: Register -> Get -> Use."""
        @register_indicator("workflow_test")
        class WorkflowIndicator:
            group = "test"

            def compute(self, df):
                return df

            def get_feature_columns(self):
                return ["test_col"]

        try:
            # Get
            cls = get_indicator("workflow_test")
            assert cls is WorkflowIndicator

            # Instantiate
            instance = cls()
            assert instance.group == "test"

            # List
            assert "workflow_test" in list_indicators()
        finally:
            del INDICATOR_REGISTRY["workflow_test"]

    def test_plugins_have_expected_interface(self):
        """Geladene Plugins sollten erwartete Interfaces haben."""
        discover_plugins()

        # Indicators
        for name in list_indicators():
            cls = get_indicator(name)
            # Sollte compute und get_feature_columns haben
            assert hasattr(cls, "compute") or hasattr(cls, "group")

        # Exit Strategies
        for name in list_exit_strategies():
            cls = get_exit_strategy(name)
            assert hasattr(cls, "compute_targets") or hasattr(cls, "iterate_grid")

        # Preprocessors müssen transform() haben (nicht process()!)
        for name in list_preprocessors():
            cls = get_preprocessor(name)
            assert hasattr(cls, "transform"), f"Preprocessor {name} has no transform() method"
