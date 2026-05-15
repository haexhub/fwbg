"""
Tests for the generic DataSource loading system.

Covers:
1. LoadResult dataclass
2. DataSourceConfig.load() on CSV/REST/WebSocket/DB
3. BaseDataLoader plugin base class
4. DataLoader registry (register/get/list)
5. Orchestrator run_data_loading()
6. MacroDataLoader plugin computation
7. StrategyConfig.get_data_loading()
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ============================================================================
# Step 1: LoadResult + load() on DataSourceConfig
# ============================================================================

class TestLoadResult:
    """LoadResult dataclass holds data + metadata from source loading."""

    def test_default_values(self):
        from fwbg.core.data_sources import LoadResult
        result = LoadResult()
        assert result.data == {}
        assert result.metadata == {}
        assert result.source_name == ""

    def test_with_data(self):
        from fwbg.core.data_sources import LoadResult
        df = pd.DataFrame({"Close": [100.0, 101.0]})
        result = LoadResult(data={"vix": df}, source_name="forexsb")
        assert "vix" in result.data
        assert result.source_name == "forexsb"

    def test_with_metadata(self):
        from fwbg.core.data_sources import LoadResult
        result = LoadResult(metadata={"api_calls": 3})
        assert result.metadata["api_calls"] == 3


class TestCSVSourceLoad:
    """CSVSourceConfig.load() reads CSV files and returns LoadResult."""

    def test_load_returns_load_result(self, tmp_path):
        from fwbg.core.data_sources import CSVSourceConfig, LoadResult

        # Create test CSV
        csv_path = tmp_path / "VIX_DAY.csv"
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Close": [20.0, 21.0, 19.5, 22.0, 20.5],
        })
        df.to_csv(csv_path, index=False)

        source = CSVSourceConfig(name="test", path=tmp_path)
        result = source.load({"VIX_DAY": "vix"})

        assert isinstance(result, LoadResult)
        assert "vix" in result.data
        assert "Close" in result.data["vix"].columns
        assert len(result.data["vix"]) == 5

    def test_load_file_not_found_skips(self, tmp_path):
        from fwbg.core.data_sources import CSVSourceConfig

        source = CSVSourceConfig(name="test", path=tmp_path)
        result = source.load({"NONEXISTENT": "missing"})

        assert "missing" not in result.data

    def test_load_parses_dates(self, tmp_path):
        from fwbg.core.data_sources import CSVSourceConfig

        csv_path = tmp_path / "TNX_DAY.csv"
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [4.0, 4.1],
        })
        df.to_csv(csv_path, index=False)

        source = CSVSourceConfig(name="test", path=tmp_path)
        result = source.load({"TNX_DAY": "tnx"})

        assert isinstance(result.data["tnx"].index, pd.DatetimeIndex)

    def test_load_multiple_items(self, tmp_path):
        from fwbg.core.data_sources import CSVSourceConfig

        for name, vals in [("VIX_DAY", [20.0]), ("TNX_DAY", [4.0])]:
            csv_path = tmp_path / f"{name}.csv"
            pd.DataFrame({
                "Date": ["2024-01-01"], "Close": vals,
            }).to_csv(csv_path, index=False)

        source = CSVSourceConfig(name="test", path=tmp_path)
        result = source.load({"VIX_DAY": "vix", "TNX_DAY": "tnx"})

        assert "vix" in result.data
        assert "tnx" in result.data


class TestRESTSourceLoad:
    """RESTSourceConfig.load() raises NotImplementedError (no live API in tests)."""

    def test_load_raises_not_implemented(self):
        from fwbg.core.data_sources import RESTSourceConfig

        source = RESTSourceConfig(name="test_api", base_url="https://example.com")
        with pytest.raises(NotImplementedError):
            source.load({"endpoint": "data"})


class TestWebSocketSourceLoad:
    """WebSocketSourceConfig.load() raises NotImplementedError (streaming only)."""

    def test_load_raises_not_implemented(self):
        from fwbg.core.data_sources import WebSocketSourceConfig

        source = WebSocketSourceConfig(name="test_ws", url="wss://example.com")
        with pytest.raises(NotImplementedError):
            source.load({"stream": "data"})


# ============================================================================
# Step 2: DBSourceConfig
# ============================================================================

class TestDBSourceConfig:
    """DBSourceConfig for database data sources."""

    def test_source_type_is_database(self):
        from fwbg.core.data_sources import DBSourceConfig, SourceType

        source = DBSourceConfig(name="test_db", connection_string="sqlite:///test.db")
        assert source.source_type == SourceType.DATABASE

    def test_register_db_source(self):
        from fwbg.core.data_sources import register_db_source, get_data_source

        source = register_db_source(
            name="test_db_reg",
            connection_string="sqlite:///test.db",
        )
        retrieved = get_data_source("test_db_reg")
        assert retrieved is source

    def test_db_source_in_data_source_union(self):
        from fwbg.core.data_sources import DBSourceConfig, DataSource
        assert DBSourceConfig in DataSource.__args__


# ============================================================================
# Step 3: BaseDataLoader
# ============================================================================

class TestBaseDataLoader:
    """BaseDataLoader is the base class for data loading plugins."""

    def test_phase_is_data_loading(self):
        from fwbg_sdk import BaseDataLoader
        from fwbg_sdk import PluginPhase
        assert BaseDataLoader.phase == PluginPhase.DATA_LOADING

    def test_is_abstract(self):
        from fwbg_sdk import BaseDataLoader
        with pytest.raises(TypeError):
            BaseDataLoader()

    def test_concrete_subclass_works(self):
        from fwbg_sdk import BaseDataLoader
        from fwbg_sdk import PipelineContext

        class TestLoader(BaseDataLoader):
            name = "test_loader"
            version = "1.0.0"

            def execute(self, ctx, **params):
                ctx.df["computed"] = 42
                return ctx

        loader = TestLoader()
        df = pd.DataFrame({"O": [1.0], "C": [2.0]})
        ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")
        result = loader.execute(ctx)
        assert "computed" in result.df.columns

    def test_exported_from_plugins(self):
        from fwbg.plugins import BaseDataLoader
        assert BaseDataLoader is not None


# ============================================================================
# Step 4: DataLoader Registry
# ============================================================================

class TestDataLoaderRegistry:
    """DataLoader registry for registering and retrieving data loaders."""

    def test_register_and_get(self):
        from fwbg.core.registry import register_data_loader, get_data_loader, DATA_LOADER_REGISTRY
        from fwbg_sdk import BaseDataLoader
        from fwbg_sdk import PipelineContext

        @register_data_loader("test_dl_reg")
        class TestDL(BaseDataLoader):
            name = "test_dl_reg"
            version = "1.0.0"
            def execute(self, ctx, **params):
                return ctx

        assert "test_dl_reg" in DATA_LOADER_REGISTRY
        assert get_data_loader("test_dl_reg") is TestDL

        # Cleanup
        del DATA_LOADER_REGISTRY["test_dl_reg"]

    def test_get_unknown_raises(self):
        from fwbg.core.registry import get_data_loader
        with pytest.raises(ValueError, match="Unknown data loader"):
            get_data_loader("nonexistent_loader")

    def test_list_data_loaders(self):
        from fwbg.core.registry import list_data_loaders, DATA_LOADER_REGISTRY

        # Registry may have items from other tests; just check it returns a list
        result = list_data_loaders()
        assert isinstance(result, list)

    def test_exported_from_core(self):
        from fwbg.core import register_data_loader, get_data_loader, list_data_loaders
        assert callable(register_data_loader)
        assert callable(get_data_loader)
        assert callable(list_data_loaders)

    def test_pipeline_registry_discovers_data_loading(self):
        """Pipeline registry scans 'data_loading' category."""
        from fwbg.pipeline.registry import PluginRegistry
        registry = PluginRegistry()
        # Just verify the category is in the scan list
        # by checking the source — actual discovery needs plugin files
        import inspect
        source = inspect.getsource(registry.discover_package)
        assert "data_loading" in source


# ============================================================================
# Step 5: Orchestrator run_data_loading()
# ============================================================================

class TestRunDataLoading:
    """run_data_loading() orchestrates DataSource I/O + Plugin computation."""

    def test_no_configs_returns_df_unchanged(self):
        from fwbg.data.loader import run_data_loading

        df = pd.DataFrame(
            {"O": [1.0, 2.0], "C": [1.1, 2.1]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        result = run_data_loading(df, [])
        pd.testing.assert_frame_equal(result, df)

    def test_csv_source_loads_and_aligns(self, tmp_path):
        from fwbg.data.loader import run_data_loading
        from fwbg.core.data_sources import register_csv_source, _DATA_SOURCES

        # Create test CSV
        csv_path = tmp_path / "VIX_DAY.csv"
        pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [20.0, 21.0],
        }).to_csv(csv_path, index=False)

        register_csv_source(name="_test_orch", path=tmp_path)

        try:
            df = pd.DataFrame(
                {"O": [1.0, 1.1, 1.2, 1.3], "C": [1.0, 1.1, 1.2, 1.3]},
                index=pd.date_range("2024-01-01", periods=4, freq="6h"),
            )

            configs = [{
                "source": "_test_orch",
                "params": {"indicators": {"VIX_DAY": "vix"}},
            }]

            result = run_data_loading(df, configs)
            assert "macro_vix" in result.columns
        finally:
            del _DATA_SOURCES["_test_orch"]

    def test_no_lookahead_bias_in_macro_alignment(self, tmp_path):
        """Macro daily data must use PREVIOUS day's close to prevent lookahead.

        On day D, the daily close is only available after market close.
        Intraday bars on day D must NOT see day D's close value.
        """
        from fwbg.data.loader import run_data_loading
        from fwbg.core.data_sources import register_csv_source, _DATA_SOURCES

        # Daily macro data: Jan 1 = 100, Jan 2 = 200, Jan 3 = 300
        csv_path = tmp_path / "TEST_DAY.csv"
        pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Close": [100.0, 200.0, 300.0],
        }).to_csv(csv_path, index=False)

        register_csv_source(name="_test_bias", path=tmp_path)

        try:
            # Hourly intraday bars spanning Jan 1-3
            df = pd.DataFrame(
                {"O": np.ones(72), "C": np.ones(72)},
                index=pd.date_range("2024-01-01", periods=72, freq="h"),
            )

            configs = [{
                "source": "_test_bias",
                "params": {"indicators": {"TEST_DAY": "test"}},
            }]

            result = run_data_loading(df, configs)

            # Bar at Jan 2 09:00 should see Jan 1's close (100),
            # NOT Jan 2's close (200) — that's only available after market close
            jan2_9am = result.loc["2024-01-02 09:00:00", "macro_test"]
            assert jan2_9am == 100.0, (
                f"Lookahead bias! Bar at Jan 2 09:00 sees value {jan2_9am}, "
                f"expected 100.0 (previous day). Got same-day close = lookahead."
            )

            # Bar at Jan 3 09:00 should see Jan 2's close (200)
            jan3_9am = result.loc["2024-01-03 09:00:00", "macro_test"]
            assert jan3_9am == 200.0, (
                f"Lookahead bias! Bar at Jan 3 09:00 sees value {jan3_9am}, "
                f"expected 200.0 (previous day)."
            )
        finally:
            del _DATA_SOURCES["_test_bias"]

    def test_release_date_column_blocks_pre_release_lookahead(self, tmp_path):
        """CSVs that carry a `release_date` column must be merged via the
        release-date contract, not the legacy prev-day shift.

        Scenario: a COT-style weekly row for the Tuesday 2024-01-02 report
        is *released* Friday 2024-01-05.  Bars on Wed-Thu (between report
        and release) must NOT see the value.
        """
        from fwbg.data.loader import run_data_loading
        from fwbg.core.data_sources import register_csv_source, _DATA_SOURCES

        csv_path = tmp_path / "COT_DEMO_DAY.csv"
        pd.DataFrame({
            "Datetime": ["2024-01-02", "2024-01-09"],  # report dates (Tue)
            "Close": [100.0, 200.0],                   # net position
            "release_date": [
                "2024-01-05 21:00:00",                  # Fri after Jan-02 report
                "2024-01-12 21:00:00",                  # Fri after Jan-09 report
            ],
        }).to_csv(csv_path, index=False)

        register_csv_source(name="_test_release", path=tmp_path)

        try:
            # Hourly bars covering Tue 2024-01-02 → Mon 2024-01-08
            df = pd.DataFrame(
                {"O": 1.0, "C": 1.0},
                index=pd.date_range("2024-01-02", periods=24 * 7, freq="h"),
            )
            configs = [{
                "source": "_test_release",
                "params": {"indicators": {"COT_DEMO_DAY": "cot_demo"}},
            }]
            result = run_data_loading(df, configs)

            # Tue 2024-01-02: report exists but not yet released → NaN.
            assert pd.isna(result.loc["2024-01-02 12:00:00", "macro_cot_demo"]), (
                "Lookahead: Tuesday bar saw COT value before Friday release."
            )
            # Wed-Thu: still pre-release → NaN.
            assert pd.isna(result.loc["2024-01-03 12:00:00", "macro_cot_demo"])
            assert pd.isna(result.loc["2024-01-04 12:00:00", "macro_cot_demo"])
            # Fri 2024-01-05 22:00 (≈ release moment) → visible.
            assert result.loc["2024-01-05 22:00:00", "macro_cot_demo"] == 100.0
            # Mon 2024-01-08: first report still in effect.
            assert result.loc["2024-01-08 12:00:00", "macro_cot_demo"] == 100.0
        finally:
            del _DATA_SOURCES["_test_release"]


# ============================================================================
# Step 7: StrategyConfig.get_data_loading()
# ============================================================================

# ============================================================================
# Step 6: MacroDataLoader Plugin
# ============================================================================

class TestMacroDataLoader:
    """MacroDataLoader computes lookbacks, derived features, and rate diffs."""

    @pytest.fixture
    def macro_df(self):
        """DataFrame with macro base columns (as loaded by orchestrator)."""
        n = 100
        index = pd.date_range("2024-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "O": np.full(n, 1.0),
            "H": np.full(n, 1.01),
            "L": np.full(n, 0.99),
            "C": np.full(n, 1.0),
            "macro_vix": np.linspace(15, 25, n),
            "macro_vvix": np.linspace(90, 110, n),
            "macro_tnx": np.linspace(3.5, 4.5, n),
            "macro_irx": np.full(n, 1.5),
            "macro_fvx": np.full(n, 3.0),
            "macro_spx": np.linspace(4900, 5100, n),
            "macro_tlt": np.full(n, 100.0),
            "macro_hyg": np.full(n, 80.0),
            "macro_lqd": np.full(n, 110.0),
            "macro_russell": np.full(n, 2000.0),
            "macro_xlk": np.full(n, 200.0),
            "macro_xlu": np.full(n, 70.0),
        }, index=index)

    def test_plugin_registered(self):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import DATA_LOADER_REGISTRY
        assert "macro_data" in DATA_LOADER_REGISTRY

    def test_hourly_lookbacks(self, macro_df):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import get_data_loader
        from fwbg_sdk import PipelineContext

        ctx = PipelineContext(df=macro_df.copy(), symbol="TEST", asset_class="FOREX")
        loader = get_data_loader("macro_data")()
        result = loader.execute(ctx, lookbacks_hours=[1, 4])

        assert "macro_vix_chg_1h" in result.df.columns
        assert "macro_vix_chg_4h" in result.df.columns

    def test_daily_lookbacks(self, macro_df):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import get_data_loader
        from fwbg_sdk import PipelineContext

        ctx = PipelineContext(df=macro_df.copy(), symbol="TEST", asset_class="FOREX")
        loader = get_data_loader("macro_data")()
        result = loader.execute(ctx, lookbacks_days=[2])

        assert "macro_vix_chg_2d" in result.df.columns

    def test_derived_features(self, macro_df):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import get_data_loader
        from fwbg_sdk import PipelineContext

        ctx = PipelineContext(df=macro_df.copy(), symbol="TEST", asset_class="FOREX")
        loader = get_data_loader("macro_data")()
        result = loader.execute(ctx)

        assert "macro_yield_curve_10y_3m" in result.df.columns
        assert "macro_vix_vvix_ratio" in result.df.columns
        # Verify computation
        expected_yc = macro_df["macro_tnx"] - macro_df["macro_irx"]
        np.testing.assert_allclose(
            result.df["macro_yield_curve_10y_3m"].values,
            expected_yc.values,
        )

    def test_empty_df_no_error(self):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import get_data_loader
        from fwbg_sdk import PipelineContext

        df = pd.DataFrame({"O": [1.0], "C": [2.0]})
        ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")
        loader = get_data_loader("macro_data")()
        result = loader.execute(ctx)
        assert isinstance(result.df, pd.DataFrame)

    def test_default_params(self):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import get_data_loader

        loader = get_data_loader("macro_data")()
        defaults = loader.get_default_params()
        assert "lookbacks_hours" in defaults
        assert "lookbacks_days" in defaults
        assert "derived_features" in defaults
        assert "interest_rates" in defaults

    def test_custom_lookbacks(self, macro_df):
        from fwbg.plugins import import_plugin_module
        import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        from fwbg.core.registry import get_data_loader
        from fwbg_sdk import PipelineContext

        ctx = PipelineContext(df=macro_df.copy(), symbol="TEST", asset_class="FOREX")
        loader = get_data_loader("macro_data")()
        result = loader.execute(ctx, lookbacks_hours=[2], lookbacks_days=[])

        assert "macro_vix_chg_2h" in result.df.columns
        # No daily lookbacks
        assert "macro_vix_chg_2d" not in result.df.columns


class TestStrategyConfigDataLoading:
    """StrategyConfig.get_data_loading() returns data loading pipeline config."""

    def test_get_data_loading_empty_default(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig()
        assert config.get_data_loading() == []

    def test_get_data_loading_from_pipeline(self):
        from fwbg.core.config import StrategyConfig
        config = StrategyConfig(pipeline={
            "data_loading": [
                {"name": "macro_data", "source": "forexsb", "params": {}}
            ]
        })
        result = config.get_data_loading()
        assert len(result) == 1
        assert result[0]["name"] == "macro_data"

    def test_get_data_loading_from_json(self, tmp_path):
        """Strategy JSON with data_loading loads correctly."""
        import json
        from fwbg.core.config import StrategyConfig

        strategy_data = {
            "name": "Test Strategy",
            "pipeline": {
                "data_loading": [
                    {"name": "macro_data", "source": "forexsb", "params": {
                        "indicators": {"VIX_DAY": "vix"}
                    }}
                ],
                "indicators": [],
            },
            "grids": {"FOREX": {"tp": [20], "sl": [20], "ct": [0.55]}},
        }

        json_path = tmp_path / "test_strategy.json"
        with open(json_path, "w") as f:
            json.dump(strategy_data, f)

        config = StrategyConfig.from_json_file(str(json_path))
        dl = config.get_data_loading()
        assert len(dl) == 1
        assert dl[0]["source"] == "forexsb"
