"""Executable SDK and namespaced registry contracts for real plugins."""

import inspect

import pandas as pd
import pytest

from fwbg.pipeline.registry import PluginRegistry


@pytest.fixture(scope="module")
def premium_registry():
    registry = PluginRegistry()
    registry.auto_discover()
    return registry


@pytest.mark.parametrize(
    "fqn",
    [
        "fwbg-premium:macro_data",
        "fwbg-premium:cot_positioning",
        "fwbg-premium:multi_timeframe",
    ],
)
def test_real_premium_plugins_resolve_by_canonical_name(premium_registry, fqn):
    plugin_cls = premium_registry.get(fqn)
    assert plugin_cls.name == fqn.split(":", 1)[1]


@pytest.mark.parametrize(
    "fqn",
    [
        "fwbg-premium:macro_data",
        "fwbg-premium:cot_positioning",
        "fwbg-premium:multi_timeframe",
    ],
)
def test_real_plugins_follow_sdk_classmethod_default_contract(premium_registry, fqn):
    plugin_cls = premium_registry.get(fqn)
    descriptor = inspect.getattr_static(plugin_cls, "get_default_params")
    assert isinstance(descriptor, classmethod)
    assert isinstance(plugin_cls.get_default_params(), dict)


def test_data_loader_getter_accepts_fqn_and_unambiguous_short_name():
    from fwbg.core.registry import get_data_loader
    from fwbg.pipeline.registry import get_registry

    fqn_cls = get_data_loader("fwbg-premium:macro_data")
    canonical = get_registry().get("fwbg-premium:macro_data")
    assert fqn_cls is canonical
    assert get_data_loader("macro_data") is canonical


def test_data_loader_falls_back_to_sdk_registry_when_name_is_ambiguous(monkeypatch):
    from fwbg.core import registry as core_registry
    from fwbg.pipeline import registry as pipeline_registry
    from fwbg_sdk import BaseDataLoader, PluginPhase

    class LegacyLoader(BaseDataLoader):
        name = "shared_loader"
        phase = PluginPhase.DATA_LOADING

        def execute(self, ctx, **params):
            return ctx

    class PipelineLoaderA(BaseDataLoader):
        name = "shared_loader"
        phase = PluginPhase.DATA_LOADING

        def execute(self, ctx, **params):
            return ctx

    class PipelineLoaderB(BaseDataLoader):
        name = "shared_loader"
        phase = PluginPhase.DATA_LOADING

        def execute(self, ctx, **params):
            return ctx

    fake_registry = pipeline_registry.PluginRegistry()
    fake_registry.register(PipelineLoaderA, "package-a")
    fake_registry.register(PipelineLoaderB, "package-b")
    monkeypatch.setattr(pipeline_registry, "get_registry", lambda: fake_registry)
    monkeypatch.setattr(core_registry, "_ensure_plugins_loaded", lambda: None)
    monkeypatch.setitem(core_registry.DATA_LOADER_REGISTRY, "shared_loader", LegacyLoader)

    assert core_registry.get_data_loader("shared_loader") is LegacyLoader


def test_data_loading_orchestrator_executes_fqn_loader(premium_registry, tmp_path):
    from fwbg.core.data_sources import _DATA_SOURCES, register_csv_source
    from fwbg.data.loader import run_data_loading

    csv_path = tmp_path / "VIX_DAY.csv"
    pd.DataFrame(
        {"Date": ["2024-01-01", "2024-01-02"], "Close": [20.0, 21.0]}
    ).to_csv(csv_path, index=False)
    register_csv_source("_contract_macro", tmp_path)
    try:
        frame = pd.DataFrame(
            {"O": 1.0, "H": 1.0, "L": 1.0, "C": 1.0},
            index=pd.date_range("2024-01-01", periods=48, freq="h"),
        )
        result = run_data_loading(
            frame,
            [
                {
                    "name": "fwbg-premium:macro_data",
                    "source": "_contract_macro",
                    "params": {
                        "indicators": {"VIX_DAY": "vix"},
                        "lookbacks_hours": [1],
                        "lookbacks_days": [],
                        "derived_features": [],
                        "interest_rate_diffs": [],
                    },
                }
            ],
        )
        assert "macro_vix_chg_1h" in result.columns
    finally:
        _DATA_SOURCES.pop("_contract_macro", None)


def test_data_loader_execution_errors_are_not_silently_skipped():
    from fwbg.core.registry import DATA_LOADER_REGISTRY, register_data_loader
    from fwbg.data.loader import run_data_loading
    from fwbg_sdk import BaseDataLoader

    @register_data_loader("_contract_error_loader")
    class ContractErrorLoader(BaseDataLoader):
        name = "_contract_error_loader"

        def execute(self, ctx, **params):
            raise RuntimeError("contract execution failure")

    try:
        frame = pd.DataFrame(
            {"O": [1.0], "H": [1.0], "L": [1.0], "C": [1.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        with pytest.raises(RuntimeError, match="contract execution failure"):
            run_data_loading(
                frame,
                [{"name": "_contract_error_loader", "params": {}}],
            )
    finally:
        DATA_LOADER_REGISTRY.pop("_contract_error_loader", None)
