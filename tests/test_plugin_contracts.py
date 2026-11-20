"""Executable SDK and namespaced registry contracts for plugins."""

import inspect

import pandas as pd
import pytest

from fwbg.pipeline.registry import PluginRegistry


@pytest.fixture(scope="module")
def discovered_registry():
    registry = PluginRegistry()
    registry.auto_discover()
    return registry


CORE_FQNS = [
    "fwbg-core:ema",
    "fwbg-core:sma",
    "fwbg-core:adx",
]


@pytest.mark.parametrize("fqn", CORE_FQNS)
def test_real_core_plugins_resolve_by_canonical_name(discovered_registry, fqn):
    plugin_cls = discovered_registry.get(fqn)
    assert plugin_cls.name == fqn.split(":", 1)[1]


@pytest.mark.parametrize("fqn", CORE_FQNS)
def test_real_plugins_follow_sdk_classmethod_default_contract(discovered_registry, fqn):
    plugin_cls = discovered_registry.get(fqn)
    descriptor = inspect.getattr_static(plugin_cls, "get_default_params")
    assert isinstance(descriptor, classmethod)
    assert isinstance(plugin_cls.get_default_params(), dict)


def _make_loader(plugin_name):
    from fwbg_sdk import BaseDataLoader, PluginPhase

    class FakeLoader(BaseDataLoader):
        name = plugin_name
        phase = PluginPhase.DATA_LOADING

        @classmethod
        def get_default_params(cls):
            return {"indicators": {}}

        def execute(self, ctx, **params):
            ctx.df["fake_loader_ran"] = 1.0
            return ctx

    return FakeLoader


@pytest.fixture
def fake_loader_registry(monkeypatch):
    """A registry holding one loader from a package that is not part of fwbg."""
    from fwbg.core import registry as core_registry
    from fwbg.pipeline import registry as pipeline_registry

    loader_cls = _make_loader("fake_loader")
    fake_registry = pipeline_registry.PluginRegistry()
    fake_registry.register(loader_cls, "fake-pkg")
    monkeypatch.setattr(pipeline_registry, "get_registry", lambda: fake_registry)
    monkeypatch.setattr(core_registry, "_ensure_plugins_loaded", lambda: None)
    return fake_registry, loader_cls


def test_data_loader_getter_accepts_fqn_and_unambiguous_short_name(
    fake_loader_registry,
):
    from fwbg.core.registry import get_data_loader

    fake_registry, loader_cls = fake_loader_registry
    fqn_cls = get_data_loader("fake-pkg:fake_loader")
    canonical = fake_registry.get("fake-pkg:fake_loader")
    assert fqn_cls is canonical is loader_cls
    assert get_data_loader("fake_loader") is canonical


def test_data_loader_getter_rejects_fqn_from_missing_package(fake_loader_registry):
    from fwbg.core.registry import get_data_loader

    with pytest.raises(ValueError, match="absent-pkg:fake_loader"):
        get_data_loader("absent-pkg:fake_loader")


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
    monkeypatch.setitem(
        core_registry.DATA_LOADER_REGISTRY, "shared_loader", LegacyLoader
    )

    assert core_registry.get_data_loader("shared_loader") is LegacyLoader


def test_data_loading_orchestrator_executes_fqn_loader(fake_loader_registry):
    from fwbg.data.loader import run_data_loading

    frame = pd.DataFrame(
        {"O": 1.0, "H": 1.0, "L": 1.0, "C": 1.0},
        index=pd.date_range("2024-01-01", periods=48, freq="h"),
    )
    result = run_data_loading(frame, [{"name": "fake-pkg:fake_loader", "params": {}}])
    assert (result["fake_loader_ran"] == 1.0).all()


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
